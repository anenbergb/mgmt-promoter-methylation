

I'm just using a "regular" CNN architecture without any fancy motifs. Let conv1 represent a standard `3x3x3` convolution, and conv2 represent a standard `3x3x3` stride-2 convolution. Then the model backbone is really just:
```
block 1 ==> conv1 - conv1 - conv1
block 2 ==> conv2 - conv1 - conv1
block 3 ==> conv2 - conv1 - conv1
block 4 ==> conv2 - conv1 - conv1
block 5 ==> conv2 - conv1 - conv1
```

This is what I will call the base `3-3-3-3-3` model (e.g., 3 conv's per block). I also play around with a `4-4-3-2-2` combination (more learning in the earlier layers tends to be slightly better). The feature maps / channel depth progress from 16 (first block) to 112 (fifth block) with a linear growth of 24 in each block.

The key is the deep supervision heads in the create_multires(...) method. Here, for each feature map (the final map per block), I am applying: 
* feature map x tumor mask (to mask out the non-tumor features, with a subsampled tumor mask as needed)
* global pool (average pooling here seems to work a bit better than max pooling)
* dense layer to a 1-element logit score for prediction

Because there are 5 blocks / feature maps, I have five separate heads / predictions.

During training, I keep track of my performance individually for each head, and independently save the best performing checkpoints for different heads. Then I also train multiple ensembles of the same model. At the very end, I will choose to combine the top `N` combinations of models and heads (e.g., any one model may contribute one or several best performing heads).
Finally I am using a `0.99` EMA decay for LR scheduling (decayed each epoch).

```python
import os, glob, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, metrics, optimizers, callbacks, mixed_precision
from jarvis.train.client import Client
from jarvis.train import models, params
from jarvis.utils import arrays as jars
from jarvis.utils.general import gpus, cupy_compatible
from sklearn.metrics import roc_auc_score

def create_model(inputs, p, **kwargs):
    """
    Method to create model backbone

    """
    # --- Inputs
    inputs = get_inputs(client)
    x = layers.Concatenate()([inputs[k] for k in get_input_keys(p)])
    y = inputs['tum']

    # --- Create blocks
    blocks = create_convs(x=x, p=p)

    # --- Create logits, losses, metric
    logits, losses, metric = create_multires(inputs, blocks, tumor=y, p=p, **kwargs)

    # --- Create models 
    backbone = Model(inputs=inputs, outputs=logits)
    training = Model(inputs=inputs, outputs={**logits, **metric})

    # =======================================================================
    # ADD LOSSES & METRICS
    # =======================================================================

    for k, val in losses.items():
        training.add_loss(val)

    for k, val in metric.items():
        training.add_metric(val, name=k)

    # --- Compile the model
    training.compile(optimizer=optimizers.Adam(learning_rate=p.get('LR')))

    return backbone, training 

def get_inputs(client, suffix=''):

    specs = client.get_specs()

    return {k: Input(
        shape=specs['xs'][k]['shape'],
        dtype=specs['xs'][k]['dtype'],
        name=k + suffix) for k in specs['xs']}

def create_multires(inputs, blocks, tumor, p, **kwargs):

    pool = {
        'avg': layers.GlobalAveragePooling3D(),
        'max': layers.GlobalMaxPooling3D(),
        'flt': layers.Flatten()}[p['pool']]

    logits = {}

    # --- Standard supervision
    logits['mgmt'] = pool(last(blocks))
    logits['mgmt'] = layers.Dense(1)(logits['mgmt'])
    ll, mm = create_losses_metric(y_true=inputs['lbl'], y_pred=logits['mgmt'], k='mgmt')

    tumor = tf.cast(tumor, tf.float32)

    # --- Deep supervision + mask
    for k, block in blocks.items():

        # --- Subsample tumor
        if int(k[1:]) > 1:
            tumor = layers.MaxPooling3D(2, dtype='float32')(tumor)

        # --- Create logits
        logits[k] = tf.cast(block, tf.float32) * tumor
        logits[k] = pool(logits[k]) 
        logits[k] = layers.Dense(1)(logits[k])

        # --- Create losses, metric
        ll_, mm_ = create_losses_metric(y_true=inputs['lbl'], y_pred=logits[k], k=k)
        ll.update(ll_)
        mm.update(mm_)

    return logits, ll, mm 

def last(x):

    return list(x.values())[-1]

def create_losses_metric(y_true, y_pred, k):

    ll, mm = {}, {}
        
    # --- Create losses
    ll['bce-{}'.format(k)] = losses.BinaryCrossentropy(from_logits=True, label_smoothing=p['label_smoothing'])(
        y_true=y_true,
        y_pred=y_pred)

    # --- Create accuracy
    mm['acc-{}'.format(k)] = metrics.binary_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        threshold=0)

    # --- Create auc
    mm['auc-{}'.format(k)] = create_auc(
        y_true=y_true,
        y_pred=y_pred) 

    return ll, mm

def create_auc_pseudo(y_true, y_pred):

    ss = tf.nn.sigmoid(y_pred)

    return tf.math.reduce_mean(tf.where(y_true == 1, ss, 1 - ss))

def create_auc(y_true, y_pred):

    return tf.py_function(calculate_auc, (y_true, y_pred), tf.double)

def calculate_auc(y_true, y_pred):

    try:
        score = roc_auc_score(y_true.ravel(), y_pred.ravel())
    except:
        score = 0.5
    finally:
        return score

def create_convs(x, p, pattern='_l', **kwargs):
    """
    Method to create standard conv pattern:

      blocks['l1'] = conv1(f1, ..., conv1(f1, ..., x))
      blocks['l2'] = conv1(f2, ..., conv2(f2, ..., x))
      blocks['l3'] = conv1(f3, ..., conv2(f3, ..., x))
      ...

    Variables in hyperparameters (p) dictionary:

      _l0, _l1, _l2 ... : number of convolutions per block
      _fs_null          : starting filter size (f1) 
      _fs_rate          : increase filter size rate per block 
      
    """
    # --- Get blocks
    conv1, conv2 = models.create_blocks(('conv1', 'conv2'), dims=3,
        gn=p.get('gn', False), groups=p.get('groups', 8), **kwargs)

    # --- Define blocks params
    blocks = {}
    output = x
    for k, v in p.items():
        if k[:len(pattern)] == pattern and type(v) is int:
            if v > 0:

                # --- Define filter channel depth
                filters = p['_fs_null'] * p['_fs_rate'] ** len(blocks) if type(p['_fs_rate']) is float else \
                          p['_fs_null'] + p['_fs_rate'] *  len(blocks)

                # --- Define block
                if len(blocks) > 0:
                    output = conv2(filters, output)
                    v -= 1

                for i in range(v):
                    output = conv1(filters, output)

                # --- Define dropout
                if p['drop_rate'] > 0 and filters > p['drop_start']:
                    output = layers.Dropout(p['drop_rate'])(output)

                name = k.replace(pattern, pattern.replace('_', ''))
                blocks[name] = output

    return blocks

def prepare_client(p):

    # --- Prepare client
    client = Client(p['_client'], configs={
        'batch': {
            'size': p['batch_size'],
            'fold': -1}},
        load=jars.create)

    # --- Prepare inputs
    for key in ['t2w', 'fla', 't1w', 't1c']:
        if not p[key]:
            client.specs['xs'].pop(key, None)

    return client

def load_data(client, p, cohort='train'):

    # --- Load into memory
    client.load_data_in_memory()

    # --- Create folds
    np.random.seed(0)
    folds = np.random.permutation(client.db.fnames.shape[0]) % 3
    train = np.array(client.db.header['cohort-{}'.format(cohort)])

    if cohort == 'train':

        # --- Prepare train
        data_train = load_data_by_fold(client, mask=(folds != p['fold']) & train)

        # --- Prepare valid
        data_valid = load_data_by_fold(client, mask=(folds == p['fold']) & train)

        return data_train, data_valid

    if cohort == 'test':

        return load_data_by_fold(client, mask=train)

def load_data_by_fold(client, mask):

    keys = get_input_keys(p)
    data = {k: [] for k in keys} 
    data['tum'] = []
    data['lbl'] = []

    for sid, fnames, header in client.db.cursor(mask=mask):

        for key in keys:

            # --- Extract mu/sd 
            mu = header['{}-096-mu'.format(key)]
            sd = header['{}-096-sd'.format(key)]

            # --- Extract and normalize 
            dd = client.data_in_memory.pop(fnames['{}-096'.format(key)]).data
            dd = (dd - mu) / sd

            data[key].append(dd.clip(min=-4, max=+4))

        # --- Extract tumor
        dd = client.data_in_memory.pop(fnames['tum-096']).data
        data['tum'].append(dd)
            
        # --- Extract MGMT
        data['lbl'].append([header['mgmt']])

    for key in data:
        data[key] = np.stack(data[key], axis=0)

    return data

def get_input_keys(p):

    return [k for k in ['t2w', 'fla', 't1w', 't1c'] if p[k]]

def evaluate_ensemble(data_valid, exps='exp-0[0]', include_only=None, threshold=0.6, keys=['l1', 'l2', 'l3', 'l4', 'l5', 'mgmt'], backbone=None):

    saveds = []
    logits = []
    aucs = []

    for exp in sorted(glob.glob('../{}*'.format(exps))):
        backbone = None
        for k in keys:
            saved = sorted(glob.glob('{}/{}/*.hdf5'.format(exp, k)))[-1]
            if (saved in include_only if include_only is not None else True):
                if float(saved.split('-')[-1][:-5]) >= threshold:

                    if backbone is None:
                        print('=======================================================')
                        print('Loading model: {}'.format(exp))
                        print('=======================================================')

                        # --- Load backbone
                        backbone = glob.glob('{}/hdf5/backbone/model_000.hdf5'.format(exp))[0] 
                        backbone = tf.keras.models.load_model(backbone, compile=False)

                    # --- Run evaluation
                    saveds.append(saved)
                    backbone.load_weights(saved)
                    y = predict(backbone, data_valid)
                    logits.append(y[k])

                    if data_valid['lbl'].min() == 0:
                        auc = calculate_auc(
                            y_true=data_valid['lbl'],
                            y_pred=logits[-1])
                        print('{}: {:0.5f}'.format(saved.ljust(40), auc))
                        aucs.append(auc)
                    else:
                        print('Created logits: {}'.format(saved))

    aucs = np.array(aucs)
    inds = np.argsort(aucs)[::-1]

    logits = np.concatenate(logits, axis=-1)

    print('=======================================================')
    print('Creating ensemble')
    print('=======================================================')

    for n in range(inds.size):

        l = np.mean(sigmoid(logits[:, inds[:n+1]]), axis=-1)
        if data_valid['lbl'].min() == 0:
            print('Ensemble (top-{}): {:0.5f}'.format(n + 1, calculate_auc(
                y_true=data_valid['lbl'],
                y_pred=l)))

    return logits, [saveds[i] for i in inds] 

def predict(backbone, data, splits=3):

    logits = {} 

    inds = np.linspace(0, data['lbl'].size, splits + 1, endpoint=True)
    inds = np.round(inds).astype('int')

    for lo, hi in zip(inds[:-1], inds[1:]):

        y = backbone.predict({k: v[lo:hi] for k, v in data.items()})
        for k, l in y.items():
            if k not in logits:
                logits[k] = []
            logits[k].append(l)

    return {k: np.concatenate(v, axis=0) for k, v in logits.items()} 

def sigmoid(arr):

    return 1 / (1 + np.exp(-arr.clip(min=-10, max=+10)))

if __name__ == '__main__':

    # --- Autoselect GPU
    gpus.autoselect(memory_limit=10000)

    # --- Prepare hyperparams
    p = params.load('./hyper.csv', row=-1)

    # --- Prepare model
    client = prepare_client(p)
    backbone, training = create_model(client, p)

    # --- Prepare data 
    data_train, data_valid = load_data(client, p)

    # --- Learning rate scheduler
    lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr : lr * p['LR_decay'])
    cb = [lr_scheduler]

    # --- Weight standardization
    if p.get('gn', False):
        cb.append(models.WeightStandardization())

    # --- Checkpoint
    cb += [callbacks.ModelCheckpoint(
        filepath='{}/{}/model'.format(p['output_dir'], key) + '-{epoch:03d}-{val_acc-' + key + ':.3f}.hdf5',
        monitor='val_acc-' + key,
        mode='max',
        save_best_only=True) for key in ['l1', 'l2', 'l3', 'l4', 'l5', 'mgmt']] 

    # --- Train
    models.train(
        x=data_train,
        batch_size=p['batch_size'],
        steps_per_epoch=int(data_train['lbl'].size / p['batch_size']),
        validation_data=data_valid,
        validation_steps=None,
        validation_batch_size=int(data_valid['lbl'].size / 3),
        validation_freq=1,
        model=training,
        graphs={'backbone': backbone},
        client=client,
        epochs=p['epochs'],
        steps=None,
        save_freq=50,
        output_dir=p['output_dir'],
        callbacks=cb)

```