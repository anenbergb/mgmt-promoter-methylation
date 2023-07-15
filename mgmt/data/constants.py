MODALITIES = ["fla", "t1w", "t1c", "t2w"]
MODALITY2NAME = {
    "fla": "FLAIR",  # "Fluid Attenuated\nInversion Recovery\n(FLAIR)",
    "t1w": "T1-Weighted Pre-contrast (T1w)",
    "t1c": "T1-Weighted Post-contrast (T1Gd)",
    "t2w": "T2-Weighted (T2)",
}
TUMOR_LABELS = {
    0: "Background",
    1: "NCR",
    2: "ED",
    4: "ET",
}
TUMOR_LABELS_LONG = {
    0: "Background",
    1: "Necrotic Tumor Core",
    2: "Peritumoral Edematous Tissue",
    4: "Gd Enhanced Tumor",
}
