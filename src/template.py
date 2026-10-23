import numpy as np

TARGET_TEMPLATE_FRAMES = 100

def resample_frames(feats, target_frames=TARGET_TEMPLATE_FRAMES):
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError(f"Fitur harus berbentuk (T, dimensi) dengan T >= 1, dapat {feats.shape}")
    if len(feats) == target_frames:
        return feats.copy()
    source = np.linspace(0.0, 1.0, len(feats))
    target = np.linspace(0.0, 1.0, target_frames)
    return np.column_stack([np.interp(target, source, feats[:, d]) for d in range(feats.shape[1])])

def build_averaged_template(feats_list, target_frames=TARGET_TEMPLATE_FRAMES):
    feats_list = list(feats_list)
    if not feats_list:
        raise ValueError("feats_list kosong: tidak ada rekaman untuk dirata-rata")
    if target_frames < 2:
        raise ValueError("target_frames minimal 2")
    dims = {np.asarray(f).shape[-1] for f in feats_list}
    if len(dims) != 1:
        raise ValueError(f"Dimensi fitur antar rekaman berbeda: {sorted(dims)}")
    return np.mean([resample_frames(f, target_frames) for f in feats_list], axis=0)

def build_templates(feats_by_vowel, target_frames=TARGET_TEMPLATE_FRAMES):
    templates = {}

    for vowel, feats_list in feats_by_vowel.items():
        template = build_averaged_template(feats_list, target_frames)
        templates[vowel] = template

    return templates