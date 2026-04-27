import numpy as np

file1 = "/home/matheus/src/datasets/RayWise/Raymobtime_s008/baseline_data/beam_output/beams_output_train.npz"
file2 = (
    "/home/matheus/src/datasets/RayWise/Raymobtime_s008/baseline_data/beam_output/beams_output_validation.npz"
)
output_file = "beams_output_s008.npz"


def summarize(name, arr):
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}"
    )


with np.load(file1, mmap_mode="r") as d1, np.load(file2, mmap_mode="r") as d2:
    merged = {}
    all_keys = set(d1.keys()) | set(d2.keys())
    for k in all_keys:
        in1 = k in d1
        in2 = k in d2
        if in1 and in2:
            a, b = d1[k], d2[k]
            if a.shape[1:] != b.shape[1:]:
                raise ValueError(f"Incompatible shapes for {k}: {a.shape} vs {b.shape}")
            dt = np.result_type(a.dtype, b.dtype)
            merged[k] = np.concatenate([a.astype(dt, copy=False), b.astype(dt, copy=False)], axis=0)
        elif in1:
            merged[k] = d1[k]
        else:
            merged[k] = d2[k]

for k, v in merged.items():
    summarize(f"merged[{k}]", v)

np.savez(output_file, **merged)
print(f"Saved to {output_file}")
