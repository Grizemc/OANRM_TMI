import numpy as np

file_paths = [
    "/code_data/cwz_szm/DeformationPyramid/checkpoints/LND_Hamlyn_85/correspondence_miccai_color/eval.npy",
    "/code_data/cwz_szm/DeformationPyramid/checkpoints/LND_Hamlyn_85RT/correspondence_miccai_color/eval.npy",
]

for file_path in file_paths:
    print("=" * 100)
    print("Loading:", file_path)
    try:
        data = np.load(file_path, allow_pickle=True)
        print("type :", type(data))
        print("shape:", getattr(data, "shape", None))
        print("dtype:", getattr(data, "dtype", None))
        print("content:")
        print(data)

        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                obj = data.item()
                print("\nitem():")
                print(obj)
            except Exception:
                pass

    except Exception as e:
        print("Failed to load:", e)