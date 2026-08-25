import glob
import numpy as np
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def process_npz_files(input_dir, output_dir,
                      z_noise_std=1.0,
                      color_noise_std=0.03,
                      random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    print(f"噪声参数：z_std={z_noise_std} mm, color_std={color_noise_std}, seed={random_seed}")
    os.makedirs(output_dir, exist_ok=True)

    for npz_file in tqdm(npz_files, desc="点云加噪中", colour='blue'):
        data = np.load(npz_file)
        noisy_data = {}
        for key in data.files:
            # if key in ('mask_point1', 'mask_point2'):
            if key in ('point1', 'point2'):
                pts = data[key].copy()
                # z 轴加高斯噪声（单位：mm）
                noise = np.random.normal(0, z_noise_std, size=pts.shape[0])
                pts[:, 2] += noise
                noisy_data[key] = pts
            elif key in ('color1', 'color2'):
                cols = data[key].copy()
                cols += np.random.normal(0, color_noise_std, size=cols.shape)
                noisy_data[key] = np.clip(cols, 0, 1)
            else:
                noisy_data[key] = data[key]

        out_path = os.path.join(output_dir, os.path.basename(npz_file))
        np.savez(out_path, **noisy_data)


if __name__ == '__main__':
    # input_dir  = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/test1"
    # base_out   = "/big_data/szm/H8amlyn_8192_Mask_3332_new_mutual_85/test1_noise"
    # input_dir = "/home/szm/Paconv_730/duibi_keshihua_huaxi_noise/322-332/origin"
    # base_out = "/home/szm/Paconv_730/duibi_keshihua_huaxi_noise/322-332/noise"
    input_dir = "/home/szm/Paconv_730/duibi_keshihua_stereo_noise/P2-5_192-377/origin"
    base_out = "/home/szm/Paconv_730/duibi_keshihua_stereo_noise/P2-5_192-377/noise"
    z_levels   = [0.5, 1.0, 2.0]      # mm
    c_levels   = [0.03, 0.06, 0.10]   # RGB noise
    # z_levels = [4.0]  # mm
    # c_levels = [0.2]  # RGB noise
    for z_std in z_levels:
        for c_std in c_levels:
            cfg = {
                "input_dir":      input_dir,
                "output_dir":     f"{base_out}/z{z_std}_c{c_std}",
                "z_noise_std":    z_std,
                "color_noise_std":c_std,
                "random_seed":    42
            }
            process_npz_files(**cfg)
