import os
import glob
import pickle
import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt


def process_and_save_dataset(raw_folder, decimate, angle_param):
    output_folder = f'data_processed_{angle_param}'

    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)

    all_files = glob.glob(os.path.join(raw_folder, '*.pkl'))

    train_count = 0
    test_count = 0
    val_count = 0

    np.random.seed(42)

    n = 0
    mean = 0.0
    M2 = 0.0
    print(f"Starting preprocessing for {len(all_files)} files...")

    for file_idx, file_path in enumerate(all_files):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for frame_idx in range(3):

            raw_rf = data['data'][frame_idx][1][0]

            for angle_idx in range(64):
                Y_2d = raw_rf[angle_idx]
                Y_2d = Y_2d[700:-301]

                comp_Y = convert_to_iq(Y_2d, decimate)

                i_channel = np.real(comp_Y)
                q_channel = np.imag(comp_Y)
                iq_data = np.stack([i_channel, q_channel], axis=0)


                is_train = (angle_idx % angle_param == 0)

                if is_train:
                    x = iq_data.ravel().astype(np.float64)
                    n_b = x.size
                    mean_b = x.mean()
                    var_b = x.var()

                    delta = mean_b - mean
                    total_n = n + n_b
                    mean += delta * n_b / total_n
                    M2 += var_b * n_b + delta**2 * n * n_b / total_n
                    n = total_n


                Y_tensor = torch.from_numpy(iq_data).float()

                if angle_idx % angle_param == 0:
                    save_path = os.path.join(output_folder, 'train', f'sample_{train_count:05d}.pt')
                    train_count += 1
                else:
                    if np.random.rand() >= 0.9:
                        save_path = os.path.join(output_folder, 'val', f'sample_{val_count:05d}.pt')
                        val_count += 1
                    else:
                        save_path = os.path.join(output_folder, 'test', f'sample_{test_count:05d}.pt')
                        test_count += 1

                torch.save(Y_tensor, save_path)

        print(f"[{file_idx + 1}/{len(all_files)}] Processed {os.path.basename(file_path)}")

    global_std = (M2 / n) ** 0.5
    stats = {'mean': float(mean), 'std': float(global_std)}
    print(f"\nGlobal train stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    # save alongside the processed data
    with open(os.path.join(output_folder, 'norm_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nDone! Saved {train_count} train samples and {test_count} test samples to '{output_folder}'.")


def convert_to_iq(data, decimate):
    fs = 65e6
    ft = 6e6

    H = data.shape[0]
    t = np.arange(H) / fs
    t_block = t[:, np.newaxis]

    data_cos = data * np.cos(2 * np.pi * ft * t_block)
    data_sin = data * np.sin(2 * np.pi * ft * t_block)

    Wn = 0.8 * ft
    sos = butter(5, Wn, 'lowpass', analog=False, fs=fs, output='sos')

    filtered_i = 2 * sosfiltfilt(sos, data_cos, axis=0)
    filtered_q = 2 * sosfiltfilt(sos, data_sin, axis=0)

    decimated_i = filtered_i[0::decimate, :]
    decimated_q = filtered_q[0::decimate, :]

    return decimated_i + 1j * decimated_q


if __name__ == '__main__':
    process_and_save_dataset(
        raw_folder='usg_string_data',
        decimate=8,
        angle_param = 5
    )