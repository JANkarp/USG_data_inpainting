import torch
import torch.nn.functional as F
from data_loader import USGDataset
from torch.utils.data import DataLoader
from UNet import UNet
import time
import numpy as np
import pandas as pd
from old_pipeline import phase_mae, MSLAELoss
from plots import reconstruct_iq, plot_val_error, plot_phase, plot_recs, amplitude_vs_phase_map, amplitude_vs_amplitude_map, plot_amplitude_extended

# U-net pipeline used with newer and larger data from the last two weeks (PJ), works only with IQ
# requires the data to be loaded first with preprocess.py, adjusted to data containing different beam angles

def new_program_loop(param, batch_size, n_features, decimate, angle_param):
    """Trains and evaluates the U-net model with IQ data

        Args:
            param: Every param-th column has been zeroed, needs to be adjusted to fit what was used in preprocess.py
            batch_size: Batch size used by the U-net model
            n_features: Number of features used by the U-net model on the first layer, doubled every layer, always used 32 or 64
            decimate: Factor by which the IQ image has been decimated, needs to be adjusted to fit what was used in preprocess.py
            angle_param: Every param-th angle has been added to the training set, needs to be adjusted to fit what was used in preprocess.py
        """

    print('Do you want to retrain the model? y/n')
    train_flag = input()

    data_folder = f'data_processed_{angle_param}'

    #create U-net model with 4 layers
    model = UNet(in_channels=2, out_channels=2, features=[n_features, n_features * 2, n_features * 4, n_features * 8])

    # Uses CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = MSLAELoss()

    train_dataset = USGDataset(data_folder, param, split='train')
    test_dataset = USGDataset(data_folder, param, split='test')
    val_dataset = USGDataset(data_folder, param, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    if train_flag == 'y':

        print(f'Used device: {device}')

        n_epochs = 300

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        # halves the learning rate every 5 epochs if the error doesn't decrease
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        patience_early_stop = 15
        best_val_loss = float('inf')
        early_stop_counter = 0
        loss_history = []
        val_loss_history = []
        epoch = 0

        while True:

            start = time.time()
            epoch = epoch + 1
            print(f'--- Training: epoch {epoch} ---')
            model.train()
            epoch_loss = 0.0
            try:
                for batch_idx, (batch_X, batch_Y, _) in enumerate(train_loader):

                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)

                    optimizer.zero_grad()

                    predictions = model(batch_X)

                    loss_I = criterion(predictions[:, 0, :, :], batch_Y[:, 0, :, :])
                    loss_Q = criterion(predictions[:, 1, :, :], batch_Y[:, 1, :, :])
                    loss = loss_I + loss_Q

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()

                    if batch_idx % 10 == 0:
                        print(f'Batch {batch_idx + 1}/{len(train_loader)}')

            except KeyboardInterrupt:
                print("\n" + "=" * 40)
                print("STOPPED TRAINING")
                print("=" * 40)

                torch.save(model.state_dict(),f'New/unet_ultrasound_model_IQ_{param}_{batch_size}_{n_features}_{decimate}_{angle_param}.pth')
                print(f"U-net weighs from {epoch} have been saved")

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch evaluation [{epoch}] -> Average MSLAE: {avg_epoch_loss:.6f}\n")
            loss_history.append(avg_epoch_loss)

            model.eval()
            val_epoch_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y, _ in val_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    val_predictions = model(batch_X)
                    test_loss_I = criterion(val_predictions[:, 0, :, :], batch_Y[:, 0, :, :])
                    test_loss_Q = criterion(val_predictions[:, 1, :, :], batch_Y[:, 1, :, :])
                    val_loss = test_loss_I + test_loss_Q

                    val_epoch_loss += val_loss.item()

            avg_val_loss = val_epoch_loss / len(val_loader)
            print(f'Validation loss: {avg_val_loss:.6f}')
            val_loss_history.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            end = time.time()
            elapsed = end - start
            print(f'Epoch lasted: {elapsed:.6f} seconds')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience_early_stop:
                    print(f"\n[Early Stopping] Stopped training in epoch {epoch}!")
                    break

            if epoch == n_epochs:
                print(f"\n[Epoch limit] Stopped training in epoch {epoch}!")
                break

        plot_val_error(loss_history, val_loss_history, f'val_error_new_{param}_{n_epochs}_{batch_size}_{n_features}_{decimate}.png')

        torch.save(model.state_dict(), f'New/unet_ultrasound_model_IQ_{param}_{batch_size}_{n_features}_{decimate}_{angle_param}.pth')
        print("Training finished!")
        print("Model weights successfully saved to disk!")

    else:
        model.load_state_dict(torch.load(f'New/unet_ultrasound_model_IQ_{param}_{batch_size}_{n_features}_{decimate}_{angle_param}.pth', map_location=device))
        print("Preparing plots")
        model.eval()

        loss_data = []
        center_freq = 6e6
        with torch.no_grad():
            for idx in range(0,len(test_dataset),4):
                if idx > 100:
                    break
                print(f"Image {idx+1} is being processed")
                sample_input, sample_target, max_val = test_dataset[idx]
                scale = max_val.item()

                sample_input_b = sample_input.unsqueeze(0).to(device)
                sample_target_b = sample_target.unsqueeze(0).to(device)

                sample_prediction = model(sample_input_b)

                pred_np = sample_prediction.cpu().numpy()[0]
                target_np = sample_target_b.cpu().numpy()[0]
                input_np = sample_input_b.cpu().numpy()[0]

                pred_img = pred_np * scale
                target_img = target_np * scale
                input_img = input_np * scale

                print('computing loss functions')
                loss_mslae = (criterion(sample_prediction[:, 0], sample_target_b[:, 0]) +
                              criterion(sample_prediction[:, 1], sample_target_b[:, 1])).item()

                loss_mae = (F.l1_loss(sample_prediction[:, 0], sample_target_b[:, 0]) +
                            F.l1_loss(sample_prediction[:, 1], sample_target_b[:, 1])).item()

                loss_mse = (F.mse_loss(sample_prediction[:, 0], sample_target_b[:, 0]) +
                            F.mse_loss(sample_prediction[:, 1], sample_target_b[:, 1])).item()

                MSLAE_loss_str = f"{loss_mslae:.6f}"
                MAE_loss_str = f"{loss_mae:.6f}"
                MSE_loss_str = f"{loss_mse:.6f}"

                complex_input = input_img[0] + 1j * input_img[1]
                complex_pred = pred_img[0] + 1j * pred_img[1]
                complex_target = target_img[0] + 1j * target_img[1]

                amplitude_input = np.abs(complex_input)
                amplitude_pred = np.abs(complex_pred)
                amplitude_target = np.abs(complex_target)

                phase_pred = np.angle(complex_pred)
                phase_target = np.angle(complex_target)

                amplitude_error = np.mean((np.abs(amplitude_pred - amplitude_target)) / (np.abs(amplitude_target) + 1e-6) * 100)
                phase_error_map = phase_mae(phase_pred, phase_target)
                phase_error = np.mean(phase_error_map)


                loss_data.append({
                    "Indeks": idx,
                    "BŁĄD MSLAE": MSLAE_loss_str,
                    "BŁĄD MAE": MAE_loss_str,
                    "BŁĄD MSE": MSE_loss_str,
                    "BŁĄD AMPLITUDY [%]": amplitude_error,
                    "BŁĄD FAZY [rad]": phase_error
                })

                current_metrics = {
                    "MSLAE": MSLAE_loss_str,
                    "MAE": MAE_loss_str,
                    "MSE": MSE_loss_str,
                    "AMPLITUDE": amplitude_error,
                    "PHASE": phase_error
                }

                print('Plotting amplitude')
                plot_amplitude_extended(amplitude_pred, amplitude_target, amplitude_input, idx, current_metrics)
                print('Plotting phase')
                plot_phase(phase_pred, phase_target, idx, current_metrics, f"New/Phase/phase_{idx}_IQ.png")
                print('Plotting amplitude vs phase')
                amplitude_vs_phase_map(amplitude_target, phase_pred, phase_target, idx)
                print('Plotting amplitude vs amplitude')
                amplitude_vs_amplitude_map(amplitude_target, amplitude_pred, idx)

                print('Reconstructing images')
                input_rec = reconstruct_iq(complex_input, decimate, center_freq, values_for_reconstruction, param)
                target_rec = reconstruct_iq(complex_target, decimate, center_freq, values_for_reconstruction)
                pred_rec = reconstruct_iq(complex_pred, decimate, center_freq, values_for_reconstruction)

                plot_recs(pred_rec, target_rec, input_rec, f'New/reconstructions/iq_reconstructions_{idx}_{param}.png')

        df = pd.DataFrame(loss_data)

        df.to_csv(f'New/loss_data_IQ_{param}_{batch_size}_{n_features}_{decimate}_{angle_param}.csv', index=False)
        print(df)

def values_for_reconstruction(frame, param, H, W, head_x, step, decimate = 1):
    """
    Calculates values needed for b-mode reconstruction, adjusted to the new pipeline's higher imaging depth

    :param frame: a frame with iq data
    :param param: only needed for an input image with missing columns
    :param H: pixel height
    :param W: pixel width
    :param head_x: number of used columns/used detectors
    :param step: every step-th column used for reconstruction (skips columns in an image with the line zeroed out)
    :param decimate: decimation factor
    :return: reconstructed image
    """

    fs = (65 * 1e6)/decimate
    c = 1480
    #physical size of the imaged region
    width = 0.04
    height = 0.075

    #width of a single detector
    det_width = 0.000245 * param
    #physical size of a pixel
    pix_len_x = width / W
    pix_len_z = height / H
    #padding to align the head with the image (move image slightly to the right)
    pad = ((head_x - 1) * det_width - (W - 1) * pix_len_x) / 2

    #distance values in every needed dimension
    z_vals = (np.arange(H) * pix_len_z)[:, np.newaxis, np.newaxis]
    x_vals = (np.arange(W) * pix_len_x + pad)[np.newaxis, :, np.newaxis]
    det_vals = (np.arange(head_x) * det_width)[np.newaxis, np.newaxis, :]


    # distances from the detector
    rtx = z_vals
    rrt = np.sqrt(z_vals ** 2 + (det_vals - x_vals) ** 2)

    t = (rtx + rrt)/c
    i = t * fs - 350 / decimate

    #index closest to calculated i
    i_0 = np.floor(i).astype(np.int32)
    i_1 = i_0 + 1
    #weight for interpolation from two nearset pixels
    w = i - i_0

    #mask for viable indexes (avoids index out of range)
    mask_0 = i_0 < frame.shape[0]
    mask_1 = i_1 < frame.shape[0]

    #indices of columns with detectors, max index will be shape[0] from frame, with the step equal to the param
    cols_1d = np.arange(head_x) * step

    #convert to 3d with correct dimensions
    cols = np.broadcast_to(cols_1d[np.newaxis, np.newaxis, :], (H, W, head_x))

    return cols, mask_0, mask_1, i_0, i_1, w, t

if __name__ == '__main__':
    new_program_loop(2, 8, 32, 8, 3)