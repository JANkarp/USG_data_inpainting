import torch
import torch.nn as nn
from old_load_data import load
from torch.utils.data import TensorDataset, DataLoader
from UNet import UNet
import time
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from plots import plot_val_error, reconstruct_rf, reconstruct_iq, amplitude_vs_phase_plots, amplitude_vs_amplitude_plots, plot_phase, plot_amplitude, plot_recs_extended, phase_mae, get_db

#U-net pipeline used for the older and smaller data set from the first two weeks (PK), works for both rf and IQ
#Along with inpainting it also is able to perform resolution extension in X axis -> from 192 channels to 384

def old_program_loop():

    print('Do you want to retrain the model? y/n')
    train_flag = input()

    print('Input the variables')
    print('Inpainting parameter. Every param-th column will be zeroed in an input')
    param = int(input())
    print('Batch size')
    batch_size = int(input())
    print('Number of features on the first layer of the U-net (32 or 64)')
    n_features = int(input())
    print('Decimation factor')
    decimate = int(input())

    print('Load RF or IQ data? r/i')
    mode_flag = input()
    if mode_flag == 'r':
        mode = 'rf'
    else:
        mode = 'IQ'

    if mode == 'rf':
        model = UNet(in_channels=1, out_channels=1,
                     features=[n_features, n_features * 2, n_features * 4, n_features * 8])
        decimate = 2
    else:
        model = UNet(in_channels=2, out_channels=2)

    train_X, val_X, test_X, train_Y, val_Y, test_Y, extended_X, test_freq, data_mean, data_std = load('usg_data', param,
                                                                                                      mode, decimate)

    # Uses CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = MSLAELoss()
    MSEcriterion = torch.nn.MSELoss()
    MAEcriterion = torch.nn.L1Loss()

    if train_flag == 'y':

        print(f'Used device: {device}')

        train_dataset = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        val_dataset = TensorDataset(val_X, val_Y)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

            for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):

                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                optimizer.zero_grad()

                predictions = model(batch_X)

                if mode == 'rf':
                    loss = criterion(predictions, batch_Y)
                else:
                    loss_I = criterion(predictions[:, 0, :, :], batch_Y[:, 0, :, :])
                    loss_Q = criterion(predictions[:, 1, :, :], batch_Y[:, 1, :, :])
                    loss = loss_I + loss_Q

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx + 1}/{len(train_loader)}')

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch evaluation [{epoch}] -> Average MSLAE: {avg_epoch_loss:.6f}\n")
            loss_history.append(avg_epoch_loss)

            model.eval()
            val_epoch_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    val_predictions = model(batch_X)
                    if mode == 'rf':
                        val_loss = criterion(val_predictions, batch_Y)
                    else:
                        val_loss_I = criterion(val_predictions[:, 0, :, :], batch_Y[:, 0, :, :])
                        val_loss_Q = criterion(val_predictions[:, 1, :, :], batch_Y[:, 1, :, :])
                        val_loss = val_loss_I + val_loss_Q

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

            if epoch == 300:
                print(f"\n[Epoch limit] Stopped training in epoch {epoch}!")
                break

        plot_val_error(loss_history, val_loss_history, f'val_error_{mode}_{param}_{batch_size}_{n_features}.png')

        if mode == 'rf':
            torch.save(model.state_dict(), f'unet_ultrasound_model_rf_{param}_{batch_size}_{n_features}.pth')
        else:
            torch.save(model.state_dict(), f'unet_ultrasound_model_IQ_{param}_{batch_size}_{n_features}_{decimate}.pth')
        print("Training finished!")
        print("Model weights successfully saved to disk!")

    else:
        if mode == 'rf':
            model.load_state_dict(
                torch.load(f"unet_ultrasound_model_rf_{param}_{batch_size}_{n_features}.pth", map_location=device))
        else:
            model.load_state_dict(
                torch.load(f'unet_ultrasound_model_IQ_{param}_{batch_size}_{n_features}_{decimate}.pth',
                           map_location=device))

        print("Model loaded and ready for evaluation!")

        print("Generating visual prediction sample...")
        model.eval()
        with torch.no_grad():
            loss_data = []
            for idx in range(test_X.shape[0]):
                sample_input = test_X[idx].unsqueeze(0).to(device)
                sample_target = test_Y[idx].unsqueeze(0).to(device)
                sample_extended = extended_X[idx].unsqueeze(0).to(device)

                sample_prediction = model(sample_input)
                extended_prediction = model(sample_extended)

                if mode == 'rf':
                    MSLAE_loss = criterion(sample_prediction, sample_target).item()
                    MAE_loss = MAEcriterion(sample_prediction, sample_target).item()
                    MSE_loss = MSEcriterion(sample_prediction, sample_target).item()

                    pred_np = sample_prediction.squeeze().cpu().numpy()
                    target_np = sample_target.squeeze().cpu().numpy()
                    input_np = sample_input.squeeze().cpu().numpy()
                    extended_np = extended_prediction.squeeze().cpu().numpy()

                else:
                    MSLAE_loss = (criterion(sample_prediction[:, 0], sample_target[:, 0]) + criterion(sample_prediction[:, 1], sample_target[:, 1])).item()
                    MAE_loss = (MAEcriterion(sample_prediction[:, 0], sample_target[:, 0]) + MAEcriterion(sample_prediction[:, 1], sample_target[:, 1])).item()
                    MSE_loss = (MSEcriterion(sample_prediction[:, 0], sample_target[:, 0]) + MSEcriterion(sample_prediction[:, 1], sample_target[:, 1])).item()

                    pred_np = sample_prediction.cpu().numpy()[0]
                    target_np = sample_target.cpu().numpy()[0]
                    input_np = sample_input.cpu().numpy()[0]
                    extended_np = extended_prediction.cpu().numpy()[0]

                target_img = (target_np * data_std) + data_mean
                pred_img = (pred_np * data_std) + data_mean
                input_img = (input_np * data_std) + data_mean
                extended_img = (extended_np * data_std) + data_mean

                if mode == 'rf':
                    complex_pred = hilbert(pred_img, axis=0)
                    complex_target = hilbert(target_img, axis=0)
                else:
                    complex_input = input_img[0] + 1j * input_img[1]
                    complex_pred = pred_img[0] + 1j * pred_img[1]
                    complex_target = target_img[0] + 1j * target_img[1]
                    complex_extended = extended_img[0] + 1j * extended_img[1]

                amplitude_pred = np.abs(complex_pred)
                amplitude_target = np.abs(complex_target)

                phase_pred = np.angle(complex_pred)
                phase_target = np.angle(complex_target)

                amplitude_error = np.mean((np.abs(amplitude_pred - amplitude_target)) / (np.abs(amplitude_target) + 1e-6) * 100)
                phase_error_map = phase_mae(phase_pred, phase_target)
                phase_error = np.mean(phase_error_map)

                MSLAE_loss_str = f"{MSLAE_loss:.6f}"
                MAE_loss_str = f"{MAE_loss:.6f}"
                MSE_loss_str = f"{MSE_loss:.6f}"

                loss_data.append({
                    "Indeks": idx,
                    "BŁĄD MSLAE": MSLAE_loss_str,
                    "BŁĄD MAE": MAE_loss_str,
                    "BŁĄD MSE": MSE_loss_str,
                    "BŁĄD AMPLITUDY [%]": amplitude_error,
                    "BłĄD FAZY [rad]": phase_error
                })

                current_metrics = {
                    "MSLAE": MSLAE_loss_str,
                    "MAE": MAE_loss_str,
                    "MSE": MSE_loss_str,
                    "AMPLITUDE": amplitude_error,
                    "PHASE": phase_error
                }

                plot_amplitude(amplitude_pred, amplitude_target, idx, current_metrics, f"{mode}/Amplitude/amplitude_{idx}_rf.png")
                plot_phase(phase_pred, phase_target, current_metrics, idx, f"{mode}/Phase/phase_{idx}_rf.png")
                amplitude_vs_phase_plots(amplitude_target, phase_pred, phase_target, idx, mode)
                amplitude_vs_amplitude_plots(amplitude_target, amplitude_pred, idx, mode)

                if mode == 'rf':
                    input_rec = reconstruct_rf(input_img, values_for_reconstruction, param)
                    target_rec = reconstruct_rf(target_img, values_for_reconstruction)
                    pred_rec = reconstruct_rf(pred_img, values_for_reconstruction)
                    extended_rec = reconstruct_rf(extended_img, values_for_reconstruction, 1 / param)
                else:
                    input_rec = reconstruct_iq(complex_input, decimate, test_freq[idx], values_for_reconstruction, param)
                    target_rec = reconstruct_iq(complex_target, decimate, test_freq[idx], values_for_reconstruction)
                    pred_rec = reconstruct_iq(complex_pred, decimate, test_freq[idx], values_for_reconstruction)
                    extended_rec = reconstruct_iq(complex_extended, decimate, test_freq[idx], values_for_reconstruction, 1 / param)

                plot_recs_extended(pred_rec, target_rec, input_rec, extended_rec, f'{mode}/reconstructions/reconstruction_{idx}_{param}.png')

        df = pd.DataFrame(loss_data)
        if mode == 'rf':
            df.to_csv(f'loss_data_rf_{param}_{batch_size}_{n_features}.csv', index=False)
            print(df)
        else:
            df.to_csv(f'loss_data_IQ_{param}_{batch_size}_{n_features}_{decimate}.csv', index=False)
            print(df)

def values_for_reconstruction(frame, param, H, W, head_x, step, decimate = 1):

    """
    Calculates values needed for b-mode reconstruction

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
    c = 1500
    #physical size of the imaged region
    width = 0.04
    height = 0.023

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
    i = t * fs

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


#   UPROSZCZONA WERSJA MSLAE
class MSLAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_log = torch.sign(pred) * torch.log(torch.abs(pred) + 1.0)
        target_log = torch.sign(target) * torch.log(torch.abs(target) + 1.0)
        return self.l1(pred_log, target_log)


if __name__ == '__main__':
    old_program_loop()

