from INR import SIREN
from preprocess import convert_to_iq
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import torch
import torch.nn as nn
from old_pipeline import MSLAELoss
from new_pipeline import values_for_reconstruction
from plots import reconstruct_iq, plot_val_error, plot_phase, plot_amplitude, plot_recs, plot_bmode_results, phase_mae
from scipy.signal import hilbert

def main(param, decimate, hidden_layers, hidden_features, hidden_omega, first_omega, patience, lr):

    file_path = 'usg_string_data/w_1.pkl'
    center_freq = 6e6

    print('Do you want to retrain the model? y/n')
    train_flag = input()

    print('Do you want to use rf, bmode or iq? r/b/i')
    data_flag = input()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')

    if data_flag == 'r':
        Y, mean, std = get_rf_data(file_path)

        model = SIREN(
            in_features=2,
            out_features=1,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega=first_omega,
            hidden_omega=30
        ).to(device)

        criterion = MSLAELoss()

    elif data_flag == 'b':
        Y, mean, std = get_bmode_data(decimate, file_path, center_freq)

        model = SIREN(in_features=2,
            out_features=1,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega=60,
            hidden_omega=30
        ).to(device)

        criterion = nn.L1Loss()
    else:

        Y, mean, std = get_iq_data(decimate, file_path)

        model = SIREN(
            in_features=2,
            out_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega=first_omega,
            hidden_omega=hidden_omega
        ).to(device)

        criterion = MSLAELoss()

    mask = torch.zeros(Y.shape[-2:], dtype=torch.bool)
    mask[:, ::param] = True

    train_dataset = PixelDataset(Y, mask, data_flag)
    train_loader = DataLoader(train_dataset, batch_size=int(len(train_dataset)/8), shuffle=True)

    test_dataset = PixelDataset(Y, None, data_flag)
    test_loader = DataLoader(test_dataset, batch_size=int(len(test_dataset)/8), shuffle=False)


    # training
    if train_flag == 'y':

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)

        n_epochs = 1000
        loss_history = []
        val_loss_history = []


        print("--- Starting INR Training ---")
        try:
            for epoch in range(n_epochs):
                model.train()
                running_train_loss = 0
                for coords_batch, targets_batch in train_loader:
                    coords_batch, targets_batch = coords_batch.to(device), targets_batch.to(device)

                    optimizer.zero_grad()
                    preds = model(coords_batch)
                    loss = criterion(preds, targets_batch)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * coords_batch.size(0)

                epoch_train_loss = running_train_loss / len(train_dataset)
                loss_history.append(epoch_train_loss)

                model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for coords_batch, targets_batch in test_loader:
                        coords_batch, targets_batch = coords_batch.to(device), targets_batch.to(device)
                        preds = model(coords_batch)
                        loss = criterion(preds, targets_batch)
                        running_val_loss += loss.item() * coords_batch.size(0)

                    epoch_val_loss = running_val_loss / len(test_dataset)
                    val_loss_history.append(epoch_val_loss)

                scheduler.step(epoch_val_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1:03d}/{n_epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        except KeyboardInterrupt:
            print("\n" + "=" * 40)
            print("STOPPED TRAINING")
            print("=" * 40)

            if data_flag == 'r':
                torch.save(model.state_dict(),f'INR/INR_ultrasound_model_rf_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')
            elif data_flag == 'b':
                torch.save(model.state_dict(),f'INR/INR_ultrasound_model_bmode_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')
            else:
                torch.save(model.state_dict(),f'INR/INR_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')

            print(f"INR weighs from {epoch} have been saved")
            plot_val_error(loss_history, val_loss_history, f'INR/val_error_INR_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.png')

        plot_val_error(loss_history, val_loss_history, f'INR/val_error_INR_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.png')

        if data_flag == 'r':
            torch.save(model.state_dict(),f'INR/INR_ultrasound_model_rf_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')
        elif data_flag == 'b':
            torch.save(model.state_dict(),f'INR/INR_ultrasound_model_bmode_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')
        else:
            torch.save(model.state_dict(),f'INR/INR_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth')
    # evaluation
    else:
        if data_flag == 'r':
            model.load_state_dict(torch.load(f'INR/INR_ultrasound_model_rf_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth', map_location=device))
        elif data_flag == 'b':
            model.load_state_dict(torch.load(f'INR/INR_ultrasound_model_bmode_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth',map_location=device))
        else:
            model.load_state_dict(torch.load(f'INR/INR_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.pth', map_location=device))
        print('Model ready for evaluation')

        model.eval()
        reconstructed_preds = []
        total_val_loss = 0
        with torch.no_grad():
            for coords_batch, targets_batch in test_loader:
                coords_batch = coords_batch.to(device)
                targets_batch = targets_batch.to(device)
                preds = model(coords_batch)
                reconstructed_preds.append(preds.cpu())
                loss = criterion(preds, targets_batch)
                total_val_loss += loss.item() * coords_batch.size(0)

            avg_val_loss = total_val_loss / len(test_dataset)
            loss_str = f"{avg_val_loss:.6f}"

        full_pred = torch.cat(reconstructed_preds, dim=0)

        H, W = Y.shape[-2], Y.shape[-1]

        if data_flag == 'r' or data_flag == 'b':
            pred_img = full_pred.reshape(H, W)
        else:
            pred_img = full_pred.reshape(H, W, 2).permute(2, 0, 1)

        pred_img = pred_img.cpu().numpy()
        target_img = Y.cpu().numpy()

        pred_img = (pred_img * std) + mean
        target_img = (target_img * std) + mean

        input_img = np.zeros_like(target_img)

        if data_flag != 'b':
            if data_flag == 'r':
                input_img[:, ::param] = target_img[:, ::param]
                complex_pred = hilbert(pred_img, axis=0)
                complex_target = hilbert(target_img, axis=0)
                complex_input = hilbert(input_img, axis=0)

            else:
                input_img[:, :, ::param] = target_img[:, :, ::param]
                complex_pred = pred_img[0] + 1j * pred_img[1]
                complex_target = target_img[0] + 1j * target_img[1]
                complex_input = input_img[0] + 1j * input_img[1]

            amplitude_target = np.abs(complex_target)
            phase_target = np.angle(complex_target)

            amplitude_pred = np.abs(complex_pred)
            phase_pred = np.angle(complex_pred)

            phase_error_map = phase_mae(phase_pred, phase_target)
            phase_error = np.mean(phase_error_map)

            amplitude_error = np.mean((np.abs(amplitude_pred - amplitude_target)) / (np.abs(amplitude_target) + 1e-6) * 100)

            current_metrics = {
                "MSLAE": loss_str,
                "AMPLITUDE": amplitude_error,
                "PHASE": phase_error
            }

            plot_amplitude(amplitude_pred, amplitude_target, 1, current_metrics, f"INR/Amplitude/amplitude_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.png")
            plot_phase(phase_pred, phase_target,  current_metrics, 1, f"INR/Phase/phase_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.png")
            target_rec = reconstruct_iq(complex_target, decimate, center_freq, values_for_reconstruction)
            pred_rec = reconstruct_iq(complex_pred, decimate, center_freq, values_for_reconstruction)
            input_rec = reconstruct_iq(complex_input, decimate, center_freq, values_for_reconstruction)
            plot_recs(pred_rec, target_rec, input_rec, f'INR/reconstructions/iq_reconstructions_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{lr}.png')
        else:

            plot_bmode_results(pred_img, target_img, loss_str)
            print(loss_str)

def get_bmode_data(decimate, file_path, center_freq):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        raw_rf = data['data'][0][1][0]

        Y_2d = raw_rf[0]
        Y_2d = Y_2d[700:-301]

        comp_Y = convert_to_iq(Y_2d, decimate)

        i_channel = np.real(comp_Y)
        q_channel = np.imag(comp_Y)
        target_img = np.stack([i_channel, q_channel], axis=0)
        complex_target = target_img[0] + 1j * target_img[1]

        target_rec = reconstruct_iq(complex_target, decimate, center_freq, values_for_reconstruction)

        mean = np.mean(target_rec)
        std = np.std(target_rec)

        rec_scaled = (target_rec - mean) / (std + 1e-8)

        rec_tensor = torch.from_numpy(rec_scaled).float()

        return rec_tensor, mean, std

def get_iq_data(decimate, file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        raw_rf = data['data'][0][1][0]

        Y_2d = raw_rf[0]
        Y_2d = Y_2d[700:-301]

        comp_Y = convert_to_iq(Y_2d, decimate)

        i_channel = np.real(comp_Y)
        q_channel = np.imag(comp_Y)
        iq_data = np.stack([i_channel, q_channel], axis=0)

        mean = np.mean(iq_data)
        std = np.std(iq_data)

        iq_scaled = (iq_data - mean) / (std + 1e-8)

        Y_tensor = torch.from_numpy(iq_scaled).float()

        return Y_tensor, mean, std

def get_rf_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        raw_rf = data['data'][0][1][0]

        Y_2d = raw_rf[0]
        Y_2d = Y_2d[700:-301]

        mean = np.mean(Y_2d)
        std = np.std(Y_2d)

        rf_scaled = (Y_2d - mean) / (std + 1e-8)

        Y_tensor = torch.from_numpy(rf_scaled).float()

        return Y_tensor, mean, std

class PixelDataset(Dataset):
    def __init__(self, img, mask, data_flag = 'i'):
        self.img = img

        if data_flag == 'r' or data_flag == 'b':
            H, W = img.shape
            targets = img.reshape(-1, 1)
        else:
            _, H, W = img.shape
            targets = img.permute(1, 2, 0).reshape(-1, 2)

        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)

        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)

        if mask is not None:
            mask_flat = mask.squeeze().reshape(-1)
            self.coords = coords[mask_flat]
            self.targets = targets[mask_flat]
        else:
            self.coords = coords
            self.targets = targets

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.targets[idx]

if __name__ == '__main__':
    #best found params: 2, 8, 3, 512, 20, 80, 15, 5e-4
    main(param = 2, decimate = 8, hidden_layers = 3, hidden_features = 512, hidden_omega = 20,
         first_omega = 90, patience = 15, lr = 5e-4)