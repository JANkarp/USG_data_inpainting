from INR import SIREN
from INR_program import PixelDataset, get_iq_data
from torch.utils.data import DataLoader
import numpy as np
import torch
from old_pipeline import MSLAELoss
from new_pipeline import values_for_reconstruction
from plots import reconstruct_iq, plot_val_error, plot_phase, plot_amplitude, plot_recs, phase_mae

def main(param, decimate, hidden_layers, hidden_features, hidden_omega, first_omega, patience, alpha_p, lr):

    file_path = 'usg_string_data/w_1.pkl'
    center_freq = 6e6

    print('Do you want to retrain the model? y/n')
    train_flag = input()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')


    Y, mean, std = get_iq_data(decimate, file_path)

    model = SIREN(
        in_features=2,
        out_features=2,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        first_omega= first_omega,
        hidden_omega=hidden_omega,
    ).to(device)

    criterion = MSLAELoss()

    mask = torch.zeros(Y.shape[-2:], dtype=torch.bool)
    mask[:, ::param] = True

    train_dataset = PixelDataset(Y, mask)
    train_loader = DataLoader(train_dataset, batch_size=int(len(train_dataset)/4), shuffle=True)

    test_dataset = PixelDataset(Y, None)
    test_loader = DataLoader(test_dataset, batch_size=int(len(test_dataset)/4), shuffle=False)

    alpha_d = 1
    # training
    if train_flag == 'y':

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)

        n_epochs = 1000
        loss_history = []
        val_loss_history = []


        print("--- Starting PINN Training ---")
        try:
            for epoch in range(n_epochs):

                model.train()
                running_train_loss = 0
                for coords_batch, targets_batch in train_loader:
                    coords_batch, targets_batch = coords_batch.to(device), targets_batch.to(device)

                    optimizer.zero_grad()
                    preds = model(coords_batch)
                    loss = criterion(preds, targets_batch)
                    pde_loss = get_pde_loss(model, coords_batch)
                    combined_loss = alpha_d * loss + alpha_p * pde_loss
                    combined_loss.backward()
                    optimizer.step()
                    running_train_loss += combined_loss.item() * coords_batch.size(0)

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


            if epoch > 200:
                torch.save(model.state_dict(),f'PINN/PINN_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.pth')

                print(f"PINN weighs from {epoch} have been saved")
                plot_val_error(loss_history, val_loss_history, f'PINN/val_error_PINN_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.png')

        plot_val_error(loss_history, val_loss_history, f'PINN/val_error_PINN_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.png')
        torch.save(model.state_dict(),f'PINN/PINN_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.pth')

    # evaluation
    else:

        model.load_state_dict(torch.load(f'PINN/PINN_ultrasound_model_IQ_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.pth', map_location=device))
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

        pred_img = full_pred.reshape(H, W, 2).permute(2, 0, 1)

        pred_img = pred_img.cpu().numpy()
        target_img = Y.cpu().numpy()

        pred_img = (pred_img * std) + mean
        target_img = (target_img * std) + mean

        input_img = np.zeros_like(target_img)

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

        plot_amplitude(amplitude_pred, amplitude_target, 1, current_metrics,f"PINN/Amplitude/amplitude_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.png")
        plot_phase(phase_pred, phase_target, current_metrics, 1,f"PINN/Phase/phase_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.png")
        target_rec = reconstruct_iq(complex_target, decimate, center_freq, values_for_reconstruction)
        pred_rec = reconstruct_iq(complex_pred, decimate, center_freq, values_for_reconstruction)
        input_rec = reconstruct_iq(complex_input, decimate, center_freq, values_for_reconstruction)
        plot_recs(pred_rec, target_rec, input_rec, f'PINN/reconstructions/iq_reconstructions_{param}_{hidden_layers}_{hidden_features}_{hidden_omega}_{first_omega}_{patience}_{alpha_p}_{lr}.png')

def get_pde_loss(model, coords, sound_speed=1480.0, center_freq = 6e6):

    coords = coords.clone().detach().requires_grad_(True)
    k0 = 2*np.pi*center_freq/sound_speed

    width = 0.04
    height = 0.075

    z_scale = 2 / height
    x_scale = 2 / width

    p = model(coords)

    p_real, p_imag = p[:, 0:1], p[:, 1:2]

    dpr = torch.autograd.grad(p_real, coords, torch.ones_like(p_real), create_graph=True)[0]
    dpi = torch.autograd.grad(p_imag, coords, torch.ones_like(p_imag), create_graph=True)[0]

    dpr_dz, dpr_dx = dpr[:, 0:1] * z_scale, dpr[:, 1:2] * x_scale
    dpi_dz, dpi_dx = dpi[:, 0:1] * z_scale, dpi[:, 1:2] * x_scale

    d2pr_dz2 = torch.autograd.grad(dpr_dz, coords, torch.ones_like(dpr_dz), create_graph=True)[0][:, 0:1] * z_scale
    d2pr_dx2 = torch.autograd.grad(dpr_dx, coords, torch.ones_like(dpr_dx), create_graph=True)[0][:, 1:2] * x_scale
    d2pi_dz2 = torch.autograd.grad(dpi_dz, coords, torch.ones_like(dpi_dz), create_graph=True)[0][:, 0:1] * z_scale
    d2pi_dx2 = torch.autograd.grad(dpi_dx, coords, torch.ones_like(dpi_dx), create_graph=True)[0][:, 1:2] * x_scale

    real = (d2pr_dz2 + d2pr_dx2) / (k0 ** 2) + (2.0 / k0) * dpi_dz
    imag = (d2pi_dz2 + d2pi_dx2) / (k0 ** 2) - (2.0 / k0) * dpr_dz

    return torch.mean(real ** 2 + imag ** 2)


if __name__ == '__main__':
    main(param=2, decimate=8, hidden_layers=3, hidden_features=512, hidden_omega=40, first_omega=60, patience=15, alpha_p = 1e-3, lr=5e-4)