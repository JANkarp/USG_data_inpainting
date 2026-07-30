import matplotlib
#for displaying images in windows
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm

def plot_val_error(loss, val_loss, filename):
    """
    param, n_epochs, batch_size, decimate, n_features, mode = None
    :param loss: training loss
    :param val_loss: validation loss
    :param filename: output filename
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(loss, color='blue', linewidth=1, label='Training loss')
    plt.plot(val_loss, color='red', linewidth=1, label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    fig.legend()
    plt.title("Validation error distribution")

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close(fig)

def reconstruct_rf(frame, func, param = 1):
    """
    Performs b-mode reconstruction of an rf frame.

    :param frame: a frame with iq data
    :param func: function used for calculating parameters for reconstruction
    :param param: only needed for an input image with missing columns
    :return: reconstructed image
    """

    #rezolution
    H, W = 600, 400

    # for extension
    if param < 1:
        step = 1
        head_x = frame.shape[1]
    # for inapainting
    else:
        step = int(param)
        head_x = int(frame.shape[1] / step)

    cols, mask_0, mask_1, i_0, i_1, w, _ = func(frame, param, H, W, head_x, step)

    # placeholders for the values calculated for both pixels needed for interpolation
    val_0 = np.zeros((H, W, head_x))
    val_1 = np.zeros((H, W, head_x))

    # gather only the values with viable indices
    val_0[mask_0] = frame[i_0[mask_0], cols[mask_0]]
    val_1[mask_1] = frame[i_1[mask_1], cols[mask_1]]

    # apply weights and sum values from both pixels
    fetched = (1 - w) * val_0 + w * val_1

    # sum values from each detector for given index
    output = np.sum(fetched, axis=2)

    # perform hilbert transform to get complex values
    complex_rec = hilbert(output, axis=0)

    # convert to dB representation
    rec = get_db(complex_rec)

    return rec

def reconstruct_iq(frame, decimate, ft, func, param = 1):
    """
    Performs b-mode reconstruction of an IQ frame.

    :param frame: a frame with iq data
    :param decimate: decimation factor
    :param ft: center frequency
    :param func: function used for calculating parameters for reconstruction
    :param param: only needed for an input image with missing columns
    :return: reconstructed image
    """

    #rezolution
    H, W = 600, 400

    #for extension
    if param < 1:
        step = 1
        head_x = frame.shape[1]
    #for inpainting
    else:
        step = int(param)
        head_x = int(frame.shape[1] / step)

    cols, mask_0, mask_1, i_0, i_1, w, t = func(frame, param, H, W, head_x, step, decimate)

    #placeholders for the values calculated for both pixels needed for interpolation
    val_0 = np.zeros((H, W, head_x), dtype=np.complex128)
    val_1 = np.zeros((H, W, head_x), dtype=np.complex128)

    #gather only the values with viable indices
    val_0[mask_0] = frame[i_0[mask_0], cols[mask_0]]
    val_1[mask_1] = frame[i_1[mask_1], cols[mask_1]]

    #apply weights and sum values from both pixels
    fetched = (1 - w) * val_0 + w * val_1

    #modulate
    fetched = np.conj(fetched) * np.exp(2 * np.pi * 1j * t * ft)

    #sum values from each detector for given index
    output = np.sum(fetched, axis=2)

    #convert to dB representation
    rec = get_db(output)

    return rec

def plot_amplitude(amp_pred, amp_target, idx, metrics, filename):
    """
    Plots a comparison between predicted and target amplitude

    :param amp_pred: predicted amplitude
    :param amp_target: ground truth amplitude
    :param idx: index of a frame
    :param metrics: evaluation metrics
    :param filename: output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 20))

    log_amp_pred = get_db(amp_pred)
    log_amp_target = get_db(amp_target)

    axes[0].imshow(log_amp_pred, cmap='hot', aspect='equal', interpolation='nearest')
    axes[0].set_title("Amplitude - Prediction")
    axes[0].axis('off')

    axes[1].imshow(log_amp_target, cmap='hot', aspect='equal', interpolation='nearest')
    axes[1].set_title("Amplitude - Ground Truth")
    axes[1].axis('off')

    amp_error = np.abs(log_amp_target-log_amp_pred)

    axes[2].imshow(amp_error, cmap='hot', aspect='equal', interpolation='nearest')
    axes[2].set_title("Amplitude error map")
    axes[2].axis('off')

    report_title = (
        f"Frame Index: {idx}\n"
        f"MSLAE loss: {metrics['MSLAE']}\n"
        f"Amplitude loss: {metrics['AMPLITUDE']}% "
    )
    fig.suptitle(report_title, fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close(fig)

def plot_amplitude_extended(amp_pred, amp_target, amp_input, idx, metrics):
    """
     Plots a comparison between predicted and target amplitude, a difference between input and output,
     a difference between three neighbouring lines in an output image

    :param amp_pred: predicted amplitude
    :param amp_target: ground truth amplitude
    :param amp_input: input amplitude
    :param idx: index of a frame
    :param metrics: evaluation metrics
    """

    log_amp_pred = get_db(amp_pred)
    log_amp_target = get_db(amp_target)
    log_amp_input = get_db(amp_input)


    amp_input_squeezed = log_amp_input[:,::2]

    fig = plt.figure(figsize=(10, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.6)

    grid[0].imshow(amp_input_squeezed, cmap='hot', aspect='equal', interpolation='nearest')
    grid[0].set_title("Amplitude - Input")
    grid[0].axis('off')

    grid[1].imshow(log_amp_pred, cmap='hot', aspect='equal', interpolation='nearest')
    grid[1].set_title("Amplitude - Prediction")
    grid[1].axis('off')

    report_title = (
        f"Frame Index: {idx}\n"
    )
    fig.suptitle(report_title, fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(f"New/Amplitude/amplitude_input_vs_output_{idx}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 8))

    axes[0].imshow(log_amp_pred, cmap='hot', aspect='equal', interpolation='nearest')
    axes[0].set_title("Amplitude - Prediction")
    axes[0].axis('off')

    axes[1].imshow(log_amp_target, cmap='hot', aspect='equal', interpolation='nearest')
    axes[1].set_title("Amplitude - Ground Truth")
    axes[1].axis('off')

    amp_error = np.abs(log_amp_target-log_amp_pred)

    im = axes[2].imshow(amp_error, cmap='hot', aspect='equal', interpolation='nearest')
    axes[2].set_title("Amplitude error map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    report_title = (
        f"Frame Index: {idx}\n"
        f"Amplitude MAE loss: {metrics['AMPLITUDE']} "
    )
    fig.suptitle(report_title, fontsize=12, fontweight='bold', y=0.98)


    plt.savefig(f"New/Amplitude/amplitude_{idx}_IQ.png", dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))


    log_amp_pred = 20 * np.log10(np.abs(amp_pred)+1e-6)
    log_amp_input = 20 * np.log10(np.abs(amp_input) + 1e-6)

    axes[0].plot(log_amp_input[:,96])
    axes[0].set_title("Amplitude input 0")

    axes[1].plot(log_amp_pred[:,97])
    axes[1].set_title("Amplitude prediction 0.5")


    axes[2].plot(log_amp_input[:,98])
    axes[2].set_title("Amplitude prediction 1")

    report_title = (
        f"Single line comparison\n"
        f"Frame Index: {idx}\n"
    )
    fig.suptitle(report_title, fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(f"New/Amplitude/one_line_amplitude_{idx}.png", dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close(fig)

def plot_phase(phase_pred, phase_target, metrics, idx, filename ):
    """
    Plots a comparison between predicted and target phase

    :param phase_pred: predicted phase
    :param phase_target: ground truth phase
    :param idx: index of a frame
    :param metrics: evaluation metrics
    :param filename: output filename
    """

    fig, axes = plt.subplots(1, 3, figsize=(10, 20))

    axes[0].imshow(phase_pred, cmap='hot', aspect='equal', interpolation='nearest')
    axes[0].set_title("Phase - Prediction")
    axes[0].axis('off')

    axes[1].imshow(phase_target, cmap='hot', aspect='equal', interpolation='nearest')
    axes[1].set_title("Phase - Ground Truth")
    axes[1].axis('off')

    phase_error = np.abs(phase_mae(phase_pred, phase_target))

    axes[2].imshow(phase_error, cmap='hot', aspect='equal', interpolation='nearest')
    axes[2].set_title("Phase error map")
    axes[2].axis('off')


    report_title = (
        f"Frame Index: {idx}\n"
        f"MSLAE loss: {metrics['MSLAE']}\n"
        f"Phase MAE loss: {metrics['PHASE']} rad"
    )
    fig.suptitle(report_title, fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def amplitude_vs_phase_map(amp_target, phase_pred, phase_target, idx):

    """
    Plots an error map showing phase error in reference to target amplitude

    :param amp_target: ground truth amplitude
    :param phase_pred: predicted phase
    :param phase_target: ground truth phase
    :param idx: index of a frame
    """

    x_amplitude = amp_target.flatten()
    y_absolute_error = phase_mae(phase_pred.flatten(), phase_target.flatten())


    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(x_amplitude, y_absolute_error, gridsize=100, cmap='viridis', mincnt=1, xscale='log')
    ax.set_xlabel("Target amplitude")
    ax.set_ylabel("Phase error")
    ax.set_title("Phase error distribution")

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of pixels (logarithmic scale)')


    plt.savefig(f"New/Amplitude-vs-phase/amp_vs_phase_err_{idx}_scatter_IQ.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def amplitude_vs_amplitude_map(amp_target, amplitude_pred, idx):
    """
    Plots an error map showing amplitude error in reference to target amplitude

    :param amp_target: ground truth amplitude
    :param amplitude_pred: predicted phase
    :param idx: index of a frame
    """

    x_amplitude = amp_target.flatten()
    y_absolute_error = np.abs(amplitude_pred.flatten() - x_amplitude)
    y_relative_error = (y_absolute_error/(x_amplitude + 1e-6))*100

    #avoids division by 0
    y_relative_error_safe = y_relative_error + 1e-6
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(x_amplitude, y_relative_error_safe, gridsize=100, cmap='viridis', mincnt=1, xscale='log', yscale='log', norm=LogNorm())
    ax.set_xlabel("Target amplitude")
    ax.set_ylabel("Relative amplitude error [%]")
    ax.set_title("Amplitude error distribution")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of pixels (logarithmic scale)')

    plt.savefig(f"New/Amplitude-vs-amplitude/amp_vs_amp_err_{idx}_scatter_IQ.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def amplitude_vs_phase_plots(amp_target, phase_pred, phase_target, idx, mode):
    """
    Plots phase error map against ground truth amplitude and average phase error against ground truth amplitude

    :param amp_target: ground truth amplitude
    :param phase_pred: predicted phase
    :param phase_target: ground truth phase
    :param idx: index of a frame
    :param mode: used mode (either rf or IQ)
    """
    x_amplitude = amp_target.flatten()
    y_absolute_error = phase_mae(phase_pred.flatten(), phase_target.flatten())

    avg_errors, bin_edges, _ = binned_statistic(
        x_amplitude,
        y_absolute_error,
        statistic='mean',
        bins=1000
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Plots average phase error
    plt.plot(bin_centers, avg_errors, '.r')
    plt.xlabel("Target amplitude")
    plt.ylabel("Average phase error (Radians)")
    plt.title("Average phase error distribution")
    plt.grid(True, linestyle='--', alpha=0.6)
    if mode == 'rf':
        plt.savefig(f"rf/Amplitude-vs-phase/amp_vs_phase_err_{idx}_line_rf.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"IQ/Amplitude-vs-phase/amp_vs_phase_err_{idx}_line_IQ.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    #plots phase error map
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(x_amplitude, y_absolute_error, gridsize=100, cmap='viridis', mincnt=1, xscale='log')
    ax.set_xlabel("Target amplitude")
    ax.set_ylabel("Phase error")
    ax.set_title("Phase error distribution")

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of pixels (logarithmic scale)')

    if mode == 'rf':
        plt.savefig(f"rf/Amplitude-vs-phase/amp_vs_phase_err_{idx}_scatter_rf.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"IQ/Amplitude-vs-phase/amp_vs_phase_err_{idx}_scatter_IQ.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def amplitude_vs_amplitude_plots(amp_target, amplitude_pred, idx, mode):
    """
    Plots amplitude error map against ground truth amplitude and average amplitude error against ground truth amplitude

    :param amp_target: ground truth amplitude
    :param amplitude_pred: predicted amplitude
    :param idx: index of a frame
    :param mode: used mode (either rf or IQ)
    """

    # Plots average amplitude error (probably broken)
    x_amplitude = amp_target.flatten()
    y_absolute_error = np.abs(amplitude_pred.flatten() - x_amplitude)
    y_relative_error = (y_absolute_error/(x_amplitude + 1e-6))*100

    avg_errors, bin_edges, _ = binned_statistic(
        x_amplitude,
        y_absolute_error,
        statistic='mean',
        bins=100
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.loglog(bin_centers, avg_errors, '.r')
    plt.xlabel("Target amplitude")
    plt.ylabel("Average amplitude error")
    plt.title("Average amplitude error distribution")
    plt.grid(True, linestyle='--', alpha=0.6)
    if mode == 'rf':
        plt.savefig(f"rf/Amplitude-vs-amplitude/amp_vs_amp_err_{idx}_line_rf.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"IQ/Amplitude-vs-amplitude/amp_vs_amp_err_{idx}_line_IQ.png", dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close()

    # avoids an error when the relativ error is equal to 0
    y_relative_error_safe = y_relative_error + 1e-6
    # Plots amplitude error map
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(x_amplitude, y_relative_error_safe, gridsize=100, cmap='viridis', mincnt=1, xscale='log', yscale='log', norm=LogNorm())
    ax.set_xlabel("Target amplitude")
    ax.set_ylabel("Relative amplitude error [%]")
    ax.set_title("Amplitude error distribution")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of pixels (logarithmic scale)')

    if mode == 'rf':
        plt.savefig(f"rf/Amplitude-vs-amplitude/amp_vs_amp_err_{idx}_scatter_rf.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"IQ/Amplitude-vs-amplitude/amp_vs_amp_err_{idx}_scatter_IQ.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_recs(rec_pred, rec_target, rec_input, filename):
    """
    Plots comparison between b-mode reconstructions
    :param rec_pred: image reconstructed from predicted data
    :param rec_target: image reconstructed from ground truth data
    :param rec_input: image reconstructed from input data
    :param filename: output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 30))

    axes[0].imshow(rec_input, cmap='gray', aspect='equal', interpolation='nearest')
    axes[0].set_title(f"Input reconstruction")
    axes[0].axis('off')

    axes[1].imshow(rec_pred, cmap='gray', aspect='equal', interpolation='nearest')
    axes[1].set_title(f"Predicted reconstruction")
    axes[1].axis('off')

    axes[2].imshow(rec_target, cmap='gray', aspect='equal', interpolation='nearest')
    axes[2].set_title(f"Target reconstruction")
    axes[2].axis('off')


    #plt.show()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_recs_extended(rec_pred, rec_target, rec_input, rec_extended, filename):
    """
    Plots comparison between b-mode reconstructions, also adds reconstruction from 'stretched' image

    :param rec_pred: image reconstructed from predicted data
    :param rec_target: image reconstructed from ground truth data
    :param rec_input: image reconstructed from input data
    :param rec_extended: image reconstructed from extended data
    :param filename: output filename
    """

    fig, axes = plt.subplots(1, 4, figsize=(10, 30))

    axes[0].imshow(rec_input, cmap='gray', aspect='equal', interpolation='nearest')
    axes[0].set_title(f"Input reconstruction")
    axes[0].axis('off')

    axes[1].imshow(rec_pred, cmap='gray', aspect='equal', interpolation='nearest')
    axes[1].set_title(f"Predicted reconstruction")
    axes[1].axis('off')

    axes[2].imshow(rec_target, cmap='gray', aspect='equal', interpolation='nearest')
    axes[2].set_title(f"Target reconstruction")
    axes[2].axis('off')

    axes[3].imshow(rec_extended, cmap='gray', aspect='equal', interpolation='nearest')
    axes[3].set_title(f"Extended reconstruction")
    axes[3].axis('off')

    #plt.show()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bmode_results(pred_img, target_img, l1_loss_str):
    """
    Plots bmode results from INR reconstruction

    :param pred_img: image reconstructed from predicted data
    :param target_img: image reconstructed from ground truth data
    :param l1_loss_str: MAE loss string
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(pred_img, cmap='gray', aspect='auto')
    axes[0].set_title("B-Mode - Prediction")
    axes[0].axis('off')

    axes[1].imshow(target_img, cmap='gray', aspect='auto')
    axes[1].set_title("B-Mode - Ground Truth")
    axes[1].axis('off')

    error_map = np.abs(target_img - pred_img)
    im = axes[2].imshow(error_map, cmap='hot', aspect='auto')
    axes[2].set_title("Absolute Error Map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"B-Mode Inpainting Evaluation | L1 MAE Loss: {l1_loss_str}",
                 fontsize=12, fontweight='bold')

    plt.savefig(f"INR/bmode_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_db(frame):
    """
    converts an image to decibel scale

    :param frame: image channel to convert
    :return: converted image
    """
    mag = np.abs(frame)
    mag_norm = mag / np.max(mag)
    log_data = 20 * np.log10(mag_norm + 1e-5)
    return log_data

def phase_mae(pred_phase, target_phase):
    """
    calculates phase MAE taking phase wrapping into account

    :param pred_phase: predicted phase
    :param target_phase: ground truth phase
    :return: phase MAE
    """

    diff = pred_phase - target_phase
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)
