import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_pipeline_256_32_256 import data_pipeline
import pymedphys
import csv

# ===================== SETTINGS =====================
input_shape = (256, 32, 256)  # [D, H, W] at 2mm resolution
test_batch_size = 1

DOSE_PERCENT_THRESHOLD = 1.0
DTA_MM_THRESHOLD       = 1.0
DOSE_CUTOFF            = 0.2
VOXEL_SIZE_MM          = 2.0

print(str(input_shape) + " test " + str(DOSE_PERCENT_THRESHOLD) + " " +
      str(DTA_MM_THRESHOLD) + "mm  " + str(DOSE_CUTOFF) + "% cutoff")

test_path = '/NFSHOME/mspezialetti/sharedFolder/MC_CT_dataset_2mm_voxel/dataset/test/'

# Output folders
prediction_folder = f"/NFSHOME/mspezialetti/sharedFolder/MC_CT_dataset_2mm_voxel/tests_and_predictions/5k_baseline_256_32_256_{DOSE_PERCENT_THRESHOLD}pct_{DTA_MM_THRESHOLD}mm_cutoff{int(DOSE_CUTOFF*100)}"
image_folder      = os.path.join(prediction_folder, "images")
os.makedirs(prediction_folder, exist_ok=True)
os.makedirs(image_folder,      exist_ok=True)

# ===================== DATA LOADER =====================
test_examples    = 89
index_list_test  = np.array([f"{i:03d}" for i in range(1, test_examples + 1)])

test_generator  = data_pipeline(path=test_path, index_list=index_list_test)
test_dataloader = DataLoader(
    test_generator,
    batch_size=test_batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True
)

# ===================== GAMMA INDEX (PyMedPhys) =====================
def compute_pymedphys_gpr(pred_np, target_np, voxel_size=VOXEL_SIZE_MM):
    """
    Compute gamma pass rate using PyMedPhys.

    Args:
        pred_np:    Evaluation dose [D, H, W] numpy array  (5k)
        target_np:  Reference dose  [D, H, W] numpy array  (1M)
        voxel_size: Isotropic voxel size in mm

    Returns:
        gpr:       Gamma pass rate (%)
        gamma_map: Gamma index map
    """
    D, H, W = target_np.shape
    coords = (
        np.arange(D) * voxel_size,
        np.arange(H) * voxel_size,
        np.arange(W) * voxel_size,
    )

    ref_max = target_np.max()
    if ref_max < 1e-10:
        return 100.0, np.zeros_like(target_np)

    valid_mask = target_np >= DOSE_CUTOFF * ref_max
    if valid_mask.sum() == 0:
        return 100.0, np.zeros_like(target_np)

    gamma_map = pymedphys.gamma(
        axes_reference            = coords,
        dose_reference            = target_np,
        axes_evaluation           = coords,
        dose_evaluation           = pred_np,
        dose_percent_threshold    = DOSE_PERCENT_THRESHOLD,
        distance_mm_threshold     = DTA_MM_THRESHOLD,
        lower_percent_dose_cutoff = DOSE_CUTOFF * 100,  # PyMedPhys expects percentage
        max_gamma                 = 10.0,
        local_gamma               = False,
        global_normalisation      = ref_max,
        ram_available              = 10**9
    )

    valid_gamma = gamma_map[valid_mask]
    valid_gamma = valid_gamma[~np.isnan(valid_gamma)]
    gpr = (valid_gamma < 1.0).sum() / len(valid_gamma) * 100

    return gpr, gamma_map


def save_slice_image(pred_np, target_np, gamma_map, idx, save_path):
    """Save a [:, 16, :] slice comparison: dose, prediction, diff, and gamma."""
    slice_idx = 16

    pred_slice   = pred_np  [:, slice_idx, :]
    target_slice = target_np[:, slice_idx, :]
    diff_slice   = np.abs(pred_slice - target_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmax = max(pred_slice.max(), target_slice.max())

    im0 = axes[0].imshow(target_slice.T, cmap='jet', aspect='auto', vmin=0, vmax=vmax)
    axes[0].set_title('Reference Dose (1M)')
    axes[0].set_xlabel('Depth (z)'); axes[0].set_ylabel('Width (x)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_slice.T, cmap='jet', aspect='auto', vmin=0, vmax=vmax)
    axes[1].set_title('Evaluation Dose (5k)')
    axes[1].set_xlabel('Depth (z)'); axes[1].set_ylabel('Width (x)')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff_slice.T, cmap='hot', aspect='auto')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Depth (z)'); axes[2].set_ylabel('Width (x)')
    plt.colorbar(im2, ax=axes[2])

    if gamma_map is not None:
        gamma_slice = gamma_map[:, slice_idx, :]
        im3 = axes[3].imshow(gamma_slice.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)
        axes[3].set_title('Gamma (PyMedPhys)')
        axes[3].set_xlabel('Depth (z)'); axes[3].set_ylabel('Width (x)')
        plt.colorbar(im3, ax=axes[3])
    else:
        axes[3].axis('off')

    plt.suptitle(f'Sample {idx}  |  5k Baseline', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===================== TESTING LOOP =====================
print("\n" + "="*60)
print("Starting evaluation (5k baseline — no model)...")
print("="*60)

pmp_gprs = []
results  = []

with tqdm(total=test_examples, desc="Testing") as pbar:
    for idx, (X, Y) in enumerate(test_dataloader):

        # Skip rotated augmentations — keep only the original orientation

        patient_idx = idx

        # X shape: [B, 2, D, H, W]  — channel 0 = CT, channel 1 = Dose_5K_norm
        # Y shape: [B, 1, D, H, W]  — Dose_1M_norm
        dose_5k_norm = X[0, 1].numpy()   # [D, H, W]
        dose_1M_norm = Y[0, 0].numpy()   # [D, H, W]

        gpr_pmp, gamma_map = compute_pymedphys_gpr(dose_5k_norm, dose_1M_norm)

        pmp_gprs.append(gpr_pmp)
        results.append({'idx': patient_idx, 'gpr_pymedphys': gpr_pmp})
        tqdm.write(f"  Sample {patient_idx:03d} | GPR: {gpr_pmp:.2f}%")

        # Save 5k dose array
        np.save(os.path.join(prediction_folder, f"dose_5k_{patient_idx:03d}.npy"), dose_5k_norm)

        # Save comparison image
        save_slice_image(
            dose_5k_norm, dose_1M_norm, gamma_map, patient_idx,
            os.path.join(image_folder, f"comparison_{patient_idx:03d}.png")
        )

        pbar.update(1)

# ===================== SUMMARY =====================
pmp_gprs = np.array(pmp_gprs)

print("\n" + "="*60)
print("RESULTS SUMMARY  —  5k Baseline")
print("="*60)
print(f"\nPyMedPhys ({DOSE_PERCENT_THRESHOLD}%/{DTA_MM_THRESHOLD}mm):")
print(f"  Mean GPR: {pmp_gprs.mean():.2f}%")
print(f"  Std GPR:  {pmp_gprs.std():.2f}%")
print(f"  Min GPR:  {pmp_gprs.min():.2f}%")
print(f"  Max GPR:  {pmp_gprs.max():.2f}%")

# Save CSV
results_csv = os.path.join(prediction_folder, "gamma_results_5k_baseline_128_16_128.csv")
with open(results_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['idx', 'gpr_pymedphys'])
    writer.writeheader()
    writer.writerows(results)
print(f"\n✔ Results saved to {results_csv}")

# Save summary text
summary_path = os.path.join(prediction_folder, "summary.txt")
with open(summary_path, 'w') as f:
    f.write("GAMMA INDEX EVALUATION SUMMARY — 5k Baseline\n")
    f.write("="*50 + "\n\n")
    f.write(f"Evaluation: Dose_5K_norm  vs  Dose_1M_norm (no model)\n")
    f.write(f"Criteria: {DOSE_PERCENT_THRESHOLD}%/{DTA_MM_THRESHOLD}mm, cutoff={DOSE_CUTOFF*100:.0f}%\n\n")
    f.write(f"PyMedPhys Results:\n")
    f.write(f"  Mean GPR: {pmp_gprs.mean():.2f}%\n")
    f.write(f"  Std GPR:  {pmp_gprs.std():.2f}%\n")
    f.write(f"  Min GPR:  {pmp_gprs.min():.2f}%\n")
    f.write(f"  Max GPR:  {pmp_gprs.max():.2f}%\n")
print(f"✔ Summary saved to {summary_path}")

# GPR distribution plot
plt.figure(figsize=(8, 5))
plt.hist(pmp_gprs, bins=20, edgecolor='black', alpha=0.75)
plt.axvline(pmp_gprs.mean(), color='r', linestyle='--',
            label=f'Mean = {pmp_gprs.mean():.1f}%')
plt.xlabel('Gamma Pass Rate (%)')
plt.ylabel('Count')
plt.title(f'GPR Distribution — 5k Baseline — '
          f'{DOSE_PERCENT_THRESHOLD}%/{DTA_MM_THRESHOLD}mm (n={len(pmp_gprs)})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(prediction_folder, "gpr_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔ Distribution plot saved")

print("\n" + "="*60)
print("✅ Evaluation complete!")
print("="*60)