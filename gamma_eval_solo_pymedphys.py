import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network_2_1 import unet
from data_pipeline_256_32_256 import data_pipeline
import pymedphys
import csv

# ===================== SETTINGS =====================
input_shape = (256, 32, 256)  # [D, H, W] at 2mm resolution
test_batch_size = 1
device = "cuda"

DOSE_PERCENT_THRESHOLD = 1.0
DTA_MM_THRESHOLD = 1.0
DOSE_CUTOFF = 0.2
VOXEL_SIZE_MM = 2.0

print(str((input_shape)) + " test" + str(DOSE_PERCENT_THRESHOLD) + " " + str(DTA_MM_THRESHOLD) + "mm  " + str(DOSE_CUTOFF) +"%"+"cutoff" + "Network 2_1_out")

test_path = '/NFSHOME/mspezialetti/sharedFolder/MC_CT_dataset_2mm_voxel/dataset/test/'
checkpoint_path = "/NFSHOME/mspezialetti/sharedFolder/MC_CT_dataset_2mm_voxel/models_and_outputs/2026-06-09_18-19-17_combined_wmse_gamma_train_2_1_out_256-32-256_zero_outside_more_data_2mm_1percent.pth"

# Output folders
prediction_folder = f"/NFSHOME/mspezialetti/sharedFolder/MC_CT_dataset_2mm_voxel/tests_and_predictions/2_1_out_256-32-256_{DOSE_PERCENT_THRESHOLD}pct_{DTA_MM_THRESHOLD}mm_cutoff{int(DOSE_CUTOFF*100)}"
image_folder = os.path.join(prediction_folder, "images")
os.makedirs(prediction_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# ===================== DATA LOADER =====================
test_examples = 89
index_list_test = np.array([f"{i:03d}" for i in range(1, test_examples + 1)])

test_generator = data_pipeline(path=test_path, index_list=index_list_test)
test_dataloader = DataLoader(
    test_generator,
    batch_size=test_batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True
)

# ===================== LOAD MODEL =====================
print("Loading model...")
model = unet(input_shape=input_shape).to(device)

# Initialize lazy layers with dummy forward pass
dummy = torch.randn(1, 2, *input_shape, device=device)
with torch.no_grad():
    _ = model(dummy)
del dummy
print(f"✔ Model initialized with {len(model.state_dict())} keys")

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"✔ Weights loaded from {checkpoint_path}")


# ===================== GAMMA INDEX (PyMedPhys) =====================
def compute_pymedphys_gpr(pred_np, target_np, voxel_size=VOXEL_SIZE_MM):
    """
    Compute gamma pass rate using PyMedPhys.

    Args:
        pred_np:    Predicted dose [D, H, W] numpy array
        target_np:  Reference dose [D, H, W] numpy array
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
        axes_reference=coords,
        dose_reference=target_np,
        axes_evaluation=coords,
        dose_evaluation=pred_np,
        dose_percent_threshold=DOSE_PERCENT_THRESHOLD,
        distance_mm_threshold=DTA_MM_THRESHOLD,
        lower_percent_dose_cutoff=DOSE_CUTOFF * 100,  # PyMedPhys expects percentage
        max_gamma=10.0,
        local_gamma=False,
        global_normalisation=ref_max,
    )

    valid_gamma = gamma_map[valid_mask]
    valid_gamma = valid_gamma[~np.isnan(valid_gamma)]
    gpr = (valid_gamma < 1.0).sum() / len(valid_gamma) * 100

    return gpr, gamma_map


def save_slice_image(pred_np, target_np, gamma_map, idx, save_path):
    """Save a central-slice comparison image: dose, prediction, diff, and gamma."""
    slice_idx = pred_np.shape[1] // 2

    pred_slice   = pred_np[:, slice_idx, :]
    target_slice = target_np[:, slice_idx, :]
    diff_slice   = np.abs(pred_slice - target_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmax = max(pred_slice.max(), target_slice.max())

    im0 = axes[0].imshow(target_slice.T, cmap='jet', aspect='auto', vmin=0, vmax=vmax)
    axes[0].set_title('Reference Dose')
    axes[0].set_xlabel('Depth (z)'); axes[0].set_ylabel('Width (x)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_slice.T, cmap='jet', aspect='auto', vmin=0, vmax=vmax)
    axes[1].set_title('Predicted Dose')
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

    plt.suptitle(f'Sample {idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===================== TESTING LOOP =====================
print("\n" + "="*60)
print("Starting evaluation...")
print("="*60)

pmp_gprs = []
results = []

with torch.no_grad():
    for idx, (X, Y) in enumerate(tqdm(test_dataloader, desc="Testing")):
        input_tensor = X.to(device, non_blocking=True)
        target = Y.to(device, non_blocking=True)

        output = model(x=input_tensor).float()
        target = target.float()

        # Normalize per-sample to [0, 1] relative to their own max
        output_norm = output / (output.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
        target_norm = target # The normalized data is coming from the datapipeline

        pred_np   = output_norm[0, 0].cpu().numpy()
        target_np = target_norm[0, 0].cpu().numpy()

        gpr_pmp, gamma_map = compute_pymedphys_gpr(pred_np, target_np)

        pmp_gprs.append(gpr_pmp)
        results.append({'idx': idx, 'gpr_pymedphys': gpr_pmp})
        tqdm.write(f"  Sample {idx:03d} | GPR: {gpr_pmp:.2f}%")
        # Save prediction array
        np.save(os.path.join(prediction_folder, f"prediction_{idx:03d}.npy"), pred_np)

        # Save comparison image
        save_slice_image(
            pred_np, target_np, gamma_map, idx,
            os.path.join(image_folder, f"comparison_{idx:03d}.png")
        )

# ===================== SUMMARY =====================
pmp_gprs = np.array(pmp_gprs)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nPyMedPhys ({DOSE_PERCENT_THRESHOLD}%/{DTA_MM_THRESHOLD}mm):")
print(f"  Mean GPR: {pmp_gprs.mean():.2f}%")
print(f"  Std GPR:  {pmp_gprs.std():.2f}%")
print(f"  Min GPR:  {pmp_gprs.min():.2f}%")
print(f"  Max GPR:  {pmp_gprs.max():.2f}%")

# Save CSV
results_csv = os.path.join(prediction_folder, "gamma_results_2_2_128_16_128_finetune.csv")
with open(results_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['idx', 'gpr_pymedphys'])
    writer.writeheader()
    writer.writerows(results)
print(f"\n✔ Results saved to {results_csv}")

# Save summary text
summary_path = os.path.join(prediction_folder, "summary.txt")
with open(summary_path, 'w') as f:
    f.write("GAMMA INDEX EVALUATION SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Checkpoint: {checkpoint_path}\n")
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
plt.axvline(pmp_gprs.mean(), color='r', linestyle='--', label=f'Mean = {pmp_gprs.mean():.1f}%')
plt.xlabel('Gamma Pass Rate (%)')
plt.ylabel('Count')
plt.title(f'GPR Distribution — {DOSE_PERCENT_THRESHOLD}%/{DTA_MM_THRESHOLD}mm (n={len(pmp_gprs)})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(prediction_folder, "gpr_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔ Distribution plot saved")

print("\n" + "="*60)
print("✅ Evaluation complete!")
print("="*60)
