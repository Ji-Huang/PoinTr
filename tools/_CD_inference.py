import os
import numpy as np
import torch
from extensions.chamfer_dist import ChamferDistanceL1

def calculate_average_cdl1(root_dir, chamfer_dist_fn, device='cuda'):
    """
    Calculates the average CDL1 between neighboring dense predictions in each trajectory using GPU.

    Args:
        root_dir (str): Root directory where inference results are stored.
        chamfer_dist_fn (callable): Chamfer Distance L1 function.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing the average CDL1 for each sample.
    """
    results = {}

    # Move the ChamferDistanceL1 module to the specified device
    chamfer_dist_fn.to(device)

    # Iterate over all samples
    for sample in os.listdir(root_dir):
        sample_dir = os.path.join(root_dir, sample)
        if not os.path.isdir(sample_dir):
            continue

        # Iterate over all trajectories
        for traj in os.listdir(sample_dir):
            traj_dir = os.path.join(sample_dir, traj)
            if not os.path.isdir(traj_dir):
                continue

            # Collect all fine predictions
            fine_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('_fine.npy')])
            cdl1_list = []

            for i in range(len(fine_files) - 1):
                file1 = os.path.join(traj_dir, fine_files[i])
                file2 = os.path.join(traj_dir, fine_files[i + 1])

                # Load predictions and move to GPU
                dense1 = torch.tensor(np.load(file1), device=device).unsqueeze(0)
                dense2 = torch.tensor(np.load(file2), device=device).unsqueeze(0)

                # Calculate Chamfer Distance L1
                cdl1 = chamfer_dist_fn(dense1, dense2)
                cdl1_list.append(cdl1.item())

            # Calculate average CDL1 for the trajectory
            if cdl1_list:
                avg_cdl1 = sum(cdl1_list) / len(cdl1_list)
                results[f"{sample}_{traj}"] = avg_cdl1
                last_key, last_value = list(results.items())[-1]
                print(results[last_key])

    return results

# Example usage
root_dir = "../inference_result_PCA_originrr_ws09r/"  # Replace with your output directory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
chamfer_dist_fn = ChamferDistanceL1()  # Initialize ChamferDistanceL1

average_cdl1_results = calculate_average_cdl1(root_dir, chamfer_dist_fn, device=device)
print(average_cdl1_results)

grouped_results = {}
for key, value in average_cdl1_results.items():
    sample, _ = key.split('_')  # Extract sample name from the key
    grouped_results[sample].append(value)
print(grouped_results)

averages = {}
for sample, values in grouped_results.items():
    averages[sample] = sum(values) / len(values)
print(averages)
