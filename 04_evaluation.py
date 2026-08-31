"""
Stage 4/4 - Evaluation Metrics

Quantifies deepfake quality on two levels:

1. Face level  - generated swapped faces vs. original extracted faces.
   Measures how much facial structure and pose survive the identity change.
2. Frame level - final blended frames vs. original video frames.
   Measures how invisibly the new face integrates into the untouched background.

Reports SSIM, PSNR and MSE for both, plus per-frame metric traces.

Input:  results/, transformed/, frames/, final/ from 03_face_swapping.py
Output: evaluation_report.txt, evaluation_metrics_plot.png
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Ensure scikit-image is installed for the evaluation metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import mean_squared_error as mse
except ImportError:
    print("Installing scikit-image...")
    subprocess.check_call(['pip', 'install', '-q', 'scikit-image'])
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import mean_squared_error as mse

def load_image(filepath):
    """Loads an image and converts it to RGB."""
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def evaluate_directory_pair(dir_orig, dir_swapped, max_frames=None):
    """
    Evaluates MSE, PSNR, and SSIM between two directories of images.
    Matches files by filename.
    """
    print(f"  -> Checking if {dir_orig} exists: {os.path.exists(dir_orig)}")
    print(f"  -> Checking if {dir_swapped} exists: {os.path.exists(dir_swapped)}")
    
    if not os.path.exists(dir_orig) or not os.path.exists(dir_swapped):
        return None

    filenames = sorted([f for f in os.listdir(dir_orig) if f.endswith(('.jpg', '.png'))])
    print(f"  -> Found {len(filenames)} images in {dir_orig}")
    
    if max_frames is not None:
        filenames = filenames[:max_frames]
    
    metrics = {'mse': [], 'psnr': [], 'ssim': []}
    
    successful_pairs = 0
    for filename in filenames:
        orig_path = os.path.join(dir_orig, filename)
        swap_path = os.path.join(dir_swapped, filename)
        
        if not os.path.exists(swap_path):
            continue
            
        orig_img = load_image(orig_path)
        swap_img = load_image(swap_path)
        
        if orig_img is None or swap_img is None:
            if successful_pairs == 0: print(f"  -> ERROR: cv2.imread failed on {filename}")
            continue
            
        # Ensure identical dimensions for metric comparison
        if orig_img.shape != swap_img.shape:
            swap_img = cv2.resize(swap_img, (orig_img.shape[1], orig_img.shape[0]))
            
        # Mean Squared Error
        current_mse = mse(orig_img, swap_img)
        metrics['mse'].append(current_mse)
        
        # Peak Signal-to-Noise Ratio
        current_psnr = psnr(orig_img, swap_img, data_range=255)
        metrics['psnr'].append(current_psnr)
        
        # Structural Similarity Index
        current_ssim = ssim(orig_img, swap_img, channel_axis=2, data_range=255)
        metrics['ssim'].append(current_ssim)
        successful_pairs += 1
        
    print(f"  -> Successfully evaluated {successful_pairs} pairs.")
    return metrics

def main():
    print("Starting Deepfake Evaluation Metrics for B.Tech Project...")
    
    # ---------------------------------------------------------
    # KAGGLE PIPELINE CONTEXT:
    # We use the exact path confirmed by the sanity check.
    # ---------------------------------------------------------
    BASE_DIR = '/kaggle/input/notebooks/ayushragh/deepfake-face-swapping/'
    
    # Fallback to local directories if not on Kaggle
    if not os.path.exists(BASE_DIR):
        print(f"Warning: Could not find {BASE_DIR}. Falling back to local directories ('./').")
        BASE_DIR = './'

    # REMOVED TRAILING SLASHES - Kaggle's virtual filesystem sometimes fails os.path.exists if these are present
    results_dir = os.path.join(BASE_DIR, 'results')
    transformed_dir = os.path.join(BASE_DIR, 'transformed')
    frames_dir = os.path.join(BASE_DIR, 'frames')
    final_dir = os.path.join(BASE_DIR, 'final')
    
    # 1. Evaluate Face-Level Transformation (results vs transformed)
    print("\n--- Evaluating Face-Level Transformation ---")
    print(f"Comparing original extracted faces with autoencoder swapped faces...")
    face_metrics = evaluate_directory_pair(results_dir, transformed_dir)
    
    # 2. Evaluate Frame-Level Blending (frames vs final)
    print("\n--- Evaluating Frame-Level Blending ---")
    print(f"Comparing original full frames with final blended frames...")
    frame_metrics = evaluate_directory_pair(frames_dir, final_dir)
    
    if face_metrics is None or len(face_metrics['ssim']) == 0:
        print(f"Could not evaluate face metrics. Ensure {results_dir} and {transformed_dir} exist and contain images.")
    
    if frame_metrics is None or len(frame_metrics['ssim']) == 0:
        print(f"Could not evaluate frame metrics. Ensure {frames_dir} and {final_dir} exist and contain images.")

        
    # Generate the text report
    with open('evaluation_report.txt', 'w') as f:
        f.write("DEEPFAKE EVALUATION METRICS REPORT\n")
        f.write("==================================\n\n")
        
        if face_metrics and len(face_metrics['ssim']) > 0:
            avg_mse_face = np.mean(face_metrics['mse'])
            avg_psnr_face = np.mean(face_metrics['psnr'])
            avg_ssim_face = np.mean(face_metrics['ssim'])
            
            print(f"Face Level Avg MSE:  {avg_mse_face:.4f}")
            print(f"Face Level Avg PSNR: {avg_psnr_face:.4f} dB")
            print(f"Face Level Avg SSIM: {avg_ssim_face:.4f}")
            
            f.write("1. FACE-LEVEL METRICS (Generation Quality)\n")
            f.write("------------------------------------------\n")
            f.write(f"Evaluated Frames: {len(face_metrics['ssim'])}\n")
            f.write(f"Average MSE:  {avg_mse_face:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr_face:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim_face:.4f}\n\n")
            
        if frame_metrics and len(frame_metrics['ssim']) > 0:
            avg_mse_frame = np.mean(frame_metrics['mse'])
            avg_psnr_frame = np.mean(frame_metrics['psnr'])
            avg_ssim_frame = np.mean(frame_metrics['ssim'])
            
            print(f"Frame Level Avg MSE:  {avg_mse_frame:.4f}")
            print(f"Frame Level Avg PSNR: {avg_psnr_frame:.4f} dB")
            print(f"Frame Level Avg SSIM: {avg_ssim_frame:.4f}")
            
            f.write("2. FRAME-LEVEL METRICS (Blending Quality)\n")
            f.write("-----------------------------------------\n")
            f.write(f"Evaluated Frames: {len(frame_metrics['ssim'])}\n")
            f.write(f"Average MSE:  {avg_mse_frame:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr_frame:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim_frame:.4f}\n\n")
            
        f.write("INTERPRETATION GUIDE\n")
        f.write("--------------------\n")
        f.write("SSIM (Structural Similarity Index):\n")
        f.write(" - Range [0, 1]. Higher is better. Measures structural information preserved.\n")
        f.write(" - Frame-level SSIM should be very high (>0.90) since only the face changes.\n")
        f.write(" - Face-level SSIM is lower because the facial identity changes entirely.\n\n")
        
        f.write("PSNR (Peak Signal-to-Noise Ratio):\n")
        f.write(" - Higher is better (less noise/distortion). Measured in dB.\n")
        f.write(" - Usually >30dB is considered good image quality.\n\n")
        
        f.write("MSE (Mean Squared Error):\n")
        f.write(" - Lower is better. Represents absolute pixel difference.\n")

    print("\nSaved evaluation report to 'evaluation_report.txt'")
    
    # Generate the plots
    if face_metrics and frame_metrics and len(face_metrics['ssim']) > 0 and len(frame_metrics['ssim']) > 0:
        plt.figure(figsize=(15, 10))
        
        # Face Metrics Plot
        plt.subplot(2, 3, 1)
        plt.plot(face_metrics['mse'], color='red')
        plt.title('Face-Level MSE')
        plt.xlabel('Frame')
        plt.ylabel('MSE')
        
        plt.subplot(2, 3, 2)
        plt.plot(face_metrics['psnr'], color='green')
        plt.title('Face-Level PSNR')
        plt.xlabel('Frame')
        plt.ylabel('PSNR (dB)')
        
        plt.subplot(2, 3, 3)
        plt.plot(face_metrics['ssim'], color='blue')
        plt.title('Face-Level SSIM')
        plt.xlabel('Frame')
        plt.ylabel('SSIM')
        
        # Frame Metrics Plot
        plt.subplot(2, 3, 4)
        plt.plot(frame_metrics['mse'], color='darkred')
        plt.title('Frame-Level MSE')
        plt.xlabel('Frame')
        plt.ylabel('MSE')
        
        plt.subplot(2, 3, 5)
        plt.plot(frame_metrics['psnr'], color='darkgreen')
        plt.title('Frame-Level PSNR')
        plt.xlabel('Frame')
        plt.ylabel('PSNR (dB)')
        
        plt.subplot(2, 3, 6)
        plt.plot(frame_metrics['ssim'], color='darkblue')
        plt.title('Frame-Level SSIM')
        plt.xlabel('Frame')
        plt.ylabel('SSIM')
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics_plot.png', dpi=150)
        print("Saved evaluation plots to 'evaluation_metrics_plot.png'")

if __name__ == '__main__':
    main()