"""
Stage 1/4 - Preprocessing

Extracts frames from the source videos, detects faces with MTCNN, crops them
with a small margin, resizes to 64x64, and writes one face dataset per identity.

Input:  four source videos (2 per identity)
Output: results_1/ (Person A faces), results_2/ (Person B faces)

Kaggle: https://www.kaggle.com/code/ayushragh/deepfakes-preprocessing
Adapted from: https://www.kaggle.com/code/sergiovirahonda/deepfakes-preprocessing
"""
import subprocess
subprocess.check_call(['pip', 'install', '-q', 'mtcnn'])
# ============================================================
# IMPORTS
# ============================================================
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mtcnn import MTCNN

FACE_SIZE = 64  # reduced from 120 for faster training

# ============================================================
# DIRECTORIES CREATION
# ============================================================
os.makedirs('frames_1', exist_ok=True)
os.makedirs('frames_2', exist_ok=True)
os.makedirs('frames_3', exist_ok=True)
os.makedirs('frames_4', exist_ok=True)
os.makedirs('results_1', exist_ok=True)
os.makedirs('results_2', exist_ok=True)

# ============================================================
# PATHS (adjust these to your local video paths)
# ============================================================
input_1 = '/kaggle/input/datasets/ayushragh/presidents-data/presidents/trump1.mp4'
input_2 = '/kaggle/input/datasets/ayushragh/presidents-data/presidents/trump2.mp4'
input_3 = '/kaggle/input/datasets/ayushragh/presidents-data/presidents/biden1.mp4'
input_4 = '/kaggle/input/datasets/ayushragh/presidents-data/presidents/biden2.mp4'
output_1 = 'frames_1/'
output_2 = 'frames_2/'
output_3 = 'frames_3/'
output_4 = 'frames_4/'

# ============================================================
# FRAME EXTRACTION
# ============================================================
def extract_frames(input_path, output_path):
    """Reads a video file and saves every frame as a JPEG image."""
    videocapture = cv2.VideoCapture(input_path)
    success, image = videocapture.read()
    count = 0
    while success:
        cv2.imwrite(output_path + "frame%d.jpg" % count, image)
        success, image = videocapture.read()
        count += 1
    videocapture.release()
    return count

total_frames_1 = extract_frames(input_1, output_1)
total_frames_2 = extract_frames(input_2, output_2)
total_frames_3 = extract_frames(input_3, output_3)
total_frames_4 = extract_frames(input_4, output_4)

print('Total frames extracted in video 1: ', total_frames_1)
print('Total frames extracted in video 2: ', total_frames_2)
print('Total frames extracted in video 3: ', total_frames_3)
print('Total frames extracted in video 4: ', total_frames_4)

# ============================================================
# FACE DETECTION AND EXTRACTION
# ============================================================
def extract_faces_from_dir(source_dir, destination, detector, start_counter=0):
    """
    Walks a directory of frame images, detects faces with MTCNN,
    crops with a clamped margin, resizes to FACE_SIZE x FACE_SIZE, and saves.
    Returns the next counter value.
    """
    counter = start_counter
    for dirname, _, filenames in os.walk(source_dir):
        for filename in sorted(filenames):
            try:
                image = cv2.imread(os.path.join(dirname, filename))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                detections = detector.detect_faces(image)
                if not detections:
                    continue
                x, y, bw, bh = detections[0]['box']
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + bw + margin)
                y2 = min(h, y + bh + margin)
                face = image[y1:y2, x1:x2]
                face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
                plt.imsave(os.path.join(destination, str(counter) + '.jpg'), face)
            except Exception:
                pass
            counter += 1
    return counter


def extract_faces(source_1, source_2, destination, detector):
    """Extracts faces from two source directories into one destination."""
    next_count = extract_faces_from_dir(source_1, destination, detector, start_counter=0)
    extract_faces_from_dir(source_2, destination, detector, start_counter=next_count)
    print(f"Extraction complete. Faces saved to {destination}")


detector = MTCNN()
extract_faces('frames_1/', 'frames_2/', 'results_1/', detector)
extract_faces('frames_3/', 'frames_4/', 'results_2/', detector)
