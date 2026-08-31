"""
Stage 3/4 - Face Swapping

Cross-decodes Person A's faces through Person B's decoder, aligns the result
to the original frame with dlib 68-point landmarks and a Procrustes transform,
blends it under a feathered convex-hull mask, and rebuilds the video.

Input:  trained weights from 02_model_training.py
Output: deepfake.avi

Kaggle: https://www.kaggle.com/code/ayushragh/deepfake-face-swapping
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import gc
import matplotlib.pyplot as plt
import cv2
import dlib
import subprocess
subprocess.check_call(['pip', 'install', '-q', 'mtcnn'])
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

FACE_SIZE = 64
LATENT_DIM = 1024

# ============================================================
# DIRECTORIES CREATION
# ============================================================
os.makedirs('frames', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('transformed', exist_ok=True)
os.makedirs('final', exist_ok=True)

# ============================================================
# REBUILD MODEL ARCHITECTURE (must match notebook 2 exactly)
# ============================================================
def build_encoder():
    inp = layers.Input(shape=(FACE_SIZE, FACE_SIZE, 3))
    x = layers.Conv2D(128, 5, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(LATENT_DIM)(x)
    x = layers.Dense(4 * 4 * 512)(x)
    x = layers.Reshape((4, 4, 512))(x)
    return keras.Model(inp, x, name="encoder")


def build_decoder(name="decoder"):
    inp = layers.Input(shape=(4, 4, 512))
    x = layers.Conv2DTranspose(512, 5, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(3, 5, strides=2, padding='same')(x)
    x = layers.Activation('sigmoid', dtype='float32')(x)
    return keras.Model(inp, x, name=name)


# ============================================================
# FRAME EXTRACTION
# ============================================================
input_path = '/kaggle/input/datasets/ayushragh/presidents-data/presidents/trump1.mp4'
output_path = 'frames/'


def extract_frames(input_path, output_path):
    videocapture = cv2.VideoCapture(input_path)
    success, image = videocapture.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_path, "%d.jpg" % count), image)
        success, image = videocapture.read()
        count += 1
    videocapture.release()
    print(f'Extracted {count} frames')
    return count


total_frames = extract_frames(input_path, output_path)

# ============================================================
# FACE EXTRACTION (using MTCNN)
# ============================================================
def extract_faces(source, destination, detector):
    counter = 0
    for dirname, _, filenames in os.walk(source):
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
                plt.imsave(os.path.join(destination, filename), face)
                counter += 1
                if counter % 100 == 0:
                    print(f"Extracted {counter} faces...")
            except Exception as e:
                print(f"Skipped {filename}: {e}")
    print(f"Face extraction done. Total: {counter}")


detector = MTCNN()
extract_faces('frames/', 'results/', detector)
del detector
gc.collect()

# ============================================================
# LOAD TRAINED WEIGHTS — adjust path to your notebook 2 output
# ============================================================
WEIGHTS_DIR = '/kaggle/input/notebooks/ayushragh/deepfakes-model-training/'

encoder   = build_encoder()
decoder_b = build_decoder(name="decoder_b")

encoder.load_weights(os.path.join(WEIGHTS_DIR, 'encoder.weights.h5'))
decoder_b.load_weights(os.path.join(WEIGHTS_DIR, 'decoder_b.weights.h5'))

print("Loaded encoder + decoder_b weights")

# ============================================================
# FACE TRANSFORMATION (encoder_A + decoder_B cross-decode)
# ============================================================
def face_transform(source, destination, encoder, decoder, batch_size=64):
    filenames = sorted([f for f in os.listdir(source) if f.endswith('.jpg')])
    total = len(filenames)

    for i in range(0, total, batch_size):
        batch_files = filenames[i:i + batch_size]
        batch_images = []

        for fname in batch_files:
            image = cv2.imread(os.path.join(source, fname))
            if image is None:
                batch_images.append(np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype='float32'))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32') / 255.0
            batch_images.append(image)

        batch_arr = np.array(batch_images)
        encoded = encoder.predict(batch_arr, verbose=0)
        decoded = decoder.predict(encoded, verbose=0)

        for j, fname in enumerate(batch_files):
            out = np.clip(decoded[j], 0, 1)
            out = (out * 255).astype(np.uint8)
            plt.imsave(os.path.join(destination, fname), out)

        if (i // batch_size) % 5 == 0:
            print(f"Transformed {min(i + batch_size, total)}/{total}")

    print("Face transformation done.")


face_transform('results/', 'transformed/', encoder, decoder_b)

# ============================================================
# FACE SWAPPING (using dlib landmarks + Procrustes alignment)
# ============================================================
PREDICTOR_PATH = '/kaggle/input/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat/shape_predictor_68_face_landmarks.dat'
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC = 0.6

dlib_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = dlib_detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                  c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def face_swap_single(head_path, face_path, output_path):
    im1, landmarks1 = read_im_and_landmarks(head_path)
    im2, landmarks2 = read_im_and_landmarks(face_path)
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite(output_path, output_im)


# ============================================================
# MASSIVE FACE SWAP
# ============================================================
def massive_face_swap(source, destination, output):
    filenames = sorted([f for f in os.listdir(source) if f.endswith('.jpg')])
    total = len(filenames)
    success_count = 0

    for i, filename in enumerate(filenames):
        src_path = os.path.join(source, filename)
        dst_path = os.path.join(destination, filename)
        out_path = os.path.join(output, filename)

        if not os.path.exists(dst_path):
            continue

        try:
            face_swap_single(dst_path, src_path, out_path)
            success_count += 1
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"Swap progress: {i+1}/{total} (success: {success_count})")

    print(f"Face swap done. {success_count}/{total} frames swapped.")


massive_face_swap('transformed/', 'frames/', 'final/')

# ============================================================
# VIDEO REBUILDING
# ============================================================
frames_list = []
height, width = None, None

for i in range(total_frames):
    image = cv2.imread(os.path.join('final/', str(i) + '.jpg'))
    if image is None:
        orig = cv2.imread(os.path.join('frames/', str(i) + '.jpg'))
        if orig is not None:
            frames_list.append(orig)
            if height is None:
                height, width, _ = orig.shape
        continue
    if height is None:
        height, width, _ = image.shape
    frames_list.append(image)

if frames_list:
    videomaker = cv2.VideoWriter('deepfake.avi',
                                 cv2.VideoWriter_fourcc(*'DIVX'),
                                 25, (width, height))
    for frame in frames_list:
        videomaker.write(frame)
    videomaker.release()
    print(f"Done! Output: deepfake.avi ({len(frames_list)} frames)")
else:
    print("No frames to assemble.")
