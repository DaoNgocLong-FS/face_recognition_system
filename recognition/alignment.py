r"""
alignment.py
------------
Face alignment using MediaPipe Face Mesh + similarity transform.

This is the CRITICAL piece that makes face recognition work properly.

Pipeline:
    YOLO bbox  ->  expand crop  ->  MediaPipe Face Mesh  ->  5 landmarks
                                                                  |
                                                                  v
    Reference 5 landmarks (ArcFace standard)  +  similarity_transform
                                                                  |
                                                                  v
                                                          Aligned 112x112 face

The 5 landmarks are: left_eye_center, right_eye_center, nose_tip,
left_mouth_corner, right_mouth_corner.

Reference landmarks (112x112) come from the InsightFace / ArcFace convention,
used by virtually every modern face recognition pretrained model.

Usage:
    from alignment import FaceAligner
    aligner = FaceAligner()
    aligned = aligner.align(face_crop_bgr)   # returns 112x112 aligned or None
"""

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "MediaPipe not installed. Run: pip install mediapipe"
    )


# Reference 5-point landmarks for 112x112 output (InsightFace / ArcFace standard)
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


# MediaPipe Face Mesh landmark indices for the 5 key points
# (MediaPipe Face Mesh has 468 landmarks; we pick 5 that map to ArcFace landmarks)
MP_INDICES = {
    "left_eye":     [33, 133, 159, 158, 145, 153],   # iris area (we'll average)
    "right_eye":    [263, 362, 386, 385, 374, 380],
    "nose":         [4],                              # nose tip
    "left_mouth":   [61],                             # left mouth corner
    "right_mouth":  [291],                            # right mouth corner
}


class FaceAligner:
    """
    Aligns a face crop (BGR) into 112x112 using MediaPipe landmarks + similarity transform.

    Args:
        confidence: minimum face detection confidence for MediaPipe (0..1)
        output_size: alignment target size (default 112)
        scale_ref: scale reference landmarks if output_size != 112
    """

    def __init__(self, confidence: float = 0.5, output_size: int = 112):
        self.confidence = confidence
        self.output_size = output_size
        self.scale = output_size / 112.0
        self.ref = REFERENCE_LANDMARKS_112 * self.scale

        # Initialize MediaPipe Face Mesh (lazy, single-face mode for speed)
        self.mp_facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.confidence,
        )

    def _extract_5_landmarks(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe Face Mesh on the face crop, extract 5 key points."""
        results = self.mp_facemesh.process(face_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        H, W = face_rgb.shape[:2]

        # Get average position for eye groups, single point for others
        points = []
        for name, indices in MP_INDICES.items():
            xs, ys = [], []
            for idx in indices:
                lm = landmarks[idx]
                xs.append(lm.x * W)
                ys.append(lm.y * H)
            points.append([float(np.mean(xs)), float(np.mean(ys))])

        return np.array(points, dtype=np.float32)

    def _similarity_transform(
        self, src: np.ndarray, dst: np.ndarray
    ) -> np.ndarray:
        """
        Compute 2x3 similarity transform matrix that maps src points to dst points.
        Uses the Umeyama algorithm (least-squares similarity).
        """
        assert src.shape == dst.shape == (5, 2)
        # Use OpenCV's estimateAffinePartial2D for similarity transform (no shear/scale-independent)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            # Fallback to a simple scaling/centering if Umeyama fails
            M = cv2.getAffineTransform(src[:3], dst[:3])
        return M

    def align(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Align a face crop into output_size x output_size.

        Args:
            face_bgr: BGR face crop (any size, expanded around bbox).
                     Recommend bbox + 30-40% margin to give MediaPipe context.

        Returns:
            Aligned face as numpy array (output_size, output_size, 3) BGR,
            or None if landmark detection fails.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_5_landmarks(face_rgb)
        if landmarks is None:
            return None

        M = self._similarity_transform(landmarks, self.ref)
        aligned = cv2.warpAffine(
            face_bgr, M, (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def align_with_landmarks(
        self, face_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Same as align() but also returns the detected 5 landmarks (for debug viz)."""
        if face_bgr is None or face_bgr.size == 0:
            return None, None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_5_landmarks(face_rgb)
        if landmarks is None:
            return None, None

        M = self._similarity_transform(landmarks, self.ref)
        aligned = cv2.warpAffine(
            face_bgr, M, (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, landmarks


def draw_landmarks(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Debug: draw 5 landmarks on the original face crop."""
    img = img_bgr.copy()
    colors = [
        (255, 0, 0),   # left eye - blue
        (0, 0, 255),   # right eye - red
        (0, 255, 0),   # nose - green
        (255, 255, 0), # left mouth - cyan
        (0, 255, 255), # right mouth - yellow
    ]
    for (x, y), color in zip(landmarks, colors):
        cv2.circle(img, (int(x), int(y)), 3, color, -1)
    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        face = cv2.imread(sys.argv[1])
        if face is None:
            print(f"Cannot read {sys.argv[1]}")
            sys.exit(1)
        aligner = FaceAligner()
        aligned, lms = aligner.align_with_landmarks(face)
        if aligned is None:
            print("Landmark detection failed (face too small or low quality?)")
        else:
            cv2.imwrite("aligned.jpg", aligned)
            cv2.imwrite("landmarks_viz.jpg", draw_landmarks(face, lms))
            print(f"OK: aligned 112x112 -> aligned.jpg")
            print(f"     landmarks viz -> landmarks_viz.jpg")
    else:
        print("Usage: python alignment.py <face_image.jpg>")
