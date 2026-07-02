import os

# Keep TensorFlow off the GPU. The emotion CNN (Mini-Xception) is small and
# runs fine on CPU; leaving TF on GPU causes CUDA-context conflicts with
# onnxruntime-gpu (used by InsightFace) and segfaults at init time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _preload_nvidia_pypi_libs():
    """
    Preload CUDA/cuDNN/cuBLAS shared objects that ship inside the
    `nvidia-*-cu12` pip packages so onnxruntime's dlopen("libcudnn.so.9") etc.
    succeed without the user having to set LD_LIBRARY_PATH.
    """
    import ctypes
    import glob
    import site

    # Most critical SONAMEs first — onnxruntime CUDA provider needs these.
    wanted = [
        ("cuda_runtime", "libcudart.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cublas", "libcublasLt.so.12"),
        ("cufft", "libcufft.so.11"),
        ("curand", "libcurand.so.10"),
        ("cusolver", "libcusolver.so.11"),
        ("cusparse", "libcusparse.so.12"),
        ("cudnn", "libcudnn.so.9"),
    ]

    search_roots = []
    for sp in site.getsitepackages():
        search_roots.append(os.path.join(sp, "nvidia"))
    user_sp = site.getusersitepackages()
    if user_sp:
        search_roots.append(os.path.join(user_sp, "nvidia"))
    # Extra root: a side-car install of cuDNN 8 for torch 2.2, which hard-links
    # libcudnn.so.8 via its RPATH. Site-packages cuDNN must stay at v9 for
    # onnxruntime-gpu 1.19, so v8 is installed with `pip install --target`
    # to this path (see the "how to install" note near the bottom of README).
    extra_cudnn8 = os.path.expanduser("~/.local/cudnn8_pkg/nvidia")
    if os.path.isdir(extra_cudnn8):
        search_roots.append(extra_cudnn8)

    for subdir, soname in wanted:
        found = None
        for root in search_roots:
            matches = glob.glob(os.path.join(root, subdir, "lib", soname))
            if matches:
                found = matches[0]
                break
        if found:
            try:
                ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    # Also preload every libcudnn*.so.8 we can find (libcudnn_ops_infer,
    # libcudnn_cnn_infer, etc). torch dlopens them lazily via their soname, and
    # having them in the already-loaded set means glibc short-circuits the
    # linker-path search.
    for root in search_roots:
        for so in glob.glob(os.path.join(root, "cudnn", "lib", "libcudnn*.so.8")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_nvidia_pypi_libs()

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import queue
import logging

# Disable TF GPU visibility *before* the emotion model loads.
import tensorflow as _tf
try:
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from importlib.resources import files as _pkg_files

from insightface.app import FaceAnalysis

from database_manager import DatabaseManager
from faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class FacialRecognitionSystem:
    """
    Consolidated facial-recognition pipeline backed by InsightFace (detection +
    bounding boxes + landmarks + 512-d embeddings + age + gender) and a
    Mini-Xception CNN for emotion classification, run directly on the
    InsightFace bbox crop (no secondary Haar-cascade detection pass).
    """

    # InsightFace buffalo_s produces L2-normalised 512-d embeddings (w600k_mbf,
    # MobileFaceNet). With normalised vectors, L2² = 2 - 2·cos_sim; see
    # `recognition_threshold` in __init__ for the chosen operating point.
    EMBEDDING_DIM = 512

    # FER-2013 class ordering — matches the softmax outputs of the emotion
    # model shipped in the `fer` package (emotion_model.hdf5).
    _EMOTION_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

    def __init__(self, db_config: Dict, camera_id: str = "0"):
        self.camera_id = camera_id

        print("Initializing database connection...")
        self.db = DatabaseManager(**db_config)
        self.db.connect()

        # buffalo_s: detection (det_500m) + recognition (w600k_mbf, 512-d
        # MobileFaceNet) + genderage head. The genderage head provides age/gender.
        print("Initializing InsightFace (buffalo_s) on CUDA...")
        # cudnn_conv_algo_search=HEURISTIC avoids the EXHAUSTIVE default, which
        # internally uses stream capture for per-algorithm timing. When we run
        # InsightFace concurrently with torch models (YOLO plate/fire/weapon,
        # SlowFast HAR) on the same GPU, the in-flight capture trips every
        # parallel torch kernel with "operation not permitted when stream is
        # capturing". HEURISTIC picks an algo from heuristics without any
        # benchmarking, so no capture and no cross-thread collision.
        self.face_app = FaceAnalysis(
            name="buffalo_s",
            providers=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        print("Loading emotion CNN (Mini-Xception) on CPU...")
        # We reuse the .hdf5 weights bundled with the `fer` package but skip
        # its API entirely: no Haar cascade, no redundant face detection. The
        # model is fed the tight InsightFace bbox crop directly.
        emotion_model_path = str(_pkg_files("fer") / "data" / "emotion_model.hdf5")
        self._emotion_classifier = _tf.keras.models.load_model(
            emotion_model_path, compile=False
        )
        # input_shape is (None, H, W, 1); cv2.resize expects (W, H).
        _, _emo_h, _emo_w, _ = self._emotion_classifier.input_shape
        self._emotion_target_size = (_emo_w, _emo_h)

        print("Initializing Faiss index...")
        self.faiss_index = FaissIndex(dimension=self.EMBEDDING_DIM)

        self._load_embeddings_from_db()

        # w600k_mbf (MobileFaceNet) embeddings in buffalo_s. For L2-normalised
        # vectors, L2² = 2 - 2·cos_sim, so L2=1.0 cos_sim=0.5. Empirically the
        # same-person distribution on this webcam sits in [0.6, 0.85].
        self.recognition_threshold = 1.0
        # Confidence is cosine similarity (0..1); 0.4 is a conservative lower
        # bound — cos_sim ≥ 0.4 is the classic ArcFace "same person" operating
        # point.
        self.min_confidence = 0.4

        # Set FR_DEBUG=1 to print the raw top-1 L2 distance for every detected
        # face — useful for calibrating `recognition_threshold` empirically.
        self._debug_distances = os.environ.get("FR_DEBUG", "") not in ("", "0", "false", "False")

        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_running = False

    # --------------------------------------------------------------------- #
    #                         INITIALIZATION HELPERS                        #
    # --------------------------------------------------------------------- #

    def _load_embeddings_from_db(self):
        """Load all existing embeddings from database into Faiss index."""
        try:
            embeddings_data = self.db.get_all_embeddings()
            if embeddings_data:
                # Guard against embeddings whose dimension does not match the model.
                first_dim = np.asarray(embeddings_data[0]["embedding"]).shape[-1]
                if first_dim != self.EMBEDDING_DIM:
                    print(
                        f"⚠  Found {len(embeddings_data)} embedding(s) with "
                        f"dim={first_dim} (expected {self.EMBEDDING_DIM}); "
                        "incompatible — skipping load."
                    )
                    return
                print(f"Loading {len(embeddings_data)} embeddings into index...")
                self.faiss_index.rebuild_index(embeddings_data)
                print("Embeddings loaded successfully")
            else:
                print("No existing embeddings found in database")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

    # --------------------------------------------------------------------- #
    #                           INTERNAL HELPERS                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _bbox_xyxy_to_xywh(bbox: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert InsightFace [x1,y1,x2,y2] float bbox to clamped (x,y,w,h) ints."""
        h_img, w_img = frame_shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img))
        y2 = max(0, min(y2, h_img))
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    def _extract_single_face(self, image: np.ndarray):
        """
        Run InsightFace on an image that should contain exactly one face.
        Returns the first `Face` object or None.
        """
        if image is None or image.size == 0:
            return None
        faces = self.face_app.get(image)
        if not faces:
            return None
        # If multiple detected, pick the largest by bbox area.
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
        return faces[0]

    def _classify_emotion(self, face_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Run the Mini-Xception emotion CNN on a tightly-cropped face (BGR).
        Returns (emotion, score) or (None, None). Preprocessing mirrors
        FER's internal pipeline: BGR→gray → resize to model input → scale
        to [-1, 1] → (1, H, W, 1) tensor.
        """
        if face_crop is None or face_crop.size == 0:
            return None, None
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self._emotion_target_size)
            x = gray.astype("float32") / 255.0
            x = (x - 0.5) * 2.0
            x = np.expand_dims(x, axis=(0, -1))
            probs = self._emotion_classifier(x, training=False).numpy()[0]
            idx = int(np.argmax(probs))
            return self._EMOTION_LABELS[idx], float(probs[idx])
        except Exception as e:
            logger.debug(f"Emotion classification failed: {e}")
            return None, None

    # --------------------------------------------------------------------- #
    #                           REGISTRATION API                            #
    # --------------------------------------------------------------------- #

    def register_person(self,
                        name: str,
                        employee_id: str,
                        face_images: List[np.ndarray],
                        department: str = None,
                        authorized_zones: List[str] = None) -> bool:
        """Register a new person in the system."""
        try:
            person_id = self.db.add_person(name, employee_id, department, authorized_zones)

            embeddings = []
            for face_image in face_images:
                face = self._extract_single_face(face_image)
                if face is None:
                    continue
                embeddings.append(np.asarray(face.normed_embedding, dtype=np.float32))

            if not embeddings:
                print(f"Failed to extract embeddings for {name}")
                return False

            for embedding in embeddings:
                self.db.add_face_embedding(person_id, embedding)
                self.faiss_index.add_embedding(embedding, person_id)

            print(f"Successfully registered {name} with {len(embeddings)} face embeddings")
            return True

        except Exception as e:
            print(f"Error registering person: {e}")
            return False

    def extract_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Helper for callers (e.g. the face-upload endpoint) that want to turn a
        full input image into a single 512-d InsightFace embedding. Returns
        None if zero or multiple faces are found.
        """
        if image is None or image.size == 0:
            return None
        faces = self.face_app.get(image)
        if len(faces) != 1:
            return None
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)

    # --------------------------------------------------------------------- #
    #                         LOAD FACES FROM FILES                         #
    # --------------------------------------------------------------------- #

    def load_faces_from_files(self,
                              image_paths: List[str],
                              num_faces: int = 20) -> List[np.ndarray]:
        """
        Load and validate face images from disk. Returns the **full** source
        images (not cropped) that contain exactly one detectable face, so the
        downstream InsightFace detect+embed pass in `register_person` has the
        surrounding context it needs to relocate the face at its natural scale.
        """
        valid_faces: List[np.ndarray] = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not read image: {path}")
                continue

            faces = self.face_app.get(img)
            if len(faces) != 1:
                print(f"[WARN] Skipping {path}: expected 1 face, found {len(faces)}")
                continue

            valid_faces.append(img)
            print(f"[INFO] Added face from {path} ({len(valid_faces)}/{num_faces})")

            if len(valid_faces) >= num_faces:
                break

        return valid_faces

    # --------------------------------------------------------------------- #
    #                       RECOGNITION & STREAMING                         #
    # --------------------------------------------------------------------- #

    def _match_embedding(self, embedding: np.ndarray) -> Optional[Dict]:
        """Search the Faiss index for the closest person; return a result dict or None."""
        # Unbounded search first so we can log the actual nearest distance for
        # threshold calibration, then apply the configured cutoff ourselves.
        raw = self.faiss_index.search(embedding, k=1, threshold=float("inf"))
        if not raw:
            if self._debug_distances:
                print("[FR_DEBUG] no candidates in index")
            return None

        person_id, distance = raw[0]

        if self._debug_distances:
            print(f"[FR_DEBUG] top-1 person_id={person_id} L2={distance:.4f} "
                  f"(threshold={self.recognition_threshold:.2f})")

        if distance >= self.recognition_threshold:
            return None

        # For L2-normalised vectors, cos_sim = 1 - L2²/2. Report it directly as
        # the confidence so the number matches the standard face-ID metric.
        cos_sim = 1.0 - (float(distance) ** 2) / 2.0
        confidence = max(0.0, min(1.0, cos_sim))
        person_info = self.db.get_person_by_id(person_id)
        if not person_info:
            return None

        return {
            "person_id": person_id,
            "name": person_info["name"],
            "employee_id": person_info["employee_id"],
            "department": person_info["department"],
            "confidence": float(confidence),
            "distance": float(distance),
        }

    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Back-compat single-face recognition helper. Runs InsightFace on the
        supplied image and matches against the Faiss index.
        """
        face = self._extract_single_face(face_image)
        if face is None:
            return None
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        return self._match_embedding(embedding)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        One InsightFace pass per frame → bbox + landmarks + embedding + age +
        gender. Emotion is classified by feeding the same InsightFace bbox
        crop directly into the Mini-Xception CNN. Returns the annotated frame
        and a list of per-face result dicts.
        """
        faces = self.face_app.get(frame)
        recognition_results: List[Dict] = []
        annotated_frame = frame.copy()

        for face in faces:
            bbox_xywh = self._bbox_xyxy_to_xywh(face.bbox, frame.shape)
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0:
                continue

            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            match = self._match_embedding(embedding)
            is_known = bool(match and match["confidence"] >= self.min_confidence)

            # Demographics (age/gender/emotion) only for unknown faces — known
            # persons already have identity info, so there's no reason to run
            # the extra emotion CNN pass on them every frame.
            if is_known:
                age, gender, emotion, emotion_conf = None, None, None, None
            else:
                age = int(round(float(face.age))) if getattr(face, "age", None) is not None else None
                # InsightFace gender: 1 = Male, 0 = Female.
                gender_raw = getattr(face, "gender", None)
                gender = "Man" if gender_raw == 1 else ("Woman" if gender_raw == 0 else None)

                emotion, emotion_conf = self._classify_emotion(frame[y:y + h, x:x + w])

            if is_known:
                result = {
                    **match,
                    "bbox": bbox_xywh,
                    "timestamp": datetime.now(),
                    "age": age,
                    "gender": gender,
                    "emotion": emotion,
                    "emotion_confidence": emotion_conf,
                }
                self.db.log_access(match["person_id"], self.camera_id, match["confidence"])
            else:
                result = {
                    "name": "Unknown",
                    "confidence": 0.0,
                    "person_id": None,
                    "employee_id": None,
                    "department": None,
                    "bbox": bbox_xywh,
                    "timestamp": datetime.now(),
                    "age": age,
                    "gender": gender,
                    "emotion": emotion,
                    "emotion_confidence": emotion_conf,
                }

            recognition_results.append(result)
            self._draw_face(annotated_frame, result)

        return annotated_frame, recognition_results

    @staticmethod
    def _draw_face(image: np.ndarray, result: Dict) -> None:
        """Draw bbox + label for one face result onto `image` in place."""
        x, y, w, h = result["bbox"]
        is_known = result.get("name") and result["name"] != "Unknown"
        color = (0, 255, 0) if is_known else (0, 165, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        name = result.get("name", "Unknown")
        conf = result.get("confidence", 0.0) or 0.0
        label = f"{name} ({conf:.2f})" if is_known else name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(image, label, (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def start_video_stream(self, source: int = 0):
        """Start processing a live camera or video file stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print("Starting video stream... (press 'q' to quit)")

        fps_start_time = time.time()
        fps_frame_count, fps = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, results = self.process_frame(frame)

            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time, fps_frame_count = fps_end_time, 0

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Facial Recognition System", annotated_frame)

            for result in results:
                print(f"Recognized: {result['name']} (ID: {result['employee_id']}) "
                      f"with confidence: {result['confidence']:.2f} "
                      f"[age={result.get('age')}, gender={result.get('gender')}, "
                      f"emotion={result.get('emotion')}]")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # --------------------------------------------------------------------- #
    #                     CAPTURE FACES FROM CAMERA                         #
    # --------------------------------------------------------------------- #

    def capture_faces_for_registration(self,
                                       source: int = 0,
                                       num_faces: int = 20) -> List[np.ndarray]:
        """Capture face images for registration via live camera or video file."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return []

        print(f"Capturing {num_faces} face images. Press 'c' to capture, 'q' to quit")

        captured_faces: List[np.ndarray] = []

        while len(captured_faces) < num_faces:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.face_app.get(frame)
            display_frame = frame.copy()
            for face in faces:
                x, y, w, h = self._bbox_xyxy_to_xywh(face.bbox, frame.shape)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(display_frame,
                        f"Captured: {len(captured_faces)}/{num_faces}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) == 1:
                # Store the full frame; register_person re-runs InsightFace on
                # the uncropped image so the detector has enough context.
                captured_faces.append(frame.copy())
                print(f"Captured face {len(captured_faces)}/{num_faces}")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured_faces

    # --------------------------------------------------------------------- #
    #                               CLEANUP                                 #
    # --------------------------------------------------------------------- #

    def cleanup(self):
        """Disconnect database and release resources."""
        self.db.disconnect()
        print("System cleaned up")
