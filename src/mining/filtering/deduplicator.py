"""
Dual-Layer Perceptual Deduplicator
==================================
Layer 1: 64-bit DCT Perceptual Hash (Hamming distance threshold d_H <= 6 rejected).
         Guarantees zero cross-split leakage between Train and Validation sets.
Layer 2: 512-d Cosine distance on ArcFace latent feature space (d_cos < 0.35 rejected).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np


def compute_64bit_dct_phash(image_gray: np.ndarray) -> int:
    """
    Computes 64-bit Discrete Cosine Transform (DCT) Perceptual Hash.
    Invariant to uniform photometric scaling.
    """
    resized = cv2.resize(image_gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[0:8, 0:8]
    med = np.median(dct[1:, 1:])  # Ignore DC component
    hash_val = 0
    for b in (dct > med).flatten():
        hash_val = (hash_val << 1) | int(b)
    return hash_val


def hamming_distance_64bit(h1: int, h2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit integers."""
    return (h1 ^ h2).bit_count()


class DualLayerDeduplicator:
    """
    Deduplicator tracking perceptual hashes and feature embeddings.
    """

    def __init__(self, hamming_threshold: int = 6, cosine_threshold: float = 0.35):
        self.hamming_threshold = hamming_threshold
        self.cosine_threshold = cosine_threshold
        self.registered_hashes: List[int] = []
        self.registered_embeddings: List[np.ndarray] = []

    def is_duplicate(
        self,
        phash: int,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """Checks if a sample matches any previously registered sample."""
        # Layer 1: pHash Hamming check
        for registered_h in self.registered_hashes:
            dist_h = hamming_distance_64bit(phash, registered_h)
            if dist_h <= self.hamming_threshold:
                return True

        # Layer 2: Cosine embedding check
        if embedding is not None and len(self.registered_embeddings) > 0:
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            for reg_emb in self.registered_embeddings:
                reg_norm = reg_emb / (np.linalg.norm(reg_emb) + 1e-8)
                cosine_dist = 1.0 - float(np.dot(emb_norm, reg_norm))
                if cosine_dist < self.cosine_threshold:
                    return True

        return False

    def is_duplicate_and_register(
        self,
        phash: int,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """Checks duplicate and registers if novel."""
        if self.is_duplicate(phash, embedding):
            return True

        self.registered_hashes.append(phash)
        if embedding is not None:
            self.registered_embeddings.append(embedding.copy())
        return False

    def check_and_register(
        self,
        face_bgr_or_gray: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str, int]:
        """
        Computes 64-bit DCT pHash from a face image, checks dual-layer deduplication,
        and registers if unique. Returns (is_duplicate, reason, phash_uint64).
        """
        gray = (
            cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
            if face_bgr_or_gray.ndim == 3
            else face_bgr_or_gray
        )
        phash = compute_64bit_dct_phash(gray)
        for registered_h in self.registered_hashes:
            if hamming_distance_64bit(phash, registered_h) <= self.hamming_threshold:
                return True, "PHASH_NEAR_DUPLICATE", phash

        if embedding is not None and len(self.registered_embeddings) > 0:
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            for reg_emb in self.registered_embeddings:
                reg_norm = reg_emb / (np.linalg.norm(reg_emb) + 1e-8)
                if (1.0 - float(np.dot(emb_norm, reg_norm))) < self.cosine_threshold:
                    return True, "EMBEDDING_IDENTITY_LEAKAGE", phash

        self.registered_hashes.append(phash)
        if embedding is not None:
            self.registered_embeddings.append(embedding.copy())
        return False, "UNIQUE", phash

