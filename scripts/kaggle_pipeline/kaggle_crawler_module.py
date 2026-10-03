"""
Kaggle In-Cloud Multi-Source Crawler & Dataset Ingestion Engine
==============================================================
Runs directly inside Kaggle compute environment.
Aggregates:
1. Large-scale public unconstrained face repositories (FairFace, CelebA, AsianFace, APPA-REAL, B3FD).
2. Automated multi-threaded demographic query crawler (Wikimedia Commons, Public APIs, Free Media).
Streams directly to /kaggle/working/raw_face_pool without disk bottlenecks.
"""

from __future__ import annotations
import os
import sys
import glob
import json
import time
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Generator


class KagglePublicDatasetHarvester:
    """
    Scans and harvests images from mounted Kaggle datasets.
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, kaggle_input_dir: str = "/kaggle/input"):
        self.input_dir = Path(kaggle_input_dir)

    def scan_mounted_datasets(self) -> List[Path]:
        """Returns list of mounted dataset folders."""
        if not self.input_dir.exists():
            return []
        return [p for p in self.input_dir.iterdir() if p.is_dir()]

    def harvest_image_paths(self, max_per_dataset: int = 50000) -> List[Path]:
        """
        Discovers all available face images across mounted Kaggle datasets.
        """
        all_images = []
        for dataset_dir in self.scan_mounted_datasets():
            count = 0
            print(f"[Harvester] Scanning {dataset_dir.name}...")
            for root, _, files in os.walk(dataset_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        all_images.append(Path(root) / f)
                        count += 1
                        if count >= max_per_dataset:
                            break
                if count >= max_per_dataset:
                    break
            print(f"[Harvester] Found {count} images in {dataset_dir.name}")
        return all_images


class InCloudDemographicCrawler:
    """
    Automated demographic face crawler running in Kaggle kernel.
    Queries open high-diversity media sources with structured demographic terms.
    """

    DEMOGRAPHIC_MATRICES = {
        "ethnicities": [
            "Vietnamese", "East Asian", "Southeast Asian", "Japanese", "Korean",
            "South Asian Indian", "Caucasian European", "African Black", "Latino Hispanic", "Middle Eastern"
        ],
        "age_brackets": [
            "newborn baby infant", "toddler preschooler 4 years old", "school child 9 years old",
            "teenager high school 15 years old", "young adult 22 years old", "person 30 years old",
            "adult 40 years old", "middle aged person 50 years old", "mature adult 60 years old",
            "senior elderly 70 years old", "centenarian 80 years old"
        ],
        "genders": ["male portrait face", "female portrait face"]
    }

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def __init__(self, output_dir: str = "/kaggle/working/raw_face_pool/web_crawled", max_workers: int = 8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def generate_query_matrix(self) -> Generator[str, None, None]:
        """Generates stratified search queries covering all demographic intersections."""
        for eth in self.DEMOGRAPHIC_MATRICES["ethnicities"]:
            for age in self.DEMOGRAPHIC_MATRICES["age_brackets"]:
                for g in self.DEMOGRAPHIC_MATRICES["genders"]:
                    yield f"portrait photo {eth} {age} {g}"

    def download_url(self, item: Dict[str, str]) -> bool:
        """Downloads a single image URL with timeout and error handling."""
        url = item['url']
        dest_path = item['dest']
        try:
            req = urllib.request.Request(url, headers=self.HEADERS)
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                    return True
        except Exception:
            return False
        return False

    def query_wikimedia_commons(self, query: str, limit: int = 20) -> List[str]:
        """
        Queries Wikimedia Commons open public domain API for high-resolution faces.
        """
        image_urls = []
        try:
            endpoint = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{query} face portrait filetype:bitmap",
                "gsrlimit": min(limit, 50),
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "format": "json"
            }
            url = f"{endpoint}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=self.HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                pages = data.get("query", {}).get("pages", {})
                for page_id, info in pages.items():
                    imageinfo = info.get("imageinfo", [])
                    if imageinfo:
                        img_url = imageinfo[0].get("url")
                        mime = imageinfo[0].get("mime", "")
                        if img_url and ("jpeg" in mime or "png" in mime or "webp" in mime):
                            image_urls.append(img_url)
        except Exception as e:
            # Tolerant to network limits
            pass
        return image_urls

    def crawl_batch(self, max_queries: int = 30, images_per_query: int = 15) -> int:
        """
        Executes parallel multi-source crawl across demographic queries.
        """
        queries = list(self.generate_query_matrix())[:max_queries]
        print(f"[Crawler] Starting crawl for {len(queries)} demographic queries...")

        tasks = []
        task_id = int(time.time())

        for q_idx, q in enumerate(queries):
            urls = self.query_wikimedia_commons(q, limit=images_per_query)
            for u_idx, u in enumerate(urls):
                ext = ".jpg"
                filename = f"crawled_q{q_idx:03d}_{u_idx:03d}_{task_id}{ext}"
                dest = str(self.output_dir / filename)
                tasks.append({'url': u, 'dest': dest})

        print(f"[Crawler] Found {len(tasks)} target URLs. Downloading with {self.max_workers} threads...")
        downloaded = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.download_url, tasks)
            for res in results:
                if res:
                    downloaded += 1

        print(f"[Crawler] Successfully downloaded {downloaded}/{len(tasks)} images.")
        return downloaded
