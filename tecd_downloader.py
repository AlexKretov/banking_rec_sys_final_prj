#!/usr/bin/env python3
"""
T-ECD Dataset Downloader
Supports selective downloading by domains and date ranges.
"""

import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import List, Tuple

import polars as pl
from huggingface_hub import snapshot_download
from tqdm import tqdm


@dataclass
class DownloadConfig:
    """Configuration for dataset download parameters."""

    token: str
    dataset_path: str = "dataset/full"
    local_dir: str = "t_ecd_full"
    domains: Tuple[str, ...] = (
        "retail",
        "marketplace",
        "offers",
        "reviews",
        "payments",
    )
    day_begin: int = 0
    day_end: int = 1308  # inclusive
    max_workers: int = 20


class DatasetDownloader:
    """Handles downloading of T-ECD dataset."""

    STATIC_FILES = ["users.pq", "brands.pq"]
    DOMAIN_ITEMS = ["retail", "marketplace", "offers"]
    ALL_DOMAINS = ["retail", "marketplace", "offers", "reviews", "payments"]

    def __init__(self, config: DownloadConfig):
        self.config = config
        self._ensure_local_dir()

    def _ensure_local_dir(self) -> None:
        """Create local directory if it doesn't exist."""
        Path(self.config.local_dir).mkdir(parents=True, exist_ok=True)

    def _generate_file_patterns(self) -> List[str]:
        """Generate all file patterns to download based on configuration."""
        patterns = []

        # Add static files
        patterns.extend(
            f"{self.config.dataset_path}/{file}" for file in self.STATIC_FILES
        )

        # Add domain-specific item files
        patterns.extend(
            f"{self.config.dataset_path}/{domain}/items.pq"
            for domain in self.config.domains
            if domain in self.DOMAIN_ITEMS
        )

        # Add daily files for each domain
        for domain in self.config.domains:
            for day in range(self.config.day_begin, self.config.day_end + 1):
                day_str = str(day).zfill(5)
                patterns.extend(self._get_domain_day_patterns(domain, day_str))

        return patterns

    def _get_domain_day_patterns(self, domain: str, day_str: str) -> List[str]:
        """Get file patterns for a specific domain and day."""
        base_path = f"{self.config.dataset_path}/{domain}"

        if domain in ["retail", "marketplace", "offers"]:
            return [f"{base_path}/events/{day_str}.pq"]
        elif domain == "payments":
            return [
                f"{base_path}/events/{day_str}.pq",
                f"{base_path}/receipts/{day_str}.pq",
            ]
        elif domain == "reviews":
            return [f"{base_path}/{day_str}.pq"]

        return []

    def _download_single_file(self, pattern: str) -> Tuple[str, bool]:
        """Download a single file."""
        try:
            snapshot_download(
                repo_id="t-tech/T-ECD",
                repo_type="dataset",
                allow_patterns=pattern,
                local_dir=self.config.local_dir,
                token=self.config.token,
            )
            return pattern, True
        except Exception as e:
            print(f"Error downloading {pattern}: {e}")
            return pattern, False

    def download(self) -> List[str]:
        """Download all specified files in parallel.
        Returns:
            List of failed download patterns
        """
        patterns = self._generate_file_patterns()
        print(f"Downloading {len(patterns)} files to {self.config.local_dir}")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            # Submit all download tasks
            future_to_pattern = {
                executor.submit(self._download_single_file, pattern): pattern
                for pattern in patterns
            }

            # Process results with progress bar
            results = []
            for future in tqdm(
                concurrent.futures.as_completed(future_to_pattern),
                total=len(patterns),
                desc="Downloading files",
            ):
                results.append(future.result())

        # Report results
        successful = sum(1 for _, status in results if status)
        failed = [pattern for pattern, status in results if not status]

        print(f"Download completed: {successful}/{len(patterns)} files successful")

        if failed:
            print("Failed downloads:")
            for pattern in sorted(failed):
                print(f"  - {pattern}")

        return failed


def create_config_from_args(args) -> DownloadConfig:
    """Create DownloadConfig from command line arguments."""
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        token = getpass("Enter your Hugging Face token: ")

    domains = args.domains if args.domains else DatasetDownloader.ALL_DOMAINS

    return DownloadConfig(
        token=token,
        dataset_path=args.dataset_path,
        local_dir=args.local_dir,
        domains=tuple(domains),
        day_begin=args.day_begin,
        day_end=args.day_end,
        max_workers=args.max_workers,
    )


def download_dataset(
    token: str,
    local_dir: str = "t_ecd_full",
    dataset_path: str = "dataset/full",
    domains: List[str] = None,
    day_begin: int = 1307,
    day_end: int = 1308,
    max_workers: int = 20,
) -> List[str]:
    """High-level function to download T-ECD dataset.
    Args:
        token: Hugging Face authentication token
        local_dir: Local directory to save dataset
        dataset_path: Path within the dataset repository
        domains: List of domains to download
        day_begin: Start day (inclusive)
        day_end: End day (inclusive)
        max_workers: Number of parallel download workers
    Returns:
        List of failed download patterns
    """
    if domains is None:
        domains = DatasetDownloader.ALL_DOMAINS

    config = DownloadConfig(
        token=token,
        local_dir=local_dir,
        dataset_path=dataset_path,
        domains=tuple(domains),
        day_begin=day_begin,
        day_end=day_end,
        max_workers=max_workers,
    )

    downloader = DatasetDownloader(config)
    return downloader.download()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download T-ECD dataset from Hugging Face Hub"
    )
    parser.add_argument(
        "--token", "-t", help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--local-dir",
        "-d",
        default="t_ecd_full",
        help="Local directory to save dataset",
    )
    parser.add_argument(
        "--dataset-path",
        "-p",
        default="dataset/full",
        help="Path within the dataset repository",
    )
    parser.add_argument(
        "--domains",
        "-m",
        nargs="+",
        choices=DatasetDownloader.ALL_DOMAINS,
        help="Domains to download (default: all)",
    )
    parser.add_argument(
        "--day-begin", "-b", type=int, default=1307, help="Start day (inclusive)"
    )
    parser.add_argument(
        "--day-end", "-e", type=int, default=1308, help="End day (inclusive)"
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=20,
        help="Number of parallel download workers",
    )

    args = parser.parse_args()

    try:
        config = create_config_from_args(args)
        downloader = DatasetDownloader(config)
        failed_downloads = downloader.download()

        exit_code = 1 if failed_downloads else 0
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

