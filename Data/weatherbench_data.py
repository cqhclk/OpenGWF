import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ERA5(Dataset):
    """NPY WeatherBench-style dataset.

    Expected file layout:
        data_folder/
          mean_std.npy or max_min.npy
          2006/2006-0000.npy
          2006/2006-0001.npy

    Each frame is stored as a `(C, H, W)` numpy array.
    """

    def __init__(
        self,
        start_time="2006-01-01 00:00:00",
        end_time="2015-12-31 23:00:00",
        interval=1,
        input_len=12,
        output_len=12,
        data_folder=None,
        normalization="meanstd",
        channels=None,
        crop=None,
    ):
        default_folder = Path(__file__).resolve().parent / "weatherbench"
        self.data_folder = Path(data_folder or os.environ.get("WEATHER_DATA_DIR", default_folder))
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.interval = int(interval)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.normalization = normalization
        self.channels = channels
        self.crop = crop

        self._load_stats()
        max_start = self.end_time - pd.Timedelta(hours=self.interval * (self.input_len + self.output_len - 1))
        self.x_time_list = pd.date_range(self.start_time, max_start, freq=f"{self.interval}h")
        self.length = len(self.x_time_list)

    @staticmethod
    def idx_in_year(time_stamp):
        first_day = pd.to_datetime(f"{time_stamp.year}-01-01 00:00:00")
        return int((time_stamp - first_day).total_seconds() / 3600)

    def time_to_path(self, time_stamp):
        return self.data_folder / str(time_stamp.year) / f"{time_stamp.year}-{self.idx_in_year(time_stamp):04d}.npy"

    def _load_stats(self):
        self.mean = self.std = self.vmin = self.vmax = None
        if self.normalization == "meanstd":
            stats = np.load(self.data_folder / "mean_std.npy").astype(np.float32)
            self.mean, self.std = stats[:, 0], stats[:, 1]
        elif self.normalization == "maxmin":
            stats = np.load(self.data_folder / "max_min.npy").astype(np.float32)
            self.vmax, self.vmin = stats[:, 0], stats[:, 1]
        elif self.normalization in {"none", None}:
            return
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        if self.channels is not None:
            self.mean = None if self.mean is None else self.mean[: self.channels]
            self.std = None if self.std is None else self.std[: self.channels]
            self.vmax = None if self.vmax is None else self.vmax[: self.channels]
            self.vmin = None if self.vmin is None else self.vmin[: self.channels]

    def _normalize(self, sample):
        if self.normalization == "meanstd":
            return (sample - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        if self.normalization == "maxmin":
            return (sample - self.vmin[None, :, None, None]) / (
                self.vmax[None, :, None, None] - self.vmin[None, :, None, None]
            )
        return sample

    def _load_frame(self, time_stamp):
        frame = np.load(self.time_to_path(time_stamp)).astype(np.float32)
        if self.channels is not None:
            frame = frame[: self.channels]
        return frame

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        t0 = self.x_time_list[index]
        x_times = [t0 + pd.Timedelta(hours=self.interval * i) for i in range(self.input_len)]
        y_times = [
            t0 + pd.Timedelta(hours=self.interval * (self.input_len + i))
            for i in range(self.output_len)
        ]
        sample_x = np.stack([self._load_frame(ts) for ts in x_times], axis=0)
        sample_y = np.stack([self._load_frame(ts) for ts in y_times], axis=0)

        sample_x = self._normalize(sample_x)
        sample_y = self._normalize(sample_y)
        if self.crop is not None:
            top, bottom, left, right = self.crop
            sample_x = sample_x[:, :, top:bottom, left:right]
            sample_y = sample_y[:, :, top:bottom, left:right]
        return torch.from_numpy(sample_x).float(), torch.from_numpy(sample_y).float()
