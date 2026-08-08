from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - model-only environments need not install xarray
    xr = None
    _XARRAY_IMPORT_ERROR = exc
else:
    _XARRAY_IMPORT_ERROR = None

from variables import FieldDefinition, VariableRegistry


LATITUDE_NAMES = ("latitude", "lat")
LONGITUDE_NAMES = ("longitude", "lon")
LEVEL_NAMES = ("level", "pressure_level", "isobaricInhPa", "plev")


def calendar_features_from_times(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise TypeError("ERA5 time coordinates must be numpy datetime64 values")
    months = values.astype("datetime64[M]").astype(np.int64) % 12 + 1
    days = values.astype("datetime64[D]")
    hours = (values.astype("datetime64[h]") - days).astype("timedelta64[h]")
    hours = hours.astype(np.int64)
    return np.stack([months, hours], axis=-1).astype(np.float32)


def _require_xarray() -> None:
    if xr is None:
        raise ImportError(
            "xarray is required for ERA5 data loading. Install requirements.txt in the training environment."
        ) from _XARRAY_IMPORT_ERROR


def _first_existing(names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    available = set(candidates)
    return next((name for name in names if name in available), None)


def _canonicalize_spatial_names(array):
    latitude = _first_existing(LATITUDE_NAMES, array.dims)
    longitude = _first_existing(LONGITUDE_NAMES, array.dims)
    if latitude is None or longitude is None:
        raise ValueError(f"Expected latitude/longitude dimensions, got {array.dims}")
    rename = {}
    if latitude != "latitude":
        rename[latitude] = "latitude"
    if longitude != "longitude":
        rename[longitude] = "longitude"
    return array.rename(rename)


def _resolve_field_array(dataset, field: FieldDefinition):
    candidates = (field.dataset_name,) + tuple(field.aliases)
    if not field.is_surface:
        candidates += tuple(f"{name}_{int(field.pressure_hpa)}" for name in candidates)

    variable_name = _first_existing(candidates, dataset.data_vars)
    if variable_name is None:
        raise KeyError(
            f"Cannot resolve field {field.key}. Tried variables {candidates}; "
            f"dataset contains {list(dataset.data_vars)[:20]}..."
        )

    array = dataset[variable_name]
    if not field.is_surface:
        level_name = _first_existing(LEVEL_NAMES, array.dims)
        if level_name is not None:
            level_values = np.asarray(array[level_name].values)
            target = field.pressure_hpa
            if np.nanmax(np.abs(level_values)) > 2000:
                target *= 100.0
            array = array.sel({level_name: target}, method="nearest")
    array = _canonicalize_spatial_names(array)
    extra_dims = [dim for dim in array.dims if dim not in {"time", "latitude", "longitude"}]
    for dim in extra_dims:
        if array.sizes[dim] != 1:
            raise ValueError(f"Unresolved dimension {dim} for {field.key}: shape={array.shape}")
        array = array.isel({dim: 0})
    return array.transpose("time", "latitude", "longitude")


def build_field_cube(dataset, registry: VariableRegistry, start: str, end: str):
    """Create lazy [time, field, latitude, longitude] data with canonical field order."""
    _require_xarray()
    dataset = dataset.sel(time=slice(start, end))
    arrays = []
    for field in registry.fields:
        array = _resolve_field_array(dataset, field)
        array = array.expand_dims(field=[field.field_id])
        arrays.append(array)
    cube = xr.concat(arrays, dim="field").transpose("time", "field", "latitude", "longitude")
    cube = cube.assign_coords(field_key=("field", [field.key for field in registry.fields]))
    return cube


def build_field_arrays(
    dataset,
    registry: VariableRegistry,
    start: str,
    end: str,
):
    """Create lazy per-field arrays without building one large xarray concat graph."""
    _require_xarray()
    selected = dataset.sel(time=slice(start, end))
    arrays = tuple(_resolve_field_array(selected, field) for field in registry.fields)
    if not arrays:
        raise ValueError("VariableRegistry must contain at least one field")

    first = arrays[0]
    latitude = np.asarray(first.latitude.values, dtype=np.float32)
    longitude = np.asarray(first.longitude.values, dtype=np.float32)
    times = np.asarray(first.time.values)
    expected_shape = (len(times), len(latitude), len(longitude))

    for field, array in zip(registry.fields, arrays):
        shape = (
            int(array.sizes["time"]),
            int(array.sizes["latitude"]),
            int(array.sizes["longitude"]),
        )
        if shape != expected_shape:
            raise ValueError(
                f"Field {field.key} has shape {shape}, expected {expected_shape}"
            )
    return arrays, times, latitude, longitude


class FieldNormalizer:
    def __init__(self, field_keys: Sequence[str], mean: Sequence[float], std: Sequence[float]):
        self.field_keys = tuple(field_keys)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        if not (len(self.field_keys) == len(self.mean) == len(self.std)):
            raise ValueError("field_keys, mean and std must have equal length")

    @classmethod
    def fit(cls, cube) -> "FieldNormalizer":
        field_keys = cube.field_key.values.tolist()
        mean: List[float] = []
        std: List[float] = []
        total = len(field_keys)
        num_times = int(cube.sizes["time"])
        time_block = min(num_times, 64)

        print(
            "Fitting normalization statistics with streaming field/time blocks "
            f"for {total} fields and {num_times} time steps.",
            flush=True,
        )
        for index, key in enumerate(field_keys):
            print(
                f"[normalization] field {index + 1:02d}/{total}: {key}",
                flush=True,
            )
            field = cube.isel(field=index)
            field_sum = 0.0
            field_sumsq = 0.0
            field_count = 0
            for start in range(0, num_times, time_block):
                stop = min(start + time_block, num_times)
                values = np.asarray(
                    field.isel(time=slice(start, stop)).values,
                    dtype=np.float64,
                )
                finite = np.isfinite(values)
                if finite.any():
                    valid = values[finite]
                    field_sum += float(valid.sum())
                    field_sumsq += float(np.square(valid).sum())
                    field_count += int(valid.size)

            if field_count == 0:
                field_mean = 0.0
                field_std = 1.0
            else:
                field_mean = field_sum / field_count
                variance = max(field_sumsq / field_count - field_mean * field_mean, 0.0)
                field_std = float(np.sqrt(variance))
            mean.append(field_mean)
            std.append(field_std)

        return cls(field_keys, mean, std)


    @staticmethod
    def _stream_statistics(array, time_block: int) -> Tuple[float, float]:
        num_times = int(array.sizes["time"])
        total = 0.0
        total_squares = 0.0
        count = 0
        for start in range(0, num_times, time_block):
            stop = min(start + time_block, num_times)
            values = np.asarray(
                array.isel(time=slice(start, stop)).values,
                dtype=np.float64,
            )
            finite = np.isfinite(values)
            if finite.any():
                valid = values[finite]
                total += float(valid.sum())
                total_squares += float(np.square(valid).sum())
                count += int(valid.size)
        if count == 0:
            return 0.0, 1.0
        average = total / count
        variance = max(total_squares / count - average * average, 0.0)
        return average, float(np.sqrt(variance))


    @classmethod
    def fit_from_dataset(
        cls,
        dataset,
        registry: VariableRegistry,
        start: str,
        end: str,
        time_block: int = 64,
    ) -> "FieldNormalizer":
        """Fit field statistics directly from an ERA5 dataset without concat."""
        _require_xarray()
        selected = dataset.sel(time=slice(start, end))
        field_keys = [field.key for field in registry.fields]
        mean: List[float] = []
        std: List[float] = []
        total = len(registry.fields)
        num_times = int(selected.sizes["time"])
        block = max(1, min(int(time_block), num_times))

        print(
            "Fitting normalization directly from dataset with streaming "
            f"field/time blocks for {total} fields and {num_times} time steps.",
            flush=True,
        )
        for index, field in enumerate(registry.fields):
            print(
                f"[normalization] field {index + 1:02d}/{total}: {field.key}",
                flush=True,
            )
            array = _resolve_field_array(selected, field)
            field_mean, field_std = cls._stream_statistics(array, block)
            mean.append(field_mean)
            std.append(field_std)

        return cls(field_keys, mean, std)


    @classmethod
    def fit_from_path(
        cls,
        dynamic_path: str,
        registry: VariableRegistry,
        start: str,
        end: str,
        time_block: int = 64,
    ) -> "FieldNormalizer":
        _require_xarray()
        dataset = xr.open_dataset(dynamic_path, cache=False)
        try:
            return cls.fit_from_dataset(dataset, registry, start, end, time_block)
        finally:
            dataset.close()

    def validate(self, registry: VariableRegistry) -> None:
        expected = tuple(field.key for field in registry.fields)
        if expected != self.field_keys:
            raise ValueError("Normalizer field order does not match VariableRegistry")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "field_keys": list(self.field_keys),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FieldNormalizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["field_keys"], payload["mean"], payload["std"])

    def with_std_floor(
        self,
        registry: Optional[VariableRegistry] = None,
        minimum_std: float = 0.0,
        variable_minimum_std: Optional[Dict[str, float]] = None,
    ) -> "FieldNormalizer":
        """Return a copy with safer lower bounds for near-constant fields."""

        std = self.std.copy()
        if minimum_std and minimum_std > 0:
            std = np.maximum(std, float(minimum_std))

        if registry is not None and variable_minimum_std:
            key_to_index = {key: index for index, key in enumerate(self.field_keys)}
            for field in registry.fields:
                floor = variable_minimum_std.get(field.variable)
                if floor is None or floor <= 0:
                    continue
                index = key_to_index.get(field.key)
                if index is not None:
                    std[index] = max(float(std[index]), float(floor))

        return FieldNormalizer(self.field_keys, self.mean, std)

    def normalize_array(self, values: np.ndarray) -> np.ndarray:
        shape = (1,) * (values.ndim - 3) + (len(self.mean), 1, 1)
        return (values - self.mean.reshape(shape)) / self.std.reshape(shape)

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        shape = (1,) * (values.ndim - 3) + (len(self.mean), 1, 1)
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype).view(shape)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype).view(shape)
        return values * std + mean

def build_static_features(
    latitude: np.ndarray,
    longitude: np.ndarray,
    static_path: Optional[str],
    static_variables: Sequence[str],
) -> np.ndarray:
    """Return requested static channels followed by five spherical geometry channels."""
    _require_xarray()
    height, width = len(latitude), len(longitude)
    channels: List[np.ndarray] = []
    static_dataset = xr.open_dataset(static_path) if static_path and Path(static_path).exists() else None

    for name in static_variables:
        if static_dataset is not None and name in static_dataset:
            array = _canonicalize_spatial_names(static_dataset[name])
            extra_dims = [dim for dim in array.dims if dim not in {"latitude", "longitude"}]
            for dim in extra_dims:
                array = array.isel({dim: 0})
            values = np.asarray(array.transpose("latitude", "longitude").values, dtype=np.float32)
            if values.shape != (height, width):
                raise ValueError(f"Static variable {name} has shape {values.shape}, expected {(height, width)}")
            finite = np.isfinite(values)
            if finite.any():
                mean = values[finite].mean()
                std = values[finite].std()
                values = (np.nan_to_num(values, nan=float(mean)) - mean) / max(float(std), 1e-6)
            else:
                values = np.zeros((height, width), dtype=np.float32)
        else:
            values = np.zeros((height, width), dtype=np.float32)
        channels.append(values)

    lat_rad = np.deg2rad(latitude.astype(np.float32))
    lon_rad = np.deg2rad(longitude.astype(np.float32))
    channels.extend(
        [
            np.broadcast_to(np.sin(lat_rad)[:, None], (height, width)),
            np.broadcast_to(np.cos(lat_rad)[:, None], (height, width)),
            np.broadcast_to(np.sin(lon_rad)[None, :], (height, width)),
            np.broadcast_to(np.cos(lon_rad)[None, :], (height, width)),
            np.broadcast_to(np.cos(lat_rad)[:, None], (height, width)),
        ]
    )
    if static_dataset is not None:
        static_dataset.close()
    return np.stack(channels, axis=0).astype(np.float32)


class VeinCastERA5Dataset(Dataset):
    """
    ERA5 dataset for fixed registry inputs and multi-step normalized targets.

    Variable-level input dropout changes ``input_present`` while keeping the
    registered 69-field tensor shape fixed for ordinary PyTorch collation.
    """

    def __init__(
        self,
        dynamic_path: str,
        registry: VariableRegistry,
        start: str,
        end: str,
        normalizer: Optional[FieldNormalizer] = None,
        fit_normalizer: bool = False,
        static_path: Optional[str] = None,
        static_variables: Sequence[str] = (),
        surface_pressure_name: str = "sp",
        data_interval_hours: int = 6,
        base_lead_hours: int = 6,
        rollout_steps: int = 1,
        target_steps: Optional[Sequence[int]] = None,
        input_dropout_probability: float = 0.0,
        training: bool = False,
        load_into_memory: bool = False,
    ) -> None:
        _require_xarray()
        super().__init__()
        if base_lead_hours % data_interval_hours != 0:
            raise ValueError("base_lead_hours must be divisible by data_interval_hours")
        self.registry = registry
        self.training = training
        self.rollout_steps = int(rollout_steps)
        if target_steps is None:
            target_steps = range(1, self.rollout_steps + 1)
        self.target_steps = tuple(sorted({int(step) for step in target_steps}))
        if (
            not self.target_steps
            or self.target_steps[0] < 1
            or self.target_steps[-1] > self.rollout_steps
        ):
            raise ValueError("target_steps must be within [1, rollout_steps]")
        self.lead_steps = base_lead_hours // data_interval_hours
        self.base_lead_hours = int(base_lead_hours)
        self.input_dropout_probability = float(input_dropout_probability)

        self.dataset = xr.open_dataset(dynamic_path, cache=False)
        if normalizer is None:
            if not fit_normalizer:
                raise ValueError("normalizer is required unless fit_normalizer=True")
            normalizer = FieldNormalizer.fit_from_dataset(
                self.dataset, registry, start, end
            )
        normalizer.validate(registry)
        self.normalizer = normalizer
        (
            self.field_arrays,
            self.times,
            self.latitude,
            self.longitude,
        ) = build_field_arrays(self.dataset, registry, start, end)
        if load_into_memory:
            self.field_arrays = tuple(array.load() for array in self.field_arrays)
        self.static = build_static_features(
            self.latitude, self.longitude, static_path, static_variables
        )

        self.surface_pressure = None
        if surface_pressure_name and surface_pressure_name in self.dataset:
            pressure = self.dataset[surface_pressure_name].sel(time=slice(start, end))
            self.surface_pressure = _canonicalize_spatial_names(pressure).transpose(
                "time", "latitude", "longitude"
            )

        required_future = self.rollout_steps * self.lead_steps
        self.n_samples = max(0, len(self.times) - required_future)

    @property
    def static_channels(self) -> int:
        return int(self.static.shape[0])

    def __len__(self) -> int:
        return self.n_samples

    def _valid_mask(self, time_index: int) -> np.ndarray:
        height, width = len(self.latitude), len(self.longitude)
        valid = np.ones((self.registry.num_fields, height, width), dtype=np.float32)
        if self.surface_pressure is None:
            return valid
        pressure_hpa = np.asarray(self.surface_pressure.isel(time=time_index).values, dtype=np.float32)
        if np.nanmedian(pressure_hpa) > 2000:
            pressure_hpa = pressure_hpa / 100.0
        for field in self.registry.fields:
            if not field.is_surface:
                valid[field.field_id] = (field.pressure_hpa <= pressure_hpa).astype(np.float32)
        return valid

    def _input_present(self) -> np.ndarray:
        present = np.ones(self.registry.num_fields, dtype=np.float32)
        if not self.training or self.input_dropout_probability <= 0:
            return present
        keep_variable = np.random.random(self.registry.num_variables) >= self.input_dropout_probability
        if keep_variable.sum() < 2:
            keep_variable[np.random.choice(self.registry.num_variables, size=2, replace=False)] = True
        for field in self.registry.fields:
            present[field.field_id] = float(keep_variable[field.variable_id])
        return present

    def _field_stack(self, time_index: int) -> np.ndarray:
        values = [
            np.asarray(array.isel(time=time_index).values, dtype=np.float32)
            for array in self.field_arrays
        ]
        return self.normalizer.normalize_array(np.stack(values, axis=0))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        state = self._field_stack(index)
        targets = []
        valid_masks = []
        for step in self.target_steps:
            target_index = index + step * self.lead_steps
            targets.append(self._field_stack(target_index))
            valid_masks.append(self._valid_mask(target_index))

        state = np.nan_to_num(state)
        targets_array = np.nan_to_num(np.stack(targets, axis=0))
        calendar_indices = [
            index + step * self.lead_steps
            for step in range(1, self.rollout_steps + 1)
        ]
        calendar_times = self.times[calendar_indices]
        calendar_features = calendar_features_from_times(calendar_times)
        present = self._input_present()
        masked_state = state * present[:, None, None]
        return {
            "state": torch.from_numpy(masked_state),
            "targets": torch.from_numpy(targets_array),
            "static": torch.from_numpy(self.static),
            "input_present": torch.from_numpy(present),
            "valid_mask": torch.from_numpy(np.stack(valid_masks, axis=0)),
            "lead_hours": torch.tensor(float(self.base_lead_hours), dtype=torch.float32),
            "calendar_features": torch.from_numpy(calendar_features),
            "target_steps": torch.tensor(self.target_steps, dtype=torch.long),
            "time_index": torch.tensor(index, dtype=torch.long),
        }

    def close(self) -> None:
        self.dataset.close()
