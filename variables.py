from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_PRESSURE_LEVELS: Tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
)

RELATION_NONE = -1
RELATION_SELF = 0
RELATION_VERTICAL = 1
RELATION_WIND_PRESSURE = 2
RELATION_THERMO_MOISTURE = 3
RELATION_MOISTURE_TRANSPORT = 4
RELATION_SURFACE_ATMOSPHERE = 5
RELATION_GENERIC = 6

RELATION_NAMES = {
    RELATION_SELF: "self",
    RELATION_VERTICAL: "vertical",
    RELATION_WIND_PRESSURE: "wind_pressure",
    RELATION_THERMO_MOISTURE: "thermo_moisture",
    RELATION_MOISTURE_TRANSPORT: "moisture_transport",
    RELATION_SURFACE_ATMOSPHERE: "surface_atmosphere",
    RELATION_GENERIC: "learned_residual",
}


@dataclass(frozen=True)
class VariableDefinition:
    name: str
    dataset_name: str
    kind: str
    units: str
    family: str
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldDefinition:
    field_id: int
    variable_id: int
    variable: str
    dataset_name: str
    pressure_hpa: float
    is_surface: bool
    units: str
    family: str
    aliases: Tuple[str, ...]

    @property
    def key(self) -> str:
        if self.is_surface:
            return self.variable
        return f"{self.variable}@{int(self.pressure_hpa)}"


def default_variables() -> Tuple[VariableDefinition, ...]:
    return (
        VariableDefinition("z", "z", "upper_air", "m2 s-2", "dynamics", ("geopotential",)),
        VariableDefinition("t", "t", "upper_air", "K", "thermodynamics", ("temperature",)),
        VariableDefinition("q", "q", "upper_air", "kg kg-1", "moisture", ("specific_humidity",)),
        VariableDefinition("u", "u", "upper_air", "m s-1", "dynamics", ("u_component_of_wind",)),
        VariableDefinition("v", "v", "upper_air", "m s-1", "dynamics", ("v_component_of_wind",)),
        VariableDefinition("t2m", "t2m", "surface", "K", "surface", ("2m_temperature",)),
        VariableDefinition("u10", "u10", "surface", "m s-1", "surface", ("10m_u_component_of_wind",)),
        VariableDefinition("v10", "v10", "surface", "m s-1", "surface", ("10m_v_component_of_wind",)),
        VariableDefinition("msl", "msl", "surface", "Pa", "surface", ("mslp", "mean_sea_level_pressure")),
    )


class VariableRegistry:
    """Canonical description of the model's variable-level fields."""

    def __init__(
        self,
        variables: Sequence[VariableDefinition] | None = None,
        pressure_levels: Sequence[int] = DEFAULT_PRESSURE_LEVELS,
    ) -> None:
        self.variables = tuple(variables or default_variables())
        self.pressure_levels = tuple(int(p) for p in pressure_levels)
        self.variable_to_id = {var.name: index for index, var in enumerate(self.variables)}
        self.fields = self._build_fields()
        self.field_to_id = {field.key: field.field_id for field in self.fields}

    def _build_fields(self) -> Tuple[FieldDefinition, ...]:
        fields: List[FieldDefinition] = []
        for variable_id, variable in enumerate(self.variables):
            levels: Iterable[float] = self.pressure_levels if variable.kind == "upper_air" else (0.0,)
            for pressure in levels:
                fields.append(
                    FieldDefinition(
                        field_id=len(fields),
                        variable_id=variable_id,
                        variable=variable.name,
                        dataset_name=variable.dataset_name,
                        pressure_hpa=float(pressure),
                        is_surface=variable.kind == "surface",
                        units=variable.units,
                        family=variable.family,
                        aliases=variable.aliases,
                    )
                )
        return tuple(fields)

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    def field(self, key_or_id: str | int) -> FieldDefinition:
        if isinstance(key_or_id, int):
            return self.fields[key_or_id]
        return self.fields[self.field_to_id[key_or_id]]

    def query_arrays(self, keys: Sequence[str] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = self.fields if keys is None else tuple(self.field(key) for key in keys)
        variable_ids = np.asarray([field.variable_id for field in selected], dtype=np.int64)
        pressures = np.asarray([field.pressure_hpa for field in selected], dtype=np.float32)
        surface = np.asarray([field.is_surface for field in selected], dtype=np.bool_)
        return variable_ids, pressures, surface

    def physical_relations(self) -> Tuple[np.ndarray, np.ndarray]:
        relation_ids = np.full((self.num_fields, self.num_fields), RELATION_NONE, dtype=np.int64)
        allowed = np.zeros((self.num_fields, self.num_fields), dtype=np.bool_)
        for target in self.fields:
            for source in self.fields:
                relation = physical_relation(source, target)
                relation_ids[target.field_id, source.field_id] = relation
                allowed[target.field_id, source.field_id] = relation != RELATION_NONE
        return relation_ids, allowed

    def to_dict(self) -> Dict[str, object]:
        return {
            "pressure_levels": list(self.pressure_levels),
            "variables": [asdict(variable) for variable in self.variables],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "VariableRegistry":
        variables = []
        for raw in payload["variables"]:
            item = dict(raw)
            item["aliases"] = tuple(item.get("aliases", ()))
            variables.append(VariableDefinition(**item))
        return cls(variables=variables, pressure_levels=payload["pressure_levels"])


def _same_or_near_pressure(source: FieldDefinition, target: FieldDefinition) -> bool:
    if source.is_surface or target.is_surface:
        return False
    return abs(np.log(source.pressure_hpa) - np.log(target.pressure_hpa)) < 0.45


def physical_relation(source: FieldDefinition, target: FieldDefinition) -> int:
    if source.key == target.key:
        return RELATION_SELF
    if source.variable == target.variable and not source.is_surface and not target.is_surface:
        return RELATION_VERTICAL

    pair = {source.variable, target.variable}
    if _same_or_near_pressure(source, target):
        if "z" in pair and ("u" in pair or "v" in pair):
            return RELATION_WIND_PRESSURE
        if pair == {"t", "q"}:
            return RELATION_THERMO_MOISTURE
        if "q" in pair and ("u" in pair or "v" in pair):
            return RELATION_MOISTURE_TRANSPORT

    if source.is_surface != target.is_surface:
        surface = source if source.is_surface else target
        upper = target if source.is_surface else source
        mapped = {
            "t2m": {"t", "q"},
            "u10": {"u", "z"},
            "v10": {"v", "z"},
            "msl": {"z", "u", "v", "t"},
        }
        if upper.pressure_hpa >= 700 and upper.variable in mapped.get(surface.variable, set()):
            return RELATION_SURFACE_ATMOSPHERE
    return RELATION_NONE


def build_query_prior(
    registry: VariableRegistry,
    query_variable_ids: np.ndarray,
    query_pressures: np.ndarray,
    query_surface: np.ndarray,
) -> np.ndarray:
    """Soft physical bias between decoder fields and registered input fields."""
    bias = np.full((len(query_variable_ids), registry.num_fields), -2.0, dtype=np.float32)
    for query_index, (variable_id, pressure, is_surface) in enumerate(
        zip(query_variable_ids, query_pressures, query_surface)
    ):
        variable = registry.variables[int(variable_id)]
        query = FieldDefinition(
            field_id=-1,
            variable_id=int(variable_id),
            variable=variable.name,
            dataset_name=variable.dataset_name,
            pressure_hpa=float(pressure),
            is_surface=bool(is_surface),
            units=variable.units,
            family=variable.family,
            aliases=variable.aliases,
        )
        for source in registry.fields:
            relation = physical_relation(source, query)
            if relation == RELATION_SELF:
                bias[query_index, source.field_id] = 2.0
            elif relation == RELATION_VERTICAL:
                distance = abs(np.log(max(pressure, 1.0)) - np.log(source.pressure_hpa))
                bias[query_index, source.field_id] = 1.5 - distance
            elif relation != RELATION_NONE:
                bias[query_index, source.field_id] = 0.75
    return bias
