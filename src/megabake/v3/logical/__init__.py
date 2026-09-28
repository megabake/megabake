"""Target-neutral logical work planning."""

from .domains import TileAxis, TileCoordinate, TileDomain, TileInstance, tile_domain
from .maps import AccessMapError, RegionMap, enumerate_region, map_indices
from .plan import LogicalExecutionPlan, LogicalPlanError, LogicalTaskFamily, lower_logical_plan
from .dependence import DependencyRelation, Readiness, TaskDependency, derive_dependencies, materialize_dependencies
from .storage import StorageLifetime, build_storage_lifetimes, can_overlay
from .verify import (LogicalVerificationReport, analyze_logical_plan, execute_schedule,
                     topological_orders, verify_logical_plan)

__all__ = ["AccessMapError", "DependencyRelation", "LogicalExecutionPlan", "LogicalPlanError",
           "LogicalTaskFamily", "LogicalVerificationReport", "Readiness", "RegionMap",
           "StorageLifetime", "TaskDependency", "TileAxis", "TileCoordinate", "TileDomain",
           "TileInstance", "analyze_logical_plan", "build_storage_lifetimes", "can_overlay",
           "derive_dependencies", "enumerate_region", "execute_schedule", "lower_logical_plan",
           "map_indices", "materialize_dependencies", "tile_domain", "topological_orders",
           "verify_logical_plan"]
