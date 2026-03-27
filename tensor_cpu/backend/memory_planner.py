"""Workspace and temporary storage planning for the stable C++ backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..ir.graph import Node


@dataclass(slots=True)
class MemoryPlannerState:
    names: Dict[int, str]
    declared_slots: set[str] = field(default_factory=set)
    workspace_slots: List[tuple[str, str]] = field(default_factory=list)
    workspace_sym: List[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    temp_owner: Dict[int, str] = field(default_factory=dict)
    free_slots: Dict[tuple[str, ...], List[str]] = field(default_factory=dict)
    slot_counter: int = 0


class MemoryPlannerMixin:
    enable_memory_planner: bool

    def _compute_use_count(self, ordered: List[Node]) -> Dict[int, int]:
        use_count: Dict[int, int] = {n.id: 0 for n in ordered}
        for node in ordered:
            for in_id in node.inputs:
                if in_id in use_count:
                    use_count[in_id] += 1
        return use_count

    def _assign_node_storage(
        self, *, node: Node, output_node: Node, state: MemoryPlannerState
    ) -> None:
        if node.id == output_node.id:
            state.names[node.id] = "out_ptr"
            return

        if self.enable_memory_planner:
            sym_key = self._sym.get(node.id, ())
            numel = self._numel_expr(node)
            bucket = state.free_slots.setdefault(sym_key, [])
            if bucket:
                slot_name = bucket.pop()
            else:
                slot_name = f"buf_s{state.slot_counter}"
                state.slot_counter += 1
                if slot_name not in state.declared_slots:
                    state.workspace_slots.append((slot_name, numel))
                    state.workspace_sym.append((slot_name, sym_key))
                    state.declared_slots.add(slot_name)
            state.names[node.id] = slot_name
            state.temp_owner[node.id] = slot_name
            return

        slot_name = f"buf_{node.id}"
        state.names[node.id] = slot_name
        numel = self._numel_expr(node)
        sym_key = self._sym.get(node.id, ())
        state.workspace_slots.append((slot_name, numel))
        state.workspace_sym.append((slot_name, sym_key))

    def _release_consumed_storage(
        self, *, node: Node, use_count: Dict[int, int], state: MemoryPlannerState
    ) -> None:
        if not self.enable_memory_planner:
            return
        for in_id in node.inputs:
            if in_id in use_count:
                use_count[in_id] -= 1
                if use_count[in_id] == 0 and in_id in state.temp_owner:
                    slot_name = state.temp_owner[in_id]
                    state.free_slots.setdefault(self._sym.get(in_id, ()), []).append(slot_name)
