from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from config import GRID_HEIGHT, GRID_WIDTH
from rl_value_net import AfterstateValueNet


@dataclass(frozen=True)
class PlacementPlan:
    x: int
    rotation: int
    landing_y: int


class RLTetrisAI:
    """
    RL-backed AI that keeps the same step-by-step control interface as TetrisAI.

    It chooses a final placement using the trained value network, then emits
    rotate / left / right / hard_drop actions one by one.
    """

    def __init__(self, controller, checkpoint_path: str, device: Optional[str] = None):
        self.game = controller
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint_meta = self.load_model(checkpoint_path, self.device)
        self.current_move: Optional[Tuple[int, int]] = None
        self.rotation_done = False
        self.error_count = 0
        self.landing_error_prob = 0.0
        self.ai_mode = "rl"
        self.checkpoint_path = checkpoint_path
        self.plan_board_signature = None

    def load_model(self, checkpoint_path: str, device: torch.device):
        active_piece = self._active_piece()
        if active_piece is None:
            raise RuntimeError("cannot initialize RL AI without an active piece")

        input_dim = len(self._afterstate_vector(self.legal_moves()[0]))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_dims = tuple(checkpoint.get("config", {}).get("hidden", [256, 256]))

        model = AfterstateValueNet(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    def reset_plan(self):
        self.current_move = None
        self.rotation_done = False
        self.plan_board_signature = None

    def _board_signature(self):
        return tuple(tuple(int(cell) for cell in row) for row in self.game.grid)

    def _refresh_plan_if_board_changed(self):
        if self.current_move is None or self.plan_board_signature is None:
            return
        if self._board_signature() != self.plan_board_signature:
            self.reset_plan()

    def next_action(self):
        if self.game.game_over or getattr(self.game, "paused", False):
            return None
        self._refresh_plan_if_board_changed()

        if self.current_move is None:
            best_plan = self.best_move()
            if best_plan is None:
                return None
            self.current_move = (best_plan.x, best_plan.rotation)
            self.rotation_done = False
            self.plan_board_signature = self._board_signature()
            self._maybe_apply_landing_error()

        target_x, remaining_rotations = self.current_move

        if not self.rotation_done:
            if remaining_rotations > 0:
                self.current_move = (target_x, remaining_rotations - 1)
                return "rotate"
            self.rotation_done = True

        if self.game.current_x < target_x:
            return "right"
        if self.game.current_x > target_x:
            return "left"

        self.current_move = None
        self.rotation_done = False
        return "hard_drop"

    def _maybe_apply_landing_error(self):
        if self.current_move is None or random.random() >= self.landing_error_prob:
            return

        orig_x, orig_rotation = self.current_move
        test_shape = self._rotate_n(self.game.get_current_shape(), orig_rotation)
        min_x = -len(test_shape[0]) + 1
        max_x = GRID_WIDTH - 1

        for _ in range(8):
            offset = random.choice([-2, -1, 1, 2])
            candidate_x = max(min_x, min(max_x, orig_x + offset))
            if not self.game.check_collision(test_shape, candidate_x, 0):
                self.current_move = (candidate_x, orig_rotation)
                self.error_count += 1
                break

    def best_move(self) -> Optional[PlacementPlan]:
        legal_moves = self.legal_moves()
        if not legal_moves:
            return None

        with torch.no_grad():
            batch = torch.tensor(
                [self._afterstate_vector(move) for move in legal_moves],
                dtype=torch.float32,
                device=self.device,
            )
            values = self.model(batch).squeeze(-1)
            best_index = int(torch.argmax(values).item())
        return legal_moves[best_index]

    def legal_moves(self) -> List[PlacementPlan]:
        active_piece = self._active_piece()
        if active_piece is None:
            return []

        moves: List[PlacementPlan] = []
        shape = self.game.get_current_shape()
        for rotation, rotated_shape in self._unique_rotations(shape):
            for x in range(-len(rotated_shape[0]), GRID_WIDTH + 1):
                if self.game.check_collision(rotated_shape, x, 0):
                    continue
                landing_y = self._simulate_drop_y(rotated_shape, x)
                moves.append(PlacementPlan(x=x, rotation=rotation, landing_y=landing_y))

        moves.sort(key=lambda move: (move.rotation, move.x))
        return moves

    def _simulate_drop_y(self, shape: List[List[int]], x: int) -> int:
        y = 0
        while not self.game.check_collision(shape, x, y + 1):
            y += 1
        return y

    def _afterstate_vector(self, move: PlacementPlan) -> List[float]:
        preview = self._preview_afterstate(move)
        return self._build_feature_vector(
            board_binary=preview["board_binary"],
            column_heights=preview["column_heights"],
            holes=preview["holes"],
            aggregate_height=preview["aggregate_height"],
            bumpiness=preview["bumpiness"],
            piece_id=preview["next_shape_id"],
        )

    def _preview_afterstate(self, move: PlacementPlan):
        active_piece = self._active_piece()
        if active_piece is None:
            raise RuntimeError("no active piece available for preview")

        shape = self._rotate_n(self.game.get_current_shape(), move.rotation)
        grid = [row[:] for row in self.game.grid]

        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell and move.landing_y + row_idx >= 0:
                    grid[move.landing_y + row_idx][move.x + col_idx] = active_piece.shape_id + 1

        grid = self._clear_full_rows(grid)
        heights = self._column_heights(grid)
        return {
            "board_binary": self._board_as_binary(grid),
            "column_heights": heights,
            "holes": self._count_holes(grid),
            "aggregate_height": sum(heights),
            "bumpiness": self._bumpiness_from_heights(heights),
            "next_shape_id": self._next_shape_id(),
        }

    def _active_piece(self):
        return self.game.shared_game.active_pieces.get(self.game.owner_id)

    def _next_shape_id(self) -> int:
        return int(self.game.shared_game.owner_states[self.game.owner_id].next_shape_id)

    def _rotate_n(self, shape: List[List[int]], count: int) -> List[List[int]]:
        rotated = [row[:] for row in shape]
        for _ in range(count % 4):
            rotated = self._rotate_shape(rotated)
        return rotated

    def _rotate_shape(self, shape: List[List[int]]) -> List[List[int]]:
        return [list(row) for row in zip(*shape[::-1])]

    def _unique_rotations(self, shape: List[List[int]]):
        rotations = []
        seen = set()
        rotated = [row[:] for row in shape]
        for rotation in range(4):
            key = tuple(tuple(row) for row in rotated)
            if key not in seen:
                rotations.append((rotation, [row[:] for row in rotated]))
                seen.add(key)
            rotated = self._rotate_shape(rotated)
        return rotations

    def _clear_full_rows(self, grid: List[List[int]]) -> List[List[int]]:
        remaining_rows = [row[:] for row in grid if not all(row)]
        cleared = GRID_HEIGHT - len(remaining_rows)
        if cleared == 0:
            return [row[:] for row in grid]
        return [[0 for _ in range(GRID_WIDTH)] for _ in range(cleared)] + remaining_rows

    def _board_as_binary(self, grid: List[List[int]]) -> List[int]:
        return [1.0 if cell else 0.0 for row in grid for cell in row]

    def _column_heights(self, grid: List[List[int]]) -> List[int]:
        heights = [0] * len(grid[0])
        for col in range(len(grid[0])):
            for row in range(len(grid)):
                if grid[row][col]:
                    heights[col] = len(grid) - row
                    break
        return heights

    def _bumpiness_from_heights(self, heights: Sequence[int]) -> int:
        return sum(abs(heights[idx] - heights[idx + 1]) for idx in range(len(heights) - 1))

    def _count_holes(self, grid: List[List[int]]) -> int:
        holes = 0
        width = len(grid[0])
        height = len(grid)
        for col in range(width):
            seen_block = False
            for row in range(height):
                if grid[row][col]:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes

    def _build_feature_vector(
        self,
        *,
        board_binary: List[float],
        column_heights: Sequence[int],
        holes: int,
        aggregate_height: int,
        bumpiness: int,
        piece_id: int,
    ) -> List[float]:
        normalized_heights = [height / GRID_HEIGHT for height in column_heights]
        one_hot_piece = [0.0] * 7
        if 0 <= piece_id < len(one_hot_piece):
            one_hot_piece[piece_id] = 1.0
        return (
            board_binary
            + normalized_heights
            + [
                holes / float(GRID_WIDTH * GRID_HEIGHT),
                aggregate_height / float(GRID_WIDTH * GRID_HEIGHT),
                bumpiness / float(GRID_WIDTH * GRID_HEIGHT),
            ]
            + one_hot_piece
        )
