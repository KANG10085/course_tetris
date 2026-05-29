from __future__ import annotations

import random
from dataclasses import dataclass, field

import pygame

from config import *
from shapes import SHAPES, SHAPE_COLORS


@dataclass
class ActivePiece:
    owner_id: str
    shape_id: int
    shape_matrix: list[list[int]]
    color: tuple[int, int, int]
    x: int
    y: int = 0
    fall_time: float = 0.0
    can_bump: bool = True
    lock_timer: float = 0.0
    grounded: bool = False

    def occupied_cells(self, shape=None, x=None, y=None):
        shape = self.shape_matrix if shape is None else shape
        x = self.x if x is None else x
        y = self.y if y is None else y
        cells = []
        for row_index, row in enumerate(shape):
            for col_index, cell in enumerate(row):
                if cell:
                    cells.append((x + col_index, y + row_index))
        return cells

    def bottom_y(self):
        return max(cell_y for _, cell_y in self.occupied_cells())


@dataclass
class OwnerState:
    owner_id: str
    label: str
    controls: list[str]
    block_color: tuple[int, int, int]
    score: int = 0
    total_lines: int = 0
    single_lines: int = 0
    double_lines: int = 0
    triple_lines: int = 0
    tetris_lines: int = 0
    eliminated: bool = False
    spawn_count: int = 0
    next_shape_id: int = 0
    next_shape_matrix: list[list[int]] = field(default_factory=list)
    next_color: tuple[int, int, int] = BLACK


@dataclass
class FloatingText:
    text: str
    x: float
    y: float
    color: tuple[int, int, int]
    ttl: float
    total_ttl: float
    rise_speed: float = 34.0


class SharedPieceController:
    def __init__(self, shared_game, owner_id):
        self.shared_game = shared_game
        self.owner_id = owner_id

    @property
    def game_over(self):
        return self.shared_game.game_over

    @property
    def paused(self):
        return self.shared_game.paused

    @property
    def grid(self):
        return self.shared_game.get_composite_grid(exclude_owner=self.owner_id)

    @property
    def current_x(self):
        piece = self.shared_game.active_pieces.get(self.owner_id)
        return 0 if piece is None else piece.x

    @property
    def current_y(self):
        piece = self.shared_game.active_pieces.get(self.owner_id)
        return 0 if piece is None else piece.y

    def get_current_shape(self):
        piece = self.shared_game.active_pieces.get(self.owner_id)
        if piece is None:
            return [[1]]
        return [row[:] for row in piece.shape_matrix]

    def check_collision(self, shape=None, x=None, y=None):
        return self.shared_game.check_collision(self.owner_id, shape=shape, x=x, y=y)

    def rotate(self):
        return self.shared_game.rotate_piece(self.owner_id)

    def move(self, dx, dy):
        return self.shared_game.move_piece(self.owner_id, dx, dy)

    def hard_drop(self):
        return self.shared_game.hard_drop_piece(self.owner_id)


class SharedBoardTetris:
    def __init__(self, screen, owner_configs, title="Shared Tetris"):
        self.screen = screen
        self.title = title
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.locked_colors = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.owner_order = [config["id"] for config in owner_configs]
        self.owner_states = {}
        self.active_pieces = {}
        self.last_locked_pieces = {}
        self.game_over = False
        self.paused = False
        self.show_grid = True
        self.show_ghost = True
        self.show_background = True
        self.background_scroll = 0
        self.background_paused = False
        self.uniform_block_colors = False
        self.capture_clean_mode = False
        self.capture_block_color = WHITE
        self.capture_background_color = BLACK
        self.level = 1
        self.fall_interval = 0.5
        self.fall_speed = 1 / self.fall_interval
        self.animation_enabled = False
        self.animation_interval = self.fall_interval
        self.animation_steps = 1
        self.animation_progress = 1.0
        self.animation_states = {}
        self.winner = None
        self.lock_delay = 0.2
        self.board_flash_timer = 0.0
        self.board_flash_total = 0.0
        self.board_flash_color = WHITE
        self.board_flash_alpha = 0
        self.board_shake_timer = 0.0
        self.board_shake_total = 0.0
        self.board_shake_vector = (0, 0)
        self.floating_texts: list[FloatingText] = []

        try:
            self.background = pygame.image.load("background1.png").convert()
            self.background = pygame.transform.scale(
                self.background,
                (SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT),
            )
        except Exception:
            self.background = pygame.Surface((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
            self.background.fill((50, 50, 100))

        self.font = pygame.font.SysFont(None, 24)
        self.feedback_font = pygame.font.SysFont(None, 32)
        self.title_font = pygame.font.SysFont(None, 34)
        self.overlay_font = pygame.font.SysFont(None, 56)
        self.common_controls = [
            "R restart",
            "Q quit",
        ]

        for config in owner_configs:
            state = OwnerState(
                owner_id=config["id"],
                label=config["label"],
                controls=config["controls"],
                block_color=config.get(
                    "color",
                    SHAPE_COLORS[len(self.owner_states) % len(SHAPE_COLORS)],
                ),
            )
            self.owner_states[state.owner_id] = state
            self.prepare_next_piece(state.owner_id)

        for owner_id in self.owner_order:
            self.spawn_piece(owner_id)
        self.reset_animation_states()

    def copy_shape(self, shape):
        return [row[:] for row in shape]

    def rotate_shape(self, shape):
        return [list(row) for row in zip(*shape[::-1])]

    def unique_rotations(self, shape):
        rotations = []
        seen = set()
        rotated = self.copy_shape(shape)
        for rotation in range(4):
            key = tuple(tuple(row) for row in rotated)
            if key not in seen:
                rotations.append((rotation, self.copy_shape(rotated)))
                seen.add(key)
            rotated = self.rotate_shape(rotated)
        return rotations

    def rotation_index_for_shape(self, shape_id, shape_matrix):
        base_shape = self.copy_shape(SHAPES[shape_id])
        for rotation, rotated_shape in self.unique_rotations(base_shape):
            if rotated_shape == shape_matrix:
                return rotation
        return 0

    def random_shape_bundle(self):
        shape_id = random.randint(0, len(SHAPES) - 1)
        return shape_id, self.copy_shape(SHAPES[shape_id])

    def prepare_next_piece(self, owner_id):
        shape_id, shape_matrix = self.random_shape_bundle()
        state = self.owner_states[owner_id]
        state.next_shape_id = shape_id
        state.next_shape_matrix = shape_matrix
        state.next_color = state.block_color

    def spawn_anchor(self, owner_id):
        return 9 if self.owner_order.index(owner_id) == 0 else 17

    def preferred_spawn_columns(self, owner_id, shape_width):
        max_x = max(0, GRID_WIDTH - shape_width)
        anchor = max(0, min(max_x, self.spawn_anchor(owner_id)))
        candidates = [anchor]
        for offset in range(1, 5):
            candidates.append(anchor - offset)
            candidates.append(anchor + offset)
        unique = []
        for candidate in candidates:
            clamped = max(0, min(max_x, candidate))
            if clamped not in unique:
                unique.append(clamped)
        return unique

    def current_board_offset(self):
        if self.board_shake_timer <= 0 or self.board_shake_total <= 0:
            return 0, 0
        scale = self.board_shake_timer / self.board_shake_total
        return (
            round(self.board_shake_vector[0] * scale),
            round(self.board_shake_vector[1] * scale),
        )

    def trigger_board_flash(self, color, alpha=90, duration=0.08):
        self.board_flash_color = color
        self.board_flash_alpha = alpha
        self.board_flash_total = duration
        self.board_flash_timer = duration

    def trigger_board_shake(self, dx, dy, duration=0.12):
        amplitude_x = 6 if dx else 0
        amplitude_y = 5 if dy else 0
        self.board_shake_vector = (
            amplitude_x * (1 if dx > 0 else -1 if dx < 0 else 0),
            amplitude_y * (1 if dy > 0 else -1 if dy < 0 else 0),
        )
        self.board_shake_total = duration
        self.board_shake_timer = duration

    def add_floating_text(self, text, x, y, color, ttl=0.55, rise_speed=34.0):
        self.floating_texts.append(
            FloatingText(
                text=text,
                x=x,
                y=y,
                color=color,
                ttl=ttl,
                total_ttl=ttl,
                rise_speed=rise_speed,
            )
        )

    def update_feedback_timers(self, dt):
        if self.board_flash_timer > 0:
            self.board_flash_timer = max(0.0, self.board_flash_timer - dt)
        if self.board_shake_timer > 0:
            self.board_shake_timer = max(0.0, self.board_shake_timer - dt)

        survivors = []
        for text in self.floating_texts:
            text.ttl -= dt
            text.y -= text.rise_speed * dt
            if text.ttl > 0:
                survivors.append(text)
        self.floating_texts = survivors

    def refresh_piece_lock_state(self, owner_id, reset_timer=False):
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return False
        reason = self.collision_reason(owner_id, x=piece.x, y=piece.y + 1)
        grounded = reason in ("floor", "stack")
        if grounded:
            if reset_timer or not piece.grounded:
                piece.lock_timer = 0.0
            piece.grounded = True
        else:
            piece.lock_timer = 0.0
            piece.grounded = False
        return grounded

    def update_frame(self, dt):
        if dt <= 0:
            return
        if not self.game_over and not self.paused:
            self.update_feedback_timers(dt)
        if self.game_over or self.paused:
            return

        to_lock = []
        for owner_id in self.owner_order:
            piece = self.active_pieces.get(owner_id)
            if piece is None:
                continue
            if self.refresh_piece_lock_state(owner_id):
                piece.lock_timer += dt
                if piece.lock_timer >= self.lock_delay:
                    to_lock.append(owner_id)

        for owner_id in to_lock:
            piece = self.active_pieces.get(owner_id)
            if piece is None:
                continue
            reason = self.collision_reason(owner_id, x=piece.x, y=piece.y + 1)
            if reason in ("floor", "stack"):
                self.lock_piece(owner_id)

    def reset_animation_states(self):
        self.animation_progress = 1.0
        self.animation_states = {}
        if not self.animation_enabled:
            return
        for owner_id, piece in self.active_pieces.items():
            self.animation_states[owner_id] = {
                "from_x": float(piece.x),
                "to_x": float(piece.x),
                "from_y": float(piece.y),
                "to_y": float(piece.y),
            }

    def set_animation_progress(self, progress):
        if not self.animation_enabled:
            self.animation_progress = 1.0
            return
        self.animation_progress = max(0.0, min(1.0, progress))

    def capture_piece_states(self):
        return {
            owner_id: {
                "x": piece.x,
                "y": piece.y,
                "shape_id": piece.shape_id,
            }
            for owner_id, piece in self.active_pieces.items()
        }

    def update_animation_states(self, before_states):
        if not self.animation_enabled:
            self.reset_animation_states()
            return
        new_states = {}
        for owner_id, piece in self.active_pieces.items():
            before = before_states.get(owner_id)
            from_x = float(piece.x)
            from_y = float(piece.y)
            if before is not None:
                dx = piece.x - before["x"]
                dy = piece.y - before["y"]
                if before["shape_id"] == piece.shape_id and abs(dx) <= 1 and abs(dy) <= 1:
                    from_x = float(before["x"])
                    from_y = float(before["y"])
            new_states[owner_id] = {
                "from_x": from_x,
                "to_x": float(piece.x),
                "from_y": from_y,
                "to_y": float(piece.y),
            }
        self.animation_states = new_states
        self.animation_progress = 0.0

    def interpolated_position(self, owner_id, piece):
        if not self.animation_enabled:
            return float(piece.x), float(piece.y)
        animation = self.animation_states.get(owner_id)
        if animation is None:
            return float(piece.x), float(piece.y)
        progress = self.animation_progress
        x = animation["from_x"] + (animation["to_x"] - animation["from_x"]) * progress
        y = animation["from_y"] + (animation["to_y"] - animation["from_y"]) * progress
        return x, y

    def spawn_piece(self, owner_id):
        state = self.owner_states[owner_id]
        if state.eliminated:
            return

        shape_matrix = self.copy_shape(state.next_shape_matrix)
        x_candidates = self.preferred_spawn_columns(owner_id, len(shape_matrix[0]))
        piece = ActivePiece(
            owner_id=owner_id,
            shape_id=state.next_shape_id,
            shape_matrix=shape_matrix,
            color=state.block_color,
            x=x_candidates[0],
        )

        placed = False
        for candidate_x in x_candidates:
            if not self.check_collision(owner_id, shape=shape_matrix, x=candidate_x, y=0):
                piece.x = candidate_x
                placed = True
                break

        if not placed:
            state.eliminated = True
            self.active_pieces.pop(owner_id, None)
            self.finish_by_overflow()
            return

        self.active_pieces[owner_id] = piece
        state.spawn_count += 1
        self.prepare_next_piece(owner_id)

    def get_spawn_count(self, owner_id):
        return self.owner_states[owner_id].spawn_count

    def get_controller(self, owner_id):
        return SharedPieceController(self, owner_id)

    def get_composite_grid(self, exclude_owner=None):
        return [row[:] for row in self.grid]

    def cell_occupied_by_other_piece(self, owner_id, x, y):
        for other_owner, piece in self.active_pieces.items():
            if other_owner == owner_id:
                continue
            for cell_x, cell_y in piece.occupied_cells():
                if cell_x == x and cell_y == y:
                    return True
        return False

    def active_collision_target(self, owner_id, shape=None, x=None, y=None, ignored_owners=None):
        piece = self.active_pieces.get(owner_id)
        if piece is None and shape is None:
            return None

        shape = piece.shape_matrix if shape is None else shape
        x = piece.x if x is None else x
        y = piece.y if y is None else y
        ignored = set(ignored_owners or ())
        ignored.add(owner_id)
        cells = set()
        for row_index, row in enumerate(shape):
            for col_index, cell in enumerate(row):
                if cell:
                    cells.add((x + col_index, y + row_index))

        for other_owner, other_piece in self.active_pieces.items():
            if other_owner in ignored:
                continue
            if cells.intersection(other_piece.occupied_cells()):
                return other_owner
        return None

    def collision_reason(self, owner_id, shape=None, x=None, y=None):
        piece = self.active_pieces.get(owner_id)
        if piece is None and shape is None:
            return "missing"
        shape = piece.shape_matrix if shape is None else shape
        x = piece.x if x is None else x
        y = piece.y if y is None else y

        for row_index, row in enumerate(shape):
            for col_index, cell in enumerate(row):
                if not cell:
                    continue
                board_x = x + col_index
                board_y = y + row_index
                if board_x < 0 or board_x >= GRID_WIDTH:
                    return "bounds"
                if board_y >= GRID_HEIGHT:
                    return "floor"
                if board_y >= 0 and self.grid[board_y][board_x]:
                    return "stack"
        return None

    def check_collision(self, owner_id, shape=None, x=None, y=None):
        return self.collision_reason(owner_id, shape=shape, x=x, y=y) is not None

    def try_bump_piece(self, attacker_owner, target_owner, dx, dy):
        if dx == 0 and dy == 0:
            return False

        attacker = self.active_pieces.get(attacker_owner)
        target = self.active_pieces.get(target_owner)
        if attacker is None or target is None or not attacker.can_bump:
            return False

        attacker_next_x = attacker.x + dx
        attacker_next_y = attacker.y + dy
        bump_distance = 5 if dx != 0 else 3
        target_next_x = target.x + dx * bump_distance
        target_next_y = target.y + dy * bump_distance

        target_reason = self.collision_reason(
            target_owner,
            shape=target.shape_matrix,
            x=target_next_x,
            y=target_next_y,
        )
        if target_reason is not None:
            return False

        attacker_cells = set(attacker.occupied_cells(x=attacker_next_x, y=attacker_next_y))
        target_cells = set(target.occupied_cells(x=target_next_x, y=target_next_y))
        if attacker_cells.intersection(target_cells):
            return False

        other_target = self.active_collision_target(
            target_owner,
            shape=target.shape_matrix,
            x=target_next_x,
            y=target_next_y,
            ignored_owners={attacker_owner},
        )
        if other_target is not None:
            return False

        target.x = target_next_x
        target.y = target_next_y
        target.can_bump = True
        attacker.can_bump = False
        self.refresh_piece_lock_state(target_owner, reset_timer=True)
        direction_text = "BUMP"
        if dx > 0:
            direction_text = "BUMP >"
        elif dx < 0:
            direction_text = "BUMP <"
        elif dy > 0:
            direction_text = "BUMP V"
        self.add_floating_text(
            direction_text,
            target.x + len(target.shape_matrix[0]) / 2,
            target.y + len(target.shape_matrix) / 2,
            self.owner_states[attacker_owner].block_color,
            ttl=0.4,
            rise_speed=18.0,
        )
        self.trigger_board_shake(dx, dy)
        return True

    def move_piece_result(self, owner_id, dx, dy):
        if self.game_over or self.paused:
            return "blocked"
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return "missing"
        next_x = piece.x + dx
        next_y = piece.y + dy

        locked_reason = self.collision_reason(owner_id, x=next_x, y=next_y)
        if locked_reason is not None:
            return locked_reason

        target_owner = self.active_collision_target(owner_id, x=next_x, y=next_y)
        if target_owner is not None:
            if self.try_bump_piece(owner_id, target_owner, dx, dy):
                piece.x = next_x
                piece.y = next_y
                self.refresh_piece_lock_state(owner_id, reset_timer=True)
                return "bumped"
            return "active_piece"

        if not self.check_collision(owner_id, x=next_x, y=next_y):
            piece.x = next_x
            piece.y = next_y
            self.refresh_piece_lock_state(owner_id, reset_timer=True)
            return "moved"
        return "blocked"

    def move_piece(self, owner_id, dx, dy):
        return self.move_piece_result(owner_id, dx, dy) in {"moved", "bumped"}

    def can_shift_horizontally(self, owner_id, dx):
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return False
        return not self.check_collision(owner_id, x=piece.x + dx, y=piece.y)

    def shift_horizontally(self, owner_id, dx):
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return False
        piece.x += dx
        return True

    def horizontal_contact_pair(self):
        active_owner_ids = [owner_id for owner_id in self.owner_order if owner_id in self.active_pieces]
        if len(active_owner_ids) != 2:
            return None

        first_owner, second_owner = active_owner_ids
        first_cells = set(self.active_pieces[first_owner].occupied_cells())
        second_cells = set(self.active_pieces[second_owner].occupied_cells())

        first_left_of_second = any((cell_x + 1, cell_y) in second_cells for cell_x, cell_y in first_cells)
        second_left_of_first = any((cell_x + 1, cell_y) in first_cells for cell_x, cell_y in second_cells)

        if first_left_of_second and not second_left_of_first:
            return first_owner, second_owner
        if second_left_of_first and not first_left_of_second:
            return second_owner, first_owner
        return None

    def can_shift_group(self, owner_ids, dx):
        return all(self.can_shift_horizontally(owner_id, dx) for owner_id in owner_ids)

    def shift_group(self, owner_ids, dx):
        for owner_id in owner_ids:
            self.shift_horizontally(owner_id, dx)

    def apply_horizontal_actions(self, action_map):
        intents = {}
        for owner_id in self.owner_order:
            action = action_map.get(owner_id)
            if action == "left":
                intents[owner_id] = -1
            elif action == "right":
                intents[owner_id] = 1
            else:
                intents[owner_id] = 0

        for owner_id in self.owner_order:
            dx = intents[owner_id]
            if dx:
                self.move_piece(owner_id, dx, 0)

    def apply_action(self, owner_id, action):
        if self.game_over or self.paused:
            return False
        if action == "left":
            return self.move_piece(owner_id, -1, 0)
        if action == "right":
            return self.move_piece(owner_id, 1, 0)
        if action == "soft_drop":
            result = self.move_piece_result(owner_id, 0, 1)
            if result in {"moved", "bumped"}:
                return True
            if result in ("floor", "stack"):
                self.refresh_piece_lock_state(owner_id)
            return False
        if action == "rotate":
            return self.rotate_piece(owner_id)
        if action == "hard_drop":
            return self.hard_drop_piece(owner_id)
        return False

    def advance_gravity(self):
        if self.game_over or self.paused:
            return

        before_states = self.capture_piece_states()
        for owner_id in self.owner_order:
            if self.game_over:
                break
            if owner_id not in self.active_pieces:
                continue
            result = self.move_piece_result(owner_id, 0, 1)
            if result in ("floor", "stack"):
                self.lock_piece(owner_id)
        self.update_animation_states(before_states)

    def step(self, action_map=None):
        if self.game_over or self.paused:
            return

        action_map = action_map or {}
        before_states = self.capture_piece_states()

        for owner_id in self.owner_order:
            if action_map.get(owner_id) == "rotate":
                self.rotate_piece(owner_id)

        self.apply_horizontal_actions(action_map)

        skip_gravity = set()
        for owner_id in self.owner_order:
            if action_map.get(owner_id) == "hard_drop":
                self.hard_drop_piece(owner_id)
                skip_gravity.add(owner_id)
            elif action_map.get(owner_id) == "soft_drop":
                self.apply_action(owner_id, "soft_drop")
                skip_gravity.add(owner_id)

        for owner_id in self.owner_order:
            if self.game_over or owner_id in skip_gravity:
                continue
            if owner_id not in self.active_pieces:
                continue
            result = self.move_piece_result(owner_id, 0, 1)
            if result in ("floor", "stack"):
                self.lock_piece(owner_id)
        self.update_animation_states(before_states)

    def rotate_piece(self, owner_id):
        if self.game_over or self.paused:
            return False
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return False
        rotated = self.rotate_shape(piece.shape_matrix)
        if not self.check_collision(owner_id, shape=rotated):
            piece.shape_matrix = rotated
            self.refresh_piece_lock_state(owner_id, reset_timer=True)
            return True
        return False

    def hard_drop_piece(self, owner_id):
        if self.game_over or self.paused:
            return False
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return False
        dropped = False
        while owner_id in self.active_pieces:
            result = self.move_piece_result(owner_id, 0, 1)
            if result in {"moved", "bumped"}:
                dropped = True
                continue
            if result in ("floor", "stack"):
                self.lock_piece(owner_id)
                dropped = True
            break
        if dropped:
            self.trigger_board_flash(WHITE, alpha=110, duration=0.08)
        return True

    def lock_piece(self, owner_id):
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return

        state = self.owner_states[owner_id]
        locked_snapshot = {
            "owner_id": owner_id,
            "shape_id": piece.shape_id,
            "shape_matrix": self.copy_shape(piece.shape_matrix),
            "rotation": self.rotation_index_for_shape(piece.shape_id, piece.shape_matrix),
            "x": piece.x,
            "y": piece.y,
            "spawn_count": state.spawn_count,
            "score_before": state.score,
            "lines_before": state.total_lines,
        }

        for cell_x, cell_y in piece.occupied_cells():
            if 0 <= cell_y < GRID_HEIGHT:
                self.grid[cell_y][cell_x] = piece.shape_id + 1
                self.locked_colors[cell_y][cell_x] = piece.color

        self.active_pieces.pop(owner_id, None)
        self.clear_lines(owner_id)
        self.resolve_active_piece_positions()
        locked_snapshot["score_after"] = state.score
        locked_snapshot["lines_after"] = state.total_lines
        locked_snapshot["score_delta"] = state.score - locked_snapshot["score_before"]
        locked_snapshot["lines_delta"] = state.total_lines - locked_snapshot["lines_before"]
        self.last_locked_pieces[owner_id] = locked_snapshot

        self.spawn_piece(owner_id)
        for active_owner_id in self.owner_order:
            self.refresh_piece_lock_state(active_owner_id, reset_timer=True)

    def clear_lines(self, owner_id):
        lines = [row_index for row_index in range(GRID_HEIGHT) if all(self.grid[row_index])]
        if not lines:
            return

        state = self.owner_states[owner_id]
        count = len(lines)
        state.total_lines += count
        if count == 1:
            state.single_lines += 1
        elif count == 2:
            state.double_lines += 1
        elif count == 3:
            state.triple_lines += 1
        elif count == 4:
            state.tetris_lines += 1

        for row_index in reversed(lines):
            del self.grid[row_index]
            del self.locked_colors[row_index]
        for _ in lines:
            self.grid.insert(0, [0] * GRID_WIDTH)
            self.locked_colors.insert(0, [None] * GRID_WIDTH)

        self.shift_active_pieces_after_clear(lines)
        state.score += count
        self.add_floating_text(
            f"+{count}",
            GRID_WIDTH / 2,
            min(lines) + 0.5,
            state.block_color,
            ttl=0.55,
            rise_speed=26.0,
        )

    def shift_active_pieces_after_clear(self, cleared_rows):
        for piece in self.active_pieces.values():
            shift = sum(1 for row_index in cleared_rows if row_index > piece.bottom_y())
            if shift:
                piece.y += shift

    def collides_with_locked_grid(self, piece):
        for cell_x, cell_y in piece.occupied_cells():
            if cell_x < 0 or cell_x >= GRID_WIDTH or cell_y >= GRID_HEIGHT:
                return True
            if cell_y >= 0 and self.grid[cell_y][cell_x]:
                return True
        return False

    def resolve_active_piece_positions(self):
        for owner_id, piece in self.active_pieces.items():
            moved = False
            while self.collides_with_locked_grid(piece):
                piece.y -= 1
                moved = True
            if moved:
                piece.fall_time = 0.0
                self.refresh_piece_lock_state(owner_id, reset_timer=True)

    def finish_by_overflow(self):
        best_score = max(state.score for state in self.owner_states.values())
        winners = [
            owner_id
            for owner_id, state in self.owner_states.items()
            if state.score == best_score
        ]
        self.winner = winners[0] if len(winners) == 1 else None
        self.game_over = True

    def update(self, dt):
        if self.game_over or self.paused:
            return

        interval = self.fall_interval
        for owner_id in self.owner_order:
            if self.game_over:
                break
            piece = self.active_pieces.get(owner_id)
            if piece is None:
                continue
            piece.fall_time += dt
            if piece.fall_time < interval:
                continue
            piece.fall_time = 0.0
            if not self.move_piece(owner_id, 0, 1):
                self.lock_piece(owner_id)

    def get_ghost_position(self, owner_id):
        piece = self.active_pieces.get(owner_id)
        if piece is None:
            return 0
        ghost_y = piece.y
        while not self.check_collision(owner_id, y=ghost_y + 1):
            ghost_y += 1
        return ghost_y

    def draw_block(self, x, y, color, alpha=255, glow_color=None):
        offset_x, offset_y = self.current_board_offset()
        rect = pygame.Rect(
            round(SIDEBAR_WIDTH + x * BLOCK_SIZE + offset_x),
            round(y * BLOCK_SIZE + offset_y),
            BLOCK_SIZE,
            BLOCK_SIZE,
        )
        if glow_color is not None:
            glow_rect = rect.inflate(8, 8)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*glow_color, 70), glow_surface.get_rect(), border_radius=8)
            self.screen.blit(glow_surface, glow_rect.topleft)
        if alpha < 255:
            surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surface, (*color, alpha), (0, 0, BLOCK_SIZE, BLOCK_SIZE))
            border_color = glow_color if glow_color is not None else BLACK
            border_width = 3 if glow_color is not None else 2
            pygame.draw.rect(surface, border_color, (0, 0, BLOCK_SIZE, BLOCK_SIZE), border_width)
            self.screen.blit(surface, rect)
        else:
            pygame.draw.rect(self.screen, color, rect)
            border_color = glow_color if glow_color is not None else BLACK
            border_width = 3 if glow_color is not None else 2
            pygame.draw.rect(self.screen, border_color, rect, border_width)

    def draw_background(self):
        if not self.show_background:
            self.screen.fill(self.capture_background_color)
            return
        y_offset = self.background_scroll % SCREEN_HEIGHT
        self.screen.blit(self.background, (0, y_offset))
        self.screen.blit(self.background, (0, y_offset - SCREEN_HEIGHT))
        if not self.background_paused:
            self.background_scroll += 1

    def render_color(self, color):
        if self.uniform_block_colors:
            return self.capture_block_color
        return color

    def draw_grid_lines(self):
        if not self.show_grid or self.capture_clean_mode:
            return
        offset_x, offset_y = self.current_board_offset()
        for x in range(GRID_WIDTH + 1):
            px = SIDEBAR_WIDTH + x * BLOCK_SIZE + offset_x
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (px, offset_y),
                (px, GRID_HEIGHT * BLOCK_SIZE + offset_y),
                1,
            )
        for y in range(GRID_HEIGHT + 1):
            py = y * BLOCK_SIZE + offset_y
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (SIDEBAR_WIDTH + offset_x, py),
                (SIDEBAR_WIDTH + GRID_WIDTH * BLOCK_SIZE + offset_x, py),
                1,
            )

    def draw_feedbacks(self):
        offset_x, offset_y = self.current_board_offset()
        if self.board_flash_timer > 0 and self.board_flash_total > 0:
            alpha_scale = self.board_flash_timer / self.board_flash_total
            alpha = round(self.board_flash_alpha * alpha_scale)
            flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.board_flash_color, alpha))
            self.screen.blit(flash_surface, (SIDEBAR_WIDTH + offset_x, offset_y))

        for text in self.floating_texts:
            alpha = 255
            if text.total_ttl > 0:
                alpha = max(0, min(255, round(255 * (text.ttl / text.total_ttl))))
            image = self.feedback_font.render(text.text, True, text.color)
            image.set_alpha(alpha)
            rect = image.get_rect(
                center=(
                    SIDEBAR_WIDTH + offset_x + round(text.x * BLOCK_SIZE),
                    offset_y + round(text.y * BLOCK_SIZE),
                )
            )
            self.screen.blit(image, rect)

    def draw_game_area(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x]:
                    self.draw_block(x, y, self.render_color(self.locked_colors[y][x] or WHITE))

        if self.show_ghost and not self.capture_clean_mode and not self.game_over:
            for owner_id in self.owner_order:
                piece = self.active_pieces.get(owner_id)
                if piece is None:
                    continue
                ghost_y = self.get_ghost_position(owner_id)
                for row_index, row in enumerate(piece.shape_matrix):
                    for col_index, cell in enumerate(row):
                        if cell:
                            self.draw_block(
                                piece.x + col_index,
                                ghost_y + row_index,
                                self.render_color(piece.color),
                                60,
                            )

        for owner_id in self.owner_order:
            piece = self.active_pieces.get(owner_id)
            if piece is None:
                continue
            draw_x, draw_y = self.interpolated_position(owner_id, piece)
            glow_color = YELLOW if piece.can_bump else None
            for row_index, row in enumerate(piece.shape_matrix):
                for col_index, cell in enumerate(row):
                    if cell:
                        self.draw_block(
                            draw_x + col_index,
                            draw_y + row_index,
                            self.render_color(piece.color),
                            glow_color=glow_color,
                        )

    def draw_panel(self, owner_id, x):
        state = self.owner_states[owner_id]
        panel_rect = pygame.Rect(x, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)
        panel_surface = pygame.Surface((SIDEBAR_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        panel_surface.fill((30, 30, 40, 180))
        self.screen.blit(panel_surface, panel_rect)

        self.screen.blit(self.title_font.render(state.label, True, WHITE), (x + 20, 20))
        self.screen.blit(self.font.render(f"Score: {state.score}", True, WHITE), (x + 20, 70))
        active_piece = self.active_pieces.get(owner_id)
        bump_ready = active_piece is not None and active_piece.can_bump
        bump_label = "READY" if bump_ready else "USED"
        bump_color = GREEN if bump_ready else GRAY
        self.screen.blit(self.font.render("Bump:", True, WHITE), (x + 20, 100))
        self.screen.blit(self.font.render(bump_label, True, bump_color), (x + 90, 100))
        self.screen.blit(self.font.render("Next:", True, WHITE), (x + 20, 140))
        next_shape = state.next_shape_matrix
        start_x = x + 20
        start_y = 170
        for row_index, row in enumerate(next_shape):
            for col_index, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        start_x + col_index * BLOCK_SIZE,
                        start_y + row_index * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE,
                    )
                    pygame.draw.rect(self.screen, self.render_color(state.next_color), rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 2)

        text_y = 300
        for line in state.controls:
            self.screen.blit(self.font.render(line, True, WHITE), (x + 20, text_y))
            text_y += 25

        text_y += 20
        for line in self.common_controls:
            self.screen.blit(self.font.render(line, True, WHITE), (x + 20, text_y))
            text_y += 25

        status_y = SCREEN_HEIGHT - 80
        if self.game_over:
            if self.winner == owner_id:
                text = f"{state.label} WIN"
            elif self.winner is None:
                text = "TIE"
            else:
                text = "LOSE"
            self.screen.blit(self.font.render(text, True, WHITE), (x + 20, status_y))

    def draw_overlay(self):
        if self.capture_clean_mode:
            return
        if self.paused and not self.game_over:
            text = "PAUSED"
        elif not self.game_over:
            return
        else:
            if self.winner is None:
                text = "TIE GAME"
            else:
                text = f"{self.owner_states[self.winner].label} WIN"
        image = self.overlay_font.render(text, True, WHITE)
        rect = image.get_rect(center=(SIDEBAR_WIDTH + SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(image, rect)

    def draw(self):
        self.draw_background()
        self.draw_game_area()
        self.draw_feedbacks()
        self.draw_grid_lines()
        if not self.capture_clean_mode:
            self.draw_panel(self.owner_order[0], 0)
            self.draw_panel(self.owner_order[1], SIDEBAR_WIDTH + SCREEN_WIDTH)
        self.draw_overlay()
