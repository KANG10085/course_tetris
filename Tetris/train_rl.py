from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import torch
import torch.nn.functional as F
from torch import nn

from config import (
    BLUE,
    GRID_HEIGHT,
    GRID_WIDTH,
    RED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SIDEBAR_WIDTH,
)
from rl_value_net import AfterstateValueNet
from shared_game import SharedBoardTetris


AGENT_ID = "agent"
OPPONENT_ID = "opponent"


@dataclass(frozen=True)
class PlacementPlan:
    x: int
    rotation: int
    landing_y: int


@dataclass
class Transition:
    feature: List[float]
    reward: float
    done: bool
    next_features: List[List[float]]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.items: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.items)

    def push(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self.items), batch_size)


class AfterstateFeatureBuilder:
    def legal_moves(self, controller) -> List[PlacementPlan]:
        active_piece = self._active_piece(controller)
        if active_piece is None:
            return []

        moves: List[PlacementPlan] = []
        shape = controller.get_current_shape()
        for rotation, rotated_shape in self._unique_rotations(shape):
            for x in range(-len(rotated_shape[0]), GRID_WIDTH + 1):
                if controller.check_collision(rotated_shape, x, 0):
                    continue
                landing_y = self._simulate_drop_y(controller, rotated_shape, x)
                moves.append(PlacementPlan(x=x, rotation=rotation, landing_y=landing_y))

        moves.sort(key=lambda move: (move.rotation, move.x))
        return moves

    def options(self, controller) -> List[Tuple[PlacementPlan, List[float]]]:
        return [(move, self.afterstate_vector(controller, move)) for move in self.legal_moves(controller)]

    def afterstate_vector(self, controller, move: PlacementPlan) -> List[float]:
        preview = self.preview_afterstate(controller, move)
        return self.build_feature_vector(
            board_binary=preview["board_binary"],
            column_heights=preview["column_heights"],
            holes=preview["holes"],
            aggregate_height=preview["aggregate_height"],
            bumpiness=preview["bumpiness"],
            piece_id=preview["next_shape_id"],
        )

    def preview_afterstate(self, controller, move: PlacementPlan):
        active_piece = self._active_piece(controller)
        if active_piece is None:
            raise RuntimeError("no active piece available for preview")

        shape = self._rotate_n(controller.get_current_shape(), move.rotation)
        grid = [row[:] for row in controller.grid]

        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                board_y = move.landing_y + row_idx
                board_x = move.x + col_idx
                if cell and 0 <= board_y < GRID_HEIGHT and 0 <= board_x < GRID_WIDTH:
                    grid[board_y][board_x] = active_piece.shape_id + 1

        grid = self.clear_full_rows(grid)
        heights = self.column_heights(grid)
        return {
            "board_binary": self.board_as_binary(grid),
            "column_heights": heights,
            "holes": self.count_holes(grid),
            "aggregate_height": sum(heights),
            "bumpiness": self.bumpiness_from_heights(heights),
            "next_shape_id": self._next_shape_id(controller),
        }

    def board_quality(self, grid: List[List[int]]) -> dict:
        heights = self.column_heights(grid)
        return {
            "holes": self.count_holes(grid),
            "aggregate_height": sum(heights),
            "bumpiness": self.bumpiness_from_heights(heights),
        }

    def _active_piece(self, controller):
        return controller.shared_game.active_pieces.get(controller.owner_id)

    def _next_shape_id(self, controller) -> int:
        return int(controller.shared_game.owner_states[controller.owner_id].next_shape_id)

    def _simulate_drop_y(self, controller, shape: List[List[int]], x: int) -> int:
        y = 0
        while not controller.check_collision(shape, x, y + 1):
            y += 1
        return y

    def _rotate_n(self, shape: List[List[int]], count: int) -> List[List[int]]:
        rotated = [row[:] for row in shape]
        for _ in range(count % 4):
            rotated = self.rotate_shape(rotated)
        return rotated

    def rotate_shape(self, shape: List[List[int]]) -> List[List[int]]:
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
            rotated = self.rotate_shape(rotated)
        return rotations

    def clear_full_rows(self, grid: List[List[int]]) -> List[List[int]]:
        remaining_rows = [row[:] for row in grid if not all(row)]
        cleared = GRID_HEIGHT - len(remaining_rows)
        if cleared == 0:
            return [row[:] for row in grid]
        return [[0 for _ in range(GRID_WIDTH)] for _ in range(cleared)] + remaining_rows

    def board_as_binary(self, grid: List[List[int]]) -> List[float]:
        return [1.0 if cell else 0.0 for row in grid for cell in row]

    def column_heights(self, grid: List[List[int]]) -> List[int]:
        heights = [0] * len(grid[0])
        for col in range(len(grid[0])):
            for row in range(len(grid)):
                if grid[row][col]:
                    heights[col] = len(grid) - row
                    break
        return heights

    def bumpiness_from_heights(self, heights: Sequence[int]) -> int:
        return sum(abs(heights[idx] - heights[idx + 1]) for idx in range(len(heights) - 1))

    def count_holes(self, grid: List[List[int]]) -> int:
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

    def build_feature_vector(
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


class OpponentPolicy:
    def choose(self, controller, feature_builder: AfterstateFeatureBuilder) -> Optional[PlacementPlan]:
        raise NotImplementedError


class RandomOpponent(OpponentPolicy):
    def choose(self, controller, feature_builder: AfterstateFeatureBuilder) -> Optional[PlacementPlan]:
        moves = feature_builder.legal_moves(controller)
        if not moves:
            return None
        return random.choice(moves)


class HeuristicOpponent(OpponentPolicy):
    def choose(self, controller, feature_builder: AfterstateFeatureBuilder) -> Optional[PlacementPlan]:
        best_move = None
        best_score = float("-inf")
        for move in feature_builder.legal_moves(controller):
            preview = feature_builder.preview_afterstate(controller, move)
            score = (
                -0.5 * preview["aggregate_height"]
                - 0.3 * preview["holes"]
                - 0.2 * preview["bumpiness"]
            )
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


class CheckpointOpponent(OpponentPolicy):
    def __init__(self, checkpoint_path: str, input_dim: int, hidden_dims: Sequence[int], device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_hidden = checkpoint.get("config", {}).get("hidden", hidden_dims)
        self.model = AfterstateValueNet(input_dim=input_dim, hidden_dims=tuple(checkpoint_hidden)).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device

    def choose(self, controller, feature_builder: AfterstateFeatureBuilder) -> Optional[PlacementPlan]:
        options = feature_builder.options(controller)
        if not options:
            return None
        features = [feature for _, feature in options]
        with torch.no_grad():
            values = self.model(torch.tensor(features, dtype=torch.float32, device=self.device)).squeeze(-1)
            index = int(torch.argmax(values).item())
        return options[index][0]


class MixedOpponent(OpponentPolicy):
    def __init__(self, policies: Iterable[OpponentPolicy]):
        self.policies = list(policies)

    def choose(self, controller, feature_builder: AfterstateFeatureBuilder) -> Optional[PlacementPlan]:
        if not self.policies:
            return None
        return random.choice(self.policies).choose(controller, feature_builder)


class SharedBoardRLEnv:
    def __init__(self, screen, opponent_policy: OpponentPolicy):
        self.screen = screen
        self.feature_builder = AfterstateFeatureBuilder()
        self.opponent_policy = opponent_policy
        self.game: Optional[SharedBoardTetris] = None
        self.agent_controller = None
        self.opponent_controller = None
        self.step_count = 0

    def reset(self) -> List[Tuple[PlacementPlan, List[float]]]:
        owner_configs = [
            {
                "id": AGENT_ID,
                "label": "AI",
                "color": BLUE,
                "controls": [],
            },
            {
                "id": OPPONENT_ID,
                "label": "OPP",
                "color": RED,
                "controls": [],
            },
        ]
        self.game = SharedBoardTetris(self.screen, owner_configs, title="RL Training")
        self.agent_controller = self.game.get_controller(AGENT_ID)
        self.opponent_controller = self.game.get_controller(OPPONENT_ID)
        self.step_count = 0
        return self.agent_options()

    def agent_options(self) -> List[Tuple[PlacementPlan, List[float]]]:
        if self.game is None or self.game.game_over:
            return []
        return self.feature_builder.options(self.agent_controller)

    def step(self, agent_plan: PlacementPlan) -> Tuple[float, bool, List[Tuple[PlacementPlan, List[float]]]]:
        if self.game is None:
            raise RuntimeError("environment must be reset before step")

        before = self._snapshot()

        self._execute_plan(AGENT_ID, agent_plan)
        if not self.game.game_over:
            opponent_plan = self.opponent_policy.choose(self.opponent_controller, self.feature_builder)
            if opponent_plan is not None:
                self._execute_plan(OPPONENT_ID, opponent_plan)

        self.step_count += 1
        after = self._snapshot()
        reward = self._reward(before, after)
        done = self.game.game_over
        return reward, done, self.agent_options()

    def scores(self) -> Tuple[int, int]:
        if self.game is None:
            return 0, 0
        return self.game.owner_states[AGENT_ID].score, self.game.owner_states[OPPONENT_ID].score

    def winner_by_score(self) -> Optional[str]:
        if self.game is None:
            return None
        if self.game.winner is not None:
            return self.game.winner
        agent_score, opponent_score = self.scores()
        if agent_score > opponent_score:
            return AGENT_ID
        if opponent_score > agent_score:
            return OPPONENT_ID
        return None

    def _execute_plan(self, owner_id: str, plan: PlacementPlan) -> None:
        if self.game is None or self.game.game_over:
            return
        controller = self.game.get_controller(owner_id)

        for _ in range(plan.rotation % 4):
            if self.game.game_over:
                return
            self.game.apply_action(owner_id, "rotate")

        guard = GRID_WIDTH + 8
        while not self.game.game_over and controller.current_x < plan.x and guard > 0:
            self.game.apply_action(owner_id, "right")
            guard -= 1
        while not self.game.game_over and controller.current_x > plan.x and guard > 0:
            self.game.apply_action(owner_id, "left")
            guard -= 1

        if not self.game.game_over:
            self.game.apply_action(owner_id, "hard_drop")

    def _snapshot(self) -> dict:
        if self.game is None:
            raise RuntimeError("environment must be reset before snapshot")
        agent = self.game.owner_states[AGENT_ID]
        opponent = self.game.owner_states[OPPONENT_ID]
        quality = self.feature_builder.board_quality(self.game.grid)
        return {
            "agent_score": agent.score,
            "opponent_score": opponent.score,
            "agent_lines": agent.total_lines,
            "opponent_lines": opponent.total_lines,
            "quality": quality,
            "winner": self.game.winner,
            "game_over": self.game.game_over,
        }

    def _reward(self, before: dict, after: dict) -> float:
        agent_delta = after["agent_score"] - before["agent_score"]
        opponent_delta = after["opponent_score"] - before["opponent_score"]
        before_diff = before["agent_score"] - before["opponent_score"]
        after_diff = after["agent_score"] - after["opponent_score"]
        diff_delta = after_diff - before_diff

        before_quality = before["quality"]
        after_quality = after["quality"]
        quality_reward = (
            0.20 * (before_quality["holes"] - after_quality["holes"])
            + 0.02 * (before_quality["aggregate_height"] - after_quality["aggregate_height"])
            + 0.02 * (before_quality["bumpiness"] - after_quality["bumpiness"])
        )

        terminal_reward = 0.0
        if after["game_over"]:
            winner = after["winner"]
            if winner == AGENT_ID:
                terminal_reward = 50.0
            elif winner == OPPONENT_ID:
                terminal_reward = -50.0

        return (
            8.0 * agent_delta
            - 8.0 * opponent_delta
            + 2.0 * diff_delta
            + quality_reward
            + terminal_reward
            - 0.01
        )


def ensure_pygame_screen():
    pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
    return pygame.display.get_surface()


def select_action(
    model: nn.Module,
    options: List[Tuple[PlacementPlan, List[float]]],
    epsilon: float,
    device: torch.device,
) -> Tuple[PlacementPlan, List[float]]:
    if random.random() < epsilon:
        return random.choice(options)

    features = [feature for _, feature in options]
    with torch.no_grad():
        values = model(torch.tensor(features, dtype=torch.float32, device=device)).squeeze(-1)
        index = int(torch.argmax(values).item())
    return options[index]


def optimize_model(
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < batch_size:
        return None

    batch = replay.sample(batch_size)
    features = torch.tensor([item.feature for item in batch], dtype=torch.float32, device=device)
    q_values = model(features).squeeze(-1)

    targets = []
    with torch.no_grad():
        for item in batch:
            if item.done or not item.next_features:
                targets.append(item.reward)
                continue
            next_tensor = torch.tensor(item.next_features, dtype=torch.float32, device=device)
            max_next = target_model(next_tensor).squeeze(-1).max().item()
            targets.append(item.reward + gamma * max_next)

    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    loss = F.smooth_l1_loss(q_values, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    total_steps: int,
    epsilon: float,
    config: dict,
    best_state: Optional[dict],
    best_metric: Optional[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "total_steps": total_steps,
            "epsilon": epsilon,
            "model_state_dict": model.state_dict(),
            "target_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "source": "shared_afterstate_rl",
            "best_state": best_state,
            "best_metric": best_metric,
        },
        path,
    )


def load_model_state(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def build_opponent_policy(
    args,
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    device: torch.device,
) -> OpponentPolicy:
    heuristic = HeuristicOpponent()
    if args.opponent == "heuristic":
        return heuristic
    if args.opponent == "random":
        return RandomOpponent()
    if args.opponent in {"checkpoint", "mixed"}:
        checkpoint_policy = None
        if args.opponent_checkpoint:
            checkpoint_path = Path(args.opponent_checkpoint)
            if checkpoint_path.exists():
                checkpoint_policy = CheckpointOpponent(str(checkpoint_path), input_dim, hidden_dims, device)
            else:
                print(f"warning: opponent checkpoint not found: {checkpoint_path}")
        if args.opponent == "checkpoint":
            return checkpoint_policy or heuristic
        policies: List[OpponentPolicy] = [heuristic, RandomOpponent()]
        if checkpoint_policy is not None:
            policies.append(checkpoint_policy)
        return MixedOpponent(policies)
    return heuristic


def moving_average(items: Sequence[float]) -> float:
    if not items:
        return 0.0
    return sum(items) / len(items)


def metric_tuple(metric: dict) -> Tuple[float, float, float, float]:
    return (
        metric["avg_win_rate"],
        metric["avg_score_diff"],
        metric["avg_agent_score"],
        metric["avg_reward"],
    )


def make_config(args) -> dict:
    config = vars(args).copy()
    config["hidden"] = list(args.hidden)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train afterstate DQN for shared-board Tetris.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--best-min-episodes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--checkpoint-dir", default="afterstate_model3_rl_v1")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--opponent", choices=["heuristic", "random", "checkpoint", "mixed"], default="heuristic")
    parser.add_argument("--opponent-checkpoint", default="")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    screen = ensure_pygame_screen()

    bootstrap_env = SharedBoardRLEnv(screen, HeuristicOpponent())
    initial_options = bootstrap_env.reset()
    if not initial_options:
        raise RuntimeError("could not build initial legal moves")
    input_dim = len(initial_options[0][1])
    hidden_dims = tuple(args.hidden)

    model = AfterstateValueNet(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    target_model = AfterstateValueNet(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_episode = 0
    total_steps = 0
    epsilon = args.epsilon_start
    best_metric = None
    best_state = None

    if args.resume_checkpoint:
        checkpoint = load_model_state(model, args.resume_checkpoint, device)
        target_model.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint.get("episode", 0))
        total_steps = int(checkpoint.get("total_steps", 0))
        epsilon = float(checkpoint.get("epsilon", epsilon))
        best_metric = checkpoint.get("best_metric")
        best_state = checkpoint.get("best_state")
    elif args.init_checkpoint:
        load_model_state(model, args.init_checkpoint, device)
        target_model.load_state_dict(model.state_dict())
    else:
        target_model.load_state_dict(model.state_dict())

    opponent_policy = build_opponent_policy(
        args,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        device=device,
    )
    env = SharedBoardRLEnv(screen, opponent_policy)
    replay = ReplayBuffer(args.buffer_size)
    checkpoint_dir = Path(args.checkpoint_dir)
    recent = deque(maxlen=args.best_window)
    last_loss = None

    for local_episode in range(1, args.episodes + 1):
        episode = start_episode + local_episode
        options = env.reset()
        episode_reward = 0.0
        steps = 0

        for _ in range(args.max_steps):
            if not options:
                break

            plan, feature = select_action(model, options, epsilon, device)
            reward, done, next_options = env.step(plan)
            next_features = [next_feature for _, next_feature in next_options]
            replay.push(
                Transition(
                    feature=feature,
                    reward=reward,
                    done=done,
                    next_features=next_features,
                )
            )

            episode_reward += reward
            total_steps += 1
            steps += 1

            if total_steps >= args.warmup and total_steps % args.train_every == 0:
                loss = optimize_model(
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    device=device,
                )
                if loss is not None:
                    last_loss = loss

            if total_steps % args.target_update == 0:
                target_model.load_state_dict(model.state_dict())

            options = next_options
            if done:
                break

        agent_score, opponent_score = env.scores()
        winner = env.winner_by_score()
        score_diff = agent_score - opponent_score
        episode_state = {
            "episode": episode,
            "reward": episode_reward,
            "agent_score": agent_score,
            "opponent_score": opponent_score,
            "score_diff": score_diff,
            "win": 1 if winner == AGENT_ID else 0,
            "steps": steps,
        }
        recent.append(episode_state)

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if len(recent) >= args.best_min_episodes:
            metric = {
                "avg_win_rate": moving_average([item["win"] for item in recent]),
                "avg_score_diff": moving_average([item["score_diff"] for item in recent]),
                "avg_reward": moving_average([item["reward"] for item in recent]),
                "avg_agent_score": moving_average([item["agent_score"] for item in recent]),
            }
            if best_metric is None or metric_tuple(metric) > metric_tuple(best_metric):
                best_metric = metric
                best_state = {
                    "episode": episode,
                    **metric,
                }
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    episode=episode,
                    total_steps=total_steps,
                    epsilon=epsilon,
                    config=make_config(args),
                    best_state=best_state,
                    best_metric=best_metric,
                )

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            episode=episode,
            total_steps=total_steps,
            epsilon=epsilon,
            config=make_config(args),
            best_state=best_state,
            best_metric=best_metric,
        )

        if episode % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"episode_{episode:05d}.pt",
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                episode=episode,
                total_steps=total_steps,
                epsilon=epsilon,
                config=make_config(args),
                best_state=best_state,
                best_metric=best_metric,
            )

        if episode % args.log_every == 0 or local_episode == 1:
            loss_text = "n/a" if last_loss is None else f"{last_loss:.4f}"
            print(
                "episode={episode} steps={steps} reward={reward:.2f} "
                "score={agent}:{opponent} diff={diff:+d} win={win} "
                "epsilon={epsilon:.3f} replay={replay} loss={loss}".format(
                    episode=episode,
                    steps=steps,
                    reward=episode_reward,
                    agent=agent_score,
                    opponent=opponent_score,
                    diff=score_diff,
                    win=1 if winner == AGENT_ID else 0,
                    epsilon=epsilon,
                    replay=len(replay),
                    loss=loss_text,
                )
            )

    pygame.quit()


if __name__ == "__main__":
    main()
