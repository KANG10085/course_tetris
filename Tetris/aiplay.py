import argparse
import sys
import time

import pygame

from ai import TetrisAI
from config import *
from input_feel import InputFeelController
from shared_game import SharedBoardTetris


DEFAULT_DIFFICULTY = "normal"
DIFFICULTY_SETTINGS = {
    "easy": {
        "label": "Easy",
        "action_interval": 0.55,
        "landing_error_prob": 0.15,
    },
    "normal": {
        "label": "Normal",
        "action_interval": 0.35,
        "landing_error_prob": 0.06,
    },
    "hard": {
        "label": "Hard",
        "action_interval": 0.2,
        "landing_error_prob": 0.0,
    },
}


def difficulty_setting(args):
    difficulty = getattr(args, "difficulty", DEFAULT_DIFFICULTY)
    return DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS[DEFAULT_DIFFICULTY])


def parse_args():
    parser = argparse.ArgumentParser(description="Player vs AI Tetris")
    parser.add_argument(
        "--ai",
        choices=["heuristic", "rl"],
        default="rl",
        help="choose which AI implementation to use",
    )
    parser.add_argument(
        "--checkpoint",
        default="shared_model3_stage2_v1/best.pt",
        help="checkpoint path used when --ai rl",
    )
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTY_SETTINGS),
        default=DEFAULT_DIFFICULTY,
        help="choose AI speed and landing error profile",
    )
    return parser.parse_args()


def build_ai(game, args):
    controller = game.get_controller("ai")
    if args.ai == "rl":
        from ai_rl import RLTetrisAI

        ai = RLTetrisAI(controller, checkpoint_path=args.checkpoint)
    else:
        ai = TetrisAI(controller)
    ai.landing_error_prob = difficulty_setting(args)["landing_error_prob"]
    return ai


def build_game(screen, args):
    difficulty = difficulty_setting(args)
    owner_configs = [
        {
            "id": "player",
            "label": "YOU",
            "color": RED,
            "controls": [
                "A left",
                "D right",
                "S down 1",
                "W hard drop",
                "E rotate",
            ],
        },
        {
            "id": "ai",
            "label": "AI",
            "color": BLUE,
            "controls": [
                f"AI {difficulty['label']}",
                f"AI step {difficulty['action_interval']:.2f}s",
                f"Error {difficulty['landing_error_prob'] * 100:.0f}%",
            ],
        },
    ]
    game = SharedBoardTetris(screen, owner_configs, title=f"Tetris Shared Board - AI {difficulty['label']}")
    game.common_controls = [
        "P pause",
        "R restart",
        "Q menu",
    ]
    ai = build_ai(game, args)
    return game, ai


def map_player_key(key):
    if key == pygame.K_a:
        return "left"
    if key == pygame.K_d:
        return "right"
    if key == pygame.K_s:
        return "soft_drop"
    if key == pygame.K_w:
        return "hard_drop"
    if key == pygame.K_e:
        return "rotate"
    return None


def sync_ai_state(game, ai, last_spawn_count):
    current_spawn_count = game.get_spawn_count("ai")
    if current_spawn_count != last_spawn_count:
        ai.reset_plan()
    return current_spawn_count


def sync_player_spawn_state(game, input_feel, last_spawn_count):
    current_spawn_count = game.get_spawn_count("player")
    if current_spawn_count != last_spawn_count:
        input_feel.clear_owner_actions("player", actions={"soft_drop", "hard_drop"})
    return current_spawn_count


def ai_piece_state_signature(controller):
    shape = controller.get_current_shape()
    shape_signature = tuple(tuple(row) for row in shape)
    return (controller.current_x, controller.current_y, shape_signature)


def maybe_reset_ai_plan_after_external_move(ai, before_state, controller):
    after_state = ai_piece_state_signature(controller)
    x_changed = after_state[0] != before_state[0]
    y_jump = abs(after_state[1] - before_state[1]) > 1
    shape_changed = after_state[2] != before_state[2]
    if x_changed or y_jump or shape_changed:
        ai.reset_plan()


def maybe_reset_ai_plan(ai, ai_action, before_x, before_shape, controller):
    if ai_action == "left" and controller.current_x == before_x:
        ai.reset_plan()
    elif ai_action == "right" and controller.current_x == before_x:
        ai.reset_plan()
    elif ai_action == "rotate" and controller.get_current_shape() == before_shape:
        ai.reset_plan()


def apply_player_action(game, ai, ai_controller, action):
    ai_before_state = ai_piece_state_signature(ai_controller)
    applied = game.apply_action("player", action)
    if applied:
        maybe_reset_ai_plan_after_external_move(ai, ai_before_state, ai_controller)
    return applied


def finish_session(embedded, destination):
    if embedded:
        return destination
    pygame.quit()
    sys.exit()


def main(embedded=False, args=None):
    if args is None:
        args = parse_args()
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Shared Board: Player vs AI")

    game, ai = build_game(screen, args)
    ai_controller = game.get_controller("ai")
    ai_spawn_count = game.get_spawn_count("ai")
    player_spawn_count = game.get_spawn_count("player")
    gravity_interval = 0.5
    ai_interval = difficulty_setting(args)["action_interval"]
    frame_interval = 1 / 60
    gravity_accumulator = 0.0
    ai_accumulator = 0.0
    last_tick = time.perf_counter()
    input_feel = InputFeelController()
    game.set_animation_progress(1.0)

    while True:
        frame_start = time.perf_counter()
        elapsed = frame_start - last_tick
        last_tick = frame_start
        elapsed = min(elapsed, gravity_interval)
        if not game.game_over and not game.paused:
            gravity_accumulator += elapsed
            ai_accumulator += elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return finish_session(embedded, "quit")
            if event.type == pygame.KEYUP:
                input_feel.release_key(event.key)
                continue
            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_q:
                return finish_session(embedded, "menu")
            elif event.key == pygame.K_p:
                game.paused = not game.paused
                last_tick = time.perf_counter()
                input_feel.clear()
            elif event.key == pygame.K_r:
                game, ai = build_game(screen, args)
                ai_controller = game.get_controller("ai")
                ai_spawn_count = game.get_spawn_count("ai")
                player_spawn_count = game.get_spawn_count("player")
                gravity_accumulator = 0.0
                ai_accumulator = 0.0
                ai_interval = difficulty_setting(args)["action_interval"]
                last_tick = time.perf_counter()
                input_feel.clear()
            else:
                mapped_action = map_player_key(event.key)
                if mapped_action is not None and not game.game_over and not game.paused:
                    action_time = time.perf_counter()
                    input_feel.note_keydown(
                        event.key,
                        "player",
                        mapped_action,
                        game.get_spawn_count("player"),
                        action_time,
                    )
                    apply_player_action(game, ai, ai_controller, mapped_action)

        now = time.perf_counter()
        input_feel.process(
            get_spawn_count=game.get_spawn_count,
            apply_action=lambda owner_id, action: apply_player_action(game, ai, ai_controller, action),
            now=now,
            enabled=not game.game_over and not game.paused,
        )

        player_spawn_count = sync_player_spawn_state(game, input_feel, player_spawn_count)
        ai_spawn_count = sync_ai_state(game, ai, ai_spawn_count)
        while ai_accumulator >= ai_interval and not game.game_over and not game.paused:
            before_x = ai_controller.current_x
            before_shape = ai_controller.get_current_shape()
            ai_action = ai.next_action()
            if ai_action is not None:
                game.apply_action("ai", ai_action)
            maybe_reset_ai_plan(ai, ai_action, before_x, before_shape, ai_controller)
            ai_accumulator -= ai_interval
            ai_spawn_count = sync_ai_state(game, ai, ai_spawn_count)
            player_spawn_count = sync_player_spawn_state(game, input_feel, player_spawn_count)
        while gravity_accumulator >= gravity_interval and not game.game_over and not game.paused:
            ai_before_state = ai_piece_state_signature(ai_controller)
            game.advance_gravity()
            maybe_reset_ai_plan_after_external_move(ai, ai_before_state, ai_controller)
            gravity_accumulator -= gravity_interval
            ai_spawn_count = sync_ai_state(game, ai, ai_spawn_count)
            player_spawn_count = sync_player_spawn_state(game, input_feel, player_spawn_count)

        game.update_frame(elapsed)
        player_spawn_count = sync_player_spawn_state(game, input_feel, player_spawn_count)
        game.set_animation_progress(1.0)

        game.draw()
        pygame.display.flip()

        frame_elapsed = time.perf_counter() - frame_start
        if frame_elapsed < frame_interval:
            time.sleep(frame_interval - frame_elapsed)


if __name__ == "__main__":
    main()
