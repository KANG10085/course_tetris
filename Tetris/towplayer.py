import sys
import time

import pygame

from config import *
from input_feel import InputFeelController
from shared_game import SharedBoardTetris


def build_game(screen):
    owner_configs = [
        {
            "id": "player1",
            "label": "P1",
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
            "id": "player2",
            "label": "P2",
            "color": BLUE,
            "controls": [
                "J left",
                "L right",
                "K down 1",
                "I hard drop",
                "O rotate",
            ],
        },
    ]
    game = SharedBoardTetris(screen, owner_configs, title="Tetris Shared Board")
    game.common_controls = [
        "P pause",
        "R restart",
        "Q menu",
    ]
    return game


def map_player1_key(key):
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


def map_player2_key(key):
    if key == pygame.K_j:
        return "left"
    if key == pygame.K_l:
        return "right"
    if key == pygame.K_k:
        return "soft_drop"
    if key == pygame.K_i:
        return "hard_drop"
    if key == pygame.K_o:
        return "rotate"
    return None


def apply_owner_action(game, owner_id, action):
    return game.apply_action(owner_id, action)


def sync_owner_spawn_state(game, input_feel, owner_id, last_spawn_count):
    current_spawn_count = game.get_spawn_count(owner_id)
    if current_spawn_count != last_spawn_count:
        input_feel.clear_owner_actions(owner_id, actions={"soft_drop", "hard_drop"})
    return current_spawn_count


def finish_session(embedded, destination):
    if embedded:
        return destination
    pygame.quit()
    sys.exit()


def main(embedded=False):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Shared Board: Two Players")

    game = build_game(screen)
    gravity_interval = 0.5
    frame_interval = 1 / 60
    gravity_accumulator = 0.0
    last_tick = time.perf_counter()
    input_feel = InputFeelController()
    spawn_counts = {
        "player1": game.get_spawn_count("player1"),
        "player2": game.get_spawn_count("player2"),
    }
    game.set_animation_progress(1.0)

    while True:
        frame_start = time.perf_counter()
        elapsed = frame_start - last_tick
        last_tick = frame_start
        elapsed = min(elapsed, gravity_interval)
        if not game.game_over and not game.paused:
            gravity_accumulator += elapsed

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
            if event.key == pygame.K_p:
                game.paused = not game.paused
                last_tick = time.perf_counter()
                input_feel.clear()
                continue
            if event.key == pygame.K_r:
                game = build_game(screen)
                gravity_accumulator = 0.0
                last_tick = time.perf_counter()
                input_feel.clear()
                spawn_counts["player1"] = game.get_spawn_count("player1")
                spawn_counts["player2"] = game.get_spawn_count("player2")
                continue

            if game.game_over:
                continue
            if game.paused:
                continue

            now = time.perf_counter()
            player1_action = map_player1_key(event.key)
            if player1_action is not None:
                input_feel.note_keydown(
                    event.key,
                    "player1",
                    player1_action,
                    game.get_spawn_count("player1"),
                    now,
                )
                apply_owner_action(game, "player1", player1_action)
                continue

            player2_action = map_player2_key(event.key)
            if player2_action is not None:
                input_feel.note_keydown(
                    event.key,
                    "player2",
                    player2_action,
                    game.get_spawn_count("player2"),
                    now,
                )
                apply_owner_action(game, "player2", player2_action)

        now = time.perf_counter()
        input_feel.process(
            get_spawn_count=game.get_spawn_count,
            apply_action=lambda owner_id, action: apply_owner_action(game, owner_id, action),
            now=now,
            enabled=not game.game_over and not game.paused,
        )
        spawn_counts["player1"] = sync_owner_spawn_state(game, input_feel, "player1", spawn_counts["player1"])
        spawn_counts["player2"] = sync_owner_spawn_state(game, input_feel, "player2", spawn_counts["player2"])

        while gravity_accumulator >= gravity_interval and not game.game_over and not game.paused:
            game.advance_gravity()
            gravity_accumulator -= gravity_interval
            spawn_counts["player1"] = sync_owner_spawn_state(game, input_feel, "player1", spawn_counts["player1"])
            spawn_counts["player2"] = sync_owner_spawn_state(game, input_feel, "player2", spawn_counts["player2"])

        game.update_frame(elapsed)
        spawn_counts["player1"] = sync_owner_spawn_state(game, input_feel, "player1", spawn_counts["player1"])
        spawn_counts["player2"] = sync_owner_spawn_state(game, input_feel, "player2", spawn_counts["player2"])
        game.set_animation_progress(1.0)
        game.draw()
        pygame.display.flip()

        frame_elapsed = time.perf_counter() - frame_start
        if frame_elapsed < frame_interval:
            time.sleep(frame_interval - frame_elapsed)


if __name__ == "__main__":
    main()
