import sys
from argparse import Namespace

import pygame

from aiplay import main as run_aiplay
from config import *
from towplayer import main as run_towplayer


class MenuButton:
    def __init__(self, label, mode_id, rect):
        self.label = label
        self.mode_id = mode_id
        self.rect = rect


def load_background():
    try:
        background = pygame.image.load("background1.png").convert()
        return pygame.transform.scale(background, (SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
    except Exception:
        surface = pygame.Surface((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
        surface.fill((25, 30, 45))
        return surface


def launch_mode(mode_id):
    if mode_id == "aiplay" or mode_id.startswith("aiplay_"):
        difficulty = "normal" if mode_id == "aiplay" else mode_id.removeprefix("aiplay_")
        args = Namespace(
            ai="rl",
            checkpoint="shared_model3_stage2_v1/best.pt",
            difficulty=difficulty,
        )
        return run_aiplay(embedded=True, args=args)
    if mode_id == "towplayer":
        return run_towplayer(embedded=True)
    return "menu"


def build_buttons():
    total_width = SCREEN_WIDTH + SIDEBAR_WIDTH * 2
    button_width = 280
    button_height = 54
    center_x = total_width // 2 - button_width // 2
    start_y = SCREEN_HEIGHT // 2 - 130
    gap = 16
    return [
        MenuButton("AI Easy", "aiplay_easy", pygame.Rect(center_x, start_y, button_width, button_height)),
        MenuButton("AI Normal", "aiplay_normal", pygame.Rect(center_x, start_y + (button_height + gap), button_width, button_height)),
        MenuButton("AI Hard", "aiplay_hard", pygame.Rect(center_x, start_y + 2 * (button_height + gap), button_width, button_height)),
        MenuButton("Two Players", "towplayer", pygame.Rect(center_x, start_y + 3 * (button_height + gap), button_width, button_height)),
    ]


def draw_button(screen, font, button, selected):
    fill = (230, 230, 230) if selected else (35, 35, 45)
    text_color = BLACK if selected else WHITE
    border = (255, 255, 255) if selected else (120, 120, 140)
    pygame.draw.rect(screen, fill, button.rect, border_radius=14)
    pygame.draw.rect(screen, border, button.rect, 3, border_radius=14)
    image = font.render(button.label, True, text_color)
    image_rect = image.get_rect(center=button.rect.center)
    screen.blit(image, image_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Menu")

    title_font = pygame.font.SysFont(None, 64)
    subtitle_font = pygame.font.SysFont(None, 28)
    button_font = pygame.font.SysFont(None, 36)

    background = load_background()
    background_scroll = 0
    buttons = build_buttons()
    selected_index = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected_index = (selected_index - 1) % len(buttons)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected_index = (selected_index + 1) % len(buttons)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    result = launch_mode(buttons[selected_index].mode_id)
                    if result == "quit":
                        pygame.quit()
                        sys.exit()
                    pygame.display.set_caption("Tetris Menu")
                    background = load_background()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.MOUSEMOTION:
                for index, button in enumerate(buttons):
                    if button.rect.collidepoint(event.pos):
                        selected_index = index
                        break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button.rect.collidepoint(event.pos):
                        result = launch_mode(button.mode_id)
                        if result == "quit":
                            pygame.quit()
                            sys.exit()
                        pygame.display.set_caption("Tetris Menu")
                        background = load_background()

        y_offset = background_scroll % SCREEN_HEIGHT
        screen.blit(background, (0, y_offset))
        screen.blit(background, (0, y_offset - SCREEN_HEIGHT))
        background_scroll += 1

        overlay = pygame.Surface((SCREEN_WIDTH + SIDEBAR_WIDTH * 2, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 120))
        screen.blit(overlay, (0, 0))

        title = title_font.render("TETRIS", True, WHITE)
        title_rect = title.get_rect(center=((SCREEN_WIDTH + SIDEBAR_WIDTH * 2) // 2, 150))
        screen.blit(title, title_rect)

        subtitle = subtitle_font.render("Choose a game mode", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=((SCREEN_WIDTH + SIDEBAR_WIDTH * 2) // 2, 205))
        screen.blit(subtitle, subtitle_rect)

        for index, button in enumerate(buttons):
            draw_button(screen, button_font, button, index == selected_index)

        hint = subtitle_font.render("Enter or click to start   Q to quit", True, WHITE)
        hint_rect = hint.get_rect(center=((SCREEN_WIDTH + SIDEBAR_WIDTH * 2) // 2, SCREEN_HEIGHT - 70))
        screen.blit(hint, hint_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
