from dataclasses import dataclass


HORIZONTAL_DAS = 0.14
HORIZONTAL_ARR = 0.045
SOFT_DROP_REPEAT = 0.05
INPUT_BUFFER_WINDOW = 0.12


def should_buffer_action(action):
    return action in ("left", "right", "rotate")


def repeat_profile(action):
    if action in ("left", "right"):
        return HORIZONTAL_DAS, HORIZONTAL_ARR
    if action == "soft_drop":
        return 0.0, SOFT_DROP_REPEAT
    return None


@dataclass
class HeldAction:
    owner_id: str
    action: str
    next_time: float
    repeat_interval: float


class InputFeelController:
    def __init__(self, buffer_window=INPUT_BUFFER_WINDOW):
        self.buffer_window = buffer_window
        self.held_actions: dict[int, HeldAction] = {}
        self.buffered_actions: dict[str, dict] = {}

    def clear(self):
        self.held_actions.clear()
        self.buffered_actions.clear()

    def release_key(self, key):
        self.held_actions.pop(key, None)

    def note_keydown(self, key, owner_id, action, spawn_count, now):
        if should_buffer_action(action):
            self.buffered_actions[owner_id] = {
                "action": action,
                "spawn_count": spawn_count,
                "expires_at": now + self.buffer_window,
            }
        profile = repeat_profile(action)
        if profile is None:
            return
        initial_delay, repeat_interval = profile
        self.held_actions[key] = HeldAction(
            owner_id=owner_id,
            action=action,
            next_time=now + initial_delay,
            repeat_interval=repeat_interval,
        )

    def process(self, get_spawn_count, apply_action, now, enabled=True):
        if not enabled:
            return

        for owner_id, buffered in list(self.buffered_actions.items()):
            if now > buffered["expires_at"]:
                self.buffered_actions.pop(owner_id, None)
                continue
            if get_spawn_count(owner_id) == buffered["spawn_count"]:
                continue
            apply_action(owner_id, buffered["action"])
            self.buffered_actions.pop(owner_id, None)

        for key, held in list(self.held_actions.items()):
            if now < held.next_time:
                continue
            apply_action(held.owner_id, held.action)
            held.next_time = now + held.repeat_interval

    def clear_owner_actions(self, owner_id, actions=None):
        actions = set(actions) if actions is not None else None

        for key, held in list(self.held_actions.items()):
            if held.owner_id != owner_id:
                continue
            if actions is not None and held.action not in actions:
                continue
            self.held_actions.pop(key, None)

        buffered = self.buffered_actions.get(owner_id)
        if buffered is None:
            return
        if actions is not None and buffered["action"] not in actions:
            return
        self.buffered_actions.pop(owner_id, None)
