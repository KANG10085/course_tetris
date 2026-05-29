# ai.py

from config import *
from shapes import *
import random

class TetrisAI:
    def __init__(self, game):
        self.game = game
        # 每个子动作（旋转/水平移动/硬降）之间的间隔（秒）。调大变慢，调小变快
        self.move_delay = 0.3
        self.timer = 0
        # 状态机状态
        self.current_move = None
        self.rotation_done = False
        # 模拟人类失误的概率：落位时有一定概率把目标横坐标随机偏移（5%）
        self.landing_error_prob = 0.03
        # 统计 AI 落位错误次数
        self.error_count = 0
        self.plan_board_signature = None

    def reset_plan(self):
        self.current_move = None
        self.rotation_done = False
        self.timer = 0.0
        self.plan_board_signature = None

    def _board_signature(self):
        return tuple(tuple(int(cell) for cell in row) for row in self.game.grid)

    def _refresh_plan_if_board_changed(self):
        if self.current_move is None or self.plan_board_signature is None:
            return
        if self._board_signature() != self.plan_board_signature:
            self.reset_plan()

    def evaluate(self, grid):
        heights=[0]*GRID_WIDTH
        holes=0
        for x in range(GRID_WIDTH):
            block=False
            for y in range(GRID_HEIGHT):
                if grid[y][x]:
                    if not block:
                        heights[x]=GRID_HEIGHT-y
                        block=True
                elif block:
                    holes+=1
        aggregate=sum(heights)
        bump=0
        for i in range(GRID_WIDTH-1):
            bump+=abs(heights[i]-heights[i+1])
        return -0.5*aggregate -0.3*holes -0.2*bump

    def simulate(self, shape, x):
        y=0
        while not self.game.check_collision(shape, x, y+1):
            y+=1
        return y

    def best_move(self):
        best=None
        best_score=-999999
        shape=self.game.get_current_shape()
        for r in range(4):
            test=shape
            for _ in range(r):
                test=[list(row) for row in zip(*test[::-1])]
            for x in range(-2, GRID_WIDTH):
                if self.game.check_collision(test, x, 0):
                    continue
                y=self.simulate(test,x)
                grid=[row[:] for row in self.game.grid]
                for rr in range(len(test)):
                    for cc in range(len(test[rr])):
                        if test[rr][cc] and 0<=y+rr<GRID_HEIGHT and 0<=x+cc<GRID_WIDTH:
                            grid[y+rr][x+cc]=1
                score=self.evaluate(grid)
                if score>best_score:
                    best_score=score
                    best=(x,r)
        return best

    def play(self, dt):
        # 不在游戏中或被暂停时不执行任何 AI 动作
        if self.game.game_over or getattr(self.game, 'paused', False):
            return
        self._refresh_plan_if_board_changed()
        # 累积时间，只有当到达 move_delay 时才执行下一项子动作
        self.timer += dt
        if self.timer < self.move_delay:
            return

        action_performed = False

        if self.current_move is None:
            self.current_move = self.best_move()
            self.rotation_done = False

            if self.current_move is None:
                self.timer = 0.0
                return
            self.plan_board_signature = self._board_signature()
            # 落位错误：以一定概率偏移目标横坐标，模拟人类落位失误
            if random.random() < self.landing_error_prob:
                orig_x, orig_r = self.current_move
                # 随机偏移 -2,-1,1,2，但只接受那些在顶部不立即碰撞的位置
                for _ in range(8):
                    offset = random.choice([-2, -1, 1, 2])
                    candidate_x = max(0, min(GRID_WIDTH-1, orig_x + offset))
                    # 计算旋转后形状用于碰撞检测
                    test_shape = self.game.get_current_shape()
                    for _r in range(orig_r):
                        test_shape = [list(row) for row in zip(*test_shape[::-1])]
                    # 如果放在顶行 candidate_x 不会立刻碰撞，则接受该偏移
                    if not self.game.check_collision(test_shape, candidate_x, 0):
                        self.current_move = (candidate_x, orig_r)
                        # 记录一次落位错误
                        try:
                            self.error_count += 1
                        except Exception:
                            pass
                        break

        x, r = self.current_move

        # 1) 旋转（每次一格旋转）
        if not self.rotation_done:
            if r > 0:
                self.game.rotate()
                r -= 1
                self.current_move = (x, r)
                action_performed = True
            else:
                self.rotation_done = True
                action_performed = True

        # 2) 水平移动（若旋转已完成）
        elif self.game.current_x < x:
            moved = self.game.move(1, 0)
            # 如果移动被阻塞（返回 False），放弃当前目标并在下次重算
            if not moved:
                self.current_move = None
            action_performed = True
        elif self.game.current_x > x:
            moved = self.game.move(-1, 0)
            if not moved:
                self.current_move = None
            action_performed = True
        else:
            # 到达目标位置 -> 硬降
            self.game.hard_drop()
            self.current_move = None
            action_performed = True

        if action_performed:
            # 执行子动作后重置计时器
            self.timer = 0.0

    def next_action(self):
        if self.game.game_over or getattr(self.game, 'paused', False):
            return None
        self._refresh_plan_if_board_changed()

        if self.current_move is None:
            self.current_move = self.best_move()
            self.rotation_done = False
            if self.current_move is None:
                return None
            self.plan_board_signature = self._board_signature()

            if random.random() < self.landing_error_prob:
                orig_x, orig_r = self.current_move
                for _ in range(8):
                    offset = random.choice([-2, -1, 1, 2])
                    candidate_x = max(0, min(GRID_WIDTH - 1, orig_x + offset))
                    test_shape = self.game.get_current_shape()
                    for _r in range(orig_r):
                        test_shape = [list(row) for row in zip(*test_shape[::-1])]
                    if not self.game.check_collision(test_shape, candidate_x, 0):
                        self.current_move = (candidate_x, orig_r)
                        self.error_count += 1
                        break

        x, r = self.current_move

        if not self.rotation_done:
            if r > 0:
                self.current_move = (x, r - 1)
                return "rotate"
            self.rotation_done = True

        if self.game.current_x < x:
            return "right"
        if self.game.current_x > x:
            return "left"

        self.current_move = None
        self.rotation_done = False
        return "hard_drop"
