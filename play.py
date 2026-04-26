import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Game2048Env:
    def __init__(self):
        self.board = np.zeros((4,4), dtype=np.int32)
        self.score = 0; self.done = False

    def reset(self):
        self.board = np.zeros((4,4), dtype=np.int32)
        self.score = 0; self.done = False
        self._spawn(); self._spawn()
        return self.board.copy()

    def get_valid_actions(self):
        valid = []
        for a in range(4):
            b, s = self.board.copy(), self.score
            if self._try_move(a): valid.append(a)
            self.board, self.score = b, s
        return valid

    def _try_move(self, a):
        orig = self.board.copy(); self._do_move(a)
        return not np.array_equal(orig, self.board)

    def step(self, action):
        if self.done: return self.board.copy(), 0.0, True, self._info()
        prev_score, prev_board = self.score, self.board.copy()
        if not self._try_move(action):
            return self.board.copy(), 0.0, self.done, self._info()
        self._spawn()
        reward = self._shaped_reward(prev_board, self.score - prev_score)
        if not self._can_move(): self.done = True
        return self.board.copy(), reward, self.done, self._info()

    def _shaped_reward(self, prev, delta):
        b = self.board.astype(np.float64)
        empty = float(np.sum(b == 0))
        mono = self._mono(b); smooth = self._smooth(b)
        corner = 3.0 if b[0,0] == b.max() else 0.0
        merge = self._merge_pot(b)
        r = delta + 0.2*empty + 1.5*mono + 1.0*smooth + corner + 0.5*merge
        return float(np.tanh(r / 1000.0))

    @staticmethod
    def _mono(b):
        s = 0.0
        for r in range(4):
            for c in range(3):
                if b[r,c] >= b[r,c+1]: s += 1
        for r in range(3):
            for c in range(4):
                if b[r,c] >= b[r+1,c]: s += 1
        return s

    @staticmethod
    def _smooth(b):
        s = 0.0; lb = np.zeros_like(b); m = b > 0; lb[m] = np.log2(b[m])
        for r in range(4):
            for c in range(3):
                if b[r,c]>0 and b[r,c+1]>0: s -= abs(lb[r,c]-lb[r,c+1])
        for r in range(3):
            for c in range(4):
                if b[r,c]>0 and b[r+1,c]>0: s -= abs(lb[r,c]-lb[r+1,c])
        return s

    @staticmethod
    def _merge_pot(b):
        c = 0
        for r in range(4):
            for cc in range(3):
                if b[r,cc]>0 and b[r,cc]==b[r,cc+1]: c += 1
        for r in range(3):
            for cc in range(4):
                if b[r,cc]>0 and b[r,cc]==b[r+1,cc]: c += 1
        return float(c)

    def _info(self): return {"score": self.score, "max_tile": int(self.board.max())}

    def _spawn(self):
        e = list(zip(*np.where(self.board == 0)))
        if e:
            r, c = random.choice(e)
            self.board[r,c] = 4 if random.random() < 0.1 else 2

    @staticmethod
    def _merge(line):
        nz = line[line != 0]; merged = []; s = 0; skip = False
        for i in range(len(nz)):
            if skip: skip = False; continue
            if i+1 < len(nz) and nz[i]==nz[i+1]:
                v = nz[i]*2; merged.append(v); s += v; skip = True
            else: merged.append(nz[i])
        merged += [0]*(4-len(merged))
        return np.array(merged, dtype=np.int32), s

    def _do_move(self, a):
        ts = 0
        if a == 0:
            for c in range(4): col,s = self._merge(self.board[:,c]); self.board[:,c]=col; ts+=s
        elif a == 1:
            for c in range(4): col,s = self._merge(self.board[::-1,c]); self.board[:,c]=col[::-1]; ts+=s
        elif a == 2:
            for r in range(4): row,s = self._merge(self.board[r,:]); self.board[r,:]=row; ts+=s
        elif a == 3:
            for r in range(4): row,s = self._merge(self.board[r,::-1]); self.board[r,:]=row[::-1]; ts+=s
        self.score += ts

    def _can_move(self):
        if 0 in self.board: return True
        for r in range(4):
            for c in range(3):
                if self.board[r,c]==self.board[r,c+1]: return True
        for r in range(3):
            for c in range(4):
                if self.board[r,c]==self.board[r+1,c]: return True
        return False


def encode_state(board):
    b = board.astype(np.float32); ch = []
    lb = np.zeros_like(b); m = b > 0; lb[m] = np.log2(b[m])
    ch.append(lb / 17.0)
    ch.append((b == 0).astype(np.float32))
    mx = b.max()
    ch.append((b == mx).astype(np.float32) if mx > 0 else np.zeros((4,4),dtype=np.float32))
    ch.append(b / max(mx, 1.0))
    mono_r = np.zeros((4,4),dtype=np.float32)
    for r in range(4):
        for c in range(3): mono_r[r,c] = 1.0 if b[r,c]>=b[r,c+1] else -1.0
    ch.append(mono_r)
    mono_c = np.zeros((4,4),dtype=np.float32)
    for r in range(3):
        for c in range(4): mono_c[r,c] = 1.0 if b[r,c]>=b[r+1,c] else -1.0
    ch.append(mono_c)
    merge = np.zeros((4,4),dtype=np.float32)
    for r in range(4):
        for c in range(3):
            if b[r,c]>0 and b[r,c]==b[r,c+1]: merge[r,c]=1; merge[r,c+1]=1
    for r in range(3):
        for c in range(4):
            if b[r,c]>0 and b[r,c]==b[r+1,c]: merge[r,c]=1; merge[r+1,c]=1
    ch.append(merge)
    dist = np.zeros((4,4),dtype=np.float32)
    for r in range(4):
        for c in range(4): dist[r,c] = 1.0 - (r+c)/6.0
    ch.append(dist)
    return np.stack(ch, axis=0)


class NoisyLinear(nn.Module):
    def __init__(self, inf, outf, sigma0=0.5):
        super().__init__()
        self.inf, self.outf = inf, outf
        self.weight_mu = nn.Parameter(torch.empty(outf, inf))
        self.weight_sigma = nn.Parameter(torch.empty(outf, inf))
        self.register_buffer("weight_eps", torch.empty(outf, inf))
        self.bias_mu = nn.Parameter(torch.empty(outf))
        self.bias_sigma = nn.Parameter(torch.empty(outf))
        self.register_buffer("bias_eps", torch.empty(outf))
        self.sigma0 = sigma0; self._reset_params(); self.reset_noise()

    def _reset_params(self):
        b = 1/math.sqrt(self.inf)
        self.weight_mu.data.uniform_(-b, b); self.bias_mu.data.uniform_(-b, b)
        self.weight_sigma.data.fill_(self.sigma0/math.sqrt(self.inf))
        self.bias_sigma.data.fill_(self.sigma0/math.sqrt(self.inf))

    @staticmethod
    def _scale(sz):
        x = torch.randn(sz); return x.sign()*x.abs().sqrt()

    def reset_noise(self):
        ei, eo = self._scale(self.inf), self._scale(self.outf)
        self.weight_eps.copy_(eo.outer(ei)); self.bias_eps.copy_(eo)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma*self.weight_eps,
                           self.bias_mu + self.bias_sigma*self.bias_eps)
        return F.linear(x, self.weight_mu, self.bias_mu)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch,ch,3,padding=1); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch,ch,3,padding=1); self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + x)


class RainbowDQN(nn.Module):
    def __init__(self, in_ch=8, n_act=4, n_atoms=51, v_min=-200., v_max=5000.):
        super().__init__()
        self.n_act, self.n_atoms = n_act, n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 128, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 2), nn.ReLU(),
            nn.Conv2d(256, 256, 1), nn.ReLU(),
        )
        self.res1 = ResBlock(256); self.res2 = ResBlock(256)
        flat = 256*2*2
        self.v_h = NoisyLinear(flat, 256); self.v_o = NoisyLinear(256, n_atoms)
        self.a_h = NoisyLinear(flat, 256); self.a_o = NoisyLinear(256, n_act*n_atoms)

    def forward(self, x):
        B = x.size(0)
        f = self.res2(self.res1(self.backbone(x))).view(B, -1)
        v = self.v_o(F.relu(self.v_h(f))).view(B, 1, self.n_atoms)
        a = self.a_o(F.relu(self.a_h(f))).view(B, self.n_act, self.n_atoms)
        return F.log_softmax(v + a - a.mean(1, keepdim=True), dim=2)

    def q_values(self, x):
        return (self.forward(x).exp() * self.support).sum(2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

    def act(self, state, valid_actions=None):
        with torch.no_grad():
            q = self.q_values(state.unsqueeze(0)).squeeze(0)
            if valid_actions:
                mask = torch.full_like(q, -1e9)
                for a in valid_actions: mask[a] = 0.
                q = q + mask
            return q.argmax().item()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_board(board, score, moves, action_name="Start"):
    clear_screen()
    print("=" * 33)
    print("       Rainbow DQN 2048")
    print("=" * 33)
    print(f" Score: {score:>7} | Moves: {moves}")
    print(f" Max:   {board.max():>7} | Last:  {action_name}")
    print("-" * 33)
    for row in board:
        row_str = " | ".join(f"{v:>4}" if v > 0 else "   ." for v in row)
        print("    " + row_str)
    print("-" * 33)
    print(" Press Ctrl+C to stop playing.")


def main():
    parser = argparse.ArgumentParser(description="Terminal 2048 AI Player")
    parser.add_argument("--model", default="rainbow_best.pth", help="Path to model weights")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between moves (sec)")
    parser.add_argument("--manual", action="store_true", help="Enter starting board manually")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found!")
        sys.exit(1)

    print(f"Loading Rainbow DQN from {args.model}...")
    model = RainbowDQN(in_ch=8).to("cpu")
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    env = Game2048Env()
    env.reset()

    if args.manual:
        print("\n" + "=" * 40)
        print("    MANUAL BOARD ENTRY")
        print(" Enter 4 numbers separated by space (0 = empty)")
        print(" Example: 0 2 0 4")
        print("=" * 40)
        custom_board = []
        for i in range(4):
            while True:
                try:
                    row_str = input(f" Row {i+1}: ").strip()
                    row = [int(x) for x in row_str.split()]
                    if len(row) != 4:
                        print(" Error: exactly 4 numbers required. Try again.")
                        continue
                    custom_board.append(row)
                    break
                except ValueError:
                    print(" Error: numbers only.")
        env.board = np.array(custom_board, dtype=np.int32)
        env.score = 0
        print("Board loaded! Starting...\n")
        time.sleep(1)

    moves = 0
    actions_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    last_action = "Start"

    try:
        while not env.done:
            print_board(env.board, env.score, moves, last_action)
            if args.delay > 0:
                time.sleep(args.delay)
            s = torch.tensor(encode_state(env.board), dtype=torch.float32)
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = model.act(s.to("cpu"), valid_actions=valid_actions)
            last_action = actions_map[action]
            _, _, done, _ = env.step(action)
            moves += 1
        print_board(env.board, env.score, moves, "GAME OVER")
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\n" + "=" * 33)
    print("             GAME OVER")
    print(f" Final Score: {env.score}")
    print(f" Max Tile:    {env.board.max()}")
    print(f" Total Moves: {moves}")
    print("=" * 33)


if __name__ == "__main__":
    main()
