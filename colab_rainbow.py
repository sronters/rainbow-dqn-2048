import copy, math, time, random
import numpy as np
from collections import deque
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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

    def render(self):
        for row in self.board: print("|".join(f"{v:>5}" if v else "    ." for v in row))
        print(f"Score: {self.score}  Max: {self.board.max()}\n")


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


class VectorEnv:
    def __init__(self, n):
        self.envs = [Game2048Env() for _ in range(n)]; self.n = n

    def reset_all(self): return [e.reset() for e in self.envs]

    def step(self, actions):
        res = [e.step(a) for e,a in zip(self.envs, actions)]
        return [list(x) for x in zip(*res)]

    def get_valid_actions(self): return [e.get_valid_actions() for e in self.envs]

    def reset_done(self):
        return [e.reset() if e.done else e.board.copy() for e in self.envs]


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


class SumTree:
    def __init__(self, cap):
        self.cap = cap; self.tree = np.zeros(2*cap-1)
        self.data = [None]*cap; self.write = 0; self.size = 0

    def _prop(self, i, ch):
        p = (i-1)//2; self.tree[p] += ch
        if p: self._prop(p, ch)

    def _get(self, i, s):
        l = 2*i+1
        if l >= len(self.tree): return i
        return self._get(l, s) if s <= self.tree[l] else self._get(l+1, s-self.tree[l])

    @property
    def total(self): return self.tree[0]

    def add(self, p, d):
        i = self.write + self.cap - 1
        self.data[self.write] = d; self.update(i, p)
        self.write = (self.write+1) % self.cap
        self.size = min(self.size+1, self.cap)

    def update(self, i, p):
        ch = p - self.tree[i]; self.tree[i] = p; self._prop(i, ch)

    def get(self, s):
        i = self._get(0, s); return i, self.tree[i], self.data[i - self.cap + 1]


class PER:
    def __init__(self, cap=500000, alpha=0.6, beta0=0.4, beta_frames=300000, n_step=5, gamma=0.99):
        self.tree = SumTree(cap); self.alpha = alpha
        self.beta0 = beta0; self.beta_frames = beta_frames; self.frame = 0
        self.max_p = 1.0; self.n_step = n_step; self.gamma = gamma
        self.nbuf = deque(maxlen=n_step)

    @property
    def beta(self): return min(1.0, self.beta0 + self.frame*(1-self.beta0)/self.beta_frames)

    def _nstep(self):
        R = sum(self.gamma**i * t[2] for i,t in enumerate(self.nbuf))
        s0,a0 = self.nbuf[0][0], self.nbuf[0][1]
        return s0, a0, R, self.nbuf[-1][3], self.nbuf[-1][4]

    def push(self, s, a, r, ns, d):
        self.nbuf.append((s, a, r, ns, d))
        if len(self.nbuf) < self.n_step and not d: return
        self.tree.add(self.max_p**self.alpha, self._nstep())
        if d:
            while len(self.nbuf) > 1:
                self.nbuf.popleft()
                if self.nbuf: self.tree.add(self.max_p**self.alpha, self._nstep())
            self.nbuf.clear()

    def sample(self, bs, dev):
        self.frame += 1; beta = self.beta
        idx, prios, batch = [], [], []
        seg = self.tree.total / bs
        for i in range(bs):
            s = np.random.uniform(seg*i, seg*(i+1))
            j, p, d = self.tree.get(s)
            if d is None:
                s = np.random.uniform(0, self.tree.total)
                j, p, d = self.tree.get(s)
            idx.append(j); prios.append(p); batch.append(d)
        pr = np.array(prios)/(self.tree.total+1e-8)
        w = (self.tree.size * pr + 1e-8)**(-beta); w /= w.max()
        s,a,r,ns,d = zip(*batch)
        return (torch.stack(s).to(dev), torch.tensor(a,dtype=torch.long,device=dev),
                torch.tensor(r,dtype=torch.float32,device=dev),
                torch.stack(ns).to(dev), torch.tensor(d,dtype=torch.float32,device=dev),
                torch.tensor(w,dtype=torch.float32,device=dev), idx)

    def update_prios(self, idx, errs):
        for i, e in zip(idx, errs):
            p = (abs(e)+1e-6)**self.alpha; self.max_p = max(self.max_p, p)
            self.tree.update(i, p)

    def __len__(self): return self.tree.size


def project_dist(next_lp, rew, done, support, gn, vmin, vmax, na, dz):
    B = rew.size(0); np_ = next_lp.exp()
    Tz = (rew.unsqueeze(1) + (1-done.unsqueeze(1)) * gn * support.unsqueeze(0)).clamp(vmin, vmax)
    b = (Tz - vmin)/dz; l = b.floor().long().clamp(0,na-1); u = b.ceil().long().clamp(0,na-1)
    proj = torch.zeros(B, na, device=rew.device)
    off = torch.arange(B, device=rew.device).unsqueeze(1)*na
    proj.view(-1).index_add_(0, (l+off).view(-1), (np_*(u.float()-b)).view(-1))
    proj.view(-1).index_add_(0, (u+off).view(-1), (np_*(b-l.float())).view(-1))
    return proj


def train(total_steps=500_000, num_envs=64, batch_size=512, lr=1e-4,
          gamma=0.99, n_step=5, target_upd=2000, warmup=20000,
          train_freq=4, log_every=100, save_every=50000):
    gamma_n = gamma ** n_step
    n_atoms, vmin, vmax = 51, -200., 5000.

    online = RainbowDQN(n_atoms=n_atoms, v_min=vmin, v_max=vmax).to(device)
    target = copy.deepcopy(online); target.eval()
    for p in target.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(online.parameters(), lr=lr, eps=1.5e-4)
    replay = PER(n_step=n_step, gamma=gamma)

    venv = VectorEnv(num_envs)
    obs = venv.reset_all()
    states = [torch.tensor(encode_state(o), dtype=torch.float32) for o in obs]

    scores, tiles, losses = [], [], []
    steps = 0; eps_done = 0; best_avg = 0; t0 = time.time()
    print(f"Rainbow DQN | {total_steps} steps | {num_envs} envs | device={device}")
    print("="*70)

    while steps < total_steps:
        va_list = venv.get_valid_actions()
        online.eval()
        actions = []
        for i in range(num_envs):
            va = va_list[i]
            actions.append(online.act(states[i].to(device), va) if va else 0)
        online.train()

        obs, rews, dones, infos = venv.step(actions)
        for i in range(num_envs):
            ns = torch.tensor(encode_state(obs[i]), dtype=torch.float32)
            replay.push(states[i], actions[i], rews[i], ns, dones[i])
            if dones[i]:
                scores.append(infos[i]["score"]); tiles.append(infos[i]["max_tile"])
                eps_done += 1
            states[i] = ns; steps += 1

        new_obs = venv.reset_done()
        for i in range(num_envs):
            if venv.envs[i].score == 0 and not venv.envs[i].done:
                states[i] = torch.tensor(encode_state(new_obs[i]), dtype=torch.float32)

        if steps >= warmup and steps % train_freq == 0 and len(replay) >= batch_size:
            online.reset_noise(); target.reset_noise()
            s,a,r,ns,d,w,idx = replay.sample(batch_size, device)
            lp = online(s); lp_a = lp[range(batch_size), a]
            with torch.no_grad():
                nq = online.q_values(ns); na_ = nq.argmax(1)
                nlp = target(ns); nlp_a = nlp[range(batch_size), na_]
                proj = project_dist(nlp_a, r, d, online.support, gamma_n, vmin, vmax, n_atoms, online.delta_z)
            loss_per = -(proj * lp_a).sum(1)
            loss = (w * loss_per).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0); opt.step()
            replay.update_prios(idx, loss_per.detach().cpu().numpy())
            losses.append(loss.item())

        if steps % target_upd == 0: target.load_state_dict(online.state_dict())

        if eps_done > 0 and eps_done % log_every == 0 and len(scores) >= log_every:
            rec = scores[-log_every:]; rt = tiles[-log_every:]
            avg = np.mean(rec)
            td = {}
            for t in rt: td[t] = td.get(t,0)+1
            r1k = sum(1 for t in rt if t >= 1024)/len(rt)*100
            r2k = sum(1 for t in rt if t >= 2048)/len(rt)*100
            el = time.time()-t0; al = np.mean(losses[-200:]) if losses else 0
            print(f"Ep {eps_done:>6} | Steps {steps:>8} | Avg {avg:>8.0f} | "
                  f"Loss {al:.4f} | 1024:{r1k:>3.0f}% 2048:{r2k:>3.0f}% | "
                  f"{dict(sorted(td.items()))} | {el:.0f}s")
            if avg > best_avg:
                best_avg = avg
                torch.save(online.state_dict(), "rainbow_best.pth")
                print(f"  -> Best! avg={best_avg:.0f}")

        if steps > 0 and steps % save_every == 0:
            torch.save(online.state_dict(), f"rainbow_{steps}.pth")

    torch.save(online.state_dict(), "rainbow_final.pth")
    print(f"\nDone! Eps={eps_done} Best={best_avg:.0f}")
    return online, scores, tiles, losses


def plot(scores, tiles, losses):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3,figsize=(18,5)); w=200
    ax[0].plot(scores,alpha=0.2); ax[0].set_title("Score")
    if len(scores)>=w:
        a=np.convolve(scores,np.ones(w)/w,"valid"); ax[0].plot(range(w-1,len(scores)),a,"r")
    ax[1].plot(tiles,alpha=0.2,color="coral"); ax[1].set_title("Max Tile")
    if len(tiles)>=w:
        a=np.convolve(tiles,np.ones(w)/w,"valid"); ax[1].plot(range(w-1,len(tiles)),a,"darkred")
    if losses:
        ax[2].plot(losses,alpha=0.2,color="green"); ax[2].set_title("Loss")
        if len(losses)>=100:
            a=np.convolve(losses,np.ones(100)/100,"valid"); ax[2].plot(range(99,len(losses)),a,"darkgreen")
    plt.tight_layout(); plt.savefig("rainbow_curves.png",dpi=150); plt.show()


def demo(path="rainbow_best.pth"):
    m = RainbowDQN().to("cpu")
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    env = Game2048Env(); obs = env.reset(); moves = 0
    while True:
        s = torch.tensor(encode_state(obs), dtype=torch.float32)
        va = env.get_valid_actions()
        if not va: break
        a = m.act(s, va); obs,_,done,info = env.step(a); moves += 1
        if done: break
    env.render()
    print(f"Score: {info['score']} | Max: {info['max_tile']} | Moves: {moves}")


if __name__ == "__main__":
    model, sc, tl, ls = train()
    plot(sc, tl, ls)
    print("\n=== Demo ===")
    demo()
