#!/usr/bin/env python3
"""
Unitree G1 tosses an apple from one hand to the other (MuJoCo).

Setup
-----
    pip install mujoco numpy imageio imageio-ffmpeg      # imageio only for --video
    git clone https://github.com/google-deepmind/mujoco_menagerie

Run
---
    python g1_apple_toss.py --scene mujoco_menagerie/unitree_g1/scene.xml
    python g1_apple_toss.py --scene ... --video toss.mp4 --seconds 20   # no window

How it works
------------
* The G1 is loaded with MjSpec; the pelvis is fixed in space (no balancing
  problem to solve) and a small "tray" (plate + rim) is bolted to each wrist so
  the apple has a reliable palm to sit on. The tray defines the palm frame, so
  the IK works no matter how the original hand mesh is oriented.
* Each arm is driven by damped-least-squares IK: the palm position follows a
  Cartesian trajectory and the palm normal is kept pointing up. The joint
  solution is sent to the (position) actuators.
* Toss: the palm accelerates along the launch direction, then brakes, so the
  apple leaves the palm with a ballistic velocity chosen to land on the
  other palm after a flight time T.
* At release, the real apple state is read and the landing point is
  re-predicted; the catching palm moves there, then cushions the apple by
  lowering. Afterwards the roles swap and the apple goes back (loops forever).
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

# --------------------------------------------------------------------------- #
# Parameters you will probably want to tune
# --------------------------------------------------------------------------- #
G = 9.81
UP = np.array([0.0, 0.0, 1.0])
APPLE_R = 0.04                              # apple radius [m]
APPLE_MASS = 0.15                           # [kg]

FLIGHT_TIME = 0.40                          # s, hand-to-hand flight time
ACCEL_DIST = 0.16                           # m, distance the palm accelerates over
BRAKE_TIME = 0.15                           # s, palm braking after release
REST_REL = np.array([0.20, 0.0, -0.18])     # palm "rest" position relative to its shoulder (world axes)

PALM_POS = np.array([0.10, 0.0, 0.0])       # palm centre in wrist_yaw_link frame
PALM_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # palm frame (z = palm normal) in wrist_yaw_link frame
TRAY_HALF = 0.05                            # half width of the palm tray

KP, KD = 100.0, 4.0                         # arm actuator gains (boosted from the menagerie defaults)
ROT_W = 0.5                                 # weight of palm-level constraint in IK

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
              "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(scene_xml):
    spec = mujoco.MjSpec.from_file(scene_xml)
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Fix the pelvis in space: drop the free joint (and keyframes that reference it).
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)

    # Stiffer arm actuators so the palms can follow fast trajectories.
    for a in spec.actuators:
        if any(k in a.name for k in ("shoulder", "elbow", "wrist")):
            a.gainprm[0] = KP
            a.biasprm[1] = -KP
            a.biasprm[2] = -KD

    # Palm tray on each wrist.
    for side in ("left", "right"):
        body = spec.body(f"{side}_wrist_yaw_link")

        def to_link(v):
            out = np.zeros(3)
            mujoco.mju_rotVecQuat(out, np.asarray(v, float), PALM_QUAT)
            return out + PALM_POS

        body.add_site(name=f"{side}_palm", pos=PALM_POS, quat=PALM_QUAT,
                      size=[0.01, 0, 0], rgba=[0, 1, 0, 1])
        common = dict(type=mujoco.mjtGeom.mjGEOM_BOX, quat=PALM_QUAT,
                      rgba=[0.75, 0.75, 0.8, 1], friction=[1.2, 0.01, 0.001])
        h = TRAY_HALF
        body.add_geom(name=f"{side}_tray", size=[h, h, 0.005],
                      pos=to_link([0, 0, -0.005]), **common)
        rims = [([h, 0, 0.008], [0.004, h, 0.008]), ([-h, 0, 0.008], [0.004, h, 0.008]),
                ([0, h, 0.008], [h, 0.004, 0.008]), ([0, -h, 0.008], [h, 0.004, 0.008])]
        for i, (p, s) in enumerate(rims):
            body.add_geom(name=f"{side}_rim{i}", size=s, pos=to_link(p), **common)

    # The apple.
    apple = spec.worldbody.add_body(name="apple", pos=[0.3, 0, 1.2])
    apple.add_joint(name="apple_free", type=mujoco.mjtJoint.mjJNT_FREE)
    apple.add_geom(name="apple_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[APPLE_R, 0, 0], mass=APPLE_MASS, rgba=[0.85, 0.1, 0.1, 1],
                   friction=[1.2, 0.01, 0.001])
    return spec.compile()


# --------------------------------------------------------------------------- #
# Arm IK
# --------------------------------------------------------------------------- #
class Arm:
    def __init__(self, model, side):
        self.side = side
        sgn = 1.0 if side == "left" else -1.0
        self.jid = [model.joint(f"{side}_{j}_joint").id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jid])
        self.lo, self.hi = model.jnt_range[self.jid].T
        self.site = model.site(f"{side}_palm").id
        act_of_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)
                        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.act = np.array([act_of_joint[j] for j in self.jid])
        self.q_seed = np.array([0.0, 0.15 * sgn, 0.0, 1.2, 0.0, 0.0, 0.0])
        self.q = self.q_seed.copy()

    def solve(self, model, scratch, data, target, iters=2):
        """A few damped-LS IK iterations (palm position + palm normal up)."""
        scratch.qpos[:] = data.qpos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(iters):
            scratch.qpos[self.qadr] = self.q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            p = scratch.site_xpos[self.site]
            R = scratch.site_xmat[self.site].reshape(3, 3)
            e = np.concatenate([target - p, ROT_W * np.cross(R[:, 2], UP)])
            mujoco.mj_jacSite(model, scratch, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], ROT_W * jacr[:, self.dadr]])
            A = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, e)
            N = np.eye(7) - J.T @ np.linalg.solve(A, J)          # null space: stay near seed posture
            dq += N @ (0.1 * (self.q_seed - self.q))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            self.q = np.clip(self.q + dq, self.lo, self.hi)


def minjerk(p0, p1, t0, dur):
    p0, p1 = np.array(p0, float), np.array(p1, float)

    def f(t):
        s = np.clip((t - t0) / dur, 0.0, 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        return p0 + (p1 - p0) * s
    return f


def const(p):
    p = np.array(p, float)
    return lambda t: p


# --------------------------------------------------------------------------- #
# Toss state machine
# --------------------------------------------------------------------------- #
class Juggler:
    def __init__(self, model, data, flight_time=FLIGHT_TIME):
        self.m, self.d = model, data
        self.T = flight_time
        self.scratch = mujoco.MjData(model)
        self.arms = {s: Arm(model, s) for s in ("left", "right")}
        self.apple_gid = model.geom("apple_geom").id
        self.apple_q = model.joint("apple_free").qposadr[0]
        self.apple_v = model.joint("apple_free").dofadr[0]
        self.tray_geoms = {
            s: {model.geom(n).id for n in [f"{s}_tray"] + [f"{s}_rim{i}" for i in range(4)]}
            for s in self.arms}
        self.reset()

    # -- helpers ----------------------------------------------------------- #
    def apple_pos(self):
        return self.d.qpos[self.apple_q:self.apple_q + 3].copy()

    def apple_vel(self):
        return self.d.qvel[self.apple_v:self.apple_v + 3].copy()

    def palm_pos(self, side):
        return self.d.site_xpos[self.arms[side].site].copy()

    def touching(self, side):
        geoms = self.tray_geoms[side]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 == self.apple_gid and c.geom2 in geoms) or \
               (c.geom2 == self.apple_gid and c.geom1 in geoms):
                return True
        return False

    # -- planning ---------------------------------------------------------- #
    def plan_toss(self):
        th, ca = self.thrower, self.catcher
        p0, tgt, T = self.rest[th], self.rest[ca], self.T
        v = (tgt - p0 + np.array([0, 0, 0.5 * G * T * T])) / T      # ballistic launch velocity
        speed = np.linalg.norm(v)
        u = v / speed
        self.toss = dict(p0=p0, v=v, u=u, a=speed**2 / (2 * ACCEL_DIST),
                         ta=2 * ACCEL_DIST / speed, p_start=p0 - u * ACCEL_DIST)

    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self.thrower, self.catcher = "left", "right"
        self.rest = {s: d.xpos[m.body(f"{s}_shoulder_pitch_link").id] + REST_REL for s in self.arms}
        self.plan_toss()
        targets = {self.thrower: self.toss["p_start"], self.catcher: self.rest[self.catcher]}

        # Hold every non-arm actuator at its starting angle (legs, waist, ...).
        for a in range(m.nu):
            d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]

        for s, arm in self.arms.items():
            arm.q = arm.q_seed.copy()
            for _ in range(400):
                arm.solve(m, self.scratch, d, targets[s], iters=1)
            d.qpos[arm.qadr] = arm.q
            d.ctrl[arm.act] = arm.q
            mujoco.mj_kinematics(m, self.scratch)
            self.scratch.qpos[:] = d.qpos
            mujoco.mj_kinematics(m, self.scratch)
            err = np.linalg.norm(self.scratch.site_xpos[arm.site] - targets[s])
            print(f"[init] {s} palm IK residual: {err * 1000:.1f} mm"
                  + ("   <-- target probably out of reach, tune REST_REL" if err > 0.01 else ""))

        mujoco.mj_kinematics(m, d)
        pt = self.toss["p_start"]
        d.qpos[self.apple_q:self.apple_q + 3] = pt + UP * (APPLE_R + 0.002)
        d.qpos[self.apple_q + 3:self.apple_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

        self.traj = {self.thrower: const(pt), self.catcher: const(self.rest[self.catcher])}
        self.state, self.t_state = "approach", d.time
        self.n_catches = 0

    # -- state transitions --------------------------------------------------- #
    def start_toss(self, t):
        s = self.toss
        t0 = t

        def f(tt):
            x = tt - t0
            if x <= s["ta"]:
                return s["p_start"] + s["u"] * 0.5 * s["a"] * x * x
            return s["p0"] + s["v"] * (x - s["ta"])          # keep speed until release is detected
        self.traj[self.thrower] = f
        self.state, self.t_state = "accel", t

    def release(self, t):
        th, ca = self.thrower, self.catcher
        p, v = self.apple_pos(), self.apple_vel()
        z_catch = self.rest[ca][2] + APPLE_R
        tl = (v[2] + np.sqrt(max(v[2]**2 + 2 * G * (p[2] - z_catch), 0.0))) / G
        land_xy = p[:2] + v[:2] * tl
        target = np.array([land_xy[0], land_xy[1], self.rest[ca][2]])
        self.traj[ca] = minjerk(self.palm_pos(ca), target, t, max(0.7 * tl, 0.1))

        pr, vr, tb, rest = self.palm_pos(th), self.toss["v"], BRAKE_TIME, self.rest[th]
        pe = pr + 0.5 * vr * tb
        back = minjerk(pe, rest, t + tb, 0.8)

        def brake(tt):
            x = tt - t
            if x < tb:
                return pr + vr * x - 0.5 * (vr / tb) * x * x
            return back(tt)
        self.traj[th] = brake
        self.state, self.t_state = "flight", t
        print(f"[t={t:5.2f}] release {th}->{ca}: v={np.round(v, 2)}, "
              f"predicted landing {np.round(target, 3)} in {tl:.2f}s")

    def update(self):
        t, d = self.d.time, self.d
        th, ca = self.thrower, self.catcher

        if self.apple_pos()[2] < 0.3:                       # dropped -> restart
            print(f"[t={t:5.2f}] apple dropped, resetting")
            self.reset()
            return

        if self.state == "approach":
            if t - self.t_state > 1.2:
                self.start_toss(t)

        elif self.state == "accel":
            x = t - self.t_state
            if x >= self.toss["ta"] and (not self.touching(th) or x > self.toss["ta"] + 0.25):
                self.release(t)

        elif self.state == "flight":
            pa, va, pp = self.apple_pos(), self.apple_vel(), self.palm_pos(ca)
            if (va[2] < 0 and pa[2] - APPLE_R - pp[2] < 0.06) or self.touching(ca):
                self.traj[ca] = minjerk(pp, pp + np.array([0, 0, -0.08]), t, 0.22)   # cushion
                self.state, self.t_state = "cushion", t

        elif self.state == "cushion":
            if t - self.t_state > 0.6:
                self.traj[ca] = minjerk(self.palm_pos(ca), self.rest[ca], t, 1.0)
                self.state, self.t_state = "carry", t
                self.n_catches += 1
                print(f"[t={t:5.2f}] caught by {ca} (#{self.n_catches})")

        elif self.state == "carry":
            if t - self.t_state > 1.3:
                self.thrower, self.catcher = ca, th         # swap roles
                self.plan_toss()
                self.traj[self.thrower] = minjerk(self.palm_pos(self.thrower),
                                                  self.toss["p_start"], t, 0.8)
                self.state, self.t_state = "approach", t

    # -- main step --------------------------------------------------------- #
    def step(self):
        self.update()
        t = self.d.time
        for s, arm in self.arms.items():
            arm.solve(self.m, self.scratch, self.d, self.traj[s](t), iters=2)
            self.d.ctrl[arm.act] = arm.q
        mujoco.mj_step(self.m, self.d)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def make_camera():
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 140, -10, 1.7
    cam.lookat[:] = [0.25, 0.0, 0.9]
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="mujoco_menagerie/unitree_g1/scene.xml")
    ap.add_argument("--flight-time", type=float, default=FLIGHT_TIME)
    ap.add_argument("--video", help="render to this mp4 instead of opening a window")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    model = build_model(args.scene)
    data = mujoco.MjData(model)
    sim = Juggler(model, data, args.flight_time)

    if args.video:
        import imageio
        renderer = mujoco.Renderer(model, 480, 640)
        cam, frames, fps = make_camera(), [], 30
        while data.time < args.seconds:
            sim.step()
            if len(frames) < data.time * fps:
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
        imageio.mimsave(args.video, frames, fps=fps)
        print("saved", args.video)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -10, 1.7
        viewer.cam.lookat[:] = [0.25, 0.0, 0.9]
        wall0, prev, n = time.time(), 0.0, 0
        while viewer.is_running():
            sim.step()
            n += 1
            if data.time < prev:                         # sim was reset
                wall0 = time.time() - data.time
            prev = data.time
            if n % 8 == 0:
                viewer.sync()
                ahead = data.time - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(ahead)


if __name__ == "__main__":
    main()