"""Build the demo MP4s from the recorded Gazebo demo episodes (DECISIONS "Demo material"; demo only).

    python -m trustmap.demo_video                      # every results/demo/<house>_<episode>/
    python -m trustmap.demo_video results/demo/200000_1 --keep-frames

Input per episode folder (written by trustmap.gazebo_demo): frames/<sim time ns>.jpg
from the fixed overhead camera, and episode.json. Output: video.mp4 (1280 x 720,
H.264), the overhead view on the left with room labels, and on the right a panel
built from episode.json and synchronised by simulation time:
  - what the robot is told at the start: target, start room, claimed room and
    displayed confidence c, rho, kappa, each room's habit weight;
  - a box "hidden from the robot": true reliability r and the object's real room;
  - per decision: chosen room, the optimum's choice in the same state, the
    decision's regret in meters, and the outcome (found / not here / failure).
The video plays at SPEED times real (simulation) time. The overhead frames are
deleted afterwards unless --keep-frames is given (they are not committed).
Needs Pillow and the ffmpeg program; runs under the project venv or ROS's Python.
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO / "results" / "demo"
SPEED = 3            # video time = simulation time / SPEED
RATE_HZ = 4          # overhead camera rate (gazebo_export.OVERHEAD_RATE_HZ)
VIEW = 720
PANEL_W = 560
HOLD_START_S = 2.0   # extra video seconds on the first and last frame, to read the panel
HOLD_END_S = 4.0
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

INK = (30, 30, 30)
MUTED = (110, 110, 110)
ACCENT = (25, 90, 170)
HIDDEN_BG = (255, 243, 224)
HIDDEN_EDGE = (200, 120, 20)
GOOD = (20, 130, 60)
BAD = (190, 40, 40)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_HEAD, F_TEXT, F_SMALL = font(19, True), font(16, True), font(15), font(13)


def to_pixel(info, x, y, z=0.0):
    """World point -> overhead image pixel (camera straight down, image up = +y, right = +x)."""
    (cx, cy, cz) = info["overhead_camera"]["xyz_m"]
    f = (VIEW / 2) / math.tan(info["overhead_camera"]["horizontal_fov"] / 2)
    return VIEW / 2 + f * (x - cx) / (cz - z), VIEW / 2 - f * (y - cy) / (cz - z)


def name(s):
    return s.replace("_", " ")


def draw_view(img, info, t):
    """Overhead frame with room labels, the map's claim, and a ring around the hidden object."""
    d = ImageDraw.Draw(img)
    labels = info["room_labels"]
    claimed = info["shown_to_robot"]["claimed_room"]
    for k, (x0, y0, x1, y1) in enumerate(info["rooms_m"]):
        # Label at the top edge of the room, so it never covers the robot at the viewpoint (room centre).
        u, v = to_pixel(info, (x0 + x1) / 2, max(y0, y1) - 0.2)
        text = labels[k] + ("\nmap: here, c = %.2f" % info["shown_to_robot"]["c"] if k == claimed else "")
        box = d.multiline_textbbox((u, v), text, font=F_SMALL, anchor="ma", align="center")
        d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(255, 255, 255))
        d.multiline_text((u, v), text, font=F_SMALL, fill=ACCENT if k == claimed else INK, anchor="ma",
                         align="center")
    ox, oy = info["object_xy_m"]
    u, v = to_pixel(info, ox, oy, 0.3)
    d.ellipse([u - 11, v - 11, u + 11, v + 11], outline=(220, 20, 20), width=3)
    d.text((u + 13, v - 8), "target", font=F_SMALL, fill=(220, 20, 20))
    return img


def panel(info, t):
    """The side panel at simulation time t."""
    img = Image.new("RGB", (PANEL_W, VIEW), (250, 250, 248))
    d = ImageDraw.Draw(img)
    labels = info["room_labels"]
    shown, hidden = info["shown_to_robot"], info["hidden_from_robot"]
    x, y = 18, 14
    d.text((x, y), f"PPO-Conf in Gazebo: house {info['house']}, episode {info['episode']}", font=F_TITLE, fill=INK)
    y += 26
    d.text((x, y), f"hand-picked example (not a reported result); {SPEED}x speed", font=F_SMALL, fill=MUTED)
    y += 28

    d.text((x, y), "What the robot is told", font=F_HEAD, fill=ACCENT)
    y += 24
    lines = [f"target: {name(shown['target'])}",
             f"start room: {labels[shown['start_room']]} (already searched)",
             f"map claims: {labels[shown['claimed_room']]}, confidence c = {shown['c']:.2f}",
             f"corruption rho = {info['rho']}, calibration kappa = {info['kappa']:g}",
             "habit weight of each room for this target:"]
    for line in lines:
        d.text((x, y), line, font=F_TEXT, fill=INK)
        y += 21
    for k, h in enumerate(shown["habit"]):
        d.text((x + 14, y), labels[k], font=F_SMALL, fill=INK)
        d.rectangle([x + 140, y + 3, x + 140 + int(200 * h), y + 13], fill=(150, 175, 210))
        d.text((x + 148 + int(200 * h), y), f"{h:.2f}", font=F_SMALL, fill=MUTED)
        y += 18
    y += 8

    d.rectangle([x - 6, y, PANEL_W - 14, y + 62], fill=HIDDEN_BG, outline=HIDDEN_EDGE, width=2)
    d.text((x + 2, y + 5), "Hidden from the robot", font=F_HEAD, fill=HIDDEN_EDGE)
    d.text((x + 2, y + 25), f"true reliability r = {hidden['r']:.2f};  object is in: "
           f"{labels[hidden['object_room']]}", font=F_TEXT, fill=INK)
    d.text((x + 2, y + 43), "map claim is " + ("correct" if hidden["claim_correct"] else "wrong"),
           font=F_SMALL, fill=MUTED)
    y += 76

    d.text((x, y), "Decisions (policy vs. optimum)", font=F_HEAD, fill=ACCENT)
    y += 24
    status = "deciding ..."
    for dec in info["decisions"]:
        if t < dec["t_decide"]:
            break
        line = f"{dec['k']}. go to {labels[dec['room']]}   (optimum: {labels[dec['optimum_room']]})"
        d.text((x, y), line, font=F_TEXT, fill=INK)
        y += 20
        if t < dec["t_arrive"]:
            status = f"driving to {labels[dec['room']]} ({dec['planned_m']:.1f} m planned)"
            d.text((x + 18, y), f"regret of this choice: {dec['regret_m']:.2f} m", font=F_SMALL, fill=MUTED)
            y += 24
            continue
        if t < dec["t_outcome"]:
            status = f"looking around in {labels[dec['room']]} (360-degree look)"
            d.text((x + 18, y), f"regret of this choice: {dec['regret_m']:.2f} m", font=F_SMALL, fill=MUTED)
            y += 24
            continue
        out = dec["outcome"]
        colour = GOOD if out == "found" else BAD if out == "failure" else MUTED
        out_text = {"found": "FOUND", "not here": "not here", "failure": "not here"}[out]
        d.text((x + 18, y), f"regret of this choice: {dec['regret_m']:.2f} m  ->  ", font=F_SMALL, fill=MUTED)
        w = d.textlength(f"regret of this choice: {dec['regret_m']:.2f} m  ->  ", font=F_SMALL)
        d.text((x + 18 + w, y - 1), out_text, font=F_HEAD if out == "found" else F_SMALL, fill=colour)
        y += 26
        if out == "failure":
            d.text((x + 18, y - 4), "FAILURE after the third search", font=F_HEAD, fill=BAD)
            y += 24
        status = "deciding ..." if out == "not here" else ""

    end = info["events"][-1]["t"]
    last = info["decisions"][-1]
    if t >= last["t_outcome"]:
        if info["success"]:
            status = (f"found in {len(info['decisions'])} decision(s): {info['planned_m']:.1f} m planned, "
                      f"{info['driven_m']:.1f} m driven")
        else:
            status = (f"not found within 3 searches: {info['planned_m']:.1f} m planned, "
                      f"{info['driven_m']:.1f} m driven")
        status2 = f"episode regret {info['regret_m']:.2f} m (0 = what the optimum would expect to do)"
    else:
        status2 = ""
    y = VIEW - 66
    d.line([x, y - 8, PANEL_W - 14, y - 8], fill=(210, 210, 210), width=1)
    d.text((x, y), status, font=F_TEXT, fill=INK)
    d.text((x, y + 22), status2, font=F_SMALL, fill=MUTED)
    d.text((PANEL_W - 14, VIEW - 18), f"sim time {max(0.0, min(t, end) - info['events'][0]['t']):5.1f} s",
           font=F_SMALL, fill=MUTED, anchor="rm")
    return img


def build(folder, keep_frames=False):
    folder = Path(folder)
    info = json.loads((folder / "episode.json").read_text())
    frames = sorted((folder / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"{folder}: no frames")
    fps = RATE_HZ * SPEED
    out = folder / "video.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VIEW + PANEL_W}x{VIEW}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for j, path in enumerate(frames):
        t = int(path.stem) / 1e9
        view = Image.open(path).convert("RGB").resize((VIEW, VIEW))
        frame = Image.new("RGB", (VIEW + PANEL_W, VIEW))
        frame.paste(draw_view(view, info, t), (0, 0))
        frame.paste(panel(info, t), (VIEW, 0))
        repeat = 1
        if j == 0:
            repeat += int(HOLD_START_S * fps)
        if j == len(frames) - 1:
            frame = Image.new("RGB", (VIEW + PANEL_W, VIEW))
            frame.paste(draw_view(view, info, t), (0, 0))
            frame.paste(panel(info, info["events"][-1]["t"] + 1), (VIEW, 0))
            repeat += int(HOLD_END_S * fps)
        data = frame.tobytes()
        for _ in range(repeat):
            proc.stdin.write(data)
            n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {folder}")
    if not keep_frames:
        shutil.rmtree(folder / "frames")
    print(f"wrote {out} ({n} frames, {n / fps:.0f} s, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*")
    p.add_argument("--keep-frames", action="store_true")
    a = p.parse_args()
    folders = a.folders or sorted(str(f.parent) for f in DEMO_DIR.glob("*_*/episode.json")
                                  if (f.parent / "frames").exists())
    for f in folders:
        build(f, a.keep_frames)


if __name__ == "__main__":
    main()
