"""Record figures/workbench.gif: headless-Chrome screenshots of the workbench at successive URL states.

Start the server first (.venv/bin/python server.py --port 8010), then

    .venv/bin/python tools/record_workbench_gif.py --base http://127.0.0.1:8010 --scan     # pick example outages
    .venv/bin/python tools/record_workbench_gif.py --base http://127.0.0.1:8010 --secure 12 --overload 34 [--newton 56]

--scan solves every N-1 state at the given load through /api/solve and prints one outage per staged outcome
(line numbers as displayed, i.e. branch id + 1). Needs Google Chrome and Pillow.
"""
import argparse
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path(__file__).resolve().parents[1] / "figures" / "workbench.gif"


def api(base, path, body=None):
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def scan(base, load, alpha):
    grid = api(base, "/api/grid")
    found = {}
    for branch in grid["branches"]:
        if branch["bridge"] or branch["rate_a"] is None:
            continue
        result = api(base, "/api/solve", {"load_scale": load, "outages": [branch["id"]], "alpha": alpha})
        if not result.get("ok") or not result.get("staged"):
            continue
        staged = result["staged"]
        key = (staged["stage"], staged["reason"], staged["verdict"], staged["check"])
        worst = max((b["loading_ac"] or 0) for b in result["branches"])
        found.setdefault(key, []).append((branch["id"] + 1, round(worst, 3), len(staged["flagged"])))
    for key, items in sorted(found.items()):
        print(key, len(items), "states; line (as displayed), worst AC loading, lines in band:", items[:12])


def shoot(chrome, url, path, size, wait_ms):
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={size[0]},{size[1]}", f"--virtual-time-budget={wait_ms}", f"--screenshot={path}", url],
                   check=True, capture_output=True, timeout=180)
    return Image.open(path).convert("RGB")


def caption(image, text, width):
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    bar = 34
    frame = Image.new("RGB", (image.width, image.height + bar), (24, 28, 36))
    frame.paste(image, (0, bar))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(frame).text((12, 7), text, fill=(240, 240, 240), font=font)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8010")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--load", type=float, default=1.1)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--secure", type=int, help="line whose outage is accepted at stage 1")
    p.add_argument("--overload", type=int, help="line whose outage causes a real overload (stage 3)")
    p.add_argument("--newton", type=int, help="line whose outage is accepted after the Newton step (stage 2), if any")
    p.add_argument("--chrome", default=CHROME)
    p.add_argument("--size", default="1600x1000")
    p.add_argument("--width", type=int, default=1100, help="width of the GIF in pixels")
    p.add_argument("--seconds", type=float, default=2.0, help="per frame")
    p.add_argument("--wait", type=int, default=15000, help="virtual time budget per screenshot, ms")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    if a.scan:
        return scan(a.base, a.load, a.alpha)
    load, mid = f"{a.load:g}", f"{(1 + a.load) / 2:g}"
    steps = [("", "Base case (N-0): solved by full AC, the anchor of the band"),
             (f"load={mid}", f"Raise the load to x{mid}"),
             (f"load={load}", f"Load x{load}: new dispatch, still no line out")]
    if a.secure:
        steps.append((f"load={load}&out={a.secure}", f"Trip line {a.secure}: no line inside the band, accepted on GENCO alone (stage 1)"))
    if a.newton:
        steps.append((f"load={load}&out={a.newton}", f"Trip line {a.newton}: a line is inside the band, one Newton step clears it (stage 2)"))
        steps.append((f"load={load}&out={a.newton}&view=newton", f"Line {a.newton} out: loadings after one Newton step"))
    if a.overload:
        steps.append((f"load={load}&out={a.overload}", f"Trip line {a.overload}: overload, the state goes to full AC (stage 3)"))
        steps.append((f"load={load}&out={a.overload}&view=newton", f"Line {a.overload} out: the view after one Newton step"))
        steps.append((f"load={load}&out={a.overload}&view=ac", f"Line {a.overload} out: the AC solution confirms the verdict"))
    size = tuple(int(v) for v in a.size.split("x"))
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (query, text) in enumerate(steps):
            sep = "&" if query else ""
            url = f"{a.base}/?{query}{sep}alpha={a.alpha:g}&lang=en"
            frames.append(caption(shoot(a.chrome, url, f"{tmp}/{i}.png", size, a.wait), f"{i + 1}/{len(steps)}  {text}", a.width))
            frames[-1].save(a.out.with_name(f".workbench_frame{i}.png"))  # to look at; delete afterwards
            print(url)
    # one shared palette keeps the file small and the colours steady between frames
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for i, frame in enumerate(frames):
        strip.paste(frame, (0, i * frame.height))
    palette = strip.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE)
    frames = [frame.quantize(palette=palette, dither=Image.NONE) for frame in frames]
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(a.seconds * 1000), loop=0, optimize=False)
    print(f"{a.out}: {len(frames)} frames, {a.seconds * len(frames):.0f} s, {a.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
