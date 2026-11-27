# Lite3 Scene Understanding

Camera perception scripts for the Group 8 Lite3 robot dog (Intelligent Robotics). They read the
robot's camera, run local YOLO object detection, decode QR codes, propose yellow warning-sign
candidates, and can optionally ask Claude to describe a scene or identify a sign.

**These scripts only observe.** None of them sends motion commands, switches robot modes or
starts navigation. Every result is an unverified observation saved to disk (or published on a
ROS topic, for `scene_node.py`).

The later mission pipeline in [Lite3-Live-Mission](https://github.com/JaimieSE/Lite3-Live-Mission)
reuses this folder's Python environment, model and camera helpers.

## Scripts

| Launcher | Script | What it does | Output |
|---|---|---|---|
| `run_direct_camera.sh` | `direct_camera.py` | Read-only RTSP viewer with live QR decoding (`--decoder opencv\|zxing\|both`) | `direct_captures/` |
| `run_yolo_camera.sh` | `direct_yolo.py` | RTSP + local YOLO detection, QR decoding and yellow-triangle proposals | `yolo_captures/` |
| `run_perception.sh` | `combined_perception.py` | YOLO + QR + sign tracking in one loop; with `--vlm`, sends new sign candidates to Claude | `perception_runs/<UTC time>/` |
| `run_vlm.sh` | `periodic_vlm.py` | Sends a camera frame to Claude every `--interval` seconds for a structured scene description (scene type, hazard signs, people, stairs, curtains) | `vlm_runs/<UTC time>/` |
| `run_signs.sh` | `sign_vlm.py` | Yellow-triangle proposals; with `--vlm`, Claude identifies the symbol on each crop | `sign_runs/<UTC time>/` |
| `run.sh` | `scene_node.py` | ROS 2 Humble node (domain 8): YOLO on `/builtin_camera/image_raw` | `/scene/annotated_image`, `/scene/detections` (JSON) |
| – | `decode_png.py` | Decode QR codes in a saved image or capture folder | stdout |
| – | `yellow_signs.py` | Shared colour/shape detector and sign tracker (proposals, not confirmed hazards) | – |

`mission_prompt.txt` holds the scene-description prompt; `sign_reference.jpg` is a reference sign image.

## Setup

The launchers expect the folder at `~/scene_understanding` with a virtual environment in `.venv/`:

```bash
cd ~/scene_understanding
python3 -m venv .venv
.venv/bin/pip install -r requirements-installed.txt anthropic zxing-cpp
```

`requirements-installed.txt` lists the CPU-only PyTorch/Ultralytics stack. `anthropic` and
`zxing-cpp` are needed as well. Put the YOLO weights at `yolo26n.pt` in this folder, or pass
`--model PATH`. The weights, the `.venv` and all capture folders are kept out of git.

`scene_node.py` also needs ROS 2 Humble with `rclpy` and `cv_bridge` (`run.sh` sources
`/opt/ros/humble/setup.bash`).

The direct-camera scripts read `rtsp://192.168.1.120:8554/test` over TCP. Change the URL in
`direct_camera.py` / `periodic_vlm.py` if your stream is elsewhere.

## Usage

```bash
./run_direct_camera.sh                 # live viewer, Q to quit
./run_yolo_camera.sh --headless --seconds 30
./run_perception.sh                    # local only, no API calls
./run_perception.sh --vlm --dry-run    # save sign triggers, no API calls
./run_perception.sh --vlm              # paid Claude sign identification
./run_vlm.sh --dry-run                 # prepare frames without calling Claude
./run_vlm.sh --interval 5 --max-requests 3
./run_signs.sh --vlm --max-requests 3 --cooldown 30
./run.sh                               # ROS 2 node
```

Common options: `--headless` (no preview window), `--seconds N` (stop after N seconds; 0 = run
until Q / Ctrl+C), `--confidence`, `--imgsz`, `--threads` for YOLO.

### Claude API

`--vlm` modes and `run_vlm.sh` call the Claude Messages API (paid). The model defaults to
`claude-opus-5-5`; override with `--model` / `--vlm-model` or the `CLAUDE_MODEL` environment
variable. Requests are capped by default (`--max-requests 3`, with a cooldown between sign
requests) so a first test can't run up cost; `--max-requests 0` removes the cap in `run_vlm.sh`.

The API key is read from `ANTHROPIC_API_KEY`. If it isn't set, the launcher asks for it at a
hidden prompt and keeps it only for that process. Never commit a key or paste it into chat.

The prompts tell Claude to report visible evidence only: hazard symbols are not treated as
proof of a hazard, posters and screens are kept apart from real people, text in the image is
data rather than instructions, and the reply never contains motion commands.
