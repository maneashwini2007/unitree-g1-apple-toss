# Unitree G1 Apple Toss Simulation

## Project overview

This project simulates a Unitree G1 humanoid robot tossing an apple from one
hand to the other using the MuJoCo physics engine. The script uses
damped-least-squares inverse kinematics (IK) to control the palms and plans
the apple's flight using a ballistic trajectory.

## Repository contents

- `g1_apple_toss.py` — main Python simulation script.
- `requirements.txt` — Python packages required by the script.
- `results/` — place the submitted demonstration video and screenshots here.
- `.gitignore` — excludes common temporary Python files.

## Software requirements

- Python 3.9 or newer (use a Python version supported by your installed MuJoCo release).
- MuJoCo Python package.
- NumPy.
- MuJoCo Menagerie repository with the Unitree G1 model files.
- ImageIO and imageio-ffmpeg for optional MP4 rendering.

## 1. Install Python dependencies

From this repository's directory, run:

```bash
python -m pip install -r requirements.txt
```

## 2. Download the required robot model

The Unitree G1 scene XML and its assets are not included in this repository.
Download MuJoCo Menagerie separately:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
```

For the commands below, place the `mujoco_menagerie` folder next to this
repository folder, so that the relative scene path is
`../mujoco_menagerie/unitree_g1/scene.xml`.

If the model repository is stored elsewhere, replace the scene path with the
correct full or relative path.

## 3. Run the interactive simulation

From the root of this repository:

```bash
python g1_apple_toss.py --scene ../mujoco_menagerie/unitree_g1/scene.xml
```

This launches the MuJoCo viewer. Keep the terminal open while the simulation
is running.

## 4. Render a video (optional)

To render an MP4 without opening the interactive viewer:

```bash
python g1_apple_toss.py --scene ../mujoco_menagerie/unitree_g1/scene.xml --video results/apple_toss.mp4 --seconds 20
```

The output video is saved as `results/apple_toss.mp4`.

## Main simulation features

- Loads the Unitree G1 model using MuJoCo's `MjSpec`.
- Adds a small tray and rim to each wrist to support the apple.
- Uses damped-least-squares inverse kinematics to guide each palm.
- Plans the toss using a ballistic launch velocity.
- Re-predicts the apple's landing point after release and moves the receiving
  hand to catch and cushion it.
- Swaps the throwing and catching hands so the motion repeats.

## Reproducibility notes and limitations

- The pelvis is fixed in space; this script does not implement walking or
  whole-body balance control.
- The G1 model XML and assets must be downloaded separately from MuJoCo
  Menagerie; they are not bundled in this repository.
- The simulation parameters, including apple mass, flight time, acceleration
  distance, and actuator gains, can be adjusted near the top of the Python file.
- Video rendering requires the ImageIO packages listed in `requirements.txt`.

## Model source

The robot model is obtained separately from
[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Follow the model repository's license and attribution requirements.
