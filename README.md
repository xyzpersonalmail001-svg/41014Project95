# Project95 Project overview

UTS 41014 Assessment 4, Track A. Hanlin Wang — Project95.

This project uses a MuJoCo TurtleBot3 Waffle Pi to build a LiDAR map, estimate pose with an encoder/vision EKF, and navigate and dock using the estimated state. The docking target is 0.55 m in front of the station.

Technical explanations, experimental discussion, the AI declaration and references are in `Project95_Integrated_Project.pdf`.

Demonstration video: [Hanlin Wang Project95](https://www.youtube.com/watch?v=2gXo7DRXXtw)
https://youtu.be/2gXo7DRXXtw
## File guide

| File or directory | Purpose |
|---|---|
| `Project95_Integrated_Project.pdf` | Technical report |
| `run_demo.py` | Complete-task entry point |
| `configs/` | Task and module configurations |
| `evaluation/` | Code, data and figures organized by Parts A–E |
| `analysis/analyze_results.py` | Recalculate metrics and generate plots from saved data |
| `third_party/README.md` | Third-party resources and licence information |

Within each part of `evaluation/`, `code/` contains programs, `data/` contains numerical records and `figures/` contains plots. Shared code is in `evaluation/common/`.

## Installation

The project requires Windows x64, Python 3.12 x64, Git and working OpenGL. Installation needs internet access to download dependencies and the robot model.

Open PowerShell in the `Project95` folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
```

The script creates a `.venv` virtual environment, installs the specified dependency versions and downloads the ROBOTIS model. Dependency and resource versions are recorded in `requirements.txt`, `resources.lock.json` and `robot_assets.lock.json`. After downloading the model, it checks the hashes of 13 asset files. If existing resources have a different version or local modifications, the script reports an error and retains those resources.

If Python or Git cannot be found, specify their installation locations. Replace the example paths below with the actual paths on your computer:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1' -PythonPath 'C:\path\to\Python312\python.exe' -GitPath 'C:\path\to\git.exe'
```

## Run the complete task

Run the following commands from the `Project95` folder.

Navigation and docking in the normal room:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_demo.py' --case normal
```

Introduce a board at simulation time 2 s to examine task performance after an obstacle appears:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_demo.py' --case obstacle
```

Occlude the camera during 0.5–1.5 s after docking begins to examine the task during a brief loss of visual measurements:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_demo.py' --case blackout
```

The program does not open an interactive viewer, but it renders and saves camera RGB images, so OpenGL is still required.

Use `--case all` to run the three cases sequentially, `--seed 1234` to specify another non-negative seed, and `--help` to view all options. The default demonstration seed is `4101901`. Repeated runs are kept separate from the report's final 24-trial evaluation.

## View the results

Results for the three cases are saved under `results/demo/<UTC-id>/` in the `Project95` folder. The terminal prints the specific output directory after `DEMO_OUTPUT`. It may temporarily show no new output while simulation and image rendering continue.

After a run finishes, open `summary.json` to check whether each case passed, then `task_paths.png` to view the paths. `DEMO_PASSED` with exit code `0` indicates a pass; `DEMO_FAILED` or `DEMO_ERROR` returns exit code `1`.

The project checks final position error ≤25 mm, absolute yaw error ≤3°, and true translational and yaw stop speeds ≤0.005 m/s and ≤0.02 rad/s, respectively. The task must also finish within 180 s, have zero obstacle contacts, and pass clearance and actuator-limit checks. Individual check results are in `summary.json`.

For a closer look at a run, use `run.json`, the effective configuration and the source snapshot to identify its conditions. Each case directory also contains camera images and sensor, estimator, control, map and truth-evaluation logs. Failed runs retain their records; program exceptions are written to `error.txt`.

## Analyze the saved data

Recalculate the supplied metrics and generate plots:

```powershell
& '.\.venv\Scripts\python.exe' '.\analysis\analyze_results.py'
```

Results are saved under `results/analysis/<UTC-id>/`, including the metric summary `evaluation_summary.json` and seven PNGs grouped by Parts B–E. `ANALYSIS_PASSED` means the recalculated metrics agree with the saved values.

Six plots correspond to report Figures 2–6 and 8; the other is a count matrix for the 24 runs. Report Figures 7 and 9 and the Part A mathematical results retain their original files.

The analysis program only needs NumPy and Matplotlib. Once dependencies are installed, it can run offline without MuJoCo, robot assets or OpenGL. It reads the supplied CSVs, `evaluation/experiments.json` and existing evaluation summaries, using all time samples without smoothing, interpolation or downsampling.

To change the input or output locations, use:

- `--output 'C:\path\to\new-folder'`: choose a new output directory. An existing directory is rejected to avoid overwriting results.
- `--data`: select another complete `evaluation/` data directory.
- `--trials`: specify a replacement trial CSV.

Replay plots use saved replay summaries, and batch plots use the original run outcomes and contact labels. This program only analyzes existing data. It does not rerun the EKF or physical simulation, or recheck raw RGB images and contact records.

## Third-party sources

Original third-party licence files are retained in `third_party/`.

Robot XML and STL assets come from ROBOTIS TurtleBot3 MuJoCo at the version specified in `resources.lock.json`. The project derives scenes from these assets in memory at runtime. The original Apache-2.0 licence is retained in `third_party/robotis/LICENSE`.

ArUco detection, corner refinement, PnP and reprojection use the OpenCV distribution. Its original notices are retained in `third_party/opencv/LICENSE.txt` and `third_party/opencv/LICENSE-3RD-PARTY.txt`. NumPy, MuJoCo, Matplotlib, Pillow and other dependencies are installed at specified versions; the installed environment and dependency binaries are not bundled.

