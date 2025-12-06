# HoopVision

**Author:** Rui Wu  
**Course / Project:** CS366 Final Project  
**Date:** 2025-12-05

**Overview**
HoopVision is a computer-vision basketball analytics system that extracts game statistics from raw video. It combines YOLOv8 object detection with classic vision algorithms to detect and track players, the ball, and the hoop, then computes possessions, shot attempts and shooting percentages.

** Directory Structure**
- `main.py`: Main script to run the analysis.
- `train.py`: Script to train YOLOv8 models.
- `config.yaml`: Configuration file for paths, thresholds, and parameters.
- `models/`: Pre-trained YOLOv8 model weights.
- `train/` and `valid/`: Datasets for training custom models.
- `requirements.txt`: Python dependencies.

**Features**
- **Multi-Object Detection:** Uses YOLOv8 models to detect players, referees, the basketball, and the hoop.
- **Team Classification:** Separates players into two teams by K-Means on jersey colors; referees are identified by low color saturation (HSV).
- **Robust Ball Tracking:** Kalman Filter smooths the ball trajectory and predicts its position during occlusions.
- **Possession Tracking:** Assigns ball possession by player proximity with a temporal stability buffer to reduce noise.
- **Shot Detection:** Detects shot attempts and made shots by analyzing ball vertical motion and proximity to the hoop.
- **Noise Filtering:** Geometric filters remove non-player objects (e.g., shot clock, scoreboards).
- **Game Summary:** Produces a post-game report with possession percentages and shooting stats saved to `game_summary.txt`.

**Installation**
1. Clone the repository:

```bash
git clone <repository-url>
cd "HoopVision"
```

2. Set up Python (3.8+ recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Usage**
- To run the analysis on a video, make sure the input path is set in `main.py` or in your config, then run:

```bash
python main.py
```

- Input: a video file (for example `1.mp4`).
- Output: a visualization window with bounding boxes, tracking lines and a HUD, plus a saved summary `game_summary.txt` after processing.

**Configuration**
- Update `config.yaml` or the configuration section in `main.py` to change input paths, model locations, thresholds, or tracking parameters.
- Model weights are stored in the `models/` directory (e.g., `ball_detector_model.pt`, `player_detector.pt`).

**Training Custom Models**
If you need to retrain YOLO models (hoop, ball, etc.):

1. Prepare your dataset in the `train/` and `valid/` folders with `images/` and `labels/`.
2. Update `config.yaml` with dataset paths and training hyperparameters.
3. Run:

```bash
python train.py
```

**Outputs & Artifacts**
- `game_summary.txt`: final game statistics report.
- `runs/`: detection and training outputs and logs.

**References**
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
- Kalman Filter Theory: https://en.wikipedia.org/wiki/Kalman_filter
- K-Means Clustering: https://en.wikipedia.org/wiki/K-means_clustering
- OpenCV Documentation: https://opencv.org/
- Training Datasets: https://universe.roboflow.com/robocon-qchql/hoops-chrie-ivyrc/dataset/3/download
- Training Tutorial: https://youtu.be/WgPbbWmnXJ8
- Inspirations for shooting detection: https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker/blob/master/README-zh.md
- YOLOv8 Training Script: see `train.py` for details.
- Game Analysis Inspiration: https://youtu.be/QqVahw9tBfw?si=YyHy0NOZknEBM32O


