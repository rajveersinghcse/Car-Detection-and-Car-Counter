# Car Detection and Car Counter
This project utilizes the YOLOv8l model for car detection and implements a Simple Online and real-time tracking (SORT) algorithm for counting the number of cars in a video stream. The system detects cars within a predefined region of interest and tracks them using unique IDs.

<o><img height="400" width="1000" src="https://github.com/rajveersinghcse/rajveersinghcse/blob/master/img/car_counter.gif" alt="car_gif"></p>

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO
- SORT (Simple Online and Realtime Tracking)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rajveersinghcse/Car-Detection-and-Car-Counter.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have a video file named `cars.mp4` in the project directory.
2. Download the YOLOv8l model weights (`yolov8l.pt`) from the Ultralytics YOLO repository and place them in the project directory.
3. Run the following command to start the car detection and counting process:

```bash
python car_counter.py
```

## Description

- `car_counter.py`: This script performs car detection and counting using YOLOv8l for object detection and SORT for object tracking. It reads frames from the `cars.mp4` video, applies a mask to isolate the region of interest, detects cars within this region, tracks them using SORT, and counts the total number of unique cars.

## Acknowledgments

- YOLOv8l: Ultralytics YOLO - [GitHub Repository](https://github.com/ultralytics/yolov5)
- SORT: Simple Online and Realtime Tracking - [GitHub Repository](https://github.com/abewley/sort)

## License

This project is licensed under the [MIT License](LICENSE).
