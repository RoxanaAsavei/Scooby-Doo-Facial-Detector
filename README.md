## Config

This project has been compiled using a virtual environment based on **Python 3.13.2**.

To compile this project, please make sure you've installed the following **libraries**:
```
numpy>=1.24.1
scipy>=1.10.0
scikit-learn>=1.6.0
matplotlib>=3.6.1
opencv-python>=4.5.0.0

tensorflow>=2.15.0
ultralytics>=8.0.0
torch>=1.8.0
torchvision>=0.9.0

```
## Recommended project structure when running the code
The project uses relative paths, so please use the follow project structure when running the project so no change is required for the paths.

```
351_Asavei_Roxana
|_ antrenare
|_ cod
|_ evaluare
|_ examples
    |_ hardNegatives
    |_ negativeExamples
    |_ positiveExamples
|_ runs
|_ save
|_ task1_yolo
    |_ cod
    |_ dataset
        |_ images
        |_ labels
        build_dataset.py
        dataset.yaml
|_ task2_yolo
    |_ cod
    |_ dataset
        |_ images
        |_ labels
        build_dataset.py
        dataset.yaml 
|_ validare
    |_ validare


```
## Running the source code
To run the code, please compile each .py file from its corresponding directory.

For the main tasks, training each CNN took ~5 hours, only using CPU.
Prediction takes ~8 seconds / image for detection.
For the bonus tasks, training the YOLO model took 6-7 hours.
