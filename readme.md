# Machine Learning Final Project - Random Forest Model
This is the code for the Random Forest model approach to the problem. There are two code locations, where decades.ipynb is a combination of testing and tuning. The final_code.py file is the final submission for the project, I just wanted to include them both to show progress steps. Included is the output of the final run of the code with the ideal parameters found after a long tuning phase. This will be found in 'model_accuracies.txt' for the details on testing and training accuracies as well as a classification report, and 'confusion_matrix.png' where you will find the final confusion matrix.

In decades.ipynb it also shows a comparison phase where I looked at Random Forest Classification vs Random Forest Regression. For Random Forest Regression I worked with the original dataset using the years as provided, and in Random Forest Classification I converted years to decades.

## Installation
To run the code you will need to follow a few simple steps.

# NOTE: This was tested on windows 11.

1. Create the dataset folder:
```bash
mkdir dataset
```

2. Run this command in dataset folder to obtain the data. This file will need to be unzipped:
```bash
wget https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip
```
Unzip with desired software or run this command in the dataset folder:
```bash
tar -xf yearpredictionmsd.zip
```

3. Install the required packages, either in virtual environment or main
```bash
pip install -r requirements.txt
```

4. Run the code:
```bash
python final_code.py
```