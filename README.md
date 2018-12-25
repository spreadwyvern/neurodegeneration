# Detecting Early-stage Neurodegeneration with 3D Convolutional Neural Network
![sample DTI](/images/neuro_project.png)
## Prediction module
This is the repository of the prediction model used in our project **"Predicting Brain Age in Health People by Brain MRI"**. The related blogpost can be found [here](/spreadwyver.github.io/projects/brain/brain/).
Here we use two measures extracted from subject's diffusion tensor images, FA and MD. The trained models achieved a result with MAE: 4.29 and Pearsons's correlation coefficient: 0.96.
![result](/images/result.png)
### Dependencies
This work uses Python 3.5.2. Before running the code, you have to instll the following.
- torch==0.2.0_3
- torchvision==0.2.1
- numpy==1.15.3

The above dependencies can be instgalled using pip by running:
```
pip install -r requirement.txt
```

### Usage
Execute brain_age.py

Use --help to see usage of ensemble_predict.py:
```
usage: ensemble_predict.py [-h] [-g]

optional arguments:
-h, --help    show help message and exit
-g, --gpu_id  assign GPU ID, default 0
-d, --dir     path of trained weights, default /model_for_prediction/model.pkl
```
