# Aeroelastic_Analysis_Using_DNNxPOD
## Title
**Development of Surrogate Model Using Deep Learning and Proper Orthogonal Decomposition**
## Problem Statements
- Predict the aerodynamic properties of airfoil NACA 64A010 using **Surrogate Model**
- The surrogate model will utilize **Deep Learning** as a model and **Proper Orthogonal Decomposition** as processing method
- The properties will be predicted are **Lift Coefficient, Drag Coefficient, Plunging Motion, and Pitching Motion**
- The model input should only be the flow characteristic (**mach**) and material characteristic (**flutter speed index**)
- The flight range will be mach 0.6 until mach 0.9
- Flutter speed range is vf 0.4 until vf 2.0

## For the report and presentation can be accessed with this [link](https://drive.google.com/drive/folders/1zJuHXgfFVUvrg6ptpNstRcuq0sfXN0bq?usp=share_link)

## Results
## Prediction Test Compared to Raw Data
Some of the prediction result is shown below. For more result can be seen from report or presentation report

<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/Test1Drag150.png">
</p>
<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/Test1Lift150.png">
</p>

### Flutter Boundary Contour Plot
#### Model with training size 50 Samples
<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/flutter_boundary_50samples.png">
</p>

#### Model with training size 100 Samples
<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/flutter_boundary_50samples.png">
  </p>

#### Model with training size 150 Samples
<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/flutter_boundary_150samples.png">
  </p>
  
### Flutter Boundary Result Compared to Previous Research
<p align="center">
  <img src="https://github.com/KukuhIksanMusyahada/Aeroelastic_Analysis_Using_DNNxPOD/blob/main/flutter_boundary_compared1.png">
</p>

