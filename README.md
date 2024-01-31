# Forest Cover Type Prediction

## Problem Overview

The problem associated with this dataset is to predict the forest cover type based on the given features. It is a multi-class classification problem where the goal is to classify each instance into one of the seven forest cover types.

![Problem Overview](images/Picture1.png)


## Dataset Overview

The dataset used for the Forest Cover Type Prediction study is sourced from the Roosevelt National Forest in northern Colorado. The data encapsulates features from the wilderness areas within this forest, with each observation representing a 30m x 30m patch of land. The primary goal is to predict the forest cover type, which is an integer classification representing one of seven possible forest cover types.

Target Variable: Forest Cover Types:
```
1 - Spruce/Fir
2 - Lodgepole Pine
3 - Ponderosa Pine
4 - Cottonwood/Willow
5 - Aspen
6 - Douglas-fir
7 – Krummholz
```
Features:
```
• Elevation : Elevation in meters
• Aspect : Aspect in degrees azimuth
• Slope : Slope in degrees
• Horizontal_Distance_To_Hydrology : Horz Dist to nearest surface water features
• Vertical_Distance_To_Hydrology : Vert Dist to nearest surface water features
• Horizontal_Distance_To_Roadways : Horz Dist to nearest roadway
• Hillshade_9am (0 to 255 index) : Hillshade index at 9am, summer solstice
• Hillshade_Noon (0 to 255 index) : Hillshade index at noon, summer solstice
• Hillshade_3pm (0 to 255 index) : Hillshade index at 3pm, summer solstice
• Horizontal_Distance_To_Fire_Points : Horz Dist to nearest wildfire ignition points
• Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) : Wilderness area designation
• Soil_Type (40 binary columns, 0 = absence or 1 = presence) : Soil Type designation
• Cover_Type (7 types, integers 1 to 7) : Forest Cover Type designation
```

## Project Flowchart

The methodology of the Forest Cover Type Prediction project encapsulates a series of steps executed using machine learning techniques. Initially, the project establishes baseline performance by exploring diverse ML models. It then seeks to enhance accuracy through meticulous feature selection and dimensionality reduction techniques. Progressing further, the project incorporates advanced ensemble strategies and integrates PKI to refine the forest cover type predictions, leading to a robust set of models. Finally, the models are evaluated and the best-performing one is selected based on its accuracy and predictive capabilities.

![Project Flowchart](images/Picture2.png)

# Exploratory Data Analysis (EDA)

The EDA section visually explores the dataset to understand the distribution, correlations, and characteristics of the features. Below are visualizations that depict these aspects:

### Distribution and Correlation
<table>
<tr>
  <td><img src="images/Picture3.png" alt="Distribution of Cover Type" width="600"/></td>
  <td><img src="images/Picture4.png" alt="Correlation Heatmap" width="600"/></td>
</tr>
<tr>
  <td align="center">Distribution of Cover Type</td>
  <td align="center">Correlation Heatmap</td>
</tr>
</table>

### Categorical and Numerical Feature Analysis
<table>
<tr>
  <td><img src="images/Picture5.png" alt="Categorical Feature Analysis" width="600"/></td>
  <td><img src="images/Picture6.png" alt="Numerical Feature Analysis" width="600"/></td>
</tr>
<tr>
  <td align="center">Categorical Feature Analysis</td>
  <td align="center">Numerical Feature Analysis</td>
</tr>
</table>

### Skewness Analysis
<table>
<tr>
  <td><img src="images/Picture7.png" alt="Skewness Analysis" width="600"/></td>
  <td><img src="images/Picture8.png" alt="Histograms for Skewness Analysis" width="600"/></td>
</tr>
<tr>
  <td align="center">Skewness of Numerical Features</td>
  <td align="center">Histograms for Skewness Analysis</td>
</tr>
</table>





