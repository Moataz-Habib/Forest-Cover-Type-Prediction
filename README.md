# Forest Cover Type Prediction

## Problem Overview

The problem associated with this dataset is to predict the forest cover type based on the given features. It is a multi-class classification problem where the goal is to classify each instance into one of the seven forest cover types.

<table>
<tr>
<td>
<img src="images/Picture1.png" alt="Problem Overview" width="1500"/>
</td>
</tr>
</table>


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

<table>
<tr>
<td>
<img src="images/Picture2.png" alt="Project Flowchart" width="1500"/>
</td>
</tr>
</table>

# Exploratory Data Analysis (EDA)

The EDA section visually explores the dataset to understand the distribution, correlations, and characteristics of the features. Below are visualizations that depict these aspects:

### Distribution and Correlation

<table>
<tr>
  <th>Distribution of Cover Type</th>
  <th>Correlation Heatmap</th>
</tr>
<tr>
  <td><img src="images/Picture3.png" alt="Distribution of Cover Type" width="600"/></td>
  <td><img src="images/Picture4.png" alt="Correlation Heatmap" width="600"/></td>
</tr>
</table>

### Categorical and Numerical Feature Analysis

<table>
<tr>
  <th>Categorical Feature Analysis</th>
  <th>Numerical Feature Analysis</th>
</tr>
<tr>
  <td><img src="images/Picture5.png" alt="Categorical Feature Analysis" width="600"/></td>
  <td><img src="images/Picture6.png" alt="Numerical Feature Analysis" width="600"/></td>
</tr>
</table>

### Skewness Analysis

<table>
<tr>
  <th>Skewness of Numerical Features</th>
  <th>Histograms for Skewness Analysis</th>
</tr>
<tr>
  <td><img src="images/Picture7.png" alt="Skewness Analysis" width="600"/></td>
  <td><img src="images/Picture8.png" alt="Histograms for Skewness Analysis" width="600"/></td>
</tr>
</table>

# Obtain Baseline Performance

To establish baseline performance, several machine learning methods were applied to the dataset. The accuracy for each model was calculated, and the best two models were identified as Decision Tree and KNN based on their performance.

### Baseline Performance Evaluation

The analysis commenced with the evaluation of multiple machine learning classifiers to determine a baseline performance. The accuracy of each model was measured after applying the following classifiers to the dataset:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

Each classifier's performance was quantified and illustrated through a bar chart. Based on the obtained results, the Decision Tree and KNN models emerged as the top performers in terms of baseline accuracy.

<table>
<tr>
<td>
<img src="images/Picture11.png" alt="Baseline Performance Results" width="1500"/>
</td>
</tr>
</table>

### Confusion Matrices and Accuracy Bar Chart

The confusion matrices provide insight into the true versus predicted labels, highlighting the performance of each classifier. The bar chart aggregates these results, showcasing the overall accuracy of each model, which informs the selection of the best-performing classifiers.

<table>
<tr>
  <th>Confusion Matrix</th>
  <th>Accuracy Bar Chart</th>
</tr>
<tr>
  <td><img src="images/Picture9.png" alt="Confusion Matrix" width="600"/></td>
  <td><img src="images/Picture10.png" alt="Bar Chart of Accuracies" width="600"/></td>
</tr>
</table>

# First Improvement Strategy: Comparing Dimensionality Reduction to Feature Selection

Advancing predictive accuracy was a key objective, achieved through feature selection using filter-based and wrapper-based methods. These methods were applied to the two models that exhibited the best baseline performance.

- **Filter Method**: Utilizes statistical tests to select relevant features. This method operates independently of machine learning algorithms, employing mutual information criteria for feature relevance.
- **Wrapper Method**: Selects feature sets by constructing multiple models and assessing their performance, adding or removing attributes iteratively to identify the most effective combination.
- **Principal Component Analysis (PCA)**: Reduces dimensionality through orthogonal transformation, targeting directions that maximize variance.

Performance comparisons were made based on the number of features versus the accuracy. Additionally, the baseline performance of each ML model was plotted as a constant dotted line with a different color for a direct comparison.

The following figures illustrate the performance of each feature selection method against the baseline, clearly indicating improvements and helping to identify the best feature subset and ML model for the subsequent stages of the project.

## Feature Selection Methods Visualizations

<table>
<tr>
  <th>Filter Method Comparison</th>
</tr>
<tr>
  <td><img src="images/Picture12.png" alt="Filter Method Comparison" width="100%"/></td>
</tr>
<tr>
  <th>Wrapper Method Comparison</th>
</tr>
<tr>
  <td><img src="images/Picture13.png" alt="Wrapper Method Comparison" width="100%"/></td>
</tr>
<tr>
  <th>PCA Comparison</th>
</tr>
<tr>
  <td><img src="images/Picture14.png" alt="PCA Comparison" width="100%"/></td>
</tr>
</table>


After evaluating the results, the **filter method with the Decision Tree model** was identified as the best performer, marking the first improvement in the project's analytic phase. This optimal feature subset and ML model will be utilized for the remaining parts of the project.

# Adding More Models

In the continuous effort to enhance the predictive models, three advanced techniques were applied using the best features determined from the Decision Tree with the filter method.

## Model Performance Results

To assess the improvements, the Random Forest, Stacking Ensemble, and Voting Ensemble techniques were utilized.The image below summarizes the accuracy obtained with each model:


<table>
<tr>
<td>
<img src="images/Picture15.png" alt="Model Results" width="1500"/>
</td>
</tr>
</table>

## Evaluation of Model Performance

The performance of the newly applied techniques was meticulously compared to the first improvement through confusion matrices, which illustrate the true versus predicted labels, and an accuracy bar chart that consolidates the results.

### Confusion Matrices and Accuracy Bar Chart

<table>
<tr>
  <th>Confusion Matrix</th>
  <th>Accuracy Bar Chart</th>
</tr>
<tr>
  <td><img src="images/Picture16.png" alt="Confusion Matrix Comparison" width="600"/></td>
  <td><img src="images/Picture17.png" alt="Accuracy Bar Chart Comparison" width="600"/></td>
</tr>
</table>

The analysis revealed that the **Stacking Ensemble model** outperformed the initial improvement, indicating that the new results will be adopted as the second improvement for subsequent analyses.


