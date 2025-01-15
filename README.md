CROP YIELD PREDICTION USING MACHINE LEARNING: 
-> This project leverages machine learning to predict crop yields, assisting farmers in making informed decisions about planting, resource allocation, and harvesting.

PROJECT OVERVIEW: 
Accurate crop yield prediction is vital for optimizing agricultural practices and ensuring food security. This project utilizes historical agricultural data to develop a predictive model, aiming to enhance productivity and resource efficiency.

DATA COLLECTION: 
The dataset comprises the following CSV files:
1. Pesticides Data: pesticides.csv
2. Rainfall Data: rainfall.csv
3. Temperature Data: temp.csv
4. Yield Data: yield.csv

DATA PREPROCESSING:  
Handling Missing Values: Replaced '..' with NaN and removed rows containing NaN values.
Encoding Categorical Variables: Converted categorical 'Area' data into numerical format.

DATA EXPLORATION: 
Visualizations BASED ON :- 
1. Pesticide usage across top countries
2. Rainfall trends over time
3. Temperature distributions
4. Crop yield changes over the years

FEATURE SELECTION: 
Selected Features:
1. Year
2. Encoded Area
3. Pesticides
4. Rainfall
5. Temperature

MODEL BUILDING: 
Algorithm Used: Linear Regression
Framework: Scikit-learn
Validation: Train-test split method

MODEL EVALUTION: 
Accuracy: Achieved 92% accuracy in predicting crop yields.

DEPLOYMENT: 
User Interface: Accepts user inputs for predictions.
Deployment: Converted to an executable (.exe) file using PyInstaller for ease of use.
                                        
INSIGHTS: 
* Pesticide Usage: Identified top countries with highest pesticide usage.
* Rainfall Trends: Visualized average rainfall over time.
* Temperature Distribution: Analyzed temperature distributions.
* Crop Yield Over Time: Assessed trends in crop yields over the years.
  
CONCLUSION: 
This machine learning model offers valuable insights for farmers, enabling informed decisions to optimize crop production and contribute to sustainable agriculture.
