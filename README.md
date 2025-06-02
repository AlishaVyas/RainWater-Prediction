# RainWater-Prediction
# Rainfall Prediction Model

This project aims to predict rainfall based on various meteorological features using a Random Forest Classifier. The model is trained and evaluated on a dataset containing historical weather information.

## Project Structure

The project is structured as a Google Colab notebook that performs the following steps:

1.  **Importing Dependencies**: Imports necessary libraries for data manipulation, visualization, model building, and evaluation.
2.  **Data Collection and Processing**:
    *   Loads the dataset from a CSV file.
    *   Performs initial data inspection (shape, head, tail, info, unique values).
    *   Cleans column names by removing extra spaces.
    *   Handles missing values by imputing with mode and median for specific columns.
    *   Converts the target variable ('rainfall') from categorical ('yes', 'no') to numerical (1, 0).
3.  **Exploratory Data Analysis (EDA)**:
    *   Sets the plot style.
    *   Provides descriptive statistics of the data.
    *   Visualizes the distribution of numerical features using histograms.
    *   Shows the distribution of the target variable using a countplot.
    *   Displays the correlation matrix using a heatmap.
    *   Visualizes potential outliers using boxplots for numerical features.
4.  **Data Preprocessing**:
    *   Drops highly correlated columns to avoid multicollinearity.
    *   Addresses class imbalance by downsampling the majority class.
    *   Shuffles the balanced dataset.
    *   Splits the data into features (X) and target (y).
    *   Splits the data into training and testing sets.
5.  **Model Training**:
    *   Initializes a Random Forest Classifier.
    *   Defines a hyperparameter grid for tuning.
    *   Performs hyperparameter tuning using GridSearchCV with cross-validation.
    *   Prints the best parameters found.
6.  **Model Evaluation**:
    *   Evaluates the best model using cross-validation on the training data.
    *   Evaluates the model performance on the test set using accuracy, classification report, and confusion matrix.
7.  **Prediction on Unknown Data**:
    *   Demonstrates how to make a prediction on a sample input.
8.  **Model Persistence**:
    *   Saves the trained model and the list of feature names to a pickle file for later use.
9.  **Loading and Using the Saved Model**:
    *   Demonstrates how to load the saved model and make a prediction using it.

## Dependencies

The project requires the following libraries:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `sklearn` (specifically `resample`, `train_test_split`, `GridSearchCV`, `cross_val_score`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `accuracy_score`)
*   `pickle`

These dependencies are typically available in a Google Colab environment. If running locally, you might need to install them using pip:
