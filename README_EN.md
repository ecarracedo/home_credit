
# Credit Risk Assessment with LightGBM

This MVP project implements a credit risk assessment system using a machine learning model (LightGBM) and an interactive interface built with Streamlit.

## Project Structure

```
├── lghtdm_model.py
├── main.py
├── requirements.txt
├── data/
│     └── data_train_new.zip
│     └── HomeCredit_columns_description.csv
|
└── model/
    └── model_lightgbm.pkl
```

- **lghtdm_model.py**: Script for training, feature engineering, and model saving.
- **main.py**: Streamlit application for interactive credit risk evaluation.
- **data/**: Contains the training datasets.
- **model/**: Folder where the trained model and artifacts are stored.
- **requirements.txt**: Project dependencies.

## Installation

1. Clone this repository and navigate to the project folder.
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Model Training

Unzip the `data_train_new.zip` file to extract the training dataset.

Run the training script to generate the model and required artifacts:

```sh
python lghtdm_model.py
```

Set the file path inside `lghtdm_model.py`:

```python
ARCHIVO_DATOS = os.path.join(BASE_DIR, 'data', 'data_train_new.csv')
```

The trained model will be saved at `model/model_lightgbm.pkl`.

## lghtdm_model.py

This script handles the entire process of training and preparing the credit risk model. Its main functions include:

- **Data loading and preprocessing**: Reads the training data, performs cleaning, transformation, and generates new relevant features.
- **Feature engineering**: Selects and transforms numerical and categorical variables, calculates financial ratios, and prepares data for the model.
- **Model training**: Uses LightGBM (or a scikit-learn pipeline) to train the classification model on the processed data.
- **Evaluation**: Computes performance metrics such as ROC-AUC, accuracy, confusion matrix, etc., to validate the model quality.
- **Artifact serialization**: Saves the trained model, preprocessor, and list of required columns/variables into a `.pkl` file inside the `model/` folder. This allows the main application (`main.py`) to load and use the model without retraining.

This file should be run whenever the model needs to be updated with new data or changes.

## HomeCredit_columns_description.csv

This file contains a description of the columns in the `data_train.csv` file. It can help enrich the model evaluation by adding more features.

## Using the Application

Launch the Streamlit app to evaluate the credit risk of new clients:

```sh
streamlit run main.py
```

Follow the interface instructions to input client data and get the risk prediction.

It is also hosted on Streamlit Cloud at the following address:

https://homecredit-mvp.streamlit.app/

## Notes

- The credit history has three components that act like external credit scores from reporting agencies. If you increase all three values, the credit evaluation becomes positive.
- Make sure the data files are placed in the `data/` folder before training the model.
- This is a personal project, but it’s also intended to showcase the model training experience for future projects I may be involved in.

## Author

Developed by Emiliano Carracedo | ecarracedo@gmail.com |
