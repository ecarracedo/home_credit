# Credit Risk Assessment with LightGBM

This project implements a credit risk assessment system using a machine learning model (LightGBM) and an interactive interface built with Streamlit.

## Project Structure

```
├── .gitignore
├── EDA.ipynb
├── lghtdm_model.py
├── main.py
├── requirements.txt
├── data/
│   ├── data_train_new.csv
│   ├── data_train_new.zip
│   └── data_train.csv
└── model/
    └── model_lightgbm.pkl
```

- **lghtdm_model.py**: Script for training, feature engineering, and model saving.
- **main.py**: Streamlit app for interactive credit risk evaluation.
- **data/**: Contains training datasets.
- **model/**: Stores the trained model and artifacts.
- **EDA.ipynb**: Exploratory Data Analysis notebook.
- **requirements.txt**: Project dependencies.

## Installation

1. Clone this repository and navigate to the project folder.
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Model Training

Run the training script to generate the model and necessary artifacts:

```sh
python lghtdm_model.py
```

The trained model will be saved as `model/model_lightgbm.pkl`.

## Application Usage

Launch the Streamlit app to evaluate the credit risk of new clients:

```sh
streamlit run main.py
```

Follow the interface instructions to input client data and obtain the risk prediction.

## Notes

- Data files are not included in the repository for privacy reasons.
- Ensure the data files are present in the `data/` folder before training the model.

## Author

Developed by [Your Name].