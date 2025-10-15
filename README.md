# Tennis Match Predictor

This project predicts the outcomes of ATP tennis matches using a machine learning model. It includes scripts for scraping data, cleaning and preparing the data, and training a model.

## Project Structure

- `data/`: Contains the raw and cleaned data in CSV format.
- `src/`: Contains the Python scripts for the project.
  - `scrape.py`: Scrapes match data from [tennisabstract.com](https://www.tennisabstract.com/reports/atp_surface_speed.html).
  - `clean.py`: Cleans the raw data and creates a mirrored dataset for training.
  - `train_model.py`: Trains a logistic regression or MLP model on the cleaned data.
  - `atp_logreg_2024.pth`: A trained model file.

## Getting Started

### Prerequisites

- Python 3.x
- The required Python packages can be found in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/tennis-predictor.git
    cd tennis-predictor
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Scrape Data

To scrape the latest ATP match data, run the `scrape.py` script.

```bash
python src/scrape.py
```

This will scrape the data and save it to a CSV file in the `data/` directory.

### 2. Clean Data

After scraping the data, you need to clean it and create the mirrored dataset.

```bash
python src/clean.py data/atp_matches_2024.csv data/atp_matches_clean_2024.csv
```

This will create a cleaned CSV file ready for training.

### 3. Train the Model

To train the model, run the `train_model.py` script with the cleaned data.

```bash
python src/train_model.py data/matches_all_mirrored_2022_2024.csv --model mlp --epochs 50 --out src/atp_mlp_2024.pth
```

This will train a new model and save it to the `src/` directory. You can choose between a logistic regression model (`logreg`) and a multi-layer perceptron (`mlp`).

## Data

The data is scraped from [tennisabstract.com](https://www.tennisabstract.com/reports/atp_surface_speed.html) and also from Jeff Sackman's GitHub repository, which can be found at [https://github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp). The data includes ATP match data from 2022, 2023 and 2024. The `clean.py` script processes this data to create a dataset suitable for training.

## Model

The model is a PyTorch model trained on the cleaned data. The `train_model.py` script can train either a logistic regression model or a multi-layer perceptron. The trained model is saved as a `.pth` file in the `src/` directory. The file `atp_logreg_2024.pth` is a pretrained model.