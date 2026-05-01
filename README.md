# Sentiment Analysis for Financial News Project

This project aims to perform sentiment analysis on financial news using deep learning models. The system is designed to analyze financial news patterns and classify them into different sentiment categories for accurate financial sentiment recognition.

## Dataset
[Financial News Dataset-Kaggle](https://www.kaggle.com/datasets)

## Features
- **Financial News Data Processing**: Tailored for processing and analyzing financial news data.
- **Deep Learning Models**: Utilizes advanced neural network architectures for sentiment classification.
- **Training on Labeled Data**: Built on a dataset labeled with sentiment categories, enabling the model to learn and predict financial sentiment accurately.
- **Performance Metrics**: Model performance evaluated using accuracy, precision, recall, and F1-score.

## Getting Started
### Prerequisites
- Python 3.x
- Required Python packages: PyTorch, Streamlit, Torchvision, Pandas, NumPy

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MEdan-US/Sentiment-Analysis-for-Financial-News.git
   cd Sentiment-Analysis-for-Financial-News
   ```
2. Create the virtual environment
   ```bash
   conda create --name sentiment_env -y
   conda activate sentiment_env
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the sentiment analysis system:

1. Run the main script:
```
python -m streamlit run app.py
```
2. Follow the prompts to input financial news data for sentiment classification.

## Acknowledgments
- Thanks to the original authors of machine learning and deep learning models for their significant contributions to sentiment classification tasks.
- We appreciate the open-source community for providing various libraries and frameworks that made this project possible.
