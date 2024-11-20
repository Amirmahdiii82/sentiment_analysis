# Persian Sentiment Analysis

This project introduces a custom Transformer-based architecture designed for sentiment analysis on Persian text reviews from the SnappFood dataset. By leveraging a pretrained BERT tokenizer for text encoding and fine-tuning a lightweight Transformer encoder, the model effectively classifies reviews as positive, neutral, or negative. This approach addresses challenges posed by Persian morphology and script, demonstrating the effectiveness of custom architectures for low-resource NLP tasks.

## Django Web App

In addition to the model, a web application has been built using **Django** to interact with the sentiment analysis model. The web app allows users to input Persian text reviews and get the sentiment classification (positive, neutral, or negative).

## Results

The repository also contains visualizations and result images from the sentiment analysis model for better understanding and presentation of the analysis.

## Running the Web App

To run the web app, follow these steps:

1. Download the model from [Google Drive](<https://drive.google.com/file/d/1UNbgB3YqWOXyzlr1aMSy6o1faDk6j2Mf/view?usp=share_link>).
2. Place the downloaded model in the `webapp/analysis/ml_model/` directory.

Once the model is in place, you can run the web app.

## Docker Image

You can pull and run the Docker image for this project using the following command:

```bash
docker pull amirmahdiaramide/sentiment_analysis_webapp
