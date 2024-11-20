# Persian Sentiment Analysis

This project introduces a custom Transformer-based architecture designed for sentiment analysis on Persian text reviews from the SnappFood dataset. By leveraging a pretrained BERT tokenizer for text encoding and fine-tuning a lightweight Transformer encoder, ethe model effectively classifies reviews as HAPPY or SAD. This approach addresses challenges posed by Persian morphology and script, demonstrating the effectiveness of custom architectures for low-resource NLP tasks.

## Django Web App

In addition to the model, a web application has been built using **Django** to interact with the sentiment analysis model. The web app allows users to input Persian text reviews and get the sentiment classification (HAPPY or SAD.).

## Results

The repository also contains visualizations and result images from the sentiment analysis model for better understanding and presentation of the analysis.

## Docker Image

You can pull and run the Docker image for this project using the following command:

```bash
docker pull amirmahdiaramide/sentiment_analysis_webapp
