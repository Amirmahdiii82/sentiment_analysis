from django.db import models

class SentimentAnalysis(models.Model):
    user_text = models.TextField("User Input Text", max_length=1000)
    sentiment = models.CharField("Predicted Sentiment", max_length=10, choices=[('HAPPY', 'Happy'), ('SAD', 'Sad')], default='HAPPY')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user_text[:50]} - {self.sentiment}"
