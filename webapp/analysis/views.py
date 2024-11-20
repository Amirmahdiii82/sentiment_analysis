from django.shortcuts import render
from .ml_model.sentiment_model import generate_prediction

def index(request):
    return render(request, 'analysis/index.html')

def analyze_text(request):
    if request.method == 'POST':
        user_text = request.POST.get('user_text', '')
        sentiment = generate_prediction(user_text)  
        
        # Prepare the context to pass to the template
        context = {
            'result': {
                'user_text': user_text,
                'sentiment': sentiment,
            }
        }
        return render(request, 'analysis/result.html', context)
    else:
        return render(request, 'analysis/index.html')
