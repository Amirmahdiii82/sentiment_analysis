from django import forms

class TextInputForm(forms.Form):
    user_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'placeholder': 'Enter text here...'}))
