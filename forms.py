from django import forms
from .models import CyberbullyingText

class TextForm(forms.ModelForm):
    class Meta:
        model = CyberbullyingText
        fields = ['text']
