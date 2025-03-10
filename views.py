from django.shortcuts import render
import joblib
from .forms import TextForm
from .models import CyberbullyingText
from train_model import clean_text

# Load trained model and vectorizer
model = joblib.load("models/cyberbullying_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def detect_cyberbullying(request):
    prediction = None
    form = TextForm()
    
    if request.method == "POST":
        form = TextForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            cleaned_text = clean_text(text)
            text_vector = vectorizer.transform([cleaned_text])
            result = model.predict(text_vector)[0]
            prediction = "Cyberbullying Detected" if result == 1 else "No Cyberbullying Detected"
            
            # Save to database
            CyberbullyingText.objects.create(text=text, is_cyberbullying=(result == 1))
    
    return render(request, "index.html", {"form": form, "prediction": prediction})
