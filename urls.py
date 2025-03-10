from django.urls import path
from .views import detect_cyberbullying

urlpatterns = [
    path('', detect_cyberbullying, name="detect"),
]
