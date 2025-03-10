from django.db import models

class CyberbullyingText(models.Model):
    text = models.TextField()
    is_cyberbullying = models.BooleanField(default=False)

    def __str__(self):
        return self.text[:50]
