from django import forms

class MolxUploadForm(forms.Form):
    file = forms.FileField(label="Upload MOLX File")
