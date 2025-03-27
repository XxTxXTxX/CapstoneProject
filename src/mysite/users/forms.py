from django import forms

class MolxUploadForm(forms.Form):
    file = forms.FileField(label="Upload MOLX File")

class ProteinSequenceForm(forms.Form):
    sequence = forms.CharField(
        widget=forms.Textarea(attrs={
            'placeholder': 'Enter protein sequence...',
            'rows': 4,
            'class': 'form-control'
        }),
        label="Protein Sequence"
    )
