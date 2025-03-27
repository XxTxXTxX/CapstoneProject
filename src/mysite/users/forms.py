from django import forms

class MolxUploadForm(forms.Form):
    file = forms.FileField(label="Upload MOLX File")

class ProteinSequenceForm(forms.Form):
    VALID_AA = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V","X", "-"]
    
    sequence = forms.CharField(
        widget=forms.Textarea(attrs={
            'placeholder': f'Enter protein sequence (valid amino acids: {", ".join(VALID_AA)})... (Length of your input: 10 < input < 512)',
            'rows': 4,
            'class': 'form-control'
        }),
        label="Protein Sequence"
    )
    pH = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'placeholder': 'Enter ph between 0 - 14',
            'class': 'form-control',
            'step': '0.1',
            'min': '0',
            'max': '14'
        }),
        label="pH Value",
        initial=7.0
    )
    temperature = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'placeholder': 'Enter tempature between 277-300 F',
            'class': 'form-control',
            'step': '0.1'
        }),
        label="Temperature (°C)",
        initial=277.0
    )

    def clean_sequence(self):
        sequence = self.cleaned_data['sequence'].upper()
        invalid_chars = set(sequence) - set(self.VALID_AA)
        if invalid_chars:
            raise forms.ValidationError(
                f"Invalid amino acids found: {', '.join(invalid_chars)}. "
                f"Valid amino acids are: {', '.join(self.VALID_AA)}"
            )
        return sequence
