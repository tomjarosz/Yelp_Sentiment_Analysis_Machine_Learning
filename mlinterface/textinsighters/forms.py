from django import forms

class BusinessName(forms.Form):
    business_name = forms.CharField(widget = forms.TextInput(attrs={
        'list': 'browsers'
    }))