from django import forms

class BusinessName(forms.Form):
    business_name = forms.CharField( widget = forms.TextInput(attrs={
        'list': 'browsers'} ), required=False )
    
    user_input = forms.CharField(widget = forms.HiddenInput(), required = False)

    case = forms.CharField(widget = forms.HiddenInput(), required = False, initial=0)