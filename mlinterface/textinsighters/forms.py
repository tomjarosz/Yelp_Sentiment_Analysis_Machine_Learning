# Machine Learning Project: Textinsighters
# Creates forms for our html pages
# Created by: Vi Nguyen, Sirui Feng, Turab Hassan, 

from django import forms
from . import extract_names

business_name = extract_names.business_data()

class BusinessName(forms.Form):
    '''
    Makes a form with business name as a drop down.
    captures other user input in the other two fields
    '''
    business_name = forms.ChoiceField \
    ( label = 'Select Your Business', choices = \
    [ (x,x) for x in business_name ], required = False )
    user_input = forms.CharField(widget = forms.HiddenInput(), required = False)
    case = forms.CharField(widget = forms.HiddenInput(), required = False, initial=0)