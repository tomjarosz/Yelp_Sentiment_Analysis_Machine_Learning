# Machine Learning Project: Textinsighters
# Creates views for our html pages
# Created by: Vi Nguyen, Sirui Feng, Turab Hassan, Tom Jarosz

from django.shortcuts import render
from django.http import HttpResponse
from .forms import BusinessName
from . import buildcontext

def suggestion(request):
    '''
    This function builds all the webpages of our website.
    Depending on the user input captured from the form
    make the page. We make three pages, the homepage,
    the insights page and then if you want just look at
    complaints, compliments, suggestions for user or 
    suggestions for business.
    '''

    if request.method == 'POST':
        form = BusinessName(request.POST)
        if form.is_valid():
            data=form.cleaned_data
            case = int(data['case'])
            if case == 0:  
                form = BusinessName()
                context = {}
                context['complaint'], context['compliments'], context['user_sugg'],\
                context['buss_sugg'] = buildcontext.context_from_bussname(data['business_name'], 0)
                context['form'] = form
                context['name'] = data['business_name']
                return render( request,'textinsighters/updatedexample.html', context)
            elif case == 1:
                context = {}
                context['complaint'] = buildcontext.context_from_bussname(data['user_input'], 1)
                return render( request,'textinsighters/complaints.html', context)
            elif case == 2:
                context = {}
                context['compliments'] = buildcontext.context_from_bussname(data['user_input'], 2)
                return render( request,'textinsighters/compliments.html', context)
            elif case == 3:
                context = {}
                context['user_sugg'] = buildcontext.context_from_bussname(data['user_input'], 3)
                return render( request,'textinsighters/usersugg.html', context)
            elif case == 4:
                context = {}
                context['buss_sugg'] = buildcontext.context_from_bussname(data['user_input'], 4)
                return render( request,'textinsighters/busssugg.html', context)                                                
    else:
        form = BusinessName()  
        context = {'form':form}
        buildcontext.make_matching_dict ()
        return render( request,'textinsighters/mainpage.html', context)