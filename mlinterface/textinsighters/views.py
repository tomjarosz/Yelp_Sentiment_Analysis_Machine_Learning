from django.shortcuts import render
from django.http import HttpResponse
from .forms import BusinessName
from . import buildcontext
# Create your views here.


def suggestion(request):
    
    #print('show first time')
    if request.method == 'POST':
        print('getting to POST')
        form = BusinessName(request.POST)
        print(form)
        if form.is_valid():
            #print('this should show')
            data=form.cleaned_data  
            print(data['business_name'])
            #context = insert_function_here(data)
            form = BusinessName()
            context = {}
            context['complaint'], context['compliments'], context['user_sugg'],\
            context['buss_sugg'] = buildcontext.context_from_bussname(data['business_name'])
            #print(context)
            print('__________________________________')
            print(len(context['complaint']))
            print('complaints',context['complaint'])
            print('compliments',context['compliments'])
            return render( request,'textinsighters/example.html', context)
    else:
        form = BusinessName()  
        context = {'form':form}
        print('first time')
        print(form)
        buildcontext.make_matching_dict ()
        return render( request,'textinsighters/mainpage.html', context)