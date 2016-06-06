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
        print('form in post',form)
        if form.is_valid():
            print('form is valid')
            data=form.cleaned_data
            case = int(data['case'])
            if case == 0:  
                print('case is 0')
                print(data)
                #print(data['business_name'])
                #context = insert_function_here(data)
                form = BusinessName()
                context = {}
                #print('form', form)
                context['complaint'], context['compliments'], context['user_sugg'],\
                context['buss_sugg'] = buildcontext.context_from_bussname(data['business_name'], 0)
                context['form'] = form
                context['name'] = data['business_name']
                #print('____________________________________')
                #print('what we have in views',context)
                return render( request,'textinsighters/updatedexample.html', context)
            elif case == 1:
                print('case is 1')
                context = {}
                context['complaint'] = buildcontext.context_from_bussname(data['user_input'], 1)
                return render( request,'textinsighters/complaints.html', context)
            elif case == 2:
                print('case is 2')
                context = {}
                context['compliments'] = buildcontext.context_from_bussname(data['user_input'], 2)
                return render( request,'textinsighters/compliments.html', context)
            elif case == 3:
                print('case is 1')
                context = {}
                context['user_sugg'] = buildcontext.context_from_bussname(data['user_input'], 3)
                return render( request,'textinsighters/usersugg.html', context)
            elif case == 4:
                print('case is 1')
                context = {}
                context['buss_sugg'] = buildcontext.context_from_bussname(data['user_input'], 4)
                return render( request,'textinsighters/busssugg.html', context)                                                
    else:
        form = BusinessName()  
        context = {'form':form}
        print('first time')
        print(form)
        buildcontext.make_matching_dict ()
        return render( request,'textinsighters/mainpage.html', context)