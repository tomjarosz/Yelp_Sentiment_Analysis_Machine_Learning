from django.shortcuts import render
from django.http import HttpResponse
from .forms import BusinessName
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
            context = buildcontext(data['business_name'])
            return render( request,'textinsighters/mainpage.html', context)
    else:
        form = BusinessName()  
        context = {'form':form}
        print('first time')
        print(form)
        return render( request,'textinsighters/mainpage.html', context)