from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^suggestion', views.suggestion, name = 'suggestion'),
]