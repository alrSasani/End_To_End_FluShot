from django.contrib import admin
from django.urls import path
from . import views


app_name = "prediction"

urlpatterns = [
    path('', views.home_view, name='home'),
    path('form/', views.form_view, name='form'),
    path('csv-upload/', views.csv_upload_view, name='csv_upload'),
    path('json-upload/', views.json_upload_view, name='json_upload'),
]
