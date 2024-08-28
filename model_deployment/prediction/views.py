import pandas as pd
import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import UserDataForm
from src.pipelines import prediction_pipeline
from src.training_config import TrainConfiguration
from .utils import (process_data_from_csv,process_data_from_json,
                    create_dataframe_from_input,COLUMN_NAMES)

#Predictor = prediction_pipeline.Predict_from_mlflow_production
Predictor = prediction_pipeline.Predict_from_saved_models

def home_view(request):
    return render(request, 'prediction/home.html')

def form_view(request):
    if request.method == 'POST':
        form = UserDataForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            df = create_dataframe_from_input(form_data)
            df['respondent_id'] = 0
            predictions = Predictor(df,artifacts_path=TrainConfiguration)
            predictions.replace([1,0],['Vaccinated','Not Vaccinated'],inplace=True)  
            return render(request, 'prediction/result.html', {'prediction': predictions.to_html()})
        else:
            return render(request, 'prediction/form.html', {'form': form})
    else:
        form = UserDataForm()
    return render(request, 'prediction/form.html', {'form': form})

def csv_upload_view(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            try:
                data,df = process_data_from_csv(csv_file)
                form = UserDataForm(data)
                if form.is_valid():
                    predictions = Predictor(df,artifacts_path=TrainConfiguration)
                    predictions.replace([1,0],['Vaccinated','Not Vaccinated'],inplace=True)  
                    return render(request, 'prediction/result.html', {'prediction': predictions.to_html()})
                else:
                    return render(request, 'prediction/form.html', {'form': form})

            except Exception as e:
                return HttpResponse(f"Error processing CSV file: {e}")
    else:
        form = UserDataForm()
    return render(request, 'prediction/csv_upload.html', {'form': form})

def json_upload_view(request):
    if request.method == 'POST':
        json_file = request.FILES.get('json_file')
        if json_file:
            try:
                data = process_data_from_json(json_file)
                form = UserDataForm(data)
                if form.is_valid():
                    form_data = form.cleaned_data
                    df = create_dataframe_from_input(form_data)
                    predictions = Predictor(df,artifacts_path=TrainConfiguration)
                    predictions.replace([1,0],['Vaccinated','Not Vaccinated'],inplace=True)
                    return render(request, 'prediction/result.html', {'prediction': predictions.to_html()})
                else:
                    return render(request, 'prediction/form.html', {'form': form})
                
            except Exception as e:
                return HttpResponse(f"Error processing JSON file: {e}")
    else:
        form = UserDataForm()
    return render(request, 'prediction/json_upload.html', {'form': form})

