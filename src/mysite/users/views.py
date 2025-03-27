import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import time
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from django.urls import reverse

from .models import User
from .forms import ProteinSequenceForm
import torch
from model.inference import run_inference, save_pdb
from model.model import ProteinStructureModel


def register(request):  # register page
    if request.method == 'POST':
        username = request.POST['username']
        password = make_password(request.POST['password'])

        # If user already registerd, popup error message and return
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return redirect('login')
        user = User.objects.create(username=username, password=password)
        user.save()
        messages.success(request, 'Registration successful! Redirecting...')
        return redirect('login')
    return render(request, 'users/register.html')


def user_login(request):  # login page
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user:
            print("YANZHENGCHENGGONG")
            login(request, user)
            messages.success(request, 'Login successful! Redirecting...')
            return redirect('sequence_input')
        else:
            print("验证失败")
            return render(request, 'users/login.html',
                          {'error': 'Invalid credentials, check your username or password'})
    return render(request, 'users/login.html')


def sequence_input(request):
    if request.method == 'POST':
        sequence = request.POST.get('sequence')
        time.sleep(3)
        return redirect('result', sequence=sequence)
    return render(request, 'users/sequence_input.html')

def result(request, sequence):
    molx_filename = "1a0r.pdb"
    molx_filepath = os.path.join(settings.MEDIA_ROOT, molx_filename)
    with open(molx_filepath, 'r') as file:
        pdb_content = file.read()
    context = {'sequence': sequence, 'pdb_content': pdb_content}
    return render(request, 'result.html', context)


def calculate_result(file_path):

    return os.path.basename(file_path)

def predict_structure(request):
    if request.method == 'POST':
        form = ProteinSequenceForm(request.POST)
        if form.is_valid():
            sequence = form.cleaned_data['sequence']
            
            try:
                pred_coords, pred_mask = run_inference(sequence)
                
                pdb_path = os.path.join(PROJECT_ROOT, "temp_structure.pdb")
                save_pdb(pred_coords, pred_mask, sequence, pdb_path)
                
                with open(pdb_path, 'r') as f:
                    pdb_content = f.read()
                
                return render(request, 'users/result.html', { 
                    'pdb_content': pdb_content
                })
            except Exception as e:
                return render(request, 'users/sequence_input.html', { 
                    'form': form,
                    'error': str(e)
                })
    else:
        form = ProteinSequenceForm()
    
    return render(request, 'users/sequence_input.html', {'form': form}) 