from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from .models import User

def register(request): # register page
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

def user_login(request): # login page
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
            return render(request, 'users/login.html', {'error': 'Invalid credentials, check your username or password'})
    return render(request, 'users/login.html')



def sequence_input(request):
    result = None
    if request.method == 'POST':
        sequence = request.POST['sequence']
        # calculate Result -> how our AI model predict value and return it
        result = calculate_result(sequence)
    return render(request, 'users/sequence_input.html', {'result': result})

def calculate_result(sequence):
    # Model put here
    return 0