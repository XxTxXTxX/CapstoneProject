import os

from django.conf import settings
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
    if request.method == "POST":
        # 获取上传的文件
        uploaded_file = request.FILES.get("molx_file")
        if uploaded_file:
            # 保存文件到服务器
            upload_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(upload_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # 返回文件名作为结果
            result = f"Uploaded file name: {uploaded_file.name}"
            return render(request, 'users/result.html', {"result": result})

    return render(request, 'users/sequence_input.html')

def calculate_result(file_path):
    """
    这个函数接收文件路径作为参数，
    返回文件的名字或者根据业务逻辑计算其他结果。
    """
    return os.path.basename(file_path)  # 返回文件名