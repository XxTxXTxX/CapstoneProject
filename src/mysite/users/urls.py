from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('sequence_input/', views.predict_structure, name='sequence_input'),
    path('result/<str:sequence>', views.result, name='result'),
    path('predict/', views.predict_structure, name='predict_structure'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) \
  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)