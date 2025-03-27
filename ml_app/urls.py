from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('simple-linear/', views.simple_linear, name='simple_linear'),
    path('multiple-linear/', views.multiple_linear, name='multiple_linear'),
    path('polynomial/', views.polynomial, name='polynomial'),
    path('logistic/', views.logistic, name='logistic'),
    path('knn/', views.knn, name='knn'),
] 