from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('coloring/', views.coloring, name='coloring'),
    path('filterON/', views.filterON, name='filterON'),
    path('palettepage/', views.palettepage, name='palettepage'),
]