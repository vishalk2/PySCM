import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleExpo():
    
    def __init__(self,n,t,alpha):
        self.periods = n
        self.period = t
        self.demand_data = ['Dt']
        self.alpha = alpha
        self.level = {}
        self.forecast = {}
        self.forecast_error = {}
        self.mean_squared_error = 'unknown'
        self.mean_absolute_deviation = 'unknown'
        self.std = 'unknown'
        
    def add_demand_data(self):
        self.demand_data += list(map(float,input().split()))
        
    def perform_analysis(self):
        
        n = self.periods
        t = self.period
        Dt = self.demand_data
        alpha = self.alpha
        Lt = self.level
        Ft = self.forecast
        Et = self.forecast_error
        
        L0 = round(sum(Dt[1:n+1])/n,3)
        Lt['L0'] = L0
        
        for i in range(1,t+1):
            Ft['F'+str(i)] = Lt['L'+str(i-1)]
            if(i == t):
                break
            else:
                Et['E'+str(i)] = round(Ft['F'+str(i)] - Dt[i],3)
                Lt['L'+str(i)] = round(alpha*Dt[i] + (1-alpha)*Lt['L'+str(i-1)],3)
        
        self.mean_squared_error = sum([i**2 for i in Et.values()])/n
        self.mean_absolute_deviation = sum([abs(i) for i in Et.values()])/n
        self.std = 1.25*self.mean_absolute_deviation
        
        print('Analysis Complete')
