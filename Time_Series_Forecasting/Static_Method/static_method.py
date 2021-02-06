import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StaticMethod():
    
    def __init__(self,p,r):
        self.periodicity = p # Periodicity
        self.seasonal_cycles = r # Number of seasonal cycles
        self.time_periods = ['t'] # Time Period (t)
        self.demand_data = ['Dt'] # Demand Data (Dt)
        self.estimated_level = 'unknown' # Level estimated from Linear Regression; initially "unknown"
        self.estimated_trend = 'unknown' # Trend estimated from Linear Regression; initially "unknown"
        self.deseasonalized_demands = [] # Deseasonalized Demand Data (DDt)
        self.seasonal_factors = [] # Seasonal Factors (SSt)
        self.seasonal_factor_for_given_period = [] # Seasonal Factors (Si)
        self.forecast_for_next_p_periods = [] # Forecast for next "p" periods
        
    def add_time_period(self): # Enables the user to add data of Time Period
        self.time_periods = self.time_periods+list(map(int,input().split()))
        
    def add_demand_data(self): # Enables the user to add Demand Data
        self.demand_data = self.demand_data+list(map(float,input().split()))
        
    def perform_analysis(self): # Performs the remaining Static Method Time-Series Forecasting Method
        
        p = self.periodicity # Period
        
        # If p is even
        if(p%2 == 0):
            Dt = self.demand_data # Demand Data
            t_initial = int(1 + (p/2)) # Initial period "t" for Deseasonalized demand
            t_final = int(len(self.time_periods) - 1 - (p/2)) # Final period "t" for Deseasonalized demand
            DDt = ['DDt'] # Initiating Deseasonalized Demand
            DDt = DDt+['unknown' for i in range(len(self.demand_data)-1)] # Deseasonalized Demand
            
            # Calculating Deseasonalized Demand from Demand Data
            for i in range(t_initial,t_final+1):
                DDt[i] = (Dt[int(i-(p/2))] + Dt[int(i+(p/2))] + 2*(sum(Dt[int(i+1-(p/2)):int(i+(p/2))])))/(2*p)
            
            self.deseasonalized_demands = DDt # Deseasonalized Demand
            
            # Running Linear Regression for finding Level & Trend
            
            sum_t = sum(range(t_initial,t_final+1)) # Sum of time periods
            sum_DDt = sum(DDt[t_initial:t_final+1]) # Sum of Deseasonalized Demands
            
            mean_t = sum_t/(len(range(t_initial,t_final+1))) # Mean of time periods
            mean_DDt = sum_DDt/len(DDt[t_initial:t_final+1]) # Mean of Deseasonalized Demand
            
            diff_t = [i-mean_t for i in range(t_initial,t_final+1)] # Difference of time periods & their mean
            diff_DDt = [DDt[i]-mean_DDt for i in range(t_initial,t_final+1)] # Difference of Deseasonalized Demand & their mean
            
            sos_t = sum([i**2 for i in diff_t]) # Sum of squares of Differences of time periods & their mean
            
            # Sum of products of 'Differences of time periods & their mean' 
            # and 'Difference of Deseasonalized Demand & their mean'
            sop = sum([diff_t[i]*diff_DDt[i] for i in range(len(diff_t))])
            
            # Slope of the Regression model between DDt & t
            slope = sop/sos_t
            self.estimated_trend = round(slope,0)
            
            # Intercept of the Regression model between DDt & t
            intercept = mean_DDt - slope*mean_t
            self.estimated_level = round(intercept,0)
            
            # Results of the Regression Model
            print('Linear Regression of Deseasonalized Demand & Time Period')
            print('DDt = {}*t + {}'.format(round(slope,0),round(intercept,0)))
            print('where -')
            print('Level = ',round(intercept,0))
            print('Trend = ',round(slope,0))
            print()
            
            # Finding "unknown" values in Deseasonalized Demand
            for i in range(len(DDt)):
                if(DDt[i] == 'unknown'):
                    DDt[i] = round(slope,0)*self.time_periods[i]+round(intercept,0)
                else:
                    pass
            
            # Estimating Seasonal Factors SSt
            SSt = ['SSt']
            SSt = SSt + list(['unknown' for i in range(len(Dt)-1)])
            
            for i in range(1,len(Dt)):
                if(DDt[i] != 0):
                    SSt[i] = round(Dt[i]/DDt[i],2)
                else:
                    pass
            self.seasonal_factors = SSt
            
            # Estimating Seasonal Factors based on seasonal cycles & periods
            Si = ['Si']
            Si = Si + ['unknown' for i in range(p)]
            for i in range(1,p+1):
                s = 0
                for j in range(self.seasonal_cycles):
                    s = s + SSt[j*p+i]
                Si[i] = round(s/self.seasonal_cycles,2)
            
            self.seasonal_factor_for_given_period = Si
            
            # Predicting Forecast for next "p" periods after final t periods
            F = {}
            for i in range(1,p+1):
                F['F'+str(self.time_periods[-1]+i)] = round((round(intercept,0)+round(slope,0)*(self.time_periods[-1]+i))*Si[i],0)
                
            self.forecast_for_next_p_periods = F
            print()
            
            # Tabulation
            df = pd.DataFrame({'Time Period':self.time_periods,
                               'Actual Demand':self.demand_data,
                               'Deseasonalized Demand':self.deseasonalized_demands,
                               'Seasonal Factors':self.seasonal_factors})
            df.set_index('Time Period',inplace=True)
            print(df)
            print()
            # Plotting Demand VS Time Period
            
            y1 = np.array(Dt[1:])
            y2 = np.array(DDt[1:])
            x = np.array(self.time_periods[1:])
            plt.plot(x,y1,label='Actual Demand')
            plt.plot(x,y2,label='Deseasonalized Demand')
            plt.xlabel('Period'),plt.ylabel('Demand'),plt.title('Demand - Time Period Graphic')
            plt.xlim([self.time_periods[1],self.time_periods[-1]]),plt.legend()
            print()
            print('Analysis Complete')
            
        elif(p%2 != 0):
            
            Dt = self.demand_data # Demand Data
            t_initial = int(1 + ((p-1)/2)) # Initial period "t" for Deseasonalized demand
            t_final = int(len(self.time_periods) - 1 - ((p-1)/2)) # Final period "t" for Deseasonalized demand
            DDt = ['DDt'] # Initiating Deseasonalized Demand
            DDt = DDt+['unknown' for i in range(len(Dt)-1)] # Deseasonalized Demand
            
            # Calculating Deseasonalized Demand from Demand Data
            for i in range(t_initial,t_final+1):
                DDt[i] = sum(Dt[t_initial:t_final+1])/p
            
            self.deseasonalized_demands = DDt # Deseasonalized Demand
            
            # Running Linear Regression for finding Level & Trend
            
            sum_t = sum(range(t_initial,t_final+1)) # Sum of time periods
            sum_DDt = sum(DDt[t_initial:t_final+1]) # Sum of Deseasonalized Demands
            
            mean_t = sum_t/(len(range(t_initial,t_final+1))) # Mean of time periods
            mean_DDt = sum_DDt/len(DDt[t_initial:t_final+1]) # Mean of Deseasonalized Demand
            
            diff_t = [i-mean_t for i in range(t_initial,t_final+1)] # Difference of time periods & their mean
            diff_DDt = [DDt[i]-mean_DDt for i in range(t_initial,t_final+1)] # Difference of Deseasonalized Demand & their mean
            
            sos_t = sum([i**2 for i in diff_t]) # Sum of squares of Differences of time periods & their mean
            
            # Sum of products of 'Differences of time periods & their mean' 
            # and 'Difference of Deseasonalized Demand & their mean'
            sop = sum([diff_t[i]*diff_DDt[i] for i in range(len(diff_t))])
            
            # Slope of the Regression model between DDt & t
            slope = sop/sos_t
            self.estimated_trend = round(slope,0)
            
            # Intercept of the Regression model between DDt & t
            intercept = mean_DDt - slope*mean_t
            self.estimated_level = round(intercept,0)
            
            # Results of the Regression Model
            print('Linear Regression of Deseasonalized Demand & Time Period')
            print('DDt = {}*t + {}'.format(round(slope,0),round(intercept,0)))
            print('where -')
            print('Level = ',round(intercept,0))
            print('Trend = ',round(slope,0))
            print()
            
            # Finding "unknown" values in Deseasonalized Demand
            for i in range(len(DDt)):
                if(DDt[i] == 'unknown'):
                    DDt[i] = round(slope,0)*self.time_periods[i]+round(intercept,0)
                else:
                    pass
            
            # Estimating Seasonal Factors SSt
            SSt = ['SSt']
            SSt = SSt + list(['unknown' for i in range(len(Dt)-1)])
            
            for i in range(1,len(Dt)):
                if(DDt[i] != 0):
                    SSt[i] = round(Dt[i]/DDt[i],2)
                else:
                    pass
            self.seasonal_factors = SSt
            
            # Estimating Seasonal Factors based on seasonal cycles & periods
            Si = ['Si']
            Si = Si + ['unknown' for i in range(p)]
            for i in range(1,p+1):
                s = 0
                for j in range(self.seasonal_cycles):
                    s = s + SSt[j*p+i]
                Si[i] = round(s/self.seasonal_cycles,2)
            
            self.seasonal_factor_for_given_period = Si
            
            # Predicting Forecast for next "p" periods after final t periods
            F = {}
            for i in range(1,p+1):
                F['F'+str(self.time_periods[-1]+i)] = round((round(intercept,0)+round(slope,0)*(self.time_periods[-1]+i))*Si[i],0)
                
            self.forecast_for_next_p_periods = F
            print()
            # Tabulation
            df = pd.DataFrame({'Time Period':self.time_periods,
                               'Actual Demand':self.demand_data,
                               'Deseasonalized Demand':self.deseasonalized_demands,
                               'Seasonal Factors':self.seasonal_factors})
            df.set_index('Time Period',inplace=True)
            print(df)
            print()
            # Plotting Demand VS Time Period
            
            y1 = np.array(Dt[1:])
            y2 = np.array(DDt[1:])
            x = np.array(self.time_periods[1:])
            plt.plot(x,y1,label='Actual Demand')
            plt.plot(x,y2,label='Deseasonalized Demand')
            plt.xlabel('Period'),plt.ylabel('Demand'),plt.title('Demand - Time Period Graphic')
            plt.xlim([self.time_periods[1],self.time_periods[-1]]),plt.legend()
            print()
            print('Analysis Complete')
            
