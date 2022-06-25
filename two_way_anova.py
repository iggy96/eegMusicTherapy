
#`` Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

# Create a dataframe
dataframe = pd.DataFrame({'Fertilizer': np.repeat(['daily', 'weekly'], 15),
						'Watering': np.repeat(['daily', 'weekly'], 15),
                        'Temperature': np.repeat(['high', 'low'], 15),
						'height': [14, 16, 15, 15, 16, 13, 12, 11,
									14, 15, 16, 16, 17, 18, 14, 13,
									14, 14, 14, 15, 16, 16, 17, 18,
									14, 13, 14, 14, 14, 15]})


# Performing two-way ANOVA
model = ols('height ~ C(Fertilizer) + C(Watering) + C(Temperature) +\
C(Fertilizer):C(Watering):C(Temperature)',
			data=dataframe).fit()
result = sm.stats.anova_lm(model, type=3)

# Print the result
print(result)
"""
independent_variable = ['Fertilizer', 'Watering']
dependent_variable = 'height'
string = dependent_variable + ' ~ ' +'C(' + independent_variable[0] + ') + C(' + independent_variable[1] + ') + \C(' + independent_variable[0] + '):C(' + independent_variable[1] + ')'
print(string)
"""

def anova(dataframe,anova_type,independent_variable,dependent_variable):
    if anova_type==2:
        string = dependent_variable + ' ~ ' +'C(' + independent_variable[0] + ') + C(' + independent_variable[1] + ') + \
            C(' + independent_variable[0] + '):C(' + independent_variable[1] + ')'
        model = ols(string,data=dataframe).fit()
        result = sm.stats.anova_lm(model, type=2)
        print(result)
    elif anova_type==3:
        string = dependent_variable + ' ~ ' +'C(' + independent_variable[0] + ') + C(' + independent_variable[1] + ') + C(' + independent_variable[2] + ') + \
            C(' + independent_variable[0] + '):C(' + independent_variable[1] + '):C(' + independent_variable[2] + ')'
        model = ols(string,data=dataframe).fit()
        result = sm.stats.anova_lm(model, type=3)
        print(result)
    return result

test = anova(dataframe,anova_type=3,independent_variable=['Fertilizer', 'Watering','Temperature'],dependent_variable='height')
