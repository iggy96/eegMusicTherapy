
#`` Importing libraries
from helper import*
from params import*


# import csv files
power_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/power_TP9_result.csv')
power_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/power_AF7_result.csv')
power_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/power_AF8_result.csv')
power_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/power_TP10_result.csv')
power_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/power_mean_result.csv')



#   run anova

#   delta
anova_power_TP9 = anova(anova_title='channel TP9',dataframe=power_TP9,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_AF7 = anova(anova_title='channel AF7',dataframe=power_AF7,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_AF8 = anova(anova_title='channel AF8',dataframe=power_AF8,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_TP10 = anova(anova_title='channel TP10',dataframe=power_TP10,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_mean = anova(anova_title='channel mean',dataframe=power_mean,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)





