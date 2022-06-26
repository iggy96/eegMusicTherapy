
#`` Importing libraries
from helper import*
from params import*


# import csv files
delta_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/delta_TP9_result.csv')
delta_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/delta_AF7_result.csv')
delta_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/delta_AF8_result.csv')
delta_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/delta_TP10_result.csv')
delta_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/delta_mean_result.csv')

theta_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_TP9_result.csv')
theta_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_AF7_result.csv')
theta_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_AF8_result.csv')
theta_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_TP10_result.csv')
theta_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_mean_result.csv')

alpha_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_TP9_result.csv')
alpha_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_AF7_result.csv')
alpha_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_AF8_result.csv')
alpha_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_TP10_result.csv')
alpha_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_mean_result.csv')

beta_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/beta_TP9_result.csv')
beta_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/beta_AF7_result.csv')
beta_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/beta_AF8_result.csv')
beta_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/beta_TP10_result.csv')
beta_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/beta_mean_result.csv')

gamma_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/gamma_TP9_result.csv')
gamma_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/gamma_AF7_result.csv')
gamma_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/gamma_AF8_result.csv')
gamma_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/gamma_TP10_result.csv')
gamma_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/gamma_mean_result.csv')

theta_beta_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_beta_TP9_result.csv')
theta_beta_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_beta_AF7_result.csv')
theta_beta_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_beta_AF8_result.csv')
theta_beta_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_beta_TP10_result.csv')
theta_beta_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/theta_beta_mean_result.csv')

alpha_beta_TP9 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_beta_TP9_result.csv')
alpha_beta_AF7 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_beta_AF7_result.csv')
alpha_beta_AF8 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_beta_AF8_result.csv')
alpha_beta_TP10 = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_beta_TP10_result.csv')
alpha_beta_mean = pd.read_csv('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/alpha_beta_mean_result.csv')


#   run anova

#   delta
anova_delta_TP9 = anova(anova_title='DELTA TP9',dataframe=delta_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_delta_AF7 = anova(anova_title='DELTA AF7',dataframe=delta_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_delta_AF8 = anova(anova_title='DELTA AF8',dataframe=delta_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_delta_TP10 = anova(anova_title='DELTA TP10',dataframe=delta_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_delta_mean = anova(anova_title='DELTA MEAN',dataframe=delta_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   theta
anova_theta_TP9 = anova(anova_title='THETA TP9',dataframe=theta_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_AF7 = anova(anova_title='THETA AF7',dataframe=theta_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_AF8 = anova(anova_title='THETA AF8',dataframe=theta_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_TP10 = anova(anova_title='THETA TP10',dataframe=theta_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_mean = anova(anova_title='THETA MEAN',dataframe=theta_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   alpha
anova_alpha_TP9 = anova(anova_title='ALPHA TP9',dataframe=alpha_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_AF7 = anova(anova_title='ALPHA AF7',dataframe=alpha_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_AF8 = anova(anova_title='ALPHA AF8',dataframe=alpha_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_TP10 = anova(anova_title='ALPHA TP10',dataframe=alpha_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_mean = anova(anova_title='ALPHA MEAN',dataframe=alpha_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   beta
anova_beta_TP9 = anova(anova_title='BETA TP9',dataframe=beta_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_beta_AF7 = anova(anova_title='BETA AF7',dataframe=beta_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_beta_AF8 = anova(anova_title='BETA AF8',dataframe=beta_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_beta_TP10 = anova(anova_title='BETA TP10',dataframe=beta_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_beta_mean = anova(anova_title='BETA MEAN',dataframe=beta_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   gamma
anova_gamma_TP9 = anova(anova_title='GAMMA TP9',dataframe=gamma_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_gamma_AF7 = anova(anova_title='GAMMA AF7',dataframe=gamma_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_gamma_AF8 = anova(anova_title='GAMMA AF8',dataframe=gamma_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_gamma_TP10 = anova(anova_title='GAMMA TP10',dataframe=gamma_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_gamma_mean = anova(anova_title='GAMMA MEAN',dataframe=gamma_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   theta_beta
anova_theta_beta_TP9 = anova(anova_title='THETA-BETA TP9',dataframe=theta_beta_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_beta_AF7 = anova(anova_title='THETA-BETA AF7',dataframe=theta_beta_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_beta_AF8 = anova(anova_title='THETA-BETA AF8',dataframe=theta_beta_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_beta_TP10 = anova(anova_title='THETA-BETA TP10',dataframe=theta_beta_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_theta_beta_mean = anova(anova_title='THETA-BETA MEAN',dataframe=theta_beta_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

#   alpha_beta
anova_alpha_beta_TP9 = anova(anova_title='ALPHA-BETA TP9',dataframe=alpha_beta_TP9,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_beta_AF7 = anova(anova_title='ALPHA-BETA AF7',dataframe=alpha_beta_AF7,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_beta_AF8 = anova(anova_title='ALPHA-BETA AF8',dataframe=alpha_beta_AF8,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_beta_TP10 = anova(anova_title='ALPHA-BETA TP10',dataframe=alpha_beta_TP10,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_alpha_beta_mean = anova(anova_title='ALPHA-BETA MEAN',dataframe=alpha_beta_mean,anova_type=3,independent_variable=['group','task','time'],dependent_variable='frequency',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

# anova plots




