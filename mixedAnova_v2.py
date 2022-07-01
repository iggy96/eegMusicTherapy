"""
statistical analysis for results from grp_analysis_v2.ipynb
"""
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
#   mixed anova analysis: group x task x time x frequency band x power
anova_power_TP9 = anova(anova_title='channel TP9',dataframe=power_TP9,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_AF7 = anova(anova_title='channel AF7',dataframe=power_AF7,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_AF8 = anova(anova_title='channel AF8',dataframe=power_AF8,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_TP10 = anova(anova_title='channel TP10',dataframe=power_TP10,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)
anova_power_mean = anova(anova_title='channel mean',dataframe=power_mean,anova_type=4,independent_variable=['group','task','time','freq'],dependent_variable='power',alphaAnova=alpha_anova,alphaPostHoc=alpha_posthoc)

# export anova results to pdf

def fn_print_pdf(df,pp): 
 total_rows, total_cols = df.shape

 rows_per_page = 30; # Number of rows per page
 rows_printed = 0
 page_number = 1
 while (total_rows >0):
    fig=plt.figure(figsize=(8.5, 11))
    plt.gca().axis('off')
    matplotlib_tab = pdplt.table(plt.gca(),df.iloc[rows_printed:rows_printed+rows_per_page],
        loc='upper center', colWidths=[0.15]*total_cols)
    #Tabular styling
    table_props=matplotlib_tab.properties()
    table_cells=table_props
    #for cell in table_cells:
    #    cell.set_height(0.024)
    #    cell.set_fontsize(12)
    # Header,Footer and Page Number
    fig.text(4.25/8.5, 10.5/11., "ANOVA & PostHoc Results", ha='center', fontsize=10)
    fig.text(4.25/8.5, 0.5/11., str(page_number), ha='center', fontsize=4)
    pp.savefig()
    plt.close()
    #Update variables 
    rows_printed += rows_per_page
    total_rows -= rows_per_page
    page_number+=1


pp = PdfPages('/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/music therapy/anova_results/channelMean.pdf')
fn_print_pdf(anova_power_mean[0],pp)
fn_print_pdf(anova_power_mean[1],pp)   
fn_print_pdf(anova_power_mean[2],pp)
fn_print_pdf(anova_power_mean[3],pp)
fn_print_pdf(anova_power_mean[4],pp)
fn_print_pdf(anova_power_mean[5],pp)
fn_print_pdf(anova_power_mean[6],pp)
fn_print_pdf(anova_power_mean[7],pp)
fn_print_pdf(anova_power_mean[8],pp)
fn_print_pdf(anova_power_mean[9],pp)
fn_print_pdf(anova_power_mean[10],pp)
fn_print_pdf(anova_power_mean[11],pp)
fn_print_pdf(anova_power_mean[12],pp)
fn_print_pdf(anova_power_mean[13],pp)
fn_print_pdf(anova_power_mean[14],pp)
fn_print_pdf(anova_power_mean[15],pp)
pp.close()

