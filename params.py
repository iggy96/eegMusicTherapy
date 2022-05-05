# SQL database parameters
hostName = "localhost"
userName = "root"
userPassword = "1A3g5m7t9n#" # IMPORTANT! Put your MySQL Terminal password here.
databaseName = 'music_therapy_eeg'
tableName_1 = 'MH_01_ES1_task1'
tableName_2 = 'MH_01_ES2_task1'
query_1 = ("% s % s"%('SELECT * FROM', tableName_1))
query_2 = ("% s % s"%('SELECT * FROM', tableName_2))


# eeg preprocessing parameters
tuneval = 2
fs = 256
collection_time = 120
fs_setting = 'resample'
plotTitles = ['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']
figSize = (8,8)
pltColor = ['r','g','b','k']
noChannels = 4
line = 60
Q = 30
lowcut = 0.1
highcut = 30
order = 4
win = 4*fs
window_size = 3072
step_size = int(window_size/4)
nfft = int(window_size/2)
noverlap = int(nfft/2)
brainwaves = dict(delta = [0.5,4],theta = [4,8],alpha = [8,13],beta = [13,32],gamma = [32,100])

# music therapy files
musicTherapyGroup_1 = [['MH_01_ES1_task1','MH_02_ES1_task1','MH_04_ES1_task1','MH_07_ES1_task1','MH_09_ES1_task1','MH_10_ES1_task1'],
                        ['MH_01_ES1_task2','MH_02_ES1_task2','MH_04_ES1_task2','MH_07_ES1_task2','MH_09_ES1_task2','MH_10_ES1_task2'],
                        ['MH_01_ES1_task3','MH_02_ES1_task3','MH_04_ES1_task3','MH_07_ES1_task3','MH_09_ES1_task3','MH_10_ES1_task3']]
musicTherapyGroup_2 = [['MH_01_ES2_task1','MH_02_ES2_task1','MH_04_ES2_task1','MH_07_ES2_task1','MH_09_ES2_task1','MH_10_ES2_task1'],
                        ['MH_01_ES2_task2','MH_02_ES2_task2','MH_04_ES2_task2','MH_07_ES2_task2','MH_09_ES2_task2','MH_10_ES2_task2'],
                        ['MH_01_ES2_task3','MH_02_ES2_task3','MH_04_ES2_task3','MH_07_ES2_task3','MH_09_ES2_task3','MH_10_ES2_task3']]

controlGroup_1 = [['MH_03_ES1_task1','MH_06_ES1_task1','MH_08_ES1_task1','MH_13_ES1_task1','MH_14_ES1_task1','MH_15_ES1_task1'],
                  ['MH_03_ES1_task2','MH_06_ES1_task2','MH_08_ES1_task2','MH_13_ES1_task2','MH_14_ES1_task2','MH_15_ES1_task2'],
                    ['MH_03_ES1_task3','MH_06_ES1_task3','MH_08_ES1_task3','MH_13_ES1_task3','MH_14_ES1_task3','MH_15_ES1_task3']]

controlGroup_2 = [['MH_03_ES2_task1','MH_06_ES2_task1','MH_08_ES2_task1','MH_13_ES2_task1','MH_14_ES2_task1','MH_15_ES2_task1'],
                    ['MH_03_ES2_task2','MH_06_ES2_task2','MH_08_ES2_task2','MH_13_ES2_task2','MH_14_ES2_task2','MH_15_ES2_task2'],
                    ['MH_03_ES2_task3','MH_06_ES2_task3','MH_08_ES2_task3','MH_13_ES2_task3','MH_14_ES2_task3','MH_15_ES2_task3']]