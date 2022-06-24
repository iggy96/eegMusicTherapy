# SQL database parameters
hostName = "localhost"
userName = "root"
userPassword = "1A3g5m7t9n#" # IMPORTANT! Put your MySQL Terminal password here.
databaseName = 'music_therapy_eeg'


# eeg preprocessing parameters
tuneval = 3
wavelet = ['sym3']
fs = 256
collection_time = 120
fs_setting = 'resample'
plotTitles = ['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']
figSize = (6,6)
pltColor = ['r','g','b','k']
noChannels = 4
line = 60
Q = 30
lowcut = 0.1
highcut = 100
order = 4
win = fs # 4*fs = 1024
# 1280 (5-seconds) 2560 (10-seconds) 3072 (12-seconds) 1024 (4-seconds) 256 (1-second) default: 2560&1280
# step size = window size / 2
window_size = 768
step_size = 256
nfft = 256
noverlap = 128
brainwaves = dict(delta = [0.5,4],theta = [4,8],alpha = [8,13],beta = [13,32],gamma = [32,100])

# music therapy files
musicTherapyGroup_1 = [['MH_01_ES1_task1','MH_02_ES1_task1','MH_04_ES1_task1','MH_07_ES1_task1','MH_09_ES1_task1','MH_10_ES1_task1'],
                        ['MH_01_ES1_task2','MH_02_ES1_task2','MH_04_ES1_task2','MH_07_ES1_task2','MH_09_ES1_task2','MH_10_ES1_task2'],
                        ['MH_01_ES1_task3','MH_02_ES1_task3','MH_04_ES1_task3','MH_07_ES1_task3','MH_09_ES1_task3','MH_10_ES1_task3']]
musicTherapyGroup_2 = [['MH_01_ES2_task1','MH_02_ES2_task1','MH_04_ES2_task1','MH_07_ES2_task1','MH_09_ES2_task1','MH_10_ES2_task1'],
                        ['MH_01_ES2_task2','MH_02_ES2_task2','MH_04_ES2_task2','MH_07_ES2_task2','MH_09_ES2_task2','MH_10_ES2_task2'],
                        ['MH_01_ES2_task3','MH_02_ES2_task3','MH_04_ES2_task3','MH_07_ES2_task3','MH_09_ES2_task3','MH_10_ES2_task3']]

controlGroup_1 = [['MH_03_ES1_task1','MH_06_ES1_task1','MH_13_ES1_task1','MH_14_ES1_task1','MH_15_ES1_task1'],
                  ['MH_03_ES1_task2','MH_06_ES1_task2','MH_13_ES1_task2','MH_14_ES1_task2','MH_15_ES1_task2'],
                    ['MH_03_ES1_task3','MH_06_ES1_task3','MH_13_ES1_task3','MH_14_ES1_task3','MH_15_ES1_task3']]

controlGroup_2 = [['MH_03_ES2_task1','MH_06_ES2_task1','MH_13_ES2_task1','MH_14_ES2_task1','MH_15_ES2_task1'],
                    ['MH_03_ES2_task2','MH_06_ES2_task2','MH_13_ES2_task2','MH_14_ES2_task2','MH_15_ES2_task2'],
                    ['MH_03_ES2_task3','MH_06_ES2_task3','MH_13_ES2_task3','MH_14_ES2_task3','MH_15_ES2_task3']]
studygroups = ['Music 1','Music 2','Control 1','Control 2']