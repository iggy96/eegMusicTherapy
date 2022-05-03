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
brainwaves = dict(
        delta = [0.5,4],
        theta = [4,8],
        alpha = [8,13],
        beta = [13,32],
        gamma = [32,100]
)