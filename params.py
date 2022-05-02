# SQL database parameters
hostName = "localhost"
userName = "root"
userPassword = "1A3g5m7t9n#" # IMPORTANT! Put your MySQL Terminal password here.
databaseName = 'music_therapy_eeg'
tableName = 'MH_01_ES1_task1'
query = ("% s % s"%('SELECT * FROM', tableName))

# eeg preprocessing parameters
tuneval = 2
fs = 256
collection_time = 120
fs_setting = 'resample'
plotTitles = ['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']
figSize = (8,8)
pltColor = ['r','g','b','k']
line = 60
Q = 30
lowcut = 0.1
highcut = 30
order = 4
win = 4*fs