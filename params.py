fs = 256
collection_time = 120
fs_setting = 'resample'
channels = ['TP9','AF7','AF8','TP10']
noChannels = 4
line = 60
Q = 30
lowcut = 0.1
highcut = 100
order = 4
win = fs # 4*fs = 1024
# 1280 (5-seconds) 2560 (10-seconds) 3072 (12-seconds) 1024 (4-seconds) 256 (1-second) default: 2560&1280
# step size = window size / 2
window_size = 15360
step_size = 15360
nfft = 1024
noverlap = 512
brainwaves = dict(delta = [0.5,4],theta = [4,7],alpha = [8,12],beta = [12.5,30],gamma = [30.5,80])

# statistics parameters
alpha_anova = 0.05
alpha_posthoc = 0.01