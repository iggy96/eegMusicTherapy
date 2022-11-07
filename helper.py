from statistics import mode
from libs import*

def singleTransformToRawEEG(data,fs,collection_time,fs_setting):
    #   Inputs  :   data    - one dataframe of unfiltered EEG data
    #   upsampling is common for muse eeg if the custom setting is utilized
    #   fs = desired sampling frequency
    #   'constant':eeg signals generated at this rate is perfect
    data = data.dropna()
    rawEEG = data
    if fs_setting == 'resample':
        rawEEG = signal.resample(rawEEG,fs*collection_time)
        t_len = len(rawEEG)
        period = (1.0/fs)
        time_s = np.arange(0, t_len * period, period)
    elif fs_setting == 'constant':
        rawEEG = rawEEG.to_numpy()
        t_len = len(rawEEG)
        time_s = np.linspace(start=0, stop=collection_time, num=len(rawEEG))
    if len(rawEEG) == int(collection_time*fs):
        rawEEG = rawEEG
    if len(rawEEG) > int(collection_time*fs):
        rawEEG = rawEEG[0:int(collection_time*fs)]
    if len(rawEEG) < int(collection_time*fs):
        l = len(rawEEG)
        while l < int(collection_time*fs):
            mean = np.mean(rawEEG,axis=0)
            mean = mean.reshape(1,len(mean))
            rawEEG = np.vstack((rawEEG,mean))
            if len(rawEEG) == int(collection_time*fs):
                break
    return rawEEG,time_s

def zipExtract(filenames,localDirectory,destDirectory,variableName,sFreq,data_collection_time,sampling_state):
    def zipExt(filename,localDirectory,destDirectory,variableName):
        zf = ZipFile(localDirectory+filename, 'r')
        path = destDirectory+variableName
        zf.extractall(path)
        zf.close()
    zipOutputs = []
    for filename in filenames:
        zipOutputs.append(zipExt(filename,localDirectory,destDirectory,variableName))
        #print(filename)
    files_dest = next(walk(destDirectory+variableName), (None, None, []))[2]
    files_dest = [f for f in files_dest if f.endswith('.csv')]
    resampled_data = []
    resampled_time = []
    for I in range(len(files_dest)):
        resampled_data.append(singleTransformToRawEEG((pd.read_csv(destDirectory+variableName+'/'+files_dest[I],low_memory=False))[['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']],sFreq,data_collection_time,sampling_state)[0])
        resampled_time.append(singleTransformToRawEEG((pd.read_csv(destDirectory+variableName+'/'+files_dest[I],low_memory=False))[['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']],sFreq,data_collection_time,sampling_state)[1])
    resampled_data = np.array(resampled_data)
    resampled_time = np.array(resampled_time)
    return files_dest,resampled_data,resampled_time

def avgBandPower(data,fs,low,high):
    #   Utilizes MultiTaper method to calculate the average power of a band
    #  Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #              fs      - sampling rate of hardware (defaults to config)
    #              low     - lower limit in Hz for the brain wave
    #              high    - upper limit in Hz for the brain wave
    #              win     - size of window to be used for sliding
    #   Output  :   3D array (columns of array,no of windows,window size)
    def absPower(data,fs,low,high): 
        psd, freqs = psd_array_multitaper(data, fs, adaptive=True,
                                            normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)  
        return bp

    avg_BandPower = []
    for i in range(len(data.T)):
        avg_BandPower.append(absPower(data[:,i],fs,low,high))
    avg_BandPower= np.array(avg_BandPower).T
    return avg_BandPower

class filters:
    # filters for EEG data
    # filtering order: adaptive filter -> notch filter -> bandpass filter (or lowpass filter, highpass filter)

    def notch(self,data,fs):
        """
        Function detects the exact frequency location of the line noise between 59 and 62Hz,then notches the PSD
            Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
                        cut     - frequency to be notched (defaults to config)
                        fs      - sampling rate of hardware (defaults to config)
                        Q       - Quality Factor (defaults to 30) that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.   
            Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
            NOTES   :   
            Todo    : report testing filter characteristics
        """
        def fn_1(data,fs):
            win = 4 * fs
            freqs,psd = signal.welch(data,fs,nperseg=win)
            noiseFreq = np.asarray(np.where(np.logical_and(freqs>=59,freqs<65))).T
            noisePSD = psd[noiseFreq[:,0]]
            max_noisePSD = np.max(noisePSD)
            idx_max_noisePSD = np.where(psd==max_noisePSD)[0]
            max_noiseFreq = freqs[idx_max_noisePSD]
            return max_noiseFreq

        def fn_2(data,line,fs,Q=30):
            cut = line
            w0 = cut/(fs/2)
            b, a = signal.iirnotch(w0, Q)
            y = signal.filtfilt(b, a, data, axis=0)
            return y

        lineNoise = []
        for i in range(len(data.T)):
            lineNoise.append(fn_1(data[:,i],fs))
        lineNoise = np.asarray(lineNoise)

        notched = []
        for i in range(len(data.T)):
            notched.append(fn_2(data[:,i],lineNoise[i],fs))
        notched = np.asarray(notched).T
        return notched

    def butterBandPass(self,data,lowcut,highcut,fs,order=4):
        #   Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
        #               low     - lower limit in Hz for the bandpass filter (defaults to config)
        #               high    - upper limit in Hz for the bandpass filter (defaults to config)
        #               fs      - sampling rate of hardware (defaults to config)
        #               order   - the order of the filter (defaults to 4)  
        #   Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
        #   NOTES   :   
        #   Todo    : report testing filter characteristics
        # data: eeg data (samples, channels)
        # some channels might be eog channels
        low_n = lowcut
        high_n = highcut
        sos = butter(order, [low_n, high_n], btype="bandpass", analog=False, output="sos",fs=fs)
        y = sosfiltfilt(sos, data, axis=0)
        return y

    def adaptive(self,eegData,eogData,nKernel=5, forgetF=0.995,  startSample=0, p = False):
        """
           Inputs:
           eegData - A matrix containing the EEG data to be filtered here each channel is a column in the matrix, and time
           starts at the top row of the matrix. i.e. size(data) = [numSamples,numChannels]
           eogData - A matrix containing the EOG data to be used in the adaptive filter
           startSample - the number of samples to skip for the calculation (i.e. to avoid the transient)
           p - plot AF response (default false)
           nKernel = Dimension of the kernel for the adaptive filter
           Outputs:
           cleanData - A matrix of the same size as "eegdata", now containing EOG-corrected EEG data.
           Adapted from He, Ping, G. Wilson, and C. Russell. "Removal of ocular artifacts from electro-encephalogram by adaptive filtering." Medical and biological engineering and computing 42.3 (2004): 407-412.
        """
        #   reshape eog array if necessary
        if len(eogData.shape) == 1:
            eogData = np.reshape(eogData, (eogData.shape[0], 1))
        # initialise Recursive Least Squares (RLS) filter state
        nEOG = eogData.shape[1]
        nEEG = eegData.shape[1]
        hist = np.zeros((nEOG, nKernel))
        R_n = np.identity(nEOG * nKernel) / 0.01
        H_n = np.zeros((nEOG * nKernel, nEEG))
        X = np.hstack((eegData, eogData)).T          # sort EEG and EOG channels, then transpose into row variables
        eegIndex = np.arange(nEEG)                              # index of EEG channels within X
        eogIndex = np.arange(nEOG) + eegIndex[-1] + 1           # index of EOG channels within X
        for n in range(startSample, X.shape[1]):
            hist = np.hstack((hist[:, 1:], X[eogIndex, n].reshape((nEOG, 1))))  # update the EOG history by feeding in a new sample
            tmp = hist.T                                                        # make it a column variable again (?)
            r_n = np.vstack(np.hsplit(tmp, tmp.shape[-1]))
            K_n = np.dot(R_n, r_n) / (forgetF + np.dot(np.dot(r_n.T, R_n), r_n))                                           # Eq. 25
            R_n = np.dot(np.power(forgetF, -1),R_n) - np.dot(np.dot(np.dot(np.power(forgetF, -1), K_n), r_n.T), R_n)       #Update R_n
            s_n = X[eegIndex, n].reshape((nEEG, 1))                   #get EEG signal and make sure it's a 1D column array
            e_nn = s_n - np.dot(r_n.T, H_n).T  #Eq. 27
            H_n = H_n + np.dot(K_n, e_nn.T)
            e_n = s_n - np.dot(r_n.T, H_n).T
            X[eegIndex, n] = np.squeeze(e_n)
        cleanData = X[eegIndex, :].T
        return cleanData

    def butter_lowpass(self,data,cutoff,fs,order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.lfilter(b, a, data)
        return y

    def butter_highpass(self,data,cutoff,fs,order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        y = signal.filtfilt(b, a, data)
        return y
       
def slidingWindow(data_2D,timing,window_size,step):
    """
    Inputs:
    1. data_2D - 2D numpy array (d0=samples, d1=channels) of data
    2. timing_array - 1D numpy array (d0 = samples) of timing data
    3. len(data_array) == len(timing_array)
    4. window_size - number of samples to use in each window in seconds e.g. 1 is 1 second
    5. step_size - the step size in seconds e.g.0.5 is 0.5 seconds 

    Outputs:    
    1. data_windows - 3D numpy array (d0=channels, d0=windows, d1=window size) of data
    """

    def params(data_1D,timing_array,window_size,step_size):
        idx_winsize = np.where(timing_array == window_size)[0][0]
        idx_stepsize = np.where(timing_array == step_size)[0][0]
        frame_len, hop_len = idx_winsize,idx_stepsize
        frames = librosa.util.frame(data_1D, frame_length=frame_len, hop_length=hop_len)
        windowed_frames = (np.hanning(frame_len).reshape(-1, 1)*frames).T
        return windowed_frames
    out_final = []
    for i in range(len(data_2D.T)):
        out_final.append(params(data_2D[:,i],timing,window_size,step))
    out_final = np.asarray(out_final).T
    out_final = out_final.transpose()
    return out_final

def plots(data,time_s,fs,figsize,subTitles,title,tickRange,timeFrequencyDomainPlots=False,frequencyDomainPlots=False,timeDomainPlots=False):
    if timeFrequencyDomainPlots:
        eeg = data
        sr = fs
        WinLength = 2     #int(0.5*sr) 
        step = 256      #int(0.025*sr) 
        myparams = dict(nperseg = 1024, noverlap = 512, scaling='density', return_onesided=True, mode='psd')
        f_1, nseg_1, Sxx_1 = signal.spectrogram(x = eeg[:,0], fs=sr, **myparams)
        f_2, nseg_2, Sxx_2 = signal.spectrogram(x = eeg[:,1], fs=sr, **myparams)
        f_3, nseg_3, Sxx_3 = signal.spectrogram(x = eeg[:,2], fs=sr, **myparams)
        f_4, nseg_4, Sxx_4 = signal.spectrogram(x = eeg[:,3], fs=sr, **myparams)

        fig, ax = plt.subplots(2,4, figsize = figsize, constrained_layout=True)
        fig.suptitle(title)
        ax[0,0].plot(time_s, eeg[:,0], lw = 1, color='C0')
        ax[0,1].plot(time_s, eeg[:,1], lw = 1, color='C1')
        ax[0,2].plot(time_s, eeg[:,2], lw = 1, color='C2')
        ax[0,3].plot(time_s, eeg[:,3], lw = 1, color='C3')
        ax[0,0].set_ylabel('Amplitude ($\mu V$)')
        ax[0,1].set_ylabel('Amplitude ($\mu V$)')
        ax[0,2].set_ylabel('Amplitude ($\mu V$)')
        ax[0,3].set_ylabel('Amplitude ($\mu V$)')
        ax[0,0].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[0,1].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[0,2].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[0,3].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[0,0].set_title(subTitles[0])
        ax[0,1].set_title(subTitles[1])
        ax[0,2].set_title(subTitles[2])
        ax[0,3].set_title(subTitles[3])
        ax[1,0].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[1,0].set_ylim(0,45)
        ax[1,1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[1,1].set_ylim(0,45)
        ax[1,2].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[1,2].set_ylim(0,45)
        ax[1,3].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[1,3].set_ylim(0,45)
        X1,X2,X3,X4 = nseg_1,nseg_2,nseg_3,nseg_4
        Y1,Y2,Y3,Y4 = f_1,f_2,f_3,f_4
        Z1,Z2,Z3,Z4 = Sxx_1,Sxx_2,Sxx_3,Sxx_4
        levels = 45
        spectrum = ax[1,0].contourf(X1,Y1,Z1,levels, cmap='jet')
        spectrum = ax[1,1].contourf(X2,Y2,Z2,levels, cmap='jet')
        spectrum = ax[1,2].contourf(X3,Y3,Z3,levels, cmap='jet')
        spectrum = ax[1,3].contourf(X4,Y4,Z4,levels, cmap='jet')
        ax[1,0].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[1,1].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[1,2].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        ax[1,3].set_xticks(np.arange(tickRange[0],tickRange[1],10))
        cbar = plt.colorbar(spectrum)#, boundaries=np.linspace(0,1,5))
        cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
    elif frequencyDomainPlots:
        eeg = data
        sr = fs
        WinLength = int(0.5*sr) 
        step = int(0.025*sr) 
        myparams = dict(nperseg = WinLength, noverlap = WinLength-step, return_onesided=True, mode='magnitude')
        f_1, nseg_1, Sxx_1 = signal.spectrogram(x = eeg[:,0], fs=sr, **myparams)
        f_2, nseg_2, Sxx_2 = signal.spectrogram(x = eeg[:,1], fs=sr, **myparams)
        f_3, nseg_3, Sxx_3 = signal.spectrogram(x = eeg[:,2], fs=sr, **myparams)
        f_4, nseg_4, Sxx_4 = signal.spectrogram(x = eeg[:,3], fs=sr, **myparams)
        fig, ax = plt.subplots(1,4, figsize=(figsize[0],figsize[1]), constrained_layout=True)
        fig.suptitle(title)
        ax[0].set_title(subTitles[0])
        ax[1].set_title(subTitles[1])
        ax[2].set_title(subTitles[2])
        ax[3].set_title(subTitles[3])
        ax[0].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[2].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[3].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        X1,X2,X3,X4 = nseg_1,nseg_2,nseg_3,nseg_4
        Y1,Y2,Y3,Y4 = f_1,f_2,f_3,f_4
        Z1,Z2,Z3,Z4 = Sxx_1,Sxx_2,Sxx_3,Sxx_4
        levels = 45
        spectrum = ax[0].contourf(X1,Y1,Z1,levels, cmap='jet')#,'linecolor','none')
        spectrum = ax[1].contourf(X2,Y2,Z2,levels, cmap='jet')#,'linecolor','none')
        spectrum = ax[2].contourf(X3,Y3,Z3,levels, cmap='jet')#,'linecolor','none')
        spectrum = ax[3].contourf(X4,Y4,Z4,levels, cmap='jet')#,'linecolor','none')
        cbar = plt.colorbar(spectrum)#, boundaries=np.linspace(0,1,5))
        cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
    elif timeDomainPlots:
        eeg = data
        sr = fs
        fig, ax = plt.subplots(1,4, figsize=(figsize[0],figsize[1]), constrained_layout=True)
        fig.suptitle(title)
        ax[0].plot(time_s, eeg[:,0], lw = 1, color='C0')
        ax[1].plot(time_s, eeg[:,1], lw = 1, color='C1')
        ax[2].plot(time_s, eeg[:,2], lw = 1, color='C2')
        ax[3].plot(time_s, eeg[:,3], lw = 1, color='C3')
        ax[0].set_ylabel('Amplitude ($\mu V$)')
        ax[1].set_ylabel('Amplitude ($\mu V$)')
        ax[2].set_ylabel('Amplitude ($\mu V$)')
        ax[3].set_ylabel('Amplitude ($\mu V$)')
        ax[0].set_title(subTitles[0])
        ax[1].set_title(subTitles[1])
        ax[2].set_title(subTitles[2])
        ax[3].set_title(subTitles[3])
        ax[0].set(xlabel='Time (s)')
        ax[1].set(xlabel='Time (s)')
        ax[2].set(xlabel='Time (s)')
        ax[3].set(xlabel='Time (s)')

def psdPlots(data,fs,titles):
# Define window length (4 seconds)
    win = 4 * fs
    freqs_1,psd_1 = signal.welch(data[:,0],fs,nperseg=win)
    freqs_2,psd_2 = signal.welch(data[:,1],fs,nperseg=win)
    freqs_3,psd_3 = signal.welch(data[:,2],fs,nperseg=win)
    freqs_4,psd_4 = signal.welch(data[:,3],fs,nperseg=win)
    fig, axs = plt.subplots(2,2,figsize=(15,8))
    axs[0, 0].plot(freqs_1,psd_1)
    axs[0, 0].set_title(titles[0])
    axs[0, 1].plot(freqs_2,psd_2, 'tab:orange')
    axs[0, 1].set_title(titles[1])
    axs[1, 0].plot(freqs_3,psd_3, 'tab:green')
    axs[1, 0].set_title(titles[2])
    axs[1, 1].plot(freqs_4,psd_4, 'tab:red')
    axs[1, 1].set_title(titles[3])
    for ax in axs.flat:
        ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB/Hz)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
def bandPowerPlots(x,y,figsize,subTitles,title,label):
    #   Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #               fs      - sampling rate of hardware (defaults to config)
    #               nfft    - number of points to use in each block (defaults to config)
    #               nOverlap- number of points to overlap between blocks (defaults to config)
    #               figsize - size of figure (defaults to config)
    #               titles  - titles for each channel (defaults to config)
    if len(y.T) % 2 != 0:
        nrows,ncols=1,int(len(y))
    elif len(y.T) % 2 == 0:
        nrows,ncols=2,int(len(y)/2)
        fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(figsize[0],figsize[1]))
        fig.suptitle(title)
        for i, axs in enumerate(axs.flatten()):
            axs.plot(x,y[i,0],c='b',marker="^",ls='--',label=label[0],fillstyle='none')
            axs.plot(x,y[i,1],c='g',marker=(8,2,0),ls='--',label=label[1])
            axs.plot(x,y[i,2],c='r',marker="v",ls='-',label=label[2])
            axs.plot(x,y[i,3],c='m',marker="o",ls='--',label=label[3],fillstyle='none')
            axs.set_title(subTitles[i])
            axs.set(xlabel='Channels', ylabel='Average Band Power')
            axs.label_outer()
            axs.legend(loc=2)

def psd(data,fs,data_1D=False,data_2D=False,data_3D=False):
    """
    Inputs: data - 1D, 2D or 3D numpy array
                    1D - single channel
                    2D - (samples,channels)
                    3D - (files,samples,channels)
            fs - sampling frequency
            data_1D - boolean, True if data is 1D
            data_2D - boolean, True if data is 2D
            data_3D - boolean, True if data is 3D
    Outputs: psd - 1D, 2D or 3D numpy array
    """
    def params_1D(dataIN,fs):
        psd, freqs = psd_array_multitaper(dataIN, fs, adaptive=True,
                                            normalization='full', verbose=0)
        return freqs,psd
    def params_2D(dataIN,fs):
        freqs,psd = [],[]
        for i in range(len(dataIN.T)):
            f,p = params_1D(dataIN[:,i],fs)
            freqs.append(f)
            psd.append(p)
        psd = np.array(psd)
        freqs = np.array(freqs)
        return freqs,psd

    if data_1D:
        freqs,psd = params_1D(data,fs)
    if data_2D:
        freqs,psd = params_2D(data,fs)
    if data_3D:
        freqs,psd = [],[]
        for i in range(len(data)):
            f,p = params_2D(data[i,:,:],fs)
            freqs.append(f)
            psd.append(p)
        freqs = np.array(freqs)
        psd = np.array(psd)
    return freqs,psd

def ar_maximumgradient(input_2D,threshold_value,timearray,len_window,step_size,choice_numwindows,channels):
    def params(data1D,threshold,time_array,winsize,step,numwindows,chan_title):
        def maxgrad2D(data2D):
            diff_succ_val = []
            for i in range(data2D.shape[0]):
                diff_succ_val.append(np.max(np.diff(data2D[i,:])))
            return np.array(diff_succ_val)

        def slidingwindow(data_1D,timing_array,window_size,step_size):
            idx_winsize = np.where(timing_array == window_size)[0][0]
            idx_stepsize = np.where(timing_array == step_size)[0][0]
            frame_len, hop_len = idx_winsize,idx_stepsize
            frames = librosa.util.frame(data_1D, frame_length=frame_len, hop_length=hop_len)
            windowed_frames = (np.hanning(frame_len).reshape(-1, 1)*frames).T
            return windowed_frames

        wins2D = slidingwindow(data1D,time_array,winsize,step)
        len_windows = wins2D.shape[0]
        maxgrads = maxgrad2D(wins2D)
        highest_maxgrads = np.amax(maxgrads)
        lowest_maxgrads = np.amin(maxgrads)
        print('maximum gradient value of worst segment for %s is %f' % (chan_title,highest_maxgrads))
        print('minimum gradient value of best segment for %s is %f' % (chan_title,lowest_maxgrads))
        idxs_badMaxGrads = np.where(maxgrads > threshold)[0]
        idxs_cleanMaxGrads = np.where(maxgrads <= threshold)[0]
        idx_highBadMaxGrad = np.where(maxgrads == highest_maxgrads)[0][0]
        idx_highCleanMaxGrad = np.where(maxgrads == lowest_maxgrads)[0][0]
        bad_dataset = wins2D[idxs_badMaxGrads,:]
        clean_dataset = wins2D[idxs_cleanMaxGrads,:]
        print('number of non-artifactual segements for %s is %d' % (chan_title,len(idxs_cleanMaxGrads)))
        print('number of artifactual segements for %s is %d' % (chan_title,len(idxs_badMaxGrads)))
        remain_windows = len_windows - len(clean_dataset)
        clean_dataset = np.concatenate((clean_dataset,np.full((remain_windows,wins2D.shape[1]),np.nan)),axis=0)
        clean_dataset = clean_dataset[0:numwindows,:]
        worst_data, best_data = wins2D[idx_highBadMaxGrad,:], wins2D[idx_highCleanMaxGrad,:]
        fig,ax = plt.subplots(1,2,figsize=(10,3))
        fig.suptitle(chan_title)
        ax[0].plot(worst_data,color='red')
        ax[0].set_title('Worst data')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(best_data,color='green')
        ax[1].set_title('Best data')
        ax[1].set_ylabel('Amplitude')
        plt.show()
        return clean_dataset

    output_2D = []
    for i in range(input_2D.shape[1]):
        output_2D.append(params(input_2D[:,i],threshold_value,timearray,len_window,step_size,choice_numwindows,channels[i]))
    return np.array(output_2D)