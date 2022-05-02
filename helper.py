from libs import*

def create_db_connection(host_name, user_name, user_password, database_name):
    # used to establish created database connection
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=database_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


def df_to_table_query(connection, query):
    # converts all sql queries user writes in python strings 
    # and passes it to cursor.execute() method to execute them
    # on the MYSQL server 
    cursor = connection.cursor()
    try:
        df = pd.read_sql(query, con=connection)
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")
    return df

def transformToRawEEG(data,fs,collection_time,fs_setting):
    # upsampling is common for muse eeg if the custom setting is utilized
    # fs = desired sampling frequency
    # 'constant':eeg signals generated at this rate is perfect
    data = data.dropna()
    rawEEG = data
    t_len = len(rawEEG)
    period = (1.0/fs)
    time_s = np.arange(0, t_len * period, period)
    if fs_setting == 'resample':
        rawEEG = signal.resample(rawEEG,fs*collection_time)
        t_len = len(rawEEG)
        period = (1.0/fs)
        time_s = np.arange(0, t_len * period, period)
    elif fs_setting == 'constant':
        pass
    return rawEEG,time_s

def plots(x,y,titles,figsize,pltclr):
    x_lim = [x[0],x[-1]]
    if len(y.T) % 2 != 0:
        nrows,ncols=1,int(len(y.T))
    elif len(y.T) % 2 == 0:
        nrows,ncols=2,int(len(y.T)/2)
    fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(figsize[0],figsize[1]))
    for i, axs in enumerate(axs.flatten()):
        axs.plot(x, y[:,i], color=pltclr[i])
        axs.set_title(titles[i])
        axs.set_ylim([np.max(y[:,i])+1000,np.min(y[:,i])-1000])
        axs.set_xlim([x_lim[0],x_lim[1]])
        axs.set(xlabel='Time (s)', ylabel='Amplitude (uV)')
        axs.label_outer()


class filters:
    # filters for EEG data
    # filtering order: adaptive filter -> notch filter -> bandpass filter (or lowpass filter, highpass filter)
    def notch(self,data,line,fs,Q):
        #   Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
        #               cut     - frequency to be notched (defaults to config)
        #               fs      - sampling rate of hardware (defaults to config)
        #               Q       - Quality Factor (defaults to 30) that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.   
        #   Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
        #   NOTES   :   
        #   Todo    : report testing filter characteristics
        cut = line
        w0 = cut/(fs/2)
        b, a = signal.iirnotch(w0, Q)
        y = signal.filtfilt(b, a, data, axis=0)
        return y

    def butterBandPass(self,data,lowcut,highcut,fs,order):
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


def rolling_window(array, window_size,freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]


def customICA(input,tuneVal):
    # algorithm uses ICA to split the eeg into components to allow for easy extraction of ocular artefacts
    # Using the idea of the ASR algorithm, the tuneVal is the number of STD above the mean
    # eeg elements that fall into the class above the mean are artefactsa and replaced with 
    # the mean value of non-artefact elements
    val = tuneVal
    ica = FastICA(len(input.T),max_iter=200,tol=0.0001,random_state=0)
    components = ica.fit_transform(input)  

    def fixICAcomps(datae,val):
        mean = np.mean(datae, axis=0)
        sd = np.std(datae, axis=0)
        final_list = [x for x in datae if (x > mean - val * sd)]
        final_list = [x for x in final_list if (x < mean + val * sd)]
        final_list = np.asarray(final_list)

        def returnNotMatches(a, b):
            a = set(a)
            b = set(b)
            return [list(b - a), list(a - b)]

        rejected = np.asarray(returnNotMatches(datae,final_list)[1])
        rejected = rejected.reshape(len(rejected),1)
        idx = np.where(datae==rejected)[1]
        idx = idx.tolist()
        #idx = idx.reshape(len(idx),1)
        #datae = datae.reshape(len(datae),1)
        replace_vals = [np.mean(final_list)] * len(idx)
        fixedComps = [replace_vals[idx.index(i)] if i in idx else datae[i] for i in range(len(datae))]
        return fixedComps

    out_final = []
    for i in range(len(components.T)):
        out_init = fixICAcomps(components[:,i],val)
        out_final.append(out_init)
    out_final = np.asarray(out_final).T
    x_restored = ica.inverse_transform(out_final)
    return x_restored

def averageBandPower(data,fs,low,high,win):
    def absPower(data,fs,low,high,win):                                                 
        freqs, psd = signal.welch(data,fs,nperseg=win)
        idx_freqBands = np.logical_and(freqs >= low, freqs <= high) 
        freq_res = freqs[1] - freqs[0]                                  
        freqBand_power = round(simps(psd[idx_freqBands],dx=freq_res),3)      
        return freqBand_power
    avgBandPower = []
    for i in range(len(data.T)):
        avgBandPower.append(absPower(data[:,i],fs,low,high,win))
    avgBandPower= np.array(avgBandPower).T
    return avgBandPower


def museEEGPipeline(data,fs,collection_time,fs_setting,tuneVal,line,Q,lowcut,highcut,order,fft_low,fft_high,win):
    rawEEG = (transformToRawEEG(data,fs,collection_time,fs_setting))[0]
    ica_data = customICA(rawEEG,tuneVal)
    noc=filters()
    notch_data = noc.notch(ica_data,line,fs,Q)
    bp = filters()
    bpData = bp.butterBandPass(notch_data,lowcut,highcut,fs,order)
    # compute average band power for each channel
    chanAvgBandPower = averageBandPower(bpData,fs,fft_low,fft_high,win)
    return chanAvgBandPower