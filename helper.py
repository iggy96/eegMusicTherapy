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


def sqlTableToDataframe(host_name,user_name,user_password,database_name,query):
    dbConnection = create_db_connection(host_name,user_name,user_password,database_name)
    sql_query = pd.read_sql_query (query,dbConnection)
    df = pd.DataFrame(sql_query)
    return df

def allSQLTableNames(hostName,userName,userPassword,databaseName):
    # Input: hostName,userName,userPassword,databaseName
    # Output: list of all table names in database
    db_connection = create_db_connection(hostName,userName,userPassword,databaseName)
    cursor = db_connection.cursor()
    cursor.execute("Show tables;")
    result = cursor.fetchall()
    result = [x[0] for x in result]
    return result


def multiSQLTablesToDataframes(hostName,userName,userPassword,databaseName,table_name):
    # Input: hostName,userName,userPassword,databaseName,table_name
    #        table_name is a list of all table names in database
    # Output: 2D array holding four channel tables
    def tableToDF(host_name,user_name,user_password,database_name,table_name):
        query = ("% s % s"%('SELECT * FROM', table_name))
        dbConnection = create_db_connection(host_name,user_name,user_password,database_name)
        sql_query = pd.read_sql_query (query,dbConnection)
        df = pd.DataFrame(sql_query)
        return df
    tables_ = []
    for i in range(len(table_name)):
        tables_.append(tableToDF(hostName,userName,userPassword,databaseName,table_name[i]))
    #tables_ = np.array(tables_,dtype=object)
    return tables_


def singleTransformToRawEEG(data,fs,collection_time,fs_setting):
    #   Inputs  :   data    - one dataframe of unfiltered EEG data
    #   upsampling is common for muse eeg if the custom setting is utilized
    #   fs = desired sampling frequency
    #   'constant':eeg signals generated at this rate is perfect
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


def multiTransformTableToRawEEG(data,fs,collection_time,fs_setting):
    #   Inputs  :   data    -multiple dataframes of unfiltered EEG data
    #   upsampling is common for muse eeg if the custom setting is utilized
    #   fs = desired sampling frequency
    #   'constant':eeg signals generated at this rate is perfect
    #   Outputs  :   rawEEG  - multiple 2D arrays of raw EEG data collapsed in a 3D array
    newRawEEG = []
    for i in range(len(data)):
        newRawEEG.append((singleTransformToRawEEG(data[i],fs,collection_time,fs_setting))[0])
    newRawEEG = np.dstack(newRawEEG)
    newRawEEG = newRawEEG.reshape(newRawEEG.shape[2],newRawEEG.shape[0],newRawEEG.shape[1])
    return newRawEEG


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

def rollingWindow(array,window_size,freq):
    #   Inputs  :   array    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #               window_size - size of window to be used for sliding
    #               freq   - step size for sliding window 
    #   Output  :   3D array (columns of array,no of windows,window size)
    def rolling_window(array, window_size,freq):
        shape = (array.shape[0] - window_size + 1, window_size)
        strides = (array.strides[0],) + array.strides
        rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
        return rolled[np.arange(0,shape[0],freq)]
    out_final = []
    for i in range(len(array.T)):
        out_final.append(rolling_window(array[:,i],window_size,freq))
    out_final = np.asarray(out_final).T
    out_final = out_final.transpose()
    return out_final

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

def averageBandPower(data,arrayType,fs,low,high,win):
    #  Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #              or data - 3D numpy array (d0 = channels, d1 = no of windows, d2 = length of windows) of unfiltered EEG data
    #              arrayType - '2D' or '3D'
    #              fs      - sampling rate of hardware (defaults to config)
    #              low     - lower limit in Hz for the brain wave
    #              high    - upper limit in Hz for the brain wave
    #              win     - size of window to be used for sliding
    #   Output  :   3D array (columns of array,no of windows,window size)
    def absPower(data,fs,low,high,win):                                                 
        freqs, psd = signal.welch(data,fs,nperseg=win)
        idx_freqBands = np.logical_and(freqs >= low, freqs <= high) 
        freq_res = freqs[1] - freqs[0]                                  
        freqBand_power = round(simps(psd[idx_freqBands],dx=freq_res),3)      
        return freqBand_power
    if arrayType=='2D':
        avgBandPower = []
        for i in range(len(data.T)):
            avgBandPower.append(absPower(data[:,i],fs,low,high,win))
        avgBandPower= np.array(avgBandPower).T
    elif arrayType=='3D':
        avgBandPower = []
        for i in range(len(data)):
            x = data[i,:,:]
            for i in range(len(x)):
                avgBandPower.append(absPower(x[i,:],fs,low,high,win))
        avgBandPower= np.array(avgBandPower)
        avgBandPower = avgBandPower.reshape(len(x),len(data))
    return avgBandPower


def spectogramPlot(data,fs,nfft,nOverlap,figsize,titles):
    #   Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #               fs      - sampling rate of hardware (defaults to config)
    #               nfft    - number of points to use in each block (defaults to config)
    #               nOverlap- number of points to overlap between blocks (defaults to config)
    #               figsize - size of figure (defaults to config)
    #               titles  - titles for each channel (defaults to config)
    y = data
    if len(y.T) % 2 != 0:
        nrows,ncols=1,int(len(y.T))
    elif len(y.T) % 2 == 0:
        nrows,ncols=2,int(len(y.T)/2)
    fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(figsize[0],figsize[1]))
    fig.suptitle('Spectogram')
    label= ["Power/Frequency"]
    for i, axs in enumerate(axs.flatten()):
        d, f, t, im = axs.specgram(data[:,i],NFFT=nfft,Fs=fs,noverlap=nOverlap)
        axs.set_title(titles[i])
        axs.set_ylim(0,50)
        axs.set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        axs.label_outer()
    fig.colorbar(im, ax=axs, shrink=0.9, aspect=10)
        


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


def pairedTTest(data1,data2,output,variableName,channelName,alpha=0.05):
    #   Inputs  :   data1   - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #               data2   - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #   Output  :   2D array (d0 = samples, d1 = channels) of paired t-test results
    def init_ttest(data1,data2,variableName,channelName):
        t_test = stats.ttest_rel(data1,data2)
        if output==True:
            if t_test[1] < alpha:
                if np.mean(data1)-np.mean(data2)<0:
                    print("for {} there is a significant difference (increase) at {} where the P-value = {}".format(variableName,channelName,round(t_test[1],3)))
                elif np.mean(data1)-np.mean(data2)>0:
                    print("for {} there is a significant difference (decrease) at {} where the P-value = {}".format(variableName,channelName,round(t_test[1],3)))
            else:
                print("for {} there is no significant difference at {} where the P-value = {}".format(variableName,channelName,round(t_test[1],3)))
        else:
            return t_test
        return t_test

    final_ttest = []
    for i in range(len(data1.T)):
        t_test = init_ttest(data1[:,i],data2[:,i],variableName,channelName[i])
        final_ttest.append(t_test)
    final_ttest = np.array(final_ttest).T
    return final_ttest