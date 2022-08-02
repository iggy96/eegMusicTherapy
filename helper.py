from statistics import mode
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
    time_s = singleTransformToRawEEG(data[0],fs,collection_time,fs_setting)[1]
    return newRawEEG,time_s

def multiChannelDWT(data,time_array,wavelet):
    """
        Inputs  :   data    -multiple dataframes of unfiltered EEG data
        upsampling is common for muse eeg if the custom setting is utilized
        fs = desired sampling frequency
        'constant':eeg signals generated at this rate is perfect
        Outputs  :   rawEEG  - multiple 2D arrays of raw EEG data collapsed in a 3D array
    """
    dwtEEG = []
    for i in range(len(data.T)):
        dwtEEG.append(singleChannelDWT(data[:,i],time_array,wavelet))
    dwtEEG = np.array(dwtEEG).T
    dwtEEG = dwtEEG.reshape(dwtEEG.shape[1],dwtEEG.shape[2])
    return dwtEEG

def plot_averageBandPower(groupA,groupB,x_labels,groups,figure_size,plot_title):
    mean_groupA_11 = np.mean(groupA[0][0],axis=0)
    mean_groupA_12 = np.mean(groupA[0][1],axis=0)
    mean_groupA_21 = np.mean(groupA[1][0],axis=0)
    mean_groupA_22 = np.mean(groupA[1][1],axis=0)
    mean_groupB_11 = np.mean(groupB[0][0],axis=0)
    mean_groupB_12 = np.mean(groupB[0][1],axis=0)
    mean_groupB_21 = np.mean(groupB[1][0],axis=0)
    mean_groupB_22 = np.mean(groupB[1][1],axis=0)
    std_groupA_11 = np.std(groupA[0][0],axis=0)
    std_groupA_12 = np.std(groupA[0][1],axis=0)
    std_groupA_21 = np.std(groupA[1][0],axis=0)
    std_groupA_22 = np.std(groupA[1][1],axis=0)
    std_groupB_11 = np.std(groupB[0][0],axis=0)
    std_groupB_12 = np.std(groupB[0][1],axis=0)
    std_groupB_21 = np.std(groupB[1][0],axis=0)
    std_groupB_22 = np.std(groupB[1][1],axis=0)

    data_mean = np.array([[mean_groupA_11[0],mean_groupA_21[0],mean_groupB_11[0],mean_groupB_21[0]],
                            [mean_groupA_11[1],mean_groupA_21[1],mean_groupB_11[1],mean_groupB_21[1]],
                            [mean_groupA_11[2],mean_groupA_21[2],mean_groupB_11[2],mean_groupB_21[2]],
                            [mean_groupA_11[3],mean_groupA_21[3],mean_groupB_11[3],mean_groupB_21[3]],
                            [mean_groupA_12[0],mean_groupA_22[0],mean_groupB_12[0],mean_groupB_22[0]],
                            [mean_groupA_12[1],mean_groupA_22[1],mean_groupB_12[1],mean_groupB_22[1]],
                            [mean_groupA_12[2],mean_groupA_22[2],mean_groupB_12[2],mean_groupB_22[2]],
                            [mean_groupA_12[3],mean_groupA_22[3],mean_groupB_12[3],mean_groupB_22[3]]])

    data_std = np.array([[std_groupA_11[0],std_groupA_21[0],std_groupB_11[0],std_groupB_21[0]],
                            [std_groupA_11[1],std_groupA_21[1],std_groupB_11[1],std_groupB_21[1]],
                            [std_groupA_11[2],std_groupA_21[2],std_groupB_11[2],std_groupB_21[2]],
                            [std_groupA_11[3],std_groupA_21[3],std_groupB_11[3],std_groupB_21[3]],
                            [std_groupA_12[0],std_groupA_22[0],std_groupB_12[0],std_groupB_22[0]],
                            [std_groupA_12[1],std_groupA_22[1],std_groupB_12[1],std_groupB_22[1]],
                            [std_groupA_12[2],std_groupA_22[2],std_groupB_12[2],std_groupB_22[2]],
                            [std_groupA_12[3],std_groupA_22[3],std_groupB_12[3],std_groupB_22[3]]])



    length = len(data_mean)

    # Set plot parameters
    fig, ax = plt.subplots(figsize=(figure_size[0],figure_size[1]))
    width = 0.2 # width of bar
    x = np.arange(length)

    ax.bar(x, data_mean[:,0], width, color='#90EE90', label=groups[0], yerr=data_std[:,0])
    ax.bar(x + width, data_mean[:,1], width, color='#013220', label=groups[1], yerr=data_std[:,1])
    ax.bar(x + (2 * width), data_mean[:,2], width, color='#C4A484', label=groups[2], yerr=data_std[:,2])
    ax.bar(x + (3 * width), data_mean[:,3], width, color='#654321', label=groups[3], yerr=data_std[:,3])

    ax.set_ylabel('Average Band Power')
    #ax.set_ylim(0,1000)
    #ax.set_yticks(np.arange(0,1000,100))
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(x_labels)
    ax.set_title(plot_title)
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    fig.tight_layout()
    plt.show()
    pass

def plots(x,y,titles,pltclr):
    x_lim = [x[0],x[-1]]
    if len(y.T) % 2 != 0:
        nrows,ncols=1,int(len(y.T))
    elif len(y.T) % 2 == 0:
        nrows,ncols=2,int(len(y.T)/2)
    fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15,8))
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
    def notch(self,data,line,fs,iterations):
        #   Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
        #               cut     - frequency to be notched (defaults to config)
        #               fs      - sampling rate of hardware (defaults to config)
        #               Q       - Quality Factor (defaults to 30) that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.   
        #   Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
        #   NOTES   :   
        #   Todo    : report testing filter characteristics
        def fn(data,line,fs,Q=30):
            cut = line
            w0 = cut/(fs/2)
            b, a = signal.iirnotch(w0, Q)
            y = signal.filtfilt(b, a, data, axis=0)
            return y
        output = fn(data,line,fs)
        for i in range(iterations):
            output = fn(output,line,fs)
        return output

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
    # eeg elements that fall into the class above the mean are artefacts and replaced with 
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
        freqBand_power = simps(psd[idx_freqBands],dx=freq_res)  
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

def spectogramPlot(data,fs,nfft,nOverlap,figsize,subTitles,title):
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
    fig.suptitle(title)
    label= ["Power/Frequency"]
    for i, axs in enumerate(axs.flatten()):
        d, f, t, im = axs.specgram(data[:,i],NFFT=nfft,Fs=fs,noverlap=nOverlap,mode='psd',scale='dB')
        axs.set_title(subTitles[i])
        axs.set_ylim(0,80)
        axs.set(xlabel='Time (s)', ylabel='Frequency (Hz)')
        axs.label_outer()
        axs
    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label('Amplitude (dB)')
    cbar.minorticks_on()

def normalityTest(data):
    #   Inputs  :   difference between data from two timepoints 
    #   Output  :   result of normality test (p-value test)
    #           :   choice of technique for significance testing

    print ("....Executing Shapiro Wilks Test.......... "'\n')

    if shapiro(data)[1] > 0.05:
        pVal = shapiro(data)[1]
        print ("Shapiro Wilks Test: data is normally distributed, P-Value=", pVal)
        print('\n'"....confirming Shapiro Wilks Test normality result with D’Agostino’s K^2 test........."'\n')
        print ("....Executing D’Agostino’s K^2 Test..........")
        if stats.normaltest(data)[1] > 0.05:
            pVal = stats.normaltest(data)[1]
            print ("D’Agostino’s K^2 Test: data is normally distributed, P-Value=", pVal)
            print('\n'"....confirming D’Agostino’s K^2 Test normality result with Anderson-Darling Test........"'\n')
            print ("....Executing Anderson-Darling Test..........")
            result = anderson(data)
            print('Statistic: %.3f' % result.statistic)
            p = 0
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < result.critical_values[i]:
                    print('%.3f: %.3f, Anderson-Darling Test: data is normally distributed' % (sl, cv))    
        test = "Paired T-test"  
        print ('\n',test,"utilized to evaluate significance of data")  
    
    if shapiro(data)[1] <= 0.05:
        pVal = shapiro(data)[1]
        print ("Shapiro Wilks Test: data is not normally distributed, P-Value=", pVal)
        print('\n'"....confirming Shapiro Wilks Test non-normality result with D’Agostino’s K^2 test......."'\n')
        print ("Executing D’Agostino’s K^2 Test...")
        if stats.normaltest(data)[1] < 0.05:
            pVal = stats.normaltest(data)[1]
            print ("D’Agostino’s K^2 Test: data is not normally distributed, P-Value=", pVal)
            print('\n'"....confirming D’Agostino’s K^2 Test non-normality result with Anderson-Darling Test......."'\n')
            print ("Executing Anderson-Darling Test...")
            result = anderson(data)
            print('Statistic: %.3f' % result.statistic)
            p = 0
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic > result.critical_values[i]:
                    print('%.3f: %.3f, Anderson-Darling Test: data is not normally distributed' % (sl, cv))
        test = "Wilcoxon Signed Test"
        print ('\n',test,"utilized to evaluate significance of data")  
    return test

def statTest(test_type,data_1,data_2,show_output,variableName,channelName,alpha=0.05):
    #   Inputs  :       data_1   - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #                   data_2   - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #   Output  :       2D array (d0 = samples, d1 = channels) of paired t-test results
    #   wilcoxon Test:  P < 0.05 : significance difference exists
    #                   P > 0.05 : no significance difference exists
    #   SD           :  Signficant difference between the two groups
    if test_type == 'Wilcoxon Signed Test': 
        def initializeTest(data_1,data_2,variableName,channelName):
            stat_test_1 = (wilcoxon(data_1,data_2))[1]
            if show_output==True:
                if stat_test_1 < alpha:
                    if np.mean(data_1)-np.mean(data_2)<0:
                        print("{} | {} | P-value = {} | SD | mean increase".format(variableName,channelName,round(stat_test_1,5)))
                    elif np.mean(data_1)-np.mean(data_2)>0:
                        print("{} | {} | P-value = {} | SD | mean decrease".format(variableName,channelName,round(stat_test_1,5)))
                else:
                    if np.mean(data_1)-np.mean(data_2)<0:
                        print("{} | {} | P-value = {} | NSD | mean increase".format(variableName,channelName,round(stat_test_1,5)))
                    elif np.mean(data_1)-np.mean(data_2)>0:
                        print("{} | {} | P-value = {} | NSD | mean decrease".format(variableName,channelName,round(stat_test_1,5)))
            else:
                return stat_test_1
            return stat_test_1

        stat_test_3 = []
        for i in range(len(data_1.T)):
            stat_test_2 = initializeTest(data_1[:,i],data_2[:,i],variableName,channelName[i])
            stat_test_3.append(stat_test_2)
        stat_test_3 = np.array(stat_test_3).T
        print("\n")

    elif test_type == 'Paired T-test':
        def initializeTest(data_1,data_2,variableName,channelName):
            stat_test_1 = (stats.ttest_rel(data_1,data_2))[1]
            if show_output==True:
                if stat_test_1 < alpha:
                    if np.mean(data_1)-np.mean(data_2)<0:
                        print("{} | {} | P-value = {} | SD | mean increase".format(variableName,channelName,round(stat_test_1,5)))
                    elif np.mean(data_1)-np.mean(data_2)>0:
                        print("{} | {} | P-value = {} | SD | mean decrease".format(variableName,channelName,round(stat_test_1,5)))
                else:
                    if np.mean(data_1)-np.mean(data_2)<0:
                        print("{} | {} | P-value = {} | NSD | mean increase".format(variableName,channelName,round(stat_test_1,5)))
                    elif np.mean(data_1)-np.mean(data_2)>0:
                        print("{} | {} | P-value = {} | NSD | mean decrease".format(variableName,channelName,round(stat_test_1,5)))
            else:
                return stat_test_1
            return stat_test_1
        
        stat_test_3 = []
        for i in range(len(data_1.T)):
            stat_test_2 = initializeTest(data_1[:,i],data_2[:,i],variableName,channelName[i])
            stat_test_3.append(stat_test_2)
        stat_test_3 = np.array(stat_test_3).T
        print("\n")
    return stat_test_3

def singleChannelDWT(data,time_array,wavelet):
    #   Probability Mapping Based Artifact Detection and Wavelet Denoising based 
    #   Artifact Removal from Scalp EEG for BCI Applications
    #  Perform DWT on the data
    #   Input: data - EEG data: 1D array (samples x channel)
    #   Output: new signal: (samples x number of wavelets)
    #           signal_global - new signal extracted after global threshold 
    #           signal_std - new signal extracted after std threshold 
    #   Reference:  choice of number of levels to threshold gotten from "Comparative Study of Wavelet-Based Unsupervised 
    #               Ocular Artifact Removal Techniques for Single-Channel EEG Data"
    
    def dwt_only(data,wavelet):
        def dwt(data,wavelet):
            coeffs = wavedec(data,wavelet,level=10)
            return np.array(coeffs,dtype=object).T

        def global_threshold(data,coeffs):
            def coeffs_approx(data,coeffs):
                return (np.median(abs(coeffs[0]))/0.6745)*(np.sqrt(2*np.log(len(data))))
            def coeffs_detail(data,coeffs):
                return (np.median(abs(coeffs[1]))/0.6745)*(np.sqrt(2*np.log(len(data))))
            arr_approx = coeffs_approx(data,coeffs)
            arr_detail = coeffs_detail(data,coeffs)
            return np.vstack((arr_approx,arr_detail))

        def apply_threshold(coeffs,threshold):
            def apply_threshold_approx(coeffs,threshold):
                #coeffs[0][abs(coeffs[0])>threshold[1]] = 0
                #coeffs_approx = coeffs[0]
                coeffs_approx = np.zeros(len(coeffs[0]))
                return coeffs_approx
            def apply_threshold_detail(coeffs,threshold):
                coeffs = coeffs[1:len(coeffs)]
                coeffs[0][abs(coeffs[0])>threshold[1]] = 0
                coeffs[1][abs(coeffs[1])>threshold[1]] = 0
                coeffs[2][abs(coeffs[2])>threshold[1]] = 0  # level 8
                coeffs[3][abs(coeffs[3])>threshold[1]] = 0  # level 7
                coeffs[4][abs(coeffs[4])>threshold[1]] = 0  # level 6
                coeffs[5][abs(coeffs[5])>threshold[1]] = 0  # level 5
                coeffs[6][abs(coeffs[6])>threshold[1]] = 0  # level 4
                coeffs[7][abs(coeffs[7])>threshold[1]] = 0  # level 3
                coeffs[8][abs(coeffs[8])>threshold[1]] = 0
                coeffs[9][abs(coeffs[9])>threshold[1]] = 0
                return coeffs
            arr_approx = apply_threshold_approx(coeffs,threshold)
            arr_detail = apply_threshold_detail(coeffs,threshold)
            arr_detail = list(np.array(arr_detail).T)
            arr_approx = arr_approx
            coefs = arr_detail
            (coefs).insert(0,arr_approx)
            return coefs

        def inv_dwt(coeffs,wavelet):
            def inverse_dwt(coeffs,wavelet):
                return waverec(coeffs,wavelet)
            arr = (inverse_dwt(list(np.array(coeffs,dtype=object)),wavelet))
            return  (np.array(arr).T)[:-1]

        coeffs = dwt(data,wavelet)
        threshold_global = global_threshold(data,coeffs)
        coeffs_global = apply_threshold(coeffs,threshold_global)
        signal_global = inv_dwt(coeffs_global,wavelet)
        return signal_global

    newEEG_global = []
    for i in range(len(wavelet)):
        newEEG_global.append((dwt_only(data,wavelet[i])))
    newEEG_global = np.array(newEEG_global).T
    if len(newEEG_global) != len(time_array):
        if len(newEEG_global) > len(time_array):
            diff = len(newEEG_global) - len(time_array)
            newEEG_global = newEEG_global[:-diff,:]
        elif len(newEEG_global) < len(time_array):
            diff = len(time_array) - len(newEEG_global)
            num_zeros = np.zeros((diff,len(newEEG_global[1])))
            newEEG_global = np.append(newEEG_global,num_zeros,axis=0)
    else:
        newEEG_global = newEEG_global
    return newEEG_global

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
    
def anova(anova_title,dataframe,anova_type,independent_variable,dependent_variable,alphaAnova,alphaPostHoc):
    
    if anova_type==2:
        print('ANOVA RESULT:',anova_title)
        string = dependent_variable + ' ~ ' +'C(' + independent_variable[0] + ') + C(' + independent_variable[1] + ') + C(' + independent_variable[0] + '):C(' + independent_variable[1] + ')'  
        model = ols(string,data=dataframe).fit()
        result = sm.stats.anova_lm(model, type=anova_type)
        filter_ = (result['PR(>F)'] <= alphaAnova)
        result = result[filter_]
        if result.empty == True:
            print('no significant interaction')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT:',anova_title)
        print('\n')
        print("Main Effect of " + independent_variable[0] + ":")
        res = stat()
        res.tukey_hsd(df=dataframe, res_var=dependent_variable, xfac_var=independent_variable[0], anova_model=string)
        result = res.tukey_summary
        filter_ = (result['p-value'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("Main Effect of " + independent_variable[1] + ":")
        res.tukey_hsd(df=dataframe, res_var=dependent_variable, xfac_var=independent_variable[1], anova_model=string)
        result = res.tukey_summary
        filter_ = (result['p-value'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[1] + ':')
        res.tukey_hsd(df=dataframe, res_var=dependent_variable, xfac_var=[independent_variable[0],independent_variable[1]], anova_model=string)
        result = res.tukey_summary
        filter_ = (result['p-value'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

    elif anova_type==3:
        print('ANOVA RESULT:',anova_title)
        string = dependent_variable + ' ~ ' +'(' + independent_variable[0] + ') + (' + independent_variable[1] + ') + (' + independent_variable[2] + ') + (' + independent_variable[0] + '):(' + independent_variable[1] + ')+ (' + independent_variable[0] + '):(' + independent_variable[2]+')+ (' + independent_variable[1] + '):(' + independent_variable[2] + ')+ (' + independent_variable[0] + '):(' + independent_variable[1] + '):(' + independent_variable[2] + ')'
        model = ols(string,data=dataframe).fit()
        result = sm.stats.anova_lm(model, type=anova_type)
        filter_ = (result['PR(>F)'] <= alphaAnova)
        result = result[filter_]
        if result.empty == True:
            print('no significant interaction')
        else:
            print(result)
        print('\n') 

        print('POST HOC RESULT:',anova_title)
        interaction_groups = dataframe[independent_variable[0]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("Main Effect of " + independent_variable[1] + ":")
        interaction_groups = dataframe[independent_variable[1]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("Main Effect of " + independent_variable[2] + ":")
        interaction_groups = dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[1] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('Interaction Effects between ' + independent_variable[1] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('Interaction Effects between ' + independent_variable[0] + ',' + independent_variable[1] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('\n')

    elif anova_type==4:
        print('ANOVA RESULT:',anova_title)
        string = dependent_variable + ' ~ ' +'(' + independent_variable[0] + ') + (' + independent_variable[1] + ') + (' + independent_variable[2] + ') + (' + independent_variable[3] + ') +(' + independent_variable[0] + '):(' + independent_variable[1] + ')+ (' + independent_variable[0] + '):(' + independent_variable[2]+') + (' + independent_variable[1] + '):(' + independent_variable[2] + ')+ (' + independent_variable[0] + '):(' + independent_variable[3] + ')+(' + independent_variable[1] + '):(' + independent_variable[3] + ')+(' + independent_variable[2] + '):(' + independent_variable[3] + ')+(' + independent_variable[0] + '):(' + independent_variable[1] + '):(' + independent_variable[2] + ') + (' + independent_variable[0] + '):(' + independent_variable[2] + '):(' + independent_variable[3] + ') + (' + independent_variable[1] + '):(' + independent_variable[2] + '):(' + independent_variable[3] + ') +(' + independent_variable[0] + '):(' + independent_variable[1] + '):(' + independent_variable[3] + ') +(' + independent_variable[0] + '):(' + independent_variable[1] + '):(' + independent_variable[2] + '):(' + independent_variable[3] + ')'
        model = ols(string,data=dataframe).fit()
        result = sm.stats.anova_lm(model, type=anova_type)
        result_anova = result
        #filter_ = (result['PR(>F)'] <= alphaAnova)
        #result = result[filter_]
        #if result.empty == True:
        #    print('no significant interaction')
        #else:
        #    print(result)
        #print('\n') 
        print(result)
        print('\n')
        
        print("POST HOC RESULT: Main Effect of " + independent_variable[0] + ":")
        interaction_groups = dataframe[independent_variable[0]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_main_1 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("POST HOC RESULT: Main Effect of " + independent_variable[1] + ":")
        interaction_groups = dataframe[independent_variable[1]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_main_2 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("POST HOC RESULT: Main Effect of " + independent_variable[2] + ":")
        interaction_groups = dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_main_3 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print("POST HOC RESULT: Main Effect of " + independent_variable[3] + ":")
        interaction_groups = dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_main_4 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[1] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_1 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_2 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[1] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_3 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_4 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[1] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_5 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[2] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[2]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_6 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ',' + independent_variable[1] + ' and ' + independent_variable[2] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        #filter_ = (result['p-adj'] <= alphaPostHoc)
        #result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_7 = result
        print(result_interaction_7)
        print('\n')
        #if result.empty == True:
        #    print('no significant effect')
        #else:
        #    print(result)
        #print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ',' + independent_variable[1] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_8 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ',' + independent_variable[2] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_9 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n') 

        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[1] + ',' + independent_variable[2] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_10 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        print('\n')
        
        print('POST HOC RESULT: Interaction Effects between ' + independent_variable[0] + ',' + independent_variable[1] + ',' + independent_variable[2] + ' and ' + independent_variable[3] + ':')
        interaction_groups = dataframe[independent_variable[0]].astype(str) + " & " + dataframe[independent_variable[1]].astype(str) + " & " + dataframe[independent_variable[2]].astype(str) + " & " + dataframe[independent_variable[3]].astype(str)
        comp = mc.MultiComparison(dataframe[dependent_variable], interaction_groups)
        post_hoc_res = comp.tukeyhsd()
        result = post_hoc_res.summary()
        results_as_html = result.as_html()
        result = pd.read_html(results_as_html)[0]
        filter_ = (result['p-adj'] <= alphaPostHoc)
        result = result[filter_]
        result.drop(result.columns[[6]], axis = 1, inplace = True)
        result_interaction_11 = result
        if result.empty == True:
            print('no significant effect')
        else:
            print(result)
        

        print('\n')
    return result_anova,result_main_1,result_main_2,result_main_3,result_main_4,result_interaction_1,result_interaction_2,result_interaction_3,result_interaction_4,result_interaction_5,result_interaction_6,result_interaction_7,result_interaction_8,result_interaction_9,result_interaction_10,result_interaction_11

def bandpowerPlots(x,y,title,label):
    fig=plt.figure()
    fig.show()
    ax=fig.add_subplot(111)
    ax.plot(x,y[0],c='b',marker="^",ls='--',label=label[0],fillstyle='none')
    ax.plot(x,y[1],c='g',marker=(8,2,0),ls='--',label=label[1])
    ax.plot(x,y[2],c='r',marker="v",ls='-',label=label[2])
    ax.plot(x,y[3],c='m',marker="o",ls='--',label=label[3],fillstyle='none')
    plt.title(title)
    plt.xlabel('Channels')
    plt.ylabel('Average Band Power')
    plt.legend(loc=2)
    plt.draw()

def ica(data,fs):
    """
    input: samples x channels
    output: samples x channels
    """

    #   Implement high pass filter @ 1Hz
    def icaHighpass(data,cutoff,fs):
        def params_fnc(data,cutoff,fs,order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            y = signal.filtfilt(b, a, data)
            return y
        filterEEG = []
        for i in range(len(data.T)):
            filterEEG.append(params_fnc(data.T[i],cutoff,fs))
        filterEEG = np.array(filterEEG).T
        return filterEEG

    def confidenceInterval(samples):
    #   At 95% significance level, tN -1 = 2.201
        means = np.mean(samples)
        std_dev = np.std(samples)
        standard_error = std_dev/np.sqrt(len(samples))
        lower_95_perc_bound = means - 2.201*standard_error
        upper_95_perc_bound = means + 2.201*standard_error
        return upper_95_perc_bound

    def setZeros(data,index):
        def params(data):
            return np.zeros(len(data))
        zeros = []
        for i in range(len(index)):
            zeros.append(params(data.T[index[i]]))
        zeros = np.array(zeros)
        return zeros

    hpEEG = icaHighpass(data,cutoff=1,fs=fs) 

    #   Computing ICA components
    ica = FastICA(n_components=4, random_state=0, tol=0.0001)
    comps = ica.fit_transform(hpEEG)
    comps_1 = comps[:,0]
    comps_2 = comps[:,1]
    comps_3 = comps[:,2]
    comps_4 = comps[:,3]

    #   Computing kurtosis of ICA weights
    comps_1_kurtosis = kurtosis(comps_1)
    comps_2_kurtosis = kurtosis(comps_2)
    comps_3_kurtosis = kurtosis(comps_3)
    comps_4_kurtosis = kurtosis(comps_4)
    comps_kurtosis = np.array([comps_1_kurtosis,comps_2_kurtosis,comps_3_kurtosis,comps_4_kurtosis])

    #   Computing skewness of ICA weights
    comps_1_skew = skew(comps_1)
    comps_2_skew = skew(comps_2)
    comps_3_skew = skew(comps_3)
    comps_4_skew = skew(comps_4)
    comps_skew = np.array([comps_1_skew,comps_2_skew,comps_3_skew,comps_4_skew])

    #   Computing sample entropy of ICA weights
    import antropy as ant
    comps_1_sampEN = ant.sample_entropy(comps_1)
    comps_2_sampEN = ant.sample_entropy(comps_2)
    comps_3_sampEN = ant.sample_entropy(comps_3)
    comps_4_sampEN = ant.sample_entropy(comps_4)
    comps_sampEN = np.array([comps_1_sampEN,comps_2_sampEN,comps_3_sampEN,comps_4_sampEN])

    #   Computing CI on to set threshold
    threshold_kurt = confidenceInterval(comps_kurtosis)
    threshold_skew = confidenceInterval(comps_skew)
    threshold_sampEN = confidenceInterval(comps_sampEN)

    "compare threshold with extracted parameter values"
    #   Extract epochs
    bool_ArtfCompsKurt = [comps_kurtosis>threshold_kurt]
    idx_ArtfCompsKurt = np.asarray(np.where(bool_ArtfCompsKurt[0]==True))
    bool_ArtfCompsSkew = [comps_skew>threshold_skew]
    idx_ArtfCompsSkew = np.asarray(np.where(bool_ArtfCompsSkew[0]==True))
    bool_ArtfCompsSampEN = [comps_sampEN>threshold_sampEN]
    idx_ArtfCompsSampEN = np.asarray(np.where(bool_ArtfCompsSampEN[0]==True))

    #   Merge index of components detected as artifacts by kurtosis, skewness, and sample entropy
    idx_artf_comps = np.concatenate((idx_ArtfCompsKurt,idx_ArtfCompsSkew,idx_ArtfCompsSampEN),axis=1)
    idx_artf_comps = np.unique(idx_artf_comps)

    "Component identified as artifact is converted to arrays of zeros"
    rejected_comps = setZeros(comps,idx_artf_comps)


    "Return zero-ed ICs into the original windows per ICs"
    for i in range(len(idx_artf_comps)):
        idx_rejected_comps = np.arange(len(rejected_comps))
        comps.T[idx_artf_comps[i]] = rejected_comps[idx_rejected_comps[i]]


    "Recover clean signal from clean ICs"
    restored = ica.inverse_transform(comps)
    return restored