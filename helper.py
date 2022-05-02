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