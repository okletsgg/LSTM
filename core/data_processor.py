import numpy as np
import pandas as pd
import cmath

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        print("split=",split)
        print("len(dataframe)=",len(dataframe))
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len]) #data_windows窗口

        data_windows = np.array(data_windows).astype(float)
        print("data_windows.shape=", data_windows.shape)
        z=data_windows[:, -1, [0]]  #这里加入z是为了得到未经过标准化的测试数据
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]

        return x,y,z

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        print("len_train=",self.len_train)
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        #Normalise window with a base value of zero
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        #print(window_data)
        for window in window_data:
            normalised_window = []
            #print("window.shape[1]=",window.shape[1])
            for col_i in range(window.shape[1]):
                t=999999 #t为最小值
                maax=-999999 #maax为最大值
                #print("range(window.shape[1])=",range(window.shape[1]))
                for p in window[:, col_i]:
                    #print("p=",p)
                    if(t>p):
                        t=p
                    if(maax<p):
                        maax=p
                #print("t=",t)
                #print("t=",t,"col_i=",col_i)
                #print("window[0, col_i]=",window[0, col_i],"t=",t,"col_i=",col_i)
                normalised_col = [(float(p-t) / (float(maax-t)+0.1)) for p in window[:, col_i]]
                #normalised_col =[float(p) for p in window[:,col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
'''  
    def normalise_windows(self, window_data, single_window=False):
        #Normalise window with a base value of zero
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        #print(window_data)
        for window in window_data:
            normalised_window = []
            #print("window.shape[1]=",window.shape[1])
            for col_i in range(window.shape[1]):
                t=999999
                #print("range(window.shape[1])=",range(window.shape[1]))
                for p in window[:, col_i]:
                    #print("p=",p)
                    if(t>p):
                        t=p
                #print("t=",t)
                if(t<0):
                    t=-t
                else:
                    t=0.1
                #print("t=",t,"col_i=",col_i)
                #print("window[0, col_i]=",window[0, col_i],"t=",t,"col_i=",col_i)
                normalised_col = [((float(p+t) / (float(window[0, col_i])+1.1*t)) -1) for p in window[:, col_i]]
                #normalised_col =[float(p) for p in window[:,col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
'''
    #继承自原项目的标准化改进过程

'''
    def normalise_windows(self, window_data, single_window=False):
        #Normalise window with a base value of zero
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        #print(window_data)
        for window in window_data:
            normalised_window = []
            #print("window.shape[1]=",window.shape[1])
            for col_i in range(window.shape[1]):
                t=0
                n=0
                dv=0
                #print("range(window.shape[1])=",range(window.shape[1]))
                for p in window[:, col_i]:
                    #print("p=",p)
                    t+=p
                    n+=1
                #print("t=",t)
                av=t/n
                for p in window[:,col_i]:
                    dv+=(p-av)*(p-av)
                sd=cmath.sqrt(dv/n) # 标准差
                #print("t=",t,"col_i=",col_i)
                #print("window[0, col_i]=",window[0, col_i],"t=",t,"col_i=",col_i)
                normalised_col = [(float((p-av)/sd)) for p in window[:, col_i]]
                #normalised_col =[float(p) for p in window[:,col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
'''

