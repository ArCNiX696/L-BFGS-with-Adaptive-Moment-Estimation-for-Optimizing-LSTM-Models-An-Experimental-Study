import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchmetrics import MeanAbsolutePercentageError , R2Score
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import gc
import os
import time

i = [0]
t_counter = 1

def create_folders(folder,folder_2, sub_folder, sub_folder_2,sub_folder_3):
    n = i[-1]  # Último elemento de la lista `i`
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(folder_2):
        os.makedirs(folder_2)

    while os.path.exists(sub_folder) or os.path.exists(sub_folder_2)or os.path.exists(sub_folder_3):
        n += 1
        sub_folder = f'./Statistics results/Training and validation losses/LSTM testing Run {n}'
        sub_folder_2 = f'./Statistics results/LSTM statistics/LSTM testing Run {n}'
        sub_folder_3 = f'./Best model weights/LSTM testing Run {n}'
    
    os.makedirs(sub_folder)
    os.makedirs(sub_folder_2)
    os.makedirs(sub_folder_3)
    i.append(n)  

    return sub_folder, sub_folder_2,sub_folder_3

directory = './Statistics results'
directory_2 = './Best model weights/'
sub_directory, sub_directory_2 ,sub_directory_3= create_folders(directory,directory_2,
                                                f'./Statistics results/Training and validation losses/LSTM testing Run  {i[-1]}',
                                                f'./Statistics results/LSTM statistics/LSTM testing Run  {i[-1]}',
                                                f'./Best model weights/LSTM testing Run {i[-1]}/')

device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PrepareDataset:
    def __init__(self):
        raw_folder='./input datasets/'
        # raw_data='TESLA stock price dataset - windows size of  t-14 days.csv'
        raw_data='TESLA stock price dataset - windows size of  t-7 days,Normalized.csv'
        self.path=os.path.join(raw_folder,raw_data)
        
    def import_dataset(self,verbose=False):
        self.df = pd.read_csv(self.path )

        if verbose == True:
            print(f'\n{self.df.head(10)}\n') 
        
        return self.df
    
    def split_data(self,split_train=float,verbose=False):
        #Import dataset
        self.df = self.import_dataset(verbose=False)

        if self.df is None:
            print('No dataset availiable for split')

        else:
            self.df = self.df.to_numpy()

            x = self.df[:,1:]
            x= dc(np.flip(x, axis= 1))#For LSTM should flip the dataset to change the order in time 
            x_col_size = x.shape[1]

            y = self.df[:,0]

        if len(x) != len (y):
            print('x  has not the same number of rows as y , then cannot split data')

        else :
            split_idx = int(len(x) * split_train)

            #Train partition
            x_train = x[:split_idx].reshape((-1 , x_col_size , 1))
            x_test = x[split_idx:].reshape((-1 , x_col_size , 1))
            #Train partition to tensor
            x_train = torch.tensor(x_train).float()
            x_test = torch.tensor(x_test).float()

            #Test partition
            y_train = y[:split_idx].reshape((-1 , 1))
            y_test = y[split_idx:].reshape((-1 , 1))
            #Train partition to tensor
            y_train = torch.tensor(y_train).float()
            y_test = torch.tensor(y_test).float()

        
        if verbose == True:
            print(f'\n{"-"* 100}\n')
            print(f'\nx dataset with type and shape : {x.dtype} , {x.shape} len {len(x)}:\n\n{x}\n') 
            print(f'\n{"-"* 100}\n')
            print(f'\ny dataset type and shape : {y.dtype} , {y.shape}  len {len(y)}:\n\n{y}\n') 
            print(f'\n{"-"* 100}\n')
            print(f'\nx_train dataset type and shape : {x_train.dtype} , {x_train.shape}  len {len(x_train)}:\n\n{x_train}\n') 
            print(f'\nx_test dataset type and shape : {x_test.dtype} , {x_test.shape}  len {len(x_test)}:\n\n{x_test}\n')
            print(f'\n{"-"* 100}\n')
            print(f'\ny_train dataset type and shape : {y_train.dtype} , {y_train.shape}  len {len(y_train)}:\n\n{y_train}\n') 
            print(f'\ny_test dataset type and shape : {y_test.dtype} , {y_test.shape}  len {len(y_test)}:\n\n{y_test}\n')

        return x_train ,x_test ,y_train , y_test


class CustomDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i],self.y[i]
    

class LstmStructure(nn.Module):
    def __init__(self, input_size , hidden_size ,staked_layers): #input_size = num of features , hidden_size = num of memory_cells ,staked_layers = num of layers in LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stacked_layers = staked_layers
        
        self.lstm = nn.LSTM(input_size,hidden_size,staked_layers,batch_first=True)
        self.fully_conected = nn.Linear(hidden_size , 1)

    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.stacked_layers, batch_size , self.hidden_size).to(device) #hidden state
        c0 = torch.zeros(self.stacked_layers, batch_size , self.hidden_size).to(device) #cell state

        out,_=self.lstm(x , (h0,c0))
        out = self.fully_conected(out[:,-1,:])
        return out 

class StockPreLSTM(nn.Module):
    def __init__(self,hidden_size,staked_layers,learning_rate,batch_size,epochs,split):
        super().__init__()
        self.hidden_size = hidden_size
        self.staked_layers = staked_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.split = split
        self.model=LstmStructure(1,self.hidden_size,self.staked_layers).to(device) #input_size = num of features , hidden_size = num of memory_cells ,staked_layers = num of layers in LSTM
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.MAPE = MeanAbsolutePercentageError().to(device)
        self.R2 = R2Score().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.best_loss = float('inf')
        self.best_model_path = sub_directory_3
        self.training_losses = []
        self.validation_losses = []
        # self.counter = t_counter

    
    def prepare_data(self,verbose=False):
        data = PrepareDataset()

        x_train ,x_test ,y_train , y_test = data.split_data(self.split) #Split data

        train_dataset = CustomDataset(x_train , y_train)
        self.train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)

        val_dataset = CustomDataset(x_test, y_test)
        self.val_loader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False)

        if verbose == True:
            for _, b in enumerate(self.train_loader):
                x_b , y_b = b[0].to(device) , b[1].to(device)
                print(f'x_b shape: {x_b.shape} ,y_b shape: {y_b.shape}')
                break

        return self.train_loader , self.val_loader
    
    def training_model(self):
        global t_counter

        for self.epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs,labels in tqdm(self.train_loader,desc=f'Epoch{self.epoch+1}/{self.epochs}'):
                inputs,labels=inputs.to(device),labels.to(device)
                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss=self.MSE(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()

            epoch_loss= running_loss / (len(self.train_loader)) 
            # print(f'Epoch {self.epoch+1}')
            print(f'\nTraining Loss : {epoch_loss:.6f}')
            self.training_losses.append(epoch_loss)
            self.validation()

        t_counter += 1
        
    def validation(self):
        self.model.eval()
        val_loss=0.0
        val_MAE = 0.0
        val_MAPE = 0.0
        real_vals = []
        predictions = []

        with torch.no_grad():
            for inputs,labels in tqdm(self.val_loader,desc=f'Calculating Validation loss for Epoch {self.epoch+1}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs=self.model(inputs)
                # print(f'\n\ninputs :\n{inputs}\n\nlabels :\n{labels}\n\nouputs :\n{outputs}\n\n')
                loss=self.MSE(outputs,labels)#MSE
                val_loss+= loss.item()
                mae_loss = self.MAE(outputs,labels)#MAE
                val_MAE+= mae_loss.item()
                MAPE_loss = self.MAPE(outputs,labels)#MAPE
                val_MAPE+= MAPE_loss
                #For R2 calculation
                real_vals.append(labels)
                predictions.append(outputs)

        rea_vals_cum = torch.cat(real_vals)
        predictions_cum = torch.cat(predictions)

        val_R2 =self.R2(predictions_cum,rea_vals_cum).item() #R2
        # print(f'\n\nrea_vals_cum :\n{rea_vals_cum}\n\npredictions_cum :\n{predictions_cum}\n\nval_R2 :\n{val_R2}\n\n')
                
        avg_loss = (val_loss / (len(self.val_loader))) * 100
        print(f'validation loss Avg : {avg_loss:.6f} %\n{"*" * 120}\n')

        self.validation_losses.append(avg_loss)

     
        if avg_loss < self.best_loss:
            print(f'New best model found in epoch: {self.epoch + 1}\n')
            self.best_loss = avg_loss
            model_name = f'{self.best_loss:.6f}.training_{t_counter}_best_model.pth'
            torch.save(self.model.state_dict(), os.path.join(self.best_model_path, model_name))

            log_name = f'{self.best_loss:.6f}.training_{t_counter}_best_model_log.txt'
            with open(os.path.join(self.best_model_path, log_name), 'a') as log_file:
                log_file.write(f'Epoch: {self.epoch + 1}, Validation Loss: {avg_loss:.6f}, MSE: {val_loss}, MAE: {val_MAE}, MAPE: {val_MAPE}, R2: {val_R2}\n')

            self.plot_eval(f'{self.best_loss:.6f}.Training and validation losses in {self.epochs} epochs',self.best_loss)

    def plot_evaluation(self, file_name,loss):
        epochs = range(1, len(self.training_losses) + 1)
        # best_val_loss = self.validation_losses[len(self.validation_losses) - 1]

        hyperparameters_label = (f'hidden_size={self.hidden_size}, staked_layers={self.staked_layers}, '
                                f'learning_rate={self.learning_rate}, batch_size={self.batch_size}, '
                                f'epochs={self.epochs}, split={self.split}, loss={loss}')

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.training_losses, label='Training losses', color='blue', marker='o')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)#row , column , number of plot in the figure
        plt.plot(epochs, self.validation_losses, label='Validation losses', color='orange', marker='x')
        plt.title('Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout(rect=[0, 0.01, 1, 0.91])  

        plt.figtext(0.5, -0.02, f'GA - {hyperparameters_label}', ha='center', fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        
        plot_path = os.path.join(sub_directory, f'{file_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        gc.collect()

    #Without GA
    def plot_eval(self, file_name,loss):
        epochs = range(1, len(self.training_losses) + 1)
        # best_val_loss = self.validation_losses[len(self.validation_losses) - 1]

        hyperparameters_label = (f'hidden_size={self.hidden_size}, staked_layers={self.staked_layers}, '
                                f'learning_rate={self.learning_rate}, batch_size={self.batch_size}, '
                                f'epochs={self.epochs}, split={self.split}, loss={loss}')

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.training_losses, label='Training losses', color='blue', marker='o')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)#row , column , number of plot in the figure
        plt.plot(epochs, self.validation_losses, label='Validation losses', color='orange', marker='x')
        plt.title('Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout(rect=[0, 0.01, 1, 0.91])  

        plt.figtext(0.5, -0.02, f'{hyperparameters_label}', ha='center', fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        
        plot_path = os.path.join(sub_directory, f'{file_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        gc.collect()


    def run_model(self):
        self.prepare_data(verbose=False)
        self.training_model()
        return self.best_loss
    
    def run(self):
        self.prepare_data(verbose=False)
        self.training_model()

    
if __name__ == '__main__':
    mdl = StockPreLSTM(hidden_size=6,staked_layers=6,learning_rate=0.01,batch_size=256,epochs=700,split=0.9)
    mdl.run()
    
