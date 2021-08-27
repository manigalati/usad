
from types import DynamicClassAttribute
import torch
from torch import optim
import torch.nn as nn
import gc

from torch.optim import optimizer

from utils import *
device = get_default_device()

class normal_model(nn.Module):
  def __init__(self):
        super().__init__()
        self.input_output_windows_result = {"input":[],"output":[],"loss":[]}

  def training_step(self, batch, n):

    loss= self.caculateMSE(batch,n)
    loss = torch.mean(loss)
    return loss

  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.parameters()))
      # print("self paramter",list(self.encoder.parameters())+list(self.decoder.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
                #Train AE1
              loss1= self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
          result = self.evaluate(val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      return history

  def testing_all(self,test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      count+=1
      print("iter ",count)
      batch=to_device(batch,device)

      with torch.no_grad():
        # w1,_ = self.encoder(batch)
        # w1=self.decoder(w1)
        if count == 1:
          loss = self.caculateMSE(batch,count,print_output=True)
        else:
          loss = self.caculateMSE(batch,count)
        results.append(loss)

      # del w1
      torch.cuda.empty_cache()
      gc.collect()
    return results

  def validation_step(self, batch, n):
    loss1=self.training_step(batch,n)
    return {'val_loss1': loss1}
        
  def validation_epoch_end(self,outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}
    
  def epoch_end(self,epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}".format(epoch, result['val_loss1']))
  def evaluate(self, val_loader, n):
      outputs = [self.validation_step(to_device(batch,device), n) for [batch] in val_loader]
      return self.validation_epoch_end(outputs)
  def get_intput_output_window_result(self):
        # y_anomaly_score=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
        #                             results[-1].flatten().detach().cpu().numpy()])
        # self.input_output_windows_result = 
        print("input_output_windows_result[input].len",len(self.input_output_windows_result["input"]))
        print("input_output_windows_result[input][0].shape",self.input_output_windows_result["input"][0].shape)
        print("input_output_windows_result[input].shape",np.array(self.input_output_windows_result["input"]).shape)
        self.input_output_windows_result["input"] = np.array(self.input_output_windows_result["input"])
        self.input_output_windows_result["output"] = np.array(self.input_output_windows_result["output"])
        self.input_output_windows_result["loss"] = np.array(self.input_output_windows_result["loss"])
        return self.input_output_windows_result


class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w

class MY_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim,window_size):
      super().__init__()
      self.window_size = window_size
      self.input_dim = input_dim
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.output_dim = output_dim
      self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
      # print("in forward x.shape",x.shape)
      x=torch.reshape(x,(-1,self.window_size,self.input_dim))
      # print("in forward x.shape",x.shape)
      # print("x.shape",x.shape)
      # print("x.size(0)",x.size(0))
      #x shape torch.Size([2048,5,1])
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      out, hidden = self.lstm(x, (h0, c0))
      self.hidden=hidden
      # print("out",out)
      # print("out shape",out.shape)
      #out shape torch.Size([2048, 5, 64])
      #out = self.fc(out[:, -1, :])
      # print("out forward x.shape",x.shape)
      # print("out forward out.shape",out.shape)
      x=torch.reshape(x,(-1,self.window_size*self.input_dim))
      out=torch.reshape(out,(-1,self.window_size*self.output_dim))
      # print("out forward x.shape",x.shape)
      # print("out forward out.shape",out.shape)
      return out
    def getHidden(self):
      return self.hidden
       
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2
  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.encoder.parameters())+list(self.decoder1.parameters()))
      optimizer2 = opt_func(list(self.encoder.parameters())+list(self.decoder2.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              #batch 是window_size*input_features_dim的一維陣列
              batch=to_device(batch,device)
              
               #Train AE1
              loss1,loss2 = self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
              #Train AE2
              loss1,loss2 = self.training_step(batch,epoch+1)
              loss2.backward()
              optimizer2.step()
              optimizer2.zero_grad()
              
          result = self.evaluate(self, val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      return history

  def testing_all(self, test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      count+=1
      print("count",count)
      batch=to_device(batch,device)
      with torch.no_grad():
        w1=self.decoder1(self.encoder(batch))
        w2=self.decoder2(self.encoder(w1))
      results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
      del w1
      del w2
      torch.cuda.empty_cache()
      gc.collect()
    return results

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}

  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))    
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder1': self.decoder1.state_dict(),
                'decoder2': self.decoder2.state_dict()
                }, "model/USAD.pth")
  def loadModel(self):
      checkpoint = torch.load("model/USAD.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder1.load_state_dict(checkpoint['decoder1'])
      self.decoder2.load_state_dict(checkpoint['decoder2'])
  def evaluate(self, val_loader, n):
      outputs = [self.validation_step(to_device(batch,device), n) for [batch] in val_loader]
      return self.validation_epoch_end(outputs)
  
class LSTM_UsadModel(nn.Module):
  def __init__(self, w_size, z_size,input_feature_dim,windows_size):
    super().__init__()
    self.windows_size =windows_size
    self.input_feature_dim = input_feature_dim
    self.num_layers=2
    self.hidden_size = int(z_size/windows_size)
    self.lstm = MY_LSTM(input_feature_dim, self.hidden_size, self.num_layers,self.hidden_size,windows_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  # def caculateOutput(self):
  #       w1 = self.decoder1(z)
  #   w2 = self.decoder2(z)
  #   w3 = self.decoder2(self.lstm(w1))

  def training_step(self, batch, n,alpha=0.5,beta=0.5):
    z = self.lstm(batch)
    # print("z.shape",z.shape)
    # z.reshape(-1,self.window_size*self.input_feature_dim)
    #print("z.shape",z.shape)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.lstm(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.lstm.parameters())+list(self.decoder1.parameters()))
      optimizer2 = opt_func(list(self.lstm.parameters())+list(self.decoder2.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              #batch 是window_size*input_features_dim的一維陣列
              batch=to_device(batch,device)
              
               #Train AEresult1
              loss1,loss2 = self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
              #Train AE2
              loss1,loss2 = self.training_step(batch,epoch+1)
              loss2.backward()
              optimizer2.step()
              optimizer2.zero_grad()
              
          result = self.evaluate(self, val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      return history

  def testing_all(self, test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      print("count",count)
      batch=to_device(batch,device)
      with torch.no_grad():
        w1=self.decoder1(self.lstm(batch))
        w2=self.decoder2(self.lstm(w1))
      results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
      if count == 0:
        print("testing_all dataset_input[0]",batch)
        print("testing_all dataset_output[0]",alpha*w1+beta*w2)
      del w1
      del w2
      torch.cuda.empty_cache()
      gc.collect()
      count+=1
    return results

  def validation_step(self, batch, n):
    loss1,loss2 = self.training_step(batch,n)
    return {'val_loss1': loss1, 'val_loss2': loss2}

  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))    
  def saveModel(self):
    torch.save({
                'lstm': self.lstm.state_dict(),
                'decoder1': self.decoder1.state_dict(),
                'decoder2': self.decoder2.state_dict()
                }, "model/USAD.pth")
  def loadModel(self):
    checkpoint = torch.load("model/USAD.pth")
    self.lstm.load_state_dict(checkpoint['lstm'])
    self.decoder1.load_state_dict(checkpoint['decoder1'])
    self.decoder2.load_state_dict(checkpoint['decoder2'])
  def evaluate(self, val_loader, n):
      outputs = [self.validation_step(to_device(batch,device), n) for [batch] in val_loader]
      return self.validation_epoch_end(outputs)
      

  
    

    

class AutoencoderModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder = Decoder(z_size, w_size)
  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                }, "model/autoencoder.pth")
  def loadModel(self):
      checkpoint = torch.load("model/autoencoder.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder(z)
    loss1 = torch.mean((batch-w1)**2)
    return loss1
  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.encoder.parameters())+list(self.decoder.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
               #Train AE1
              loss1= self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
          result = self.evaluate(self, val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      return history

  def testing_all(self, test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      count+=1
      print("count",count)
      batch=to_device(batch,device)
      with torch.no_grad():
        w1=self.decoder(self.encoder(batch))
      results.append(torch.mean((batch-w1)**2,axis=1))
      del w1
      torch.cuda.empty_cache()
      gc.collect()
    return results

  def validation_step(self, batch, n):
    loss1=self.training_step(batch,n)
    return {'val_loss1': loss1}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}".format(epoch, result['val_loss1']))
  def evaluate(self, val_loader, n):
      outputs = [self.validation_step(to_device(batch,device), n) for [batch] in val_loader]
      return self.validation_epoch_end(outputs)


class LSTM_AutoencoderModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder = Decoder(z_size, w_size)
  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                }, "model/autoencoder.pth")
  def loadModel(self):
      checkpoint = torch.load("model/autoencoder.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder(z)
    loss1 = torch.mean((batch-w1)**2)
    return loss1
  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.encoder.parameters())+list(self.decoder.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
               #Train AE1
              loss1= self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
          result = self.evaluate(self, val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      return history

  def testing_all(self, test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      count+=1
      print("count",count)
      batch=to_device(batch,device)
      with torch.no_grad():
        w1=self.decoder(self.encoder(batch))
      results.append(torch.mean((batch-w1)**2,axis=1))
      del w1
      torch.cuda.empty_cache()
      gc.collect()
    return results

  def validation_step(self, batch, n):
    loss1=self.training_step(batch,n)
    return {'val_loss1': loss1}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}".format(epoch, result['val_loss1']))

class LSTM_VAE_ENCODER(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, num_layers, latent_size, windows_size,bidirection=False):
        super(LSTM_VAE_ENCODER, self).__init__()

        self.windows_size =windows_size
        self.num_layers = num_layers
        self.hidden_dim= hidden_dim
        self.input_feature_dim = input_feature_dim
        self.num_layers=2

        self.lstm = MY_LSTM(input_feature_dim, self.hidden_dim, self.num_layers,self.hidden_dim,windows_size)
        self.hidden2mean = nn.Linear(hidden_dim, latent_size)
        self.hidden2logv = nn.Linear(hidden_dim, latent_size)

    def forward(self, x):

        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out= self.lstm(x)
        hidden = self.lstm.getHidden()
        hidden = hidden[0]
        hidden = hidden[0]
        # print("hidden.shape",hidden.shape)

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps
        # print("z.shape",z.shape)
        z.reshape(z.shape[0],1,z.shape[1])
        # print("z.shape",z.shape)

        kld = -0.5 * torch.mean(1 + logv - mean.pow(2) - logv.exp()).to('cuda')

        return z, kld,hidden


class LSTM_VAE_DECODER(nn.Module):
    def __init__(self, latent_size, num_layers, output_feature_dim, windows_size,bidirection=False):
        super(LSTM_VAE_DECODER, self).__init__()
        if bidirection == True:
            self.bidirection = 2
        else:
            self.bidirection = 1

        self.windows_size =windows_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.output_feature_dim = output_feature_dim
        self.num_layers=2
        self.hidden_size = output_feature_dim
        self.input_feature_dim = latent_size

        #self.lstm = MY_LSTM(latent_size ,self.output_feature_dim, self.num_layers,self.output_feature_dim,windows_size)
        self.lstm = nn.LSTM(self.input_feature_dim, self.output_feature_dim, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hidden = self.lstm(x, (h0, c0))
        return out


class dynamic_anomaly_threshold(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.dynamic_anomaly_threshold_NN = nn.Linear(hidden_size,1)
    self.loss_hidden_set=[]
    # self.testing_loss_hidden_set=[]
  def add_data(self,loss,hidden):
    # print("loss",loss,"hidden",hidden)
    # print("type loss",type(loss),"type hidden",type(hidden))
    # print("loss.shape",loss.shape,"hidden.shape",hidden.shape)
    self.loss_hidden_set.append({"loss":loss.detach(),"hidden":hidden.detach()})
    # self.loss_hidden_set.append({"loss":torch.randn(1).cuda(),"hidden":torch.randn(1,46).cuda()})
    
  # def add_testing_data(self,hidden):
  #   self.testing_loss_hidden_set.append({"hidden":hidden})
  def training_all(self):
    optimizer = torch.optim.Adam(list(self.parameters()))
    count=0
    for epoch in range(40):
          print("epoch ",epoch)
          for loss_hidden in self.loss_hidden_set:
                count+=1
                pred=  self.dynamic_anomaly_threshold_NN(loss_hidden["hidden"])
                loss = torch.mean((loss_hidden["loss"]- pred)**2)
                print("count",count,"loss",loss,"pred",pred[:10],"label",loss_hidden["loss"][:10])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
  def testing_all(self):
      result = []
      with torch.no_grad():
        for loss_hidden in self.loss_hidden_set:
          result.append(self.dynamic_anomaly_threshold_NN(loss_hidden["hidden"]))
    
      return result
  def getThreshold(self):
        return self.testing_all()
      
                
          
    


class LSTM_VAE(nn.Module):
  def __init__(self, hidden_size,latent_size,input_feature_dim,windows_size):
    super().__init__()
    self.windows_size =windows_size
    self.input_feature_dim = input_feature_dim
    self.num_layers=2
    self.hidden_size = hidden_size
    self.latent_size = latent_size
    self.output_feature_dim = input_feature_dim

    # self.anomaly_threshold_estimater = dynamic_anomaly_threshold(self.hidden_size)
    self.encoder= LSTM_VAE_ENCODER(input_feature_dim, self.hidden_size, self.num_layers,self.latent_size,windows_size)
    self.decoder= LSTM_VAE_DECODER(self.latent_size,self.num_layers,self.output_feature_dim,windows_size)

  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                # 'anomaly_threshold_estimater': self.anomaly_threshold_estimater.state_dict()
                }, "model/LSTM_VAE.pth")
  def loadModel(self):
      checkpoint = torch.load("model/LSTM_VAE.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])
      # self.anomaly_threshold_estimater.load_state_dict(checkpoint['anomaly_threshold_estimater'])


  def caculateMSE(self,batch,n,print_output=False):
    # print("batch.shape ",batch.shape)
    # print("batch",batch)
    latent,kld ,hidden= self.encoder(batch)
    # hidden_history_list=[]
    # if action == "training":
    #       hidden_history_list.append(hidden)
    # print("latent.shape ",latent.shape)
    latent=torch.reshape(latent,(latent.shape[0],1,latent.shape[1]))
    w1 = self.decoder(latent)
    w1 = torch.reshape(w1,(w1.shape[0],w1.shape[2]))
    batchFinal=torch.reshape(batch,(w1.shape[0],-1,w1.shape[1]))[:,-1,:]
    # print("w1.shape",w1.shape)
    # print("batchFinal.shape ",batchFinal.shape)
    loss1 = torch.mean((batchFinal-w1)**2,axis=1) 
    if print_output == True:
        print("testcase batchFinal",batchFinal)
        print("w1",w1)

    # print("hidden",hidden.shape)
    # print("loss1.shape",loss1.shape)
    
    # self.anomaly_threshold_estimater.add_data(loss1,hidden) 
    return loss1,kld



  def training_step(self, batch, n,kld_times):

    loss,kld = self.caculateMSE(batch,n)
    loss = torch.mean(loss)
    loss += kld_times * kld
    return loss


  def VAE_printResult(self,y_test,y_pred):
      threshold = [ x  for x in self.anomaly_threshold_estimater.getThreshold()]
      print("============== VAE result ==================")
      print("threshold[:5]",threshold[:5])
      # print("threshold.shape",threshold.shape)
      print("threshold len",len(threshold),"y_pred len",len(y_pred))
      y_pred=[ 1 if(x>=threshold[index]) else 0 for index,x in enumerate(y_pred)]

      precision, recall, fscore, support = score(y_test, y_pred)

      print('precision: {}'.format(precision[0]))
      print('recall: {}'.format(recall[0]))
      print('f1score: {}'.format(fscore[0]))
      print("============== result ==================")
  def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(self.encoder.parameters())+list(self.decoder.parameters()))
      kld_times = 0
      for epoch in range(epochs):
          if epoch >10 and kld_times < 1:
              kld_times += 0.1
          print("epoch ",epoch,"kld_times:",kld_times)
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
               #Train AE1
              loss1= self.training_step(batch,epoch+1,kld_times)

              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()

              # dynamic_anomaly_threshold_loss.backward()
              # optimizer2.step()
              # optimizer2.zero_grad()
              
              
          result = self.evaluate(self, val_loader, epoch+1)
          self.epoch_end(epoch, result)
          history.append(result)
      print("finish training")
      # self.anomaly_threshold_estimater.training_all()
      return history

  def testing_all(self, test_loader, alpha=0.5,beta=0.5):
    count=0
    results=[]
    for [batch] in test_loader:
      # print("batch shape",batch.shape)
      count+=1
      print("count",count)
      batch=to_device(batch,device)

      with torch.no_grad():
        # w1,_ = self.encoder(batch)
        # w1=self.decoder(w1)
        print_output=False
        if count == 1:
              print_output = True
        loss,_ = self.caculateMSE(batch,count,print_output)
        results.append(loss)


      # del w1
      torch.cuda.empty_cache()
      gc.collect()
    return results

  def validation_step(self, batch, n):
    loss1=self.training_step(batch,n,0)
    return {'val_loss1': loss1}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}".format(epoch, result['val_loss1']))
  def evaluate(self,model, val_loader, n):
      outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
      return model.validation_epoch_end(outputs)



class CNN_LSTM(normal_model):
  ### need to modify this when building new model
  def __init__(self, latent_size,input_feature_dim,windows_size):
    super().__init__()

    self.windows_size =windows_size
    self.input_feature_dim = input_feature_dim
    self.latent_size = latent_size
    self.output_feature_dim = input_feature_dim
    self.num_layers=2

    kernel_size = 3
    padding = int((kernel_size-1)/2)
    self.conv1 = nn.Conv1d(in_channels=input_feature_dim,out_channels=self.latent_size,kernel_size=kernel_size,padding=padding)
    self.lstm = nn.LSTM(self.latent_size, self.output_feature_dim, self.num_layers, batch_first=True)
  

  ### need to modify this when building new model
  def caculateMSE(self,batch,n,print_output=False):

    # convert batch shape to [epoch,window_size,input_feature_dim]
    batch=batch.reshape(batch.shape[0],-1,self.input_feature_dim)
    # print("batch.shape ",batch.shape)
    latent = batch.permute(0,2,1)
    # print("latent.shape ",latent.shape)

    latent= self.conv1(latent)

    latent = latent.permute(0,2,1)
    # print("latent.shape ",latent.shape)

    h0 = torch.zeros(self.num_layers, latent.size(0), self.output_feature_dim).to(device)
    c0 = torch.zeros(self.num_layers, latent.size(0), self.output_feature_dim).to(device)
    out, hidden = self.lstm(latent, (h0, c0))

    loss1 = torch.mean((batch-out)**2,axis=1) 
    self.input_output_windows_result["input"].extend(batch.detach().cpu().numpy())
    self.input_output_windows_result["output"].extend(out.detach().cpu().numpy())
    self.input_output_windows_result["loss"].extend(loss1.detach().cpu().numpy())
    loss1 = torch.mean(loss1,axis=1) 
    # self.input_output_windows_result["input"].extend(batch.detach().cpu().numpy()),"output":out.detach().cpu().numpy(),"loss":loss1.detach().cpu().numpy()})
    return loss1
  

  ### need to modify this when building new model
  def saveModel(self):
    torch.save({
                'conv1': self.conv1.state_dict(),
                'lstm': self.lstm.state_dict(),
                }, "model/CNN_LSTM.pth")
  ### need to modify this when building new model
  def loadModel(self):
      checkpoint = torch.load("model/CNN_LSTM.pth")
      self.conv1.load_state_dict(checkpoint['conv1'])
      self.lstm.load_state_dict(checkpoint['lstm'])
  ### In most case, you do not need to modify below function when building new model
  # def training_step(self, batch, n):
  #   return general_training_step(self,batch,n)

  # def training_all(self,epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
  #   return general_training_all(self,epochs,train_loader,val_loader,opt_func)

  # def testing_all(self, test_loader, alpha=0.5,beta=0.5):
  #   return general_testing_all(self,test_loader,alpha,beta)

  # def validation_step(self, batch, n):
  #   return general_validation_step(self,batch,n)
        
  # def validation_epoch_end(self, outputs):
  #   return general_validation_epoch_end(outputs)
    
  # def epoch_end(self, epoch, result):
  #   general_epoch_end(epoch,result)

################################################### normal model template

#############################################3