import torch
import torch.nn as nn
import gc

from utils import *
device = get_default_device()

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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
      super().__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      x.reshape(-1,self.window_size,self.input_size)
      print("in forward x.shape",x.shape)
      # print("x.shape",x.shape)
      # print("x.size(0)",x.size(0))
      #x shape torch.Size([2048,5,1])
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      out, _ = self.lstm(x, (h0, c0))
      # print("out",out)
      # print("out shape",out.shape)
      #out shape torch.Size([2048, 5, 64])
      #out = self.fc(out[:, -1, :])
      return out
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder1': self.decoder1.state_dict(),
                'decoder2': self.decoder2.state_dict()
                }, "USAD.pth")
  def loadModel(self):
      checkpoint = torch.load("USAD.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder1.load_state_dict(checkpoint['decoder1'])
      self.decoder2.load_state_dict(checkpoint['decoder2'])

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2
  def training_all(self,epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
      optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
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
              
          result = evaluate(self, val_loader, epoch+1)
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
  
class LSTM_UsadModel(nn.Module):
  def __init__(self, w_size, z_size,input_feature_dim,windows_size):
    super().__init__()
    self.windows_size =windows_size
    self.input_feature_dim = input_feature_dim
    self.num_layers=2
    self.lstm = MY_LSTM(input_feature_dim, z_size , self.num_layers,z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n,alpha=0.5,beta=0.5):
    z = self.lstm(batch)
    # print("z.shape",z.shape)
    # z.reshape(-1,self.window_size*self.input_feature_dim)
    print("z.shape",z.shape)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.lstm(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2
  
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
    

    

class AutoencoderModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder = Decoder(z_size, w_size)
  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                }, "autoencoder.pth")
  def loadModel(self):
      checkpoint = torch.load("autoencoder.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder(z)
    loss1 = torch.mean((batch-w1)**2)
    return loss1
  def training_all(self,epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
               #Train AE1
              loss1= self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
          result = evaluate(self, val_loader, epoch+1)
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


class LSTM_AutoencoderModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder = Decoder(z_size, w_size)
  
  def saveModel(self):
    torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                }, "autoencoder.pth")
  def loadModel(self):
      checkpoint = torch.load("autoencoder.pth")
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])

  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder(z)
    loss1 = torch.mean((batch-w1)**2)
    return loss1
  def training_all(self,epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
      history = []
      optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
      for epoch in range(epochs):
          for [batch] in train_loader:
              batch=to_device(batch,device)
              
               #Train AE1
              loss1= self.training_step(batch,epoch+1)
              loss1.backward()
              optimizer1.step()
              optimizer1.zero_grad()
              
              
          result = evaluate(self, val_loader, epoch+1)
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



def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)