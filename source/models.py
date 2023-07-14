import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_split import *
import matplotlib.pyplot as plt
import math
plt.rcParams["figure.figsize"] = (18,7)

class AutoEncoder_ElasticNet(nn.Module):
    def __init__(self, n_features, n_cycles=49, alpha=0.5):
        super(AutoEncoder_ElasticNet, self).__init__()

        self.alpha = alpha
        self.n_features = n_features
        self.n_cycles = n_cycles

        self.encoder = nn.Sequential(
            nn.Linear(n_features*n_cycles, n_features*32),
            nn.ReLU(),
            nn.Linear(n_features*32, n_features*16),
            # nn.ReLU(),
            # nn.Linear(n_features*16, n_features*8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # nn.Linear(n_features*8, n_features*16),
            # nn.ReLU(),
            nn.Linear(n_features*16, n_features*32),
            nn.ReLU(),
            nn.Linear(n_features*32, n_features*n_cycles)
        )

        self.prediction = nn.Linear(n_features*16, 1)

    def elastic_net_loss(self):

        l1_norm = self.prediction.weight.abs().sum()
        l2_norm = self.prediction.weight.pow(2).sum()

        return (1-self.alpha)/2 * l2_norm + self.alpha * l1_norm
    
    def elastic_net_predict(self, x):
        return self.prediction(self.encoder(x))
    
    def forward(self, x):
        self.eval()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def evaluate(self, x, y):
        mses = [0.0, 0.0, 0.0]
        for i in train_ind:
            mses[0] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2
        for i in test_ind:
            mses[1] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2
        for i in secondary_ind:
            mses[2] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2

        mses[0] = np.sqrt(mses[0] / len(train_ind))
        mses[1] = np.sqrt(mses[1] / len(test_ind))
        mses[2] = np.sqrt(mses[2] / len(secondary_ind))

        for mse, label in zip(mses, ["train", "test", "sec"]):
            print(label, "\t", mse[0])


    def plotter(self, x, y, id):
        fig, axs = plt.subplots(1, 2)
        for i in range(self.n_features):
            axs[0].plot(x[id].detach().numpy()[i*self.n_cycles:(i+1)*self.n_cycles], 
                        self.forward(x[id]).detach().numpy()[i*self.n_cycles:(i+1)*self.n_cycles], ".", label = "feature {}".format(i+1))
        axs[0].plot(np.linspace(-5.5, 1.5, 3), np.linspace(-5.5, 1.5, 3), "k", alpha = 0.5)

        axs[0].legend()
        axs[0].set_xlabel("True input")
        axs[0].set_ylabel("Decoded input")
        axs[0].set_title("Decoder performance, cell "+str(id))

        axs[1].plot(y[train_ind], self.elastic_net_predict(x[train_ind]).detach().numpy(), ".", label = "train")
        axs[1].plot(y[test_ind], self.elastic_net_predict(x[test_ind]).detach().numpy(), ".", label = "test")
        axs[1].plot(y[secondary_ind], self.elastic_net_predict(x[secondary_ind]).detach().numpy(), ".", label = "secondary")
        axs[1].plot(np.linspace(2.15, 3.4, 3), np.linspace(2.15, 3.4, 3), "k", alpha = 0.5)

        axs[1].legend()
        axs[1].set_xlabel("True input")
        axs[1].set_ylabel("Decoded input")
        axs[1].set_title("Prediction performance")

        plt.show()

    def fit(self, x, y, train_policy, verbose = True, plots = True, log_loss = False):

        self.train()
        loss_function = nn.MSELoss()
        train_data = TensorDataset(torch.Tensor(x[train_ind]),torch.Tensor(y[train_ind]))

        num_stages = train_policy["num_stages"]
        for stage in range(num_stages):

            if verbose: print("\nStage {}:\n".format(stage+1))

            epochs = train_policy["epochs"][stage]
            lr = train_policy["learning_rates"][stage]
            batch_size = train_policy["batch_sizes"][stage]
            prediction_weight = train_policy["prediction_weights"][stage]
            decoding_weight = train_policy["decoding_weights"][stage]
            en_weight = train_policy["decoding_weights"][stage]

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            optimiser = torch.optim.Adam(self.parameters(), lr=lr)

            for ep in range(epochs):
                for batch in train_loader:

                    train_inputs, train_labels = batch
                    outputs = self.forward(train_inputs)
                    predictions = self.elastic_net_predict(train_inputs)

                    if log_loss:
                        loss = loss_function(train_labels, predictions[:, 0])*prediction_weight 
                    else:
                        loss = loss_function(10**train_labels, 10**predictions[:, 0])*prediction_weight
                    loss += self.elastic_net_loss()*en_weight + loss_function(train_inputs, outputs)*decoding_weight

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                if verbose:    
                    if (ep+1) % int(epochs / 10) == 0: print(f"Epoch {ep+1}/{epochs}, loss: {loss.item():.2f}")

            if verbose: self.evaluate(x, y)
            if plots:
                self.plotter(x, y, 30)

class AttentionModel(torch.nn.Module):
    def __init__(self, d_model, feat_dim, vdim=1, num_heads=1, attn_model="softmax", beta=1, skip_connect=0):
        super(AttentionModel, self).__init__()
        '''d_model: embedding dimension; can be chosen independently of input data dimensions
           feat_dim: number of cycles x number of features / length of collapsed input vector for one battery 
           vdim: dimension of output, 1 for our regression problem
           num_heads: default 1; can theoretically be increased for multihead attention (not supported by code yet)
           attn_model: default softmax; code also supports batch normalized attention with keyword "batch_norm"
           beta: if using batch normalized attention, beta is the weight placed on the mean
           skip_connect: whether or not to add a skip connection. If 0, no skip connection. If 1, H=AV+B where B
           is a trainable projection of the input X. If 2, H=AV+V'''
        self.W_q = nn.Linear(feat_dim, d_model)
        self.W_k = nn.Linear(feat_dim, d_model)
        self.W_v = nn.Linear(feat_dim, vdim)
        self.W_b = nn.Linear(feat_dim, vdim)
        self.d_model = d_model
        self.attn_model = attn_model
        self.beta = beta
        self.skip_connect = skip_connect

    def reshape_input(self,X): 
        '''collapses cycle and feature dimensions into a single dimension'''
        return X.reshape(X.shape[0], -1)

    def scaled_dot_product_attention(self, Q, K, V, B): 
        '''softmax attention'''
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_probs = torch.softmax(attn_scores, dim=-1) # attention matrix, dimensionality (batch size, batch size)
        output = torch.matmul(attn_probs, V)
        if self.skip_connect == 1:
            output = output + B
        elif self.skip_connect == 2:
            output = output + V
        return output
    
    def batch_normalized_attention(self, Q, K, V, B):
        '''batch normalized attention'''
        mu = torch.mean(K,0)
        s = torch.std(K,0,correction=0)
        attn_scores = torch.matmul(torch.mul(Q-self.beta*mu,s), torch.mul(K-self.beta*mu,s).transpose(-2,-1)) / math.sqrt(self.d_model)
        attn_probs = torch.softmax(attn_scores, dim=-1) # attention matrix, dimensionality (batch size, batch size)
        output = torch.matmul(attn_probs, V)
        if self.skip_connect == 1:
            output = output + B
        elif self.skip_connect == 2:
            output = output + V
        return output
    
    def forward(self, X):
        X = self.reshape_input(X)
        Q = self.W_q(X) # create query matrix, dimensionality (batch size, d_model)
        K = self.W_k(X) # create key matrix, dimensionality (batch size, d_model)
        V = self.W_v(X) # create value matrix, dimensionality (batch size, 1)
        B = self.W_b(X) # create matrix for skip connection

        if self.attn_model=="softmax": attn_output = self.scaled_dot_product_attention(Q, K, V, B)
        elif self.attn_model=="batch_norm": attn_output = self.batch_normalized_attention(Q, K, V, B)
        return attn_output