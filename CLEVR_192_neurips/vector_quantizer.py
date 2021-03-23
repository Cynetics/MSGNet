import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = ("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, embedding_prior_input=False, decay=0.99):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._train_prior = embedding_prior_input
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        
        # Uncomment code below to activate exponential moving averages
        # Although, we found that this sometimes hampered performance....
        """
        self._embedding.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = 1e-5
        """

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        indices = None
        # convert inputs from BCHW -> BHWC
        if self._train_prior:
            with torch.no_grad():
                embedding_squared = torch.sum(self._embedding.weight ** 2, dim=1)
                flat_input_squared = torch.sum(flat_input ** 2, dim=1, keepdim=True)
                distances = torch.addmm(embedding_squared + flat_input_squared, flat_input, self._embedding.weight.t(), alpha=-2.0, beta=1.0)
                _, indices_flatten = torch.min(distances, dim=1)
                indices = indices_flatten.view(*input_shape[:-1])
                
                return None, None, None, None, indices
            
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        #Exponential Moving Averages
        """
        self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)
    
        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n
            )
            
        dw = torch.matmul(encodings.t(), flat_input)
        self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
        self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        """
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, indices
