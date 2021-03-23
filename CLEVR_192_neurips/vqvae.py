import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantizer import *
from residual_block import *
import numpy as np


class Self_Attn(nn.Module):
    """
    Self attention Layer
    From https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py


    """
    def __init__(self,in_dim,activation="relu", attn_div=8):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//attn_div , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//attn_div , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out


class VQ(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, embedding_prior_input=False):
        super(VQ, self).__init__()
        self._train_prior = embedding_prior_input
        self._embedding_dim = embedding_dim
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=embedding_dim,
                                    kernel_size=1,
                                    stride=1)


        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, embedding_prior_input)


        self._post_vq_conv = nn.Conv2d(in_channels=embedding_dim,
                                 out_channels=num_hiddens,
                                 kernel_size=1,
                                 stride=1)

    def forward(self, bottom_encoding):

        encoded_img = self._pre_vq_conv(bottom_encoding)

        loss, quantized, perplexity, _, indices = self._vq_vae(encoded_img)

        if self._train_prior:
            return None, None, None, indices

        global_decode = self._post_vq_conv(quantized)
        return loss, global_decode, perplexity, indices

class Model(nn.Module):
    def __init__(self, input_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, n_labels=10, embedding_prior_input=False):
        super(Model, self).__init__()
        self._train_prior = embedding_prior_input
        self._embedding_dim = embedding_dim


        # Encode Image to 32x32
        self._conv_1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,
                                     out_channels=num_hiddens//4,
                                     kernel_size=4,
                                     stride=2, padding=1), nn.BatchNorm2d(num_hiddens//4), nn.LeakyReLU(0.2))

        # Encoding Self Attention Path
        self._sa_conv_2 = nn.Sequential(nn.Conv2d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1), nn.BatchNorm2d(num_hiddens//2), nn.LeakyReLU(0.2),
                                 Self_Attn(num_hiddens//2, attn_div=2), nn.LeakyReLU(0.2))

        self._sa_conv_3 = nn.Sequential(nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1), nn.BatchNorm2d(num_hiddens), nn.LeakyReLU(0.2),
                                 Self_Attn(num_hiddens,attn_div=2), nn.LeakyReLU(0.2))


        self._sa_residual_stack_enc = ResidualStack(in_channels=num_hiddens,    #num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)


        self._vq_attention = VQ(num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens,
                                    num_embeddings,
                                    embedding_dim,
                                    commitment_cost,
                                    embedding_prior_input=self._train_prior)

        #  Encoding Normal Path
        self._conv_2 = nn.Sequential(nn.Conv2d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1), nn.BatchNorm2d(num_hiddens//2), nn.LeakyReLU(0.2))

        self._conv_3 = nn.Sequential(nn.Conv2d(in_channels=(num_hiddens//2)*2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1), nn.BatchNorm2d(num_hiddens), nn.LeakyReLU(0.2))


        self._residual_stack_enc = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)


        self._vq_normal = VQ(num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens,
                                    num_embeddings,
                                    embedding_dim,
                                    commitment_cost,
                                    embedding_prior_input=self._train_prior)

        # Decode Attention Path
        self._sa_residual_stack_dec = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._attention_dec_1 = Self_Attn(num_hiddens, attn_div=2)

        self._sa_deconv_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1),
                                       nn.LeakyReLU(0.2),
                                       Self_Attn(num_hiddens//2, attn_div=4), nn.LeakyReLU(0.2))

        self._sa_deconv_2 = nn.Sequential(nn.ConvTranspose2d(in_channels=(num_hiddens//2)*2,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1),
                                       nn.LeakyReLU(0.2))


        # Decode normal path
        self._residual_stack_dec = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._deconv_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=num_hiddens*2,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1),
                                       nn.LeakyReLU(0.2))

        self._deconv_2 = nn.Sequential(nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1),
                                       nn.LeakyReLU(0.2))

        self._deconv_reconstruction = nn.Sequential(nn.ConvTranspose2d(in_channels=(num_hiddens//4)*2,
                                                out_channels=input_channels,
                                                kernel_size=4,
                                                stride=2, padding=1),
                                       nn.Tanh())



    def forward(self, images):

        global_encoding = self._conv_1(images)
        # Forward Attention Encoding
        sa_conv_enc_1 = self._sa_conv_2(global_encoding)

        sa_conv_enc_2 = self._sa_conv_3(sa_conv_enc_1)

        sa_res_enc = self._sa_residual_stack_enc(sa_conv_enc_2)


        # Forwarding normal encoding
        norm_conv_enc = self._conv_2(global_encoding)
        norm_conv_enc = self._conv_3(torch.cat([norm_conv_enc, sa_conv_enc_1], dim=1))
        norm_res_enc = self._residual_stack_enc(norm_conv_enc)
        norm_vq_loss, norm_encoding, norm_perplexity, norm_quant = self._vq_normal(norm_res_enc)

        # Attention VQ Decode
        sa_vq_loss, sa_encoding, sa_perplexity, sa_quant = self._vq_attention(sa_res_enc)

        # If we are training the prior we don't need to decode the latents
        # We would like to return the latents instead.
        if self._train_prior:
            return None, None, sa_quant, norm_quant

        # Reconstruct
        reconstruction = self.decode_latents(sa_encoding, norm_encoding, discrete=False)

        return sa_vq_loss+norm_vq_loss, reconstruction, norm_perplexity, sa_perplexity


    def decode_latents(self, sa_encoding, norm_encoding, discrete=False):

        # When generating images, we need to decode the discrete latents
        # these latents are of size (B, H, W)
        if discrete:
            sa_encoding = self._vq_attention._vq_vae._embedding(sa_encoding)
            norm_encoding = self._vq_normal._vq_vae._embedding(norm_encoding)
            sa_encoding = sa_encoding.permute(0,3,1,2)
            sa_encoding = self._vq_attention._post_vq_conv(sa_encoding)
            norm_encoding = norm_encoding.permute(0,3,1,2)
            norm_encoding = self._vq_normal._post_vq_conv(norm_encoding)

        # If we are training the model the decoding of the discrete latents
        # already happens implicitly, these latents are of size (B, embedding_dim, H, W)

        # Decode attention path
        sa_res_dec = self._sa_residual_stack_dec(sa_encoding)
        att_1_dec = self._attention_dec_1(sa_res_dec)
        sa_dec_1 = self._sa_deconv_1(att_1_dec)

        # Decode normal path
        norm_res_dec = self._residual_stack_dec(norm_encoding)
        norm_dec_1 = self._deconv_1(torch.cat([norm_res_dec, att_1_dec], dim=1))
        norm_dec_2 = self._deconv_2(norm_dec_1)


        sa_dec_2 = self._sa_deconv_2(torch.cat([norm_dec_1, sa_dec_1], dim=1))
        # Reconstruct
        reconstruction = self._deconv_reconstruction(torch.cat([sa_dec_2, norm_dec_2], dim=1))

        return reconstruction

