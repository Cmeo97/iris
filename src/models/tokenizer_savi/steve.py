import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from steve.transformer import TransformerEncoder, TransformerDecoder #TODO: use transformer defined in transformer.py?

def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft, index
    else:
        return y_soft

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m

class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')
    
    def forward(self, x):
        x = self.m(x)
        return F.relu(x)
    
def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m

class dVAE(nn.Module):
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1)
        )
        
        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )

    def forward(self, x, tau, hard):
        z_logits = F.log_softmax(self.encoder(x), dim=1)                           # B * T, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                             # B * T, vocab_size, H_enc, W_enc
        z_hard, index = gumbel_softmax(z_logits, tau, True, dim=1).detach()                    # B * T, vocab_size, H_enc, W_enc
        return z_soft, z_hard, index


class STEVEEncoder(nn.Module):
    def __init__(self, 
                 img_channels, 
                 cnn_hidden_size, 
                 image_size,
                 d_model,
                 num_iterations,
                 num_slots,
                 slot_size,
                 mlp_hidden_size,
                 num_predictor_blocks,
                 num_predictor_heads,
                 predictor_dropout):
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv2dBlock(img_channels, cnn_hidden_size, 5, 1 if image_size == 64 else 2, 2),
            Conv2dBlock(cnn_hidden_size, cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(cnn_hidden_size, cnn_hidden_size, 5, 1, 2),
            conv2d(cnn_hidden_size, d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(d_model, image_size if image_size == 64 else image_size // 2)

        self.layer_norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            linear(d_model, d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(d_model, d_model))

        self.savi = SlotAttentionVideo(
            num_iterations, num_slots,
            d_model, slot_size, mlp_hidden_size,
            num_predictor_blocks, num_predictor_heads, predictor_dropout)

        self.slot_proj = linear(slot_size, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        conv_output = self.cnn(x)                # B * T, cnn_hidden_size, H, W
        out = self.pos(out)                      # B * T, cnn_hidden_size, H, W
        out = out.flatten(2, 3).permute(0, 2, 1) # B * T, H * W, cnn_hidden_size
        out = self.mlp(self.layer_norm(out))     # B * T, H * W, cnn_hidden_size
        out = out.reshape(conv_output.shape)
        return out

class STEVEDecoder(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model,
                 image_size,
                 num_decoder_blocks,
                 num_decoder_heads,
                 dropout):
        super().__init__()

        self.dict = OneHotDictionary(vocab_size, d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self.bos)

        self.pos = LearnedPositionalEmbedding1D(1 + (image_size // 4) ** 2, d_model)

        self.tf = TransformerDecoder(
            num_decoder_blocks, (image_size // 4) ** 2, d_model, num_decoder_heads, dropout)

        self.head = linear(d_model, vocab_size, bias=False)


class SlotAttentionVideo(nn.Module):
    
    def __init__(self, 
                 num_iterations,
                 num_slots,
                 input_size,
                 slot_size,
                 mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if i < self.num_iterations - 1:
                    slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs
    

@dataclass
class SteveConfig:
    num_iterations: int = 2
    num_slots: int = 15
    cnn_hidden_size: int = 64
    slot_size: int = 192
    mlp_hidden_size: int = 192
    img_channels: int = 3
    image_size: int = 128
    vocab_size: int = 4096
    d_model: int = 192
    dropout: float = 0.1
    num_decoder_blocks: int = 8
    num_decoder_heads: int = 4
    num_predictor_blocks: int = 1
    num_predictor_heads: int = 4 
    predictor_dropout: float = 0.0


class STEVE(nn.Module):
    
    def __init__(self, config: SteveConfig):
        super().__init__()
        
        self.num_iterations = config.num_iterations
        self.num_slots = config.num_slots
        self.cnn_hidden_size = config.cnn_hidden_size
        self.slot_size = config.slot_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.img_channels = config.img_channels
        self.image_size = config.image_size
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.num_decoder_blocks = config.num_decoder_blocks
        self.num_decoder_heads = config.num_decoder_heads
        self.num_predictor_blocks = config.num_predictor_blocks
        self.num_predictor_heads = config.num_predictor_heads
        self.predictor_dropout = config.predictor_dropout

        # encoder networks
        self.steve_encoder = STEVEEncoder(self.img_channels,
                                    self.cnn_hidden_size,
                                    self.image_size,
                                    self.d_model,
                                    self.num_iterations,
                                    self.num_slots,
                                    self.slot_size,
                                    self.mlp_hidden_size,
                                    self.num_predictor_blocks,
                                    self.num_predictor_heads,
                                    self.predictor_dropout)

        # decoder networks
        self.steve_decoder = STEVEDecoder(self.d_model,
                                    self.image_size,
                                    self.num_decoder_blocks,
                                    self.num_decoder_heads,
                                    self.dropout)

    def forward(self, video, z_hard):
        B, T, C, H, W = video.shape

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W

        # dvae encode
        #z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
        #z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc
        #z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * T, vocab_size, H_enc, W_enc
        #z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * T, H_enc * W_enc, vocab_size
        z_emb = self.steve_decoder.dict(z_hard)                                                     # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model
                   

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

        # decode
        slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
        pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
        pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
        #cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (B * T)                         # 1

        return (pred, attns)

    def encode(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
                                                            # attns: B, T, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

        attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

        return slots, attns_vis, attns

    def decode(self, slots):
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.steve_encoder.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.steve_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.steve_decoder.tf(
                self.steve_decoder.pos(input),
                slots
            )
            z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, video):
        """
        image: batch_size x img_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots, attns, _ = self.encode(video)
        recon_transformer = self.decode(slots.flatten(end_dim=1))
        recon_transformer = recon_transformer.reshape(B, T, C, H, W)

        return recon_transformer

