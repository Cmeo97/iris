import torch
import torch.nn as nn
import torch.nn.functional as F

from steve.transformer import TransformerEncoder, TransformerDecoder

def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
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
        z_logits = F.log_softmax(self.dvae.encoder(x), dim=1)                           # B * T, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                             # B * T, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()                    # B * T, vocab_size, H_enc, W_enc
        return z_soft, z_hard


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
