import torch
import torch.nn as nn
import numpy as np


# Customized Transformer structure for signal processing, based on https://github.com/hyunwoongko/transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, F_in, F_int, n_head, att_type="Add"):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = Attention(F_int // n_head, att_type)

        self.w_qkv = nn.Sequential(
            nn.Conv1d(3 * F_in, 3 * F_int, kernel_size=(1,),
                      stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(3 * F_int)
        )

        self.w_concat = nn.Sequential(
            nn.Conv1d(F_int, F_in, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_in)
        )

    def forward(self, q, k, v, mask=None):

        # 1. dot product with weight matrices
        qkv = self.w_qkv(torch.cat([q, k, v], dim=1))
        q, k, v = qkv.split(split_size=qkv.size(1) // 3, dim=1)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, d_model, length]
        :return: [batch_size, head, d_tensor, length]
        """
        batch_size, d_model, length = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, d_tensor, length)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, d_tensor, length]
        :return: [batch_size, d_model, length]
        """
        batch_size, head, d_tensor, length = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.contiguous().view(batch_size, d_model, length)
        return tensor


class Attention(nn.Module):
    """
    compute scale dot product attention

    Query : given samples that we focused on (decoder)
    Key : every sample to check relationship with Query(encoder)
    Value : every sample same with Key (encoder)
    """

    def __init__(self, d_tensor, att_type="Add"):
        super(Attention, self).__init__()

        if att_type == "Mult":
            self.psi = nn.Softmax(dim=-1)
        elif att_type == "Add":
            self.psi = nn.Sequential(
                nn.Conv1d(d_tensor, 1, kernel_size=(1,),
                          stride=(1,), padding=0, bias=True),
                nn.Sigmoid()
            )

        self.relu = nn.LeakyReLU()
        self.att_type = att_type

    def forward(self, q, k, v, mask=None):
        batch_size, n_head, d_tensor, length = q.size()

        if self.att_type == "Mult":
            psi = self.relu(torch.matmul(q.transpose(2, 3), k) / np.sqrt(d_tensor))
        elif self.att_type == "Add":
            psi = self.relu(q + k)
            psi = psi.view(batch_size * n_head, psi.size(2), psi.size(3))

        psi = self.psi(psi)

        if self.att_type == "Mult":
            out = torch.matmul(psi, v.transpose(2, 3)).transpose(2, 3)
        elif self.att_type == "Add":
            psi = psi.view(batch_size, n_head, psi.size(1), psi.size(2))
            out = psi * v

        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        var = x.var(-2, unbiased=False, keepdim=True)
        # '-2' means one before last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out.transpose(1, 2) + self.beta
        return out.transpose(1, 2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W1 = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=(1,),
                      stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(hidden)
        )
        self.W2 = nn.Sequential(
            nn.Conv1d(hidden, d_model, kernel_size=(1,),
                      stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(d_model)
        )
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, F_in, F_int, F_hid, n_head, drop_prob, att_type):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(F_in, F_int, n_head, att_type)
        self.norm1 = LayerNorm(d_model=F_in)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=F_in, hidden=F_hid,
                                           drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=F_in)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = x + _x
        x = self.norm1(x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = x + _x
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, F_in, F_int, F_hid, n_head, drop_prob, att_type, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(F_in=F_in, F_int=F_int, F_hid=F_hid,
                                                  n_head=n_head, drop_prob=drop_prob,
                                                  att_type=att_type)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, F_in, F_int, F_hid, n_head, drop_prob, att_type):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(F_in, F_int, n_head, att_type)
        self.norm1 = LayerNorm(d_model=F_in)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(F_in, F_int, n_head, att_type)
        self.norm2 = LayerNorm(d_model=F_in)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=F_in, hidden=F_hid,
                                           drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=F_in)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = x + _x
        x = self.norm1(x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = x + _x
            x = self.norm2(x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = x + _x
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, F_in, F_int, F_hid, n_head, drop_prob, att_type, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(F_in=F_in, F_int=F_int, F_hid=F_hid,
                                                  n_head=n_head, drop_prob=drop_prob,
                                                  att_type=att_type)
                                     for _ in range(n_layers)])

        self.W_final = nn.Sequential(
            nn.Conv1d(F_in, F_in, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_in)
        )

    def forward(self, trg, src, trg_mask, src_mask):

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.W_final(trg)
        return output


class Transformer(nn.Module):
    def __init__(self, F_in_enc, F_in_dec, F_int, F_hid, n_head, drop_prob, att_type, n_layers):
        super().__init__()

        self.encoder = Encoder(F_in_enc, F_int, F_hid, n_head, drop_prob, att_type,
                               n_layers)
        self.decoder = Decoder(F_in_dec, F_int, F_hid, n_head, drop_prob, att_type,
                               n_layers)

    def forward(self, enc, dec, enc_mask=None, dec_mask=None):

        enc_out = self.encoder(enc, enc_mask)
        output = self.decoder(dec, enc_out, dec_mask, enc_mask)

        return output
