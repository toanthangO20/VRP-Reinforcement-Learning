import torch
import torch.nn as nn
import torch.nn.functional as F

# Kiểm tra xem GPU có sẵn không và thiết lập thiết bị tính toán mặc định.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Lớp Encoder dùng để mã hóa trạng thái tĩnh và động sử dụng Convolution 1 chiều.
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        # Tạo một lớp Convolution 1d với kích thước kernel là 1.
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        # Áp dụng lớp convolution lên đầu vào và trả về kết quả.
        output = self.conv(input)
        return output  # Kích thước đầu ra: (batch, hidden_size, seq_len)


# Lớp Attention dùng để tính toán sự chú ý trên các nút đầu vào dựa trên trạng thái hiện tại.
class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # Khởi tạo các tham số cho quá trình tính toán sự chú ý.
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        # Tính toán sự chú ý dựa trên trạng thái tĩnh, động và trạng thái của decoder.
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # Kích thước đầu ra: (batch, seq_len)
        return attns


# Lớp Pointer dùng để tính toán trạng thái tiếp theo dựa trên trạng thái trước đó và nhúng đầu vào.
class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()
        
        # Khởi tạo các tham số và lớp GRU cho quá trình dự đoán trạng thái tiếp theo.
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Tính toán khả năng lựa chọn trạng thái tiếp theo
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Tính toán biểu diễn đầu ra của trạng thái hiện tại
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        # Tính toán trạng thái tiếp theo và sự chú ý của encoder.
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Luôn dropout output của mạng RNN
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # Nếu số layers > 1 thì đã thực hiện dropout
            last_hh = self.drop_hh(last_hh) 

        # Tìm context của input từ output
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # Kích thước đầu ra: (B, 1, num_feats)

        # Tính toán đầu ra tiếp theo
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # Kích thước đầu ra: (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh



if __name__ == '__main__':
    raise Exception('Cannot be called from main')
