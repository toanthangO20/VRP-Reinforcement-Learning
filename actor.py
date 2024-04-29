import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models import Encoder, Pointer, Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRL4TSP(nn.Module):
    
    # Phương thức khởi tạo của lớp DRL4TSP
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        # Lưu trữ các hàm cập nhật và mặt nạ vào biến thể hiện
        # để sử dụng sau này trong quá trình lan truyền tiến
        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Khởi tạo các encoder tĩnh và động, decoder và con trỏ
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        # Khởi tạo trọng số của mô hình bằng cách sử dụng khởi tạo Xavier
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Tạo một trạng thái ban đầu giả định cho encoder nếu không có đầu vào cụ thể nào được cung cấp.
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    # Lan truyền tiến
    def forward(self, static, dynamic, decoder_input=None, last_hh=None):

        # Lấy kích thước của dữ liệu đầu vào static để xác định
        # batch_size, input_size, và sequence_size
        batch_size, input_size, sequence_size = static.size()

        # Nếu không có đầu vào cụ thể cho bộ giải mã,
        # sử dụng trạng thái ban đầu x0 đã được khởi tạo trước đó.
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Khởi tạo mặt nạ với giá trị 1 cho mỗi phần tử trong chuỗi
        # Mặt nạ này sẽ được cập nhật để ngăn chọn lại các điểm đã được thăm.
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Khởi tạo danh sách để lưu chỉ số của các điểm thăm và log xác suất của chúng.
        tour_idx, tour_logp = [], []

        # max_steps là số bước tối đa mà mô hình sẽ thực hiện
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Mã hóa các yếu tố tĩnh và động để có được hidden state của chúng.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        # Bắt đầu vòng lặp qua số bước tối đa.
        for _ in range(max_steps):

            # Nếu tất cả các phần tử trong mặt nạ đều là 0 
            # (không còn điểm nào để thăm), thoát khỏi vòng lặp.
            if not mask.byte().any():
                break

            # Mã hóa đầu vào của bộ giải mã để có được hidden state.
            decoder_hidden = self.decoder(decoder_input)

            # Sử dụng con trỏ để tính toán xác suất cho mỗi điểm tiếp theo
            # và cập nhật trạng thái ẩn cuối cùng
            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            
            # Áp dụng softmax để chuyển đổi logit thành xác suất, đồng thời tính toán với mặt nạ.
            probs = F.softmax(probs + mask.log(), dim=1)

            # Trong quá trình huấn luyện, chọn điểm tiếp theo dựa trên phân phối xác suất
            # Trong quá trình đánh giá, chọn điểm có xác suất cao nhất (phương pháp tham lam).
            if self.training:
                m = torch.distributions.Categorical(probs)

                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # Nếu có hàm cập nhật, cập nhật biểu diễn động sau mỗi bước.
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # Nếu có hàm mặt nạ, cập nhật mặt nạ để ngăn chọn lại các điểm đã thăm.
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            # Lưu log xác suất và chỉ số của điểm được chọn vào danh sách.
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            # Cập nhật đầu vào cho bộ giải mã với thông tin của điểm mới được chọn.
            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        # Kết hợp các chỉ số và log xác suất thành một tensor duy nhất cho mỗi lô.
        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        # Trả về chỉ số của chuỗi và log xác suất tương ứng.
        return tour_idx, tour_logp