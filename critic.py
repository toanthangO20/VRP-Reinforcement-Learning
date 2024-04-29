import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Định nghĩa lớp StateCritic, kế thừa từ nn.Module, để ước lượng độ phức tạp của vấn đề
class StateCritic(nn.Module):

    # Phương thức khởi tạo của lớp StateCritic
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        # Khởi tạo hai bộ encoder để xử lý dữ liệu tĩnh và động.
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Định nghĩa ba lớp kết nối đầy đủ (fully connected layers)
        # để xử lý biểu diễn ẩn từ bộ mã hóa.
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        # Khởi tạo trọng số của mô hình bằng cách sử dụng khởi tạo Xavier.
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    # Phương thức forward xử lý dữ liệu đầu vào qua mô hình.
    def forward(self, static, dynamic):

        # Mã hóa dữ liệu tĩnh và động để có được biểu diễn ẩn.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        # Kết hợp biểu diễn ẩn của dữ liệu tĩnh và động.
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        # Áp dụng hàm kích hoạt ReLU và truyền qua các lớp kết nối đầy đủ, sau đó tổng hợp kết quả.
        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)

        # Trả về kết quả đầu ra, là ước lượng độ phức tạp của vấn đề.
        return output

# Định nghĩa lớp Critic, tương tự như StateCritic, nhưng chỉ sử dụng một loại đầu vào.
class Critic(nn.Module):

    # Phương thức khởi tạo của lớp Critic.
    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Định nghĩa các lớp kết nối đầy đủ cho lớp Critic.
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    # Phương thức forward của lớp Critic.
    def forward(self, input):

        # Xử lý đầu vào qua các lớp kết nối đầy đủ và áp dụng hàm kích hoạt ReLU.
        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)

        # Trả về kết quả đầu ra, là ước lượng độ phức tạp của vấn đề.
        return output