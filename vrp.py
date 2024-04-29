"""Định nghĩa bài toán VRP

Bài toán VRP trong trường hợp này được xác định với những ràng buộc sau:
    1. Nhu cầu của mỗi thành phố nằm trong khoảng [1, 9]
    2. Phương tiện có 1 sức chứa nhất định có thể quy định từ trước, phải đi qua tất cả các thành phố
    3. Khi số lượng hàng trong xe bằng 0, xe bắt buộc phải trở về kho
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Định nghĩa lớp VehicleRoutingDataset, kế thừa từ Dataset của PyTorch, để tạo dữ liệu cho VRP.
class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()

        # Kiểm tra để đảm bảo rằng tải trọng tối đa lớn hơn nhu cầu tối đa.
        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        # Thiết lập seed cho việc tạo số ngẫu nhiên để đảm bảo tính nhất quán.
        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Lưu trữ số mẫu, tải trọng tối đa và nhu cầu tối đa vào biến thể hiện.
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        # Tạo và lưu trữ vị trí ngẫu nhiên cho các điểm, bao gồm cả kho.
        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations

        # Tạo và lưu trữ tải trọng ban đầu cho mỗi xe.
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # Tạo và lưu trữ nhu cầu ngẫu nhiên cho mỗi điểm,
        # sau đó chuẩn hóa dựa trên tải trọng tối đa và đặt nhu cầu của kho là 0.
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)
        demands[:, 0, 0] = 0  # depot starts with a demand of 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    # Phương thức trả về số lượng mẫu trong dataset.
    def __len__(self):
        return self.num_samples

    # Phương thức lấy một mẫu từ dataset dựa trên chỉ số.
    def __getitem__(self, idx):
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    # Phương thức cập nhật mặt nạ để ẩn các trạng thái không hợp lệ dựa trên tải trọng và nhu cầu.
    def update_mask(self, mask, dynamic, chosen_idx=None):

        # Lấy thông tin tải trọng và nhu cầu từ tensor dynamic.
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # Nếu không còn nhu cầu dương nào (tất cả các nhu cầu đều đã được đáp ứng),
        # kết thúc chuyến đi bằng cách trả về một mặt nạ chỉ chứa giá trị 0.
        if demands.eq(0).all():
            return demands * 0.

        # Tạo mặt nạ mới, cho phép di chuyển đến
        # các điểm có nhu cầu lớn hơn 0 và nhỏ hơn tải trọng hiện tại.
        new_mask = demands.ne(0) * demands.lt(loads)

        # Kiểm tra xem điểm vừa được chọn có phải là kho (điểm xuất phát) hay không.
        repeat_home = chosen_idx.ne(0)

        # Nếu điểm vừa chọn không phải là kho,
        # tránh việc chọn kho liên tiếp bằng cách cập nhật mặt nạ.
        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (~repeat_home).any():
            new_mask[(~repeat_home).nonzero(), 0] = 0.

        # Kiểm tra xem xe có còn tải trọng hay không
        # và liệu tất cả các điểm còn lại có nhu cầu hay không.
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        # Kết hợp hai điều kiện trên để xác định xem liệu
        # có cần phải chờ tất cả các mẫu trong minibatch hoàn thành không.
        combined = (has_no_load + has_no_demand).gt(0)

        # Nếu cần phải chờ, cập nhật mặt nạ để chỉ cho phép di chuyển về kho
        # và không cho phép di chuyển đến bất kỳ điểm nào khác.
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        # Trả về mặt nạ mới dưới dạng tensor số thực.
        return new_mask.float()

    # Phương thức cập nhật giá trị tải trọng và nhu cầu sau mỗi lần chọn điểm.
    def update_dynamic(self, dynamic, chosen_idx):

        # Xác định xem điểm được chọn có phải là kho (điểm xuất phát) hay không
        # visit: tensor boolean cho biết các điểm không phải kho
        # depot: tensor boolean cho biết kho
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Sao chép thông tin tải trọng và nhu cầu để không làm thay đổi đồ thị tính toán gốc.
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        # Lấy thông tin tải trọng và nhu cầu cụ thể cho điểm được chọn từ tensor dynamic
        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Kiểm tra xem có bất kỳ xe nào chọn thăm một thành phố không (không phải kho)
        if visit.any():

            # Cập nhật tải trọng mới và nhu cầu mới dựa trên lượng hàng hóa được giao và nhận
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Lấy chỉ số của các điểm được chọn không phải là kho
            visit_idx = visit.nonzero().squeeze()

            # Cập nhật tải trọng và nhu cầu cho các điểm được chọn.
            # Đặt nhu cầu của kho là -1 cộng với tải trọng mới để biểu thị việc trở về kho.
            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Trở về kho để tiếp tục lấy hàng
        if depot.any():
            # Đặt lại tải trọng là đầy (1) và nhu cầu của kho là 0 cho các xe trở về kho.
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        # Kết hợp thông tin tải trọng và nhu cầu thành một tensor mới.
        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)

        # Trả về tensor cập nhật với thông tin tải trọng và nhu cầu mới,
        # đảm bảo tensor này ở cùng thiết bị với tensor dynamic ban đầu.
        return torch.tensor(tensor.data, device=dynamic.device)


def reward(static, tour_indices):

    # Mở rộng tour_indices để có cùng kích thước với static
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)

    # Lấy tọa độ của các thành phố từ static dựa trên tour_indices và hoán đổi các chiều
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Lấy tọa độ của điểm xuất phát (kho) và thêm một chiều để có thể nối nó vào đầu và cuối của tour.
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Tính khoảng cách Euclid giữa mỗi cặp điểm liên tiếp trong tour
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    # Tính tổng khoảng cách của toàn bộ tour 
    return tour_len.sum(1)


def render(static, tour_indices, save_path):
    # Vẽ và hiển thị các giải pháp tìm được

    # Đóng tất cả các cửa sổ đồ họa hiện có để tránh xung đột khi vẽ mới.
    plt.close('all')

    # Xác định số lượng đồ họa cần vẽ dựa trên số lượng tour.
    # Nếu có đủ tour, sẽ vẽ 3x3 đồ họa; nếu không, chỉ vẽ một đồ họa.
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    # Tạo một lưới các đồ họa con với số hàng và cột tương ứng với num_plots
    # chia sẻ trục x và y giữa các đồ họa con.
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    # Duyệt qua từng đồ họa con để vẽ từng tour.
    for i, ax in enumerate(axes):

        # Lấy chỉ số của tour thứ i và đảm bảo rằng nó có hai chiều.
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # Mở rộng chỉ số để khớp với kích thước của static và lấy tọa độ tương ứng từ static.
        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        # Lấy tọa độ điểm xuất phát và nối chúng với tọa độ của các điểm trong tour,
        # đảm bảo tour bắt đầu và kết thúc tại điểm xuất phát.
        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Thêm chỉ số của điểm xuất phát vào đầu và cuối của mảng chỉ số
        # và tìm vị trí của các điểm xuất phát trong mảng.
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            # Nếu hai chỉ số liên tiếp chỉ đến cùng một điểm, bỏ qua và không vẽ.
            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        # Vẽ đường đi giữa các điểm từ low đến high.
        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)

        # Vẽ các điểm trên tour và đánh dấu điểm xuất phát bằng một ngôi sao lớn
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        # Đặt giới hạn cho trục x và y
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Tối ưu hóa bố cục để tất cả nội dung đều vừa vặn trong khung hình
    plt.tight_layout()

    # Lưu hình ảnh vào đường dẫn đã cho với độ phân giải cao
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
