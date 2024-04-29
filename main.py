import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from actor import DRL4TSP
import vrp
from vrp import VehicleRoutingDataset
from critic import StateCritic

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):

    actor.eval()

    # Kiểm tra nếu thư mục save_dir không tồn tại, tạo mới thư mục đó.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Khởi tạo một danh sách để lưu trữ các giá trị phần thưởng từ mỗi lô dữ liệu.
    rewards = []

    # Lặp qua từng lô dữ liệu trong data_loader.
    for batch_idx, batch in enumerate(data_loader):

        # Giải nén lô dữ liệu thành các biến static, dynamic, và x0
        static, dynamic, x0 = batch

        # Chuyển các tensor đến thiết bị được chỉ định (device), có thể là CPU hoặc GPU.
        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        # Thực hiện lan truyền tiến (forward pass) mà không tính toán gradient
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        # Tính toán phần thưởng trung bình cho lô dữ liệu
        reward = reward_fn(static, tour_indices).mean().item()

        # Thêm phần thưởng vào danh sách rewards
        rewards.append(reward)

        # Nếu hàm render_fn được cung cấp và số lô nhỏ hơn num_plot
        # vẽ giải pháp và lưu nó với tên được định dạng theo chỉ số lô và phần thưởng.
        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()

    # Trả về phần thưởng trung bình trên tất cả các lô dữ liệu đã xác thực.
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    
    # Tạo thư mục lưu trữ dựa trên thời gian hiện tại để lưu các checkpoint và kết quả.
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    # In thông báo bắt đầu quá trình huấn luyện.
    print('Starting training')

    # Tạo thư mục checkpoints để lưu trữ các trạng thái của mô hình trong quá trình huấn luyện.
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Khởi tạo bộ tối ưu hóa Adam cho cả actor và critic với tốc độ học được chỉ định.
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    # Tạo DataLoader cho dữ liệu huấn luyện và xác thực.
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    # Khởi tạo biến để theo dõi mô hình tốt nhất và phần thưởng tốt nhất.
    best_params = None
    best_reward = np.inf

    # Bắt đầu vòng lặp qua các epoch, ở đây chỉ chạy 2 epoch cho ví dụ.
    for epoch in range(2):

        actor.train()
        critic.train()

        # Khởi tạo danh sách để theo dõi thời gian,
        # mất mát, phần thưởng, và ước lượng phần thưởng của critic.
        times, losses, rewards, critic_rewards = [], [], [], []

        # Ghi lại thời gian bắt đầu của epoch.
        epoch_start = time.time()
        start = epoch_start

        # Lặp qua từng lô dữ liệu trong DataLoader.
        for batch_idx, batch in enumerate(train_loader):

            # Giải nén lô dữ liệu thành các biến static, dynamic, và x0.
            static, dynamic, x0 = batch

            # Chuyển các tensor đến thiết bị được chỉ định (device), có thể là CPU hoặc GPU.
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Thực hiện lan truyền tiến qua bộ dữ liệu
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Tính toán phần thưởng cho lô dữ liệu hiện tại.
            reward = reward_fn(static, tour_indices)

            # Lấy ước lượng phần thưởng từ mô hình critic.
            critic_est = critic(static, dynamic).view(-1)

            # Tính toán lợi thế và mất mát cho cả actor và critic.
            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            # Thực hiện quá trình cập nhật trọng số cho mô hình actor.
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            # Thực hiện quá trình cập nhật trọng số cho mô hình critic.
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            # Lưu trữ các giá trị phần thưởng và mất mát vào danh sách tương ứng.
            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            # In thông tin sau mỗi 100 lô dữ liệu được xử lý.
            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                # Tính toán mất mát và phần thưởng trung bình cho epoch.
                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Tạo đường dẫn cho thư mục checkpoint của epoch hiện tại.
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        # Lưu trạng thái của mô hình critic.
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Lưu hiển thị của chu trình hợp lệ
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        # Thực hiện xác thực và lấy phần thưởng trung bình trên tập dữ liệu xác thực.
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        # Kiểm tra xem kết quả xác thực có tốt hơn kết quả tốt nhất trước đó không.
        if mean_valid < best_reward:

            # Cập nhật phần thưởng tốt nhất.
            best_reward = mean_valid

            # Lưu trạng thái tốt nhất của mô hình actor.
            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            # Lưu trạng thái tốt nhất của mô hình critic
            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        # In thông tin tổng kết về mất mát và phần thưởng trung bình của epoch,
        # cùng với thời gian huấn luyện.
        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))


def train_vrp(args):


    print('Starting VRP training')

    # Xác định tải trọng của xe tương ứng với số nút trong các trường hợp
    LOAD_DICT = {5: 10, 10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)
    print('Actor: {} '.format(actor))

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()    
    
    train_vrp(args)