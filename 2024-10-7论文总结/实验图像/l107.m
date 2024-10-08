% 参数定义
num_pulses = 200;           % 矩形脉冲数量
pulse_width = 0.05;         % 每个脉冲的宽度，单位为秒
pulse_period = 0.1;         % 脉冲周期，单位为秒
t_total = num_pulses * pulse_period;  % 总时间
Fs = 100;                   % 采样频率
t_rect = linspace(-t_total/2, t_total/2, Fs * t_total); % 对称时间向量

% 初始化脉冲信号
signal = zeros(size(t_rect));

% 生成关于 x=0 对称的矩形脉冲
for n = -num_pulses/2:num_pulses/2-1
    pulse_start = n * pulse_period; % 每个脉冲的开始时间
    pulse_end = pulse_start + pulse_width; % 每个脉冲的结束时间
    signal(t_rect >= pulse_start & t_rect < pulse_end) = 1; % 将脉冲赋值为1
end
% 线性调频信号部分
T = 10e-6;                  % 脉冲持续时间 10微秒
B = 5e6;                    % 线性调频信号的带宽 5MHz
K = B / T;                  % 调频斜率
Fs_lfm = length(t_rect);    % 确保线性调频信号的采样点数与t_rect相同
t_lfm = linspace(-T/2, T/2, Fs_lfm); % 时间向量用于LFM信号
mu = B / T;                 % mu值
f0 = 0;                     % 中心频率
% 生成线性调频信号
st = exp(1j * (2 * pi * f0 * t_lfm + pi * mu * t_lfm.^2)); % 线性调频信号公式
% 计算 xt
xt = signal .* st; % 矩形脉冲信号和 LFM 信号相乘
% 计算频谱
Xt = fft(xt);
St = fft(st);
% 频率范围 (将其转换为 MHz)
freq = linspace(-Fs/2, Fs/2, length(Xt));
% 绘制时域和频域信号
figure;
subplot(2, 2, 1);
plot(t_lfm, real(st));
title('LFM 信号时域');
xlabel('时间 (秒)');
ylabel('幅度');
grid on;
subplot(2, 2, 2);
plot(freq, fftshift(abs(St)) / max(abs(St)));
title('LFM 信号的频谱');
xlabel('频率 (MHz)');
ylabel('归一化幅度');
grid on;
xlim([-8 8]);  % 设置 X 轴范围为 -8 MHz 到 8 MHz
subplot(2, 2, 3);
plot(t_rect, real(xt));
title('xt 信号时域');
xlabel('时间 (秒)');
ylabel('幅度');
grid on;
subplot(2, 2, 4);
plot(freq, fftshift(abs(Xt)) / max(abs(Xt)));
title('xt 信号的频谱');
xlabel('频率 (MHz)');
ylabel('归一化幅度');
grid on;





freq1 = linspace(-Fs/10, Fs/10, Fs_lfm);
%原始信号回波脉压结果
ht = conj(fliplr(st));  % 时域匹配滤波为发射信号时间反转再取共轭
s1 = conv(st, ht, 'same');  % 线性调频信号经过匹配滤波器后的输出 (时域卷积)
figure;
% 假设速度为v，计算距离
v = 3e8; % 例如，光速 (米/秒)
distance = freq1 / 1e6 * v; % 计算距离
% 计算偏移量
offsets = [1500, 3500, 8500]; % 定义偏移位置
hold on; % 保持当前图形
% 找到abs(s1)中的最大绝对值
max_abs_s1 = max(abs(s1));
% 绘制每个偏移位置的原始结果 (红色，不透明)
for i = 1:length(offsets)
    plot(distance + offsets(i), abs(s1) / max_abs_s1, 'b', 'DisplayName', sprintf('原始结果 (偏移 %dm)', offsets(i)), 'LineWidth', 1); % 红色，不透明
end
title('原始结果 ');
legend('show');
figure;
% 之前的参数定义和信号生成部分
ht = conj(fliplr(st));  % 时域匹配滤波为发射信号时间反转再取共轭
s1 = conv(st, ht, 'same');  % 线性调频信号经过匹配滤波器后的输出 (时域卷积)
% 添加高斯白噪声
bandwidth = 8e6; % 带宽为8 MHz
signal_power = mean(abs(s1).^2); % 计算s1的信号功率
target_snr = -5; % 目标信噪比，调低以减少噪声强度
noise_power = signal_power / (10^(target_snr / 10)); % 计算噪声功率
noise_std = sqrt(noise_power); % 计算噪声标准差
% 生成高斯白噪声
fs = 20e6; % 采样频率（请根据实际情况设置）
t_noise = (0:length(s1)-1) / fs; % 生成噪声的时间向量
noise = noise_std * randn(size(s1)); % 生成高斯白噪声
% 将噪声添加到s1中
s1_noisy = s1 + noise; % 添加噪声到信号
% 假设速度为v，计算距离
v = 3e8; % 例如，光速 (米/秒)
distance = freq1 / 1e6 * v; % 计算距离
% 计算偏移量
offsets = [1500, 3500, 8500]; % 定义偏移位置
hold on; % 保持当前图形
% 找到abs(s1_noisy)中的最大绝对值
max_abs_s1_noisy = max(abs(s1_noisy));
% 绘制每个偏移位置的有噪声结果 (蓝色，不透明)
for i = 1:length(offsets)
    plot(distance + offsets(i), abs(s1_noisy) / max_abs_s1_noisy, 'b', 'DisplayName', sprintf('有噪声结果 (偏移 %dm)', offsets(i)), 'LineWidth', 1); % 蓝色
end
% 绘制每个偏移位置的原始结果 (红色，不透明)
max_abs_s1 = max(abs(s1)); % 原始结果的最大绝对值
title('匹配滤波结果及噪声');
xlabel('距离 (米)');
ylabel('归一化幅度');
legend('show');
grid on;







figure;
f_n = 1e6;  % 设置频率偏移
hs_t = ht.* exp(1j * 2 * pi * f_n * t_lfm);  % 频移操作
y_x=conv(st,hs_t,'same');  
D = K / T;
n = -100:100;  % 定义n的范围
C = n * pi / sin(n * D * pi);
y_compensated = C.* y_x;
% 假设速度为 v，计算距离
v = 3e8; % 例如，光速 (米/秒)
distance = freq1 / 1e6 * v; % 计算距离
% 计算偏移量
offsets = [1500, 3500, 8500]; % 定义偏移位置
hold on; % 保持当前图形
% 绘制每个偏移位置的原始结果 (红色，不透明)
for i = 1:length(offsets)
    plot(distance + offsets(i), abs(y_x), 'k', 'DisplayName', sprintf('原始结果 (偏移 %dm)', offsets(i)), 'LineWidth', 1.5); 
end
% 绘制每个偏移位置的补偿处理结果 (蓝色，透明度为0.5)
for i = 1:length(offsets)
    plot(distance + offsets(i), abs(y_compensated), 'Color', [1, 0, 0, 0.3], 'DisplayName', sprintf('补偿处理结果 (偏移 %dm)', offsets(i))); 
end
% 添加标题和标签
title('信号比较');
xlabel('距离 (米)'); % 修改单位为距离
ylabel('幅度');
grid on; % 添加网格
% 添加图例
legend('show'); % 显示图例




%%

T_p = 10e-6; % 脉宽，单位为秒
mu = B / T_p;
% f = linspace(-10e6, 10e6, 1000);
st_1_m = exp(1j * (2 * pi * f0 * t_lfm + pi * mu * t_lfm.^2)); % 线性调频信号公式
xt_1_m = signal .* st_1_m; 
fs = 20e6; % 频移，单位为Hz，添加一个频移值以进行计算
st_m = exp(1j * (2 * pi * (f0+fs) * t_lfm + pi * mu * t_lfm.^2)); % 线性调频信号公式
xt_m=signal.*st_m;
st_1_m_f = fft(xt_1_m);
st_m_f = fft(xt_m);
% 绘制合成图形
figure;
% 计算归一化因子
magnitude_st1 = fftshift(abs(st_1_m_f));
magnitude_st2 = fftshift(abs(st_m_f));
max_st1 = max(magnitude_st1);
max_st2 = max(magnitude_st2);
% 绘制原始参考信号频谱
hold on; % 保持当前图形
plot(freq, magnitude_st1 / max_st1, 'b', 'DisplayName', '原始信号'); % 归一化，蓝色线
plot(freq, magnitude_st2 / max_st2, 'r', 'DisplayName', '频移后信号'); % 归一化，红色线

% 添加标题和标签
title('信号频谱比较');
xlabel('频率 (MHz)');
ylabel('幅度 (归一化)');
grid on;

% 添加图例
legend('show'); % 显示图例




%% 添加干扰和匹配滤波器
fs = 20e6; % 采样率
t = (0:length(xt)-1) / fs; % 确保时间向量与信号长度匹配
signal_power = mean(abs(xt).^2); % 计算信号功率
target_snr = -25; % 目标信噪比
noise_power = signal_power / (10^(target_snr / 10)); % 计算噪声功率
noise_std = sqrt(noise_power); % 计算噪声标准差
noise = noise_std * randn(size(xt)); % 生成高斯噪声
s_j_t = xt + noise; % 添加噪声到信号

% 匹配滤波器
ht = conj(fliplr(st));  % 匹配滤波器
y_s2 = conv(s_j_t, ht, 'same');  % 卷积结果

% 对滤波器输出进行归一化处理
y_s2_norm = abs(y_s2) / max(abs(y_s2)); % 将结果归一化

% 绘制归一化后的结果
v = 3e8; % 光速 (米/秒)
distance = freq1 / 1e6 * v; % 计算距离
offsets = [1500, 3500, 8500]; % 定义偏移位置
figure;
hold on;
for i = 1:length(offsets)
    plot(distance + offsets(i), y_s2_norm, 'b'); % 绘制归一化后的结果
end
title('-25信噪比的图像');
xlabel('距离 (米)');
ylabel('归一化幅度');
ylim([0 1]); % 限制Y轴范围在0到1之间
grid on;


