clear;
close all;
%% 信号参数
B = 2e6;
Ts = 1e-4;
Tp = 1e-3;
K = B/Ts;
fs = 2 * B;
N = Tp*fs;
t = linspace(-Tp/2,Tp/2,N);
f = (-N/2:N/2-1) * (fs / N);  % 计算频率轴
%% LFM信号
st = rectpuls(t,Ts).*exp(1i*pi*K.*t.^2);
St_f = fftshift(fft(st));
figure;
plot(f, abs(St_f) / max(abs(St_f)));  % 绘制归一化后的频谱幅度
title('LFM 信号的频域幅度谱');
xlabel('频率 (Hz)');
ylabel('归一化幅度');
grid on;
% figure;
% plot(t,real(st))
% title('LFM 信号的实部')  % 为LFM信号的实部加上标题
% xlabel('时间 (秒)')
% ylabel('幅度')
% 匹配滤波器
Hf = conj(fft(st,N));
f_n = 1e6;  % 设置频率偏移
hs_f = Hf.* exp(1j * 2 * pi * f_n * t);  % 频移操作
% 脉冲压缩
sout = fftshift(ifft(fft(st).*Hf));
% 脉冲压缩偏移
sout1 = fftshift(ifft(fft(st).*hs_f));
figure;
plot(t,abs(sout1))
title('LFM 信号的脉冲压缩频移结果')  % 为脉冲压缩结果加上标题
xlabel('时间 (秒)')
ylabel('幅度')

figure;
plot(t,abs(sout))
title('LFM 信号的脉冲压缩结果')  % 为脉冲压缩结果加上标题
xlabel('时间 (秒)')
ylabel('幅度')
%% 加噪
st_n = awgn(st,-10);
figure; 
plot(t,real(st_n))
title('加噪后的LFM信号的实部')  % 为加噪后的信号加上标题
xlabel('时间 (秒)')
ylabel('幅度')
%% 脉冲压缩后的加噪信号
sout_n = fftshift(ifft(fft(st_n).*Hf));
figure;
plot(t,abs(sout_n))
title('加噪后的LFM信号的脉冲压缩结果')  % 为加噪后的脉冲压缩结果加上标题
xlabel('时间 (秒)')
ylabel('幅度')
%% 分贝处理
abs_st = abs(sout_n);
st_max = max(abs_st);
nor_st = abs_st / st_max;
db_st = 10 * log10(nor_st);
figure;
plot(t, db_st)
title('分贝处理后的图像')  % 添加标题
xlabel('时间 (秒)')  % 添加横轴标签
ylabel('幅度 (dB)')  % 添加纵轴标签
grid on;  % 添加网格线
P_fa = 1e-3;          % Desired probability of false alarm (Pfa)
num_training = 12;    % Number of training cells
num_guard = 4;        % Number of guard cells
half_window = num_training + num_guard;   % Half window size (left or right)
cfar_output = zeros(1, length(db_st));
threshold = zeros(1, length(db_st));
for i = half_window+1:length(db_st)-half_window
    % Define the training and guard cells
    training_cells = [db_st(i-half_window:i-num_guard-1), db_st(i+num_guard+1:i+half_window)];
    
    % Estimate the noise level (mean of training cells)
    noise_level = mean(training_cells);
    
    % Determine the threshold (based on the P_fa)
    threshold(i) = noise_level * (P_fa^(-1/(2*num_training)) - 1);
    
    % Compare with the CUT (Cell Under Test)
    if db_st(i) > threshold(i)
        cfar_output(i) = 1;  % Detected
    else
        cfar_output(i) = 0;  % Not detected
    end
end
figure;
hold on;
plot(db_st, 'b', 'LineWidth', 1);
plot(threshold, 'r--', 'LineWidth', 1);
position1 = find(cfar_output == 1);  
plot(position1, db_st(position1), 'go', 'MarkerSize', 5, 'LineWidth', 1);  % 在检测到的目标位置绘制红色圆圈
title('CFAR 检测结果（带白噪声）');
hold off;
legend('带噪声的信号', '检测阈值', '检测到的目标');
grid on;
hold off;