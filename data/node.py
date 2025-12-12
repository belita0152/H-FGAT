import torch
import numpy as np
import torch.nn as nn


def np_to_var(x, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
  if not hasattr(x, "__len__"):
      x = [x]
  x = np.asarray(x)  # scalar and array를 처리할 때 일괄적으로 np.array로 반환
  if dtype is not None:
      x = x.astype(dtype)  # 특정 데이터 타입이 지정되어 있으면, numpy 단계에서 먼저 타입 변환
  x_tensor = torch.tensor(x, requires_grad=requires_grad, **tensor_kwargs)  # np.array -> torch.tensor
  if pin_memory:
      x_tensor = x_tensor.pin_memory()
  return x_tensor


class Conv2dWithConstraint(nn.Conv2d):  # EEG처럼 noise가 많은 데이터 학습 시, 필터가 너무 예민하게 반응하지 않도록 강제
  def __init__(self, *args, max_norm=1, **kwargs):
      self.max_norm = max_norm
      super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

  def forward(self, x):
      self.weight.data = torch.renorm(
          self.weight.data, p=2, dim=0, maxnorm=self.max_norm
      )  # kernel 하나하나 L2 norm으로 가중치 크기를 재서, max_norm 보다 크면 max_norm이 되도록 비율을 맞춰 줄임
      return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
  def __init__(self, f1, f2, d, input_time_length, embedding_dim, dropout_rate, sampling_rate, classes):
    super(EEGNet, self).__init__()

    # channel_size = 1  # 채널별 처리를 위해 spatial kernel의 높이는 1로 설정
    cnn_kernel_1 = int(sampling_rate // 2)

    self.classes = classes
    self.cnn = nn.Sequential()

    # Block 1: Temporal Conv
    self.cnn.add_module(
      name='conv_temporal',
      module=Conv2dWithConstraint(
          in_channels=1,
          out_channels=f1,
          kernel_size=(1, cnn_kernel_1),
          stride=1,
          bias=False,
          padding=(0, int(cnn_kernel_1 // 2))
      )
    )
    self.cnn.add_module(
        name='batch_normalization_1',
        module=nn.BatchNorm2d(f1)
    )

    # Block 2: Spatial Conv (1x1 conv)
    self.cnn.add_module(
        name='conv_spatial',
        module=Conv2dWithConstraint(
            in_channels=f1,
            out_channels=f1 * d,
            kernel_size=(1, 1),
            max_norm=1,
            stride=1,
            bias=False,
            groups=f1,
            padding=(0, 0))
    )
    self.cnn.add_module(
        name='batch_normalization_2',
        module=nn.BatchNorm2d(f1 * d, momentum=0.01, affine=True, eps=1e-3)
    )
    self.cnn.add_module(
        name='activation_1',
        module=nn.ELU()
    )
    self.cnn.add_module(
        name='average_pool_2d_1',
        module=nn.AvgPool2d(kernel_size=(1, 4))
    )
    self.cnn.add_module(
        name='dropout_rate_1',
        module=nn.Dropout(dropout_rate)
    )

    # Block 3: Separable Depthwise Convolution (채널별 3x3 -> 합쳐서 1x1)
    self.cnn.add_module(
        name='conv_separable_depth',
        module=nn.Conv2d(
            in_channels=f1 * d,
            out_channels=f1 * d,
            kernel_size=(1, 16),
            stride=1,
            bias=False,
            groups=f1 * d,
            padding=(0, 16 // 2),
        )
    )
    self.cnn.add_module(
        name='batch_normalization_3',
        module=nn.BatchNorm2d(f2),
    )
    self.cnn.add_module(
        name='activation_2',
        module=nn.ELU()
    )
    self.cnn.add_module(
        name='average_pool_2d_2',
        module=nn.AvgPool2d(kernel_size=(1, 8))
    )
    self.cnn.add_module(
        name='dropout_rate_2',
        module=nn.Dropout(dropout_rate)
    )
    out = self.cnn(
            np_to_var(
                np.ones(
                    (1, 1, 1, input_time_length),
                    dtype=np.float32,
                )
            )
    )
    final_length = out.reshape(-1).shape[0]

    self.fc = nn.Sequential()
    self.fc.add_module(
        name='fully_connected',
        module=nn.Linear(
            in_features=final_length,
            out_features=embedding_dim
        )
    )

  def forward(self, x):
    b, c, t = x.size()  # (64, 30, 384)
    x = x.view(b * c, 1, 1, t)

    x = self.cnn(x)

    x = x.flatten(start_dim=1) # (B*C, flatten_dim)  torch.Size([1920, 192])
    x = self.fc(x)  # (B*C, out_dim)

    x = x.view(b, c, -1)
    return x
