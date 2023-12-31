#### 将数据集解压后放置在该目录下
- 数据集下载：[数据下载链接](https://XXXXX) 提取码：1234
- 数据集说明：数据集包含了训练数据和验证数据
  - 训练数据：包含了训练数据和数据标签，训练数据用于模型训练
  - 验证数据：包含了验证数据和数据标签，验证数据用于模型调参，**也可以用于模型训练**
- 数据集结构
  - dataset: 数据集文件夹
    - train：训练数据文件夹
      - XXX1：子文件夹
        - XXX1.csv: 具体训练数据文件1
        - XXX2.csv: 具体训练数据文件2
        - ...
      - XXX2：子文件夹
        - XXX1.csv: 具体训练数据文件1
        - XXX2.csv: 具体训练数据文件2
        - ...
      - ...
    - test：验证数据文件夹
      - XXX1：子文件夹
        - XXX1.csv: 具体验证数据文件1
        - XXX2.csv: 具体验证数据文件2
        - ...
      - XXX2：子文件夹
        - XXX1.csv: 具体验证数据文件1
        - XXX2.csv: 具体验证数据文件2
        - ...
      - ...
- 数据载入: 参考train.py中的IterDataset类，该类实现了数据的载入，参赛者可以根据自己的需要进行修改
- CSV文件内各列含义：
  - XXX_pos: XXX关节角度
  - XXX_vel: XXX关节角速度
  - XXX_pwm: XXX关节PWM，液压驱动信号，用于控制挖机关节动作
  - XXX_next_pos: XXX关节下一时刻角度角度（类似于label）
  - XXX_next_vel: XXX关节下一时刻角速度（类似于label）
  - state_time: 收到pos，vel等信息后多久发出了PWM信号（单位：秒）
  - action_time: 发出PWM信号后多久收到了pos，vel等信息（单位：秒） 
