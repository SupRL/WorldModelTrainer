import math
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, "client"))

class MultiJointEnv():
    def __init__(self, *args, **kwargs):
        self.state_round_num = kwargs.get("state_round_num", 5)
        self.joint_name = kwargs.get("joint_name", 'boom_arm_bucket_swing')
        self.policy_joints = self.joint_name.split("_")
        self.pos_min_max_dic = {'boom': [-math.pi, math.pi], 'arm': [-math.pi, math.pi],
                                'bucket': [-math.pi, math.pi], 'swing': [-math.pi, math.pi]}
        self.vel_min_max_dic = {'boom': [-0.4, 0.5], 'arm': [-0.55, 0.65],
                                'bucket': [-1.0, 1.0], 'swing': [-0.82, 0.82]}
        self.pwm_min_max = {'boom': {"negative": [-800, -250], "positive": [250, 800]},
                            'arm': {"negative": [-800, -250], "positive": [250, 800]},
                            'bucket': {"negative": [-600, -250], "positive": [250, 600]},
                            'swing': {"negative": [-450, -180], "positive": [180, 450]}}
        self.seq_length = kwargs.get("state_seq_length", 20)
        
    @property
    def joint_num(self):
        return len(self.policy_joints)
        



