from kuka import KukaCamEnv1, KukaCamEnv2, KukaCamEnv3
import numpy as np
import time

def actor(s):
    x_n = s[0]
    y_n = s[1]
    height = s[2]            # 机械臂末端位置
    gripper_angle = s[5]     # 末端执行器绕z轴角度
    finger_angle = s[6]          # 夹爪角度
    finger_force = s[7]         # 夹爪力
    # 两物体位姿，1为操作物体，2为目标物体
    pos1 = s[8:11]
    orn1 = s[11:14]
    pos2 = s[14:17]
    orn2 = s[17:20]
    action = np.zeros(5)

    #############################################################################
    # 控制器代码编写
    ##############################################################################
    action[0] = 0
    action[1] = 0
    action[2] = 0
    action[3] = 0.1
    action[4] = 0
    ##############################################################################

    return action


def test_actor(n_episodes=100, render=True, add_noise=False):
    env = KukaCamEnv2(renders=render, image_output=False)
    max_steps = 100
    success_count = 0
    sum_L = 0
    misbehavior_count =0
    print("*******************************************")
    for n in range(n_episodes):
        o, s = env.reset() # 环境初始化

        frame = 0
        R = 0
        while True:
            time.sleep(0.1)
            a = actor(s)   # 控制信号
            if add_noise:
                a += 0.1 * np.random.normal(0, 1, 5)
            o_next, s_next, r, done,_ = env.step(a) # 执行控制指令，获取新的状态
            s = s_next
            frame += 1
            R += r
            if done or frame >= max_steps:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break

    print('Average time in executing the task is', sum_L / success_count, ';\n'
                                                                          'Success rate in', n_episodes,
          'episodes is', success_count / n_episodes, ';\n'
                                                     'Misbehavior rate in', n_episodes, 'episodes is',
          misbehavior_count / n_episodes, ';\n')
    print("*******************************************")


if __name__ == '__main__':
    test_actor(10, render=True,add_noise=False)
