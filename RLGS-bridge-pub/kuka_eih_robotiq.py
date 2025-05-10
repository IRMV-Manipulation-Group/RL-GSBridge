import pybullet as p
import numpy as np
import time
from utils.view_utils import obj2world_transform, matrix_to_quaternion
from collections import namedtuple

def setCameraPicAndGetPic(robot_id: int, width: int = 512, height: int = 512, physicsClientId: int = 0, fov: int = 50):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    state = p.getLinkState(robot_id, physicsClientId)
    basePos = state[0]
    baseOrientation = state[1]

    #basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)

    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴
    ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的z轴

    basePos = np.array(basePos)
    # 摄像头的位置
    # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
    # BASE_RADIUS = 0.05
    # BASE_THICKNESS = 0.0
    # cameraPos = basePos + BASE_RADIUS * tx_vec - 0.5 * BASE_THICKNESS * ty_vec
    # targetPos = cameraPos + 1 * tz_vec
    BASE_RADIUS = 0.065
    BASE_THICKNESS = 0.0
    cameraPos = basePos + BASE_RADIUS * tx_vec - 0.5 * BASE_THICKNESS * ty_vec - 0.045 * tz_vec
    targetPos = cameraPos + 1 * tz_vec - 0.087 * tx_vec
    #print("end effector pos:", basePos6)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tx_vec,
        physicsClientId=physicsClientId
    )
    #print("cam pos", cameraPos)
    #print(np.array(viewMatrix).reshape(4, 4).T)
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=fov,  # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,  # 摄像头焦距下限
        farVal=20,  # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    img_arr = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
    )
    #print(segImg)

    extrinsic = np.array(viewMatrix).reshape(4, 4).T
    # P_matrix = np.array(projectionMatrix).reshape(4, 4)
    # fx = P_matrix[0, 0] * width / 2
    # fy = P_matrix[1, 1] * height / 2
    # cx = width / 2
    # cy = height / 2
    # intrinsic = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    #print(intrinsic)
    return img_arr, extrinsic

class Kuka:
    def __init__(self, urdfPath):
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # rest poses for null space
        self.rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]
        #self.kukaUid = p.loadSDF(urdfPath)[0]
        self.kukaUid = p.loadURDF(urdfPath)#, useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)#[0]
        print("SDF model loaded")
        self.numJoints = p.getNumJoints(self.kukaUid)
        print("num joints:, ", self.numJoints)
        self.endEffectorPos = [0.55, 0.0, 0.6]
        self.endEffectorAngle = np.pi#np.pi#np.pi/2
        self.fingerAngle = 0.0
        # self.jointPositions = [0.0070825, 0.380528, - 0.009961, - 1.363558, 0.0037537, 1.397523, - 0.00280725,
        #                        np.pi, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0]
        self.jointPositions = [0.0070825, 0.380528, - 0.009961, - 1.363558, 0.0037537, 1.397523, - 0.00280725,
                               np.pi, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0, 0.0, 0.0]
        self.motorIndices = []
        self.__parse_joint_info__()
        self.__post_load__()

        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            qIndex = p.getJointInfo(self.kukaUid, jointIndex)[3]
            #print(p.getJointInfo(self.kukaUid, jointIndex))
            #print("written")
            #print(jointIndex)
            if qIndex > -1:
                self.motorIndices.append(jointIndex)

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.kukaUid)
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.kukaUid, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.kukaUid, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint' ### 电机位置
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}


        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            #print('joint name:', joint_id)
            c = p.createConstraint(self.kukaUid, self.mimic_parent_id,
                                   self.kukaUid, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def reset(self):
        self.endEffectorPos = [0.55, 0.0, 0.6]
        self.endEffectorAngle = np.pi#np.pi/4
        self.fingerAngle = 0.0
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.kukaUid, 7)
        finger_state = p.getJointState(self.kukaUid, 8)
        
        finger_angle = -finger_state[0]
        finger_force = finger_state[3]
        
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
        observation.extend([finger_angle, finger_force])
        return observation

    def applyAction(self, motorCommands):
        dx = motorCommands[0]
        dy = motorCommands[1]
        dz = motorCommands[2]
        da = motorCommands[3]
        df = motorCommands[4]
        self.endEffectorPos[0] = min(max(self.endEffectorPos[0] + dx, 0.35), 0.75)
        self.endEffectorPos[1] = min(max(self.endEffectorPos[1] + dy, -0.2), 0.2)
        self.endEffectorPos[2] = min(max(self.endEffectorPos[2] + dz, 0.25), 0.65)
        self.fingerAngle = min(max(self.fingerAngle + df, 0.0), 0.8)
        #print('angel:', self.fingerAngle)
        self.endEffectorAngle += da
        pos = self.endEffectorPos
        orn = p.getQuaternionFromEuler([np.pi, 0, np.pi])
        self.setInverseKine(pos, orn, self.fingerAngle )

    def setInverseKine(self, pos, orn, fingerAngle):
        jointPoses = p.calculateInverseKinematics(self.kukaUid, 6, pos, orn,
                                                  self.ll, self.ul, self.jr, self.rp)
        #print(jointPoses)
        for i in range(7):
            p.setJointMotorControl2(bodyUniqueId=self.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                    maxVelocity=1.0, positionGain=0.3, velocityGain=1)

        fingerAngle = 0.8 - fingerAngle
        #### 这里输入0是张开，0.8是全闭合
        #print("now finger angle:", fingerAngle)
        p.setJointMotorControl2(self.kukaUid, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=fingerAngle,
                                force=2.5, #self.joints[self.mimic_parent_id].maxForce,
                                maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

        p.setJointMotorControl2(self.kukaUid, 7, p.POSITION_CONTROL,
                                targetPosition=self.endEffectorAngle, force=200)


class Object:
    def __init__(self, urdfPath, block, pos=[0, 0, 0], euler=[0, 0, 0], trans=None):
        self.path = urdfPath
        self.block = block
        self.id = p.loadURDF(urdfPath)
        #self.half_height = 0.025 if block else 0.0745
        self.half_height = pos[2] if block else 0.07453946168118829
        #self.half_height = 0.021957018536702427 if block else 0.07453946168118829
        #self.half_height = 0 if block else 0.07453946168118829
        quat = p.getQuaternionFromEuler(euler)
        self.init_trans = obj2world_transform(np.array(pos), quat)
        if trans is not None:
            self.init_trans = trans

    def reset(self, scale=1, rand_center = None, plate = False):
        if scale != 1:
            p.removeBody(self.id)
            self.id = p.loadURDF(self.path, globalScaling=scale)
            #self.half_height = 0.025 if block else 0.0745
            #self.half_height = 0.021957018536702427 if self.block else 0.07453946168118829
            self.half_height = self.half_height * scale
            #self.half_height = 0 if block else 0.07453946168118829
        #print('rand_center:', rand_center is None)
        while True:
            if rand_center is None:
                rand_x = np.random.uniform(0.4, 0.7)
                rand_y = np.random.uniform(-0.15, 0.15)
                rand_yaw = np.random.uniform(-np.pi / 4, np.pi / 4)
            else:
                # rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05 , 0.15), 0.4, 0.7) ### -0.05 0.1
                # rand_y = np.clip(rand_center[1] + np.random.uniform(-0.15, 0.15), -0.15, 0.15) ### -0.1 0.1
                # rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05 , 0.1), 0.4, 0.7) ### -0.05 0.1
                # rand_y = np.clip(rand_center[1] + np.random.uniform(-0.1, 0.1), -0.15, 0.15) ### -0.1 0.1
                rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05, 0.25), 0.4, 0.7)  ### -0.05 0.1
                rand_y = np.clip(rand_center[1] + np.random.uniform(-0.15, 0.15), -0.15, 0.15)  ### -0.1 0.1
                rand_yaw = np.random.uniform(-np.pi / 4, np.pi / 4)
            if plate and rand_x<0.565 and rand_y >-0.13 and rand_y<0.13 :
                continue
            else:
                break

        p.resetBasePositionAndOrientation(self.id,
                                        np.array([rand_x, rand_y,
                                                    self.half_height]),
                                        p.getQuaternionFromEuler([0, 0, rand_yaw]))

        return [rand_x, rand_y, rand_yaw]

    def reset_quad(self, quad, scale=1, rand_center = None, plate = False):
        if quad == 1:
            bond1 = 0.55
            bond2 = 0.7
            bond3 = -0.15
            bond4 = 0
        elif quad ==2:
            bond1 = 0.55
            bond2 = 0.7
            bond3 = 0
            bond4 = 0.15
        elif quad ==3:
            bond1 = 0.4
            bond2 = 0.55
            bond3 = -0.15
            bond4 = 0
        elif quad ==4:
            bond1 = 0.4
            bond2 = 0.55
            bond3 = 0
            bond4 = 0.15  
            
        if scale != 1:
            p.removeBody(self.id)
            self.id = p.loadURDF(self.path, globalScaling=scale)
            #self.half_height = 0.025 if block else 0.0745
            #self.half_height = 0.021957018536702427 if self.block else 0.07453946168118829
            self.half_height = self.half_height * scale
            #self.half_height = 0 if block else 0.07453946168118829
        #print('rand_center:', rand_center is None)
        while True:
            if rand_center is None:
                rand_x = np.random.uniform(bond1, bond2)
                rand_y = np.random.uniform(bond3, bond4)
                rand_yaw = np.random.uniform(-np.pi / 4, np.pi / 4)
            else:
                # rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05 , 0.15), 0.4, 0.7) ### -0.05 0.1
                # rand_y = np.clip(rand_center[1] + np.random.uniform(-0.15, 0.15), -0.15, 0.15) ### -0.1 0.1
                # rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05 , 0.1), 0.4, 0.7) ### -0.05 0.1
                # rand_y = np.clip(rand_center[1] + np.random.uniform(-0.1, 0.1), -0.15, 0.15) ### -0.1 0.1
                rand_x = np.clip(rand_center[0] + np.random.uniform(-0.05, 0.25), 0.4, 0.7)  ### -0.05 0.1
                rand_y = np.clip(rand_center[1] + np.random.uniform(-0.15, 0.15), -0.15, 0.15)  ### -0.1 0.1
                rand_yaw = np.random.uniform(-np.pi / 4, np.pi / 4)
            if plate and rand_x<0.565 and rand_y >-0.13 and rand_y<0.13 :
                continue
            else:
                break

        p.resetBasePositionAndOrientation(self.id,
                                        np.array([rand_x, rand_y,
                                                    self.half_height]),
                                        p.getQuaternionFromEuler([0, 0, rand_yaw]))

        return [rand_x, rand_y, rand_yaw]

    def reset_norand(self, pos, euler):
        p.resetBasePositionAndOrientation(self.id,
                                          pos,
                                          p.getQuaternionFromEuler(euler))
        
        
    def pos_and_rot(self):
        pos, quat = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(quat)
        trans = obj2world_transform(np.array(pos), quat)
        trans_r = trans @ np.linalg.inv(self.init_trans)
        rot_mat_r = trans_r[:3, :3]
        quat_r = matrix_to_quaternion(rot_mat_r)
        #print("trans_r:", trans_r)
        #print("pos height:", pos[2])
        #print("initial pos:", np.linalg.inv(transform_matrix))
        return pos, euler, quat_r, trans_r


def check_pairwise_collisions(bodies):
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


class KukaCamEnvBase:
    def __init__(self, object1_path, object1_shape, obj1_pos, obj1_euler, object2_path, object2_shape, obj2_pos, obj2_euler, 
                 renders=False, image_output=True, mode='de', width=512, fov = 50.0, loadmesh = False, bg_pos=[0, 0, -0.625], bg_urdf=None, task=1):
        self._timeStep = 0.02
        self._renders = renders
        self._image_output = image_output
        self._mode = mode
        self._width = width
        self._height = self._width
        self._fov = fov
        self._p = p
        self.task = task
        self.ok_grasp = False
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.0, 230, -40, [0.55, 0, 0])
        else:
            p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)
        if bg_urdf is not None:
            print('To be implemented')
        if loadmesh and task==3:
            p.loadURDF("models/floor.urdf", [0, 0, bg_pos[2]], useFixedBase=True)
            p.loadURDF("mesh_models/bg_plate.urdf", [0., 0, bg_pos[2]], useFixedBase=True)
            print('load table mesh')
        else:
            p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
            p.loadURDF("models/table_collision/table.urdf", [0.6, 0, -0.625], useFixedBase=True)

        self._kuka = Kuka("models/kuka_iiwa/kuka_robotiq_140.urdf")


        print('obj1_pos', obj1_pos)
        self._object1 = Object(object1_path, block=object1_shape, pos = obj1_pos, euler = obj1_euler)
        self._object2 = Object(object2_path, block=object2_shape, pos = obj2_pos, euler = obj2_euler)

    def reset(self, scale1 = 1, scale2 = 1):
        if self.task == 3 or self.task == 2:
            self.ok_grasp = False
        collision = True
        self.scale = scale1
        while collision:
            if self.task ==3:
                self._object1.reset(scale=scale1, plate=True)
                self._object2.reset(scale=scale2, plate=True)
            elif self.task ==2:
                rand_center = self._object1.reset(scale=scale1)
                #self._object2.reset(scale=scale2)
                self._object2.reset(scale=scale2, rand_center=rand_center)
            else:
                rand_center = self._object1.reset(scale=scale1)
                #self._object2.reset(scale=scale2)
                self._object2.reset(scale=scale2)
            collision = check_pairwise_collisions([self._object1.id, self._object2.id])
        self._kuka.reset()
        p.stepSimulation()
        return self.getExtendedObservation()

    def reset_fix(self, pos1, eul1, pos2, eul2,scale1 = 1, scale2 = 1):
        if self.task == 3 or self.task == 2:
            self.ok_grasp = False
        collision = True
        self.scale = scale1
        while collision:
            if self.task ==3:
                self._object1.reset_norand(pos1, eul1)
                self._object2.reset_norand(pos2, eul2)
            elif self.task ==2:
                rand_center = self._object1.reset(scale=scale1)
                #self._object2.reset(scale=scale2)
                self._object2.reset(scale=scale2, rand_center=rand_center)
            else:
                self._object1.reset_norand(pos1, eul1)
                self._object2.reset_norand(pos2, eul2)
            collision = check_pairwise_collisions([self._object1.id, self._object2.id])
        self._kuka.reset()
        p.stepSimulation()
        return self.getExtendedObservation()
    
    def reset_quad(self, quad, scale1 = 1, scale2 = 1):
        if self.task == 3 or self.task == 2:
            self.ok_grasp = False
        collision = True
        self.scale = scale1
        while collision:
            if self.task ==3:
                self._object1.reset_quad(quad, scale=scale1, plate=True)
                self._object2.reset(scale=scale2, plate=True)
            elif self.task ==2:
                rand_center = self._object1.reset_quad(quad, scale=scale1)
                #self._object2.reset(scale=scale2)
                self._object2.reset(scale=scale2, rand_center=rand_center)
            else:
                rand_center = self._object1.reset_quad(quad, scale=scale1)
                #self._object2.reset(scale=scale2)
                self._object2.reset(scale=scale2)
            collision = check_pairwise_collisions([self._object1.id, self._object2.id])
        self._kuka.reset()
        p.stepSimulation()
        return self.getExtendedObservation()

    def __del__(self):
        p.disconnect()

    def getExtendedObservation(self):
        observation = np.zeros((4 if self._mode == 'rgbd' else 6, self._height, self._width), dtype=np.uint8)
        if self._image_output:  # for speeding up test, image output can be turned off
            camEyePos = [0.55, 0, 0]
            distance = 0.8
            pitch = -60
            yaw = 180
            roll = 0
            upAxisIndex = 2
            nearPlane = 0.01
            farPlane = 1000
            fov = 45
            viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
            projMatrix = p.computeProjectionMatrixFOV(fov, 1, nearPlane, farPlane, physicsClientId=0)

            #img_arr = p.getCameraImage(width=self._width, height=self._height,
            #                           viewMatrix=viewMat, projectionMatrix=projMatrix)
            img_arr, extrinsic = setCameraPicAndGetPic(self._kuka.kukaUid, self._width, self._height, physicsClientId=7, fov = self._fov)
            rgb = img_arr[2]
            mask = img_arr[4]
            observation[0] = rgb[:, :, 0]
            observation[1] = rgb[:, :, 1]
            observation[2] = rgb[:, :, 2]
            if self._mode == 'rgbd':
                depth_buffer = img_arr[3].reshape(self._height, self._width)
                observation[3] = np.round(255*farPlane*nearPlane / ((farPlane-(farPlane-nearPlane)*depth_buffer)*1.1))
            else :
                #viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)
                #img_arr2 = p.getCameraImage(width=self._width, height=self._height,
                #                            viewMatrix=viewMat2, projectionMatrix=projMatrix)
                rgb2 = rgb
                observation[3] = rgb2[:, :, 0]
                observation[4] = rgb2[:, :, 1]
                observation[5] = rgb2[:, :, 2]
            
        additional_observation = self._kuka.getObservation()
        Pos1, Euler1, quat1, trans1 = self._object1.pos_and_rot()
        Pos2, Euler2, quat2, trans2 = self._object2.pos_and_rot()
        #Pos3, Euler3, quat3, trans3 = self._object5.pos_and_rot()
        additional_observation.extend(list(Pos1))
        additional_observation.extend(list(Euler1))
        additional_observation.extend(list(Pos2))
        additional_observation.extend(list(Euler2))
        additional_observation = np.array(additional_observation, dtype=np.float32)
        
        if self._image_output:
            mask_gripp =np.zeros_like(mask)
            mask_gripp[mask==2] = 1
            simdata = {
                    'rgb': rgb, 
                    'mask': mask_gripp, 
                    'extrinsic': extrinsic, 
                    'obj_trans_list': [trans1, trans2], 
                    'obj_rot_list': [quat1, quat2]
                }
        else:
            simdata = None
        
        return observation, additional_observation, simdata

    def step(self, action):
        action = np.clip(action, -1, 1)
        dv = 0.008
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        da = action[3] * 0.05
        df = action[4] * 0.1
        realAction = [dx, dy, dz, da, df]
        for i in range(3):
            self._kuka.applyAction(realAction)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        observation, additional_observation, simdata = self.getExtendedObservation()
        done, reward = self.reward()
        return observation, additional_observation, reward, done, simdata

    def reward(self):
        raise NotImplementedError


class KukaCamEnv1(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=512, fov = 50.0, loadmesh = False):
        if loadmesh:
            box_path = "mesh_models/cube.urdf"
            cup_path = "mesh_models/cup.urdf"
        else:
            box_path = "models/box_green.urdf"
            cup_path = "models/cup/cup.urdf"
        pos1 = [0., 0., 0.021957018536702427]
        euler1 = [-0.001635076225049408, -0.005217522706643496, -0.4642806075635298]
        pos2 = [0., 0., 0.07453946168118829]
        euler2 = [-7.035301340448207e-06, -2.980837841155693e-05, -0.4850510993572782]
        super().__init__(box_path, True, pos1, euler1, 
                         cup_path, False, pos2, euler2, 
                         renders=renders, image_output=image_output, mode=mode, width=width, fov = fov, loadmesh = loadmesh)

    def reward(self):
        blockPos, blockOrn, _, _ = self._object1.pos_and_rot()
        cupPos, cupOrn, _, _ = self._object2.pos_and_rot()
        if abs(cupOrn[0]) > 1 or abs(cupOrn[1]) > 1:
            return True, 0.0
        dist = np.sqrt((blockPos[0] - cupPos[0]) ** 2 + (blockPos[1] - cupPos[1]) ** 2)
        #if dist < 0.01 and blockPos[2] - cupPos[2] < 0.05 and abs(cupOrn[0]) < 0.2 and abs(cupOrn[1]) < 0.2:
        if dist < 0.01 and blockPos[2] - cupPos[2] < 0.11 and abs(cupOrn[0]) < 0.2 and abs(cupOrn[1]) < 0.2:
            return True, 1.0
        # print('dist:', dist < 0.01)
        # print('z:', blockPos[2] - cupPos[2] < 0.11)
        # print('abs1:', abs(cupOrn[0]) < 0.2)
        # print('abs2:', abs(cupOrn[1]) < 0.2)
        return False, 0.0


class KukaCamEnv2(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=512, fov = 50.0, loadmesh = False,  bg_pos = None, bg_urdf = None, obj_infos = None):
        if loadmesh:
            if obj_infos is not None:
                box_path = obj_infos[0][0]
                cup_path = obj_infos[0][1]
            else:
                box_path = "mesh_models/cake_2.urdf"
                cup_path = "mesh_models/gift.urdf"
        else:
            box_path = "models/box_green.urdf"
            cup_path = "models/box_purple.urdf"
        if obj_infos is not None:
            pos1 = [0., 0., obj_infos[1][0]]
            euler1 = [0, 0, 0]
            pos2 = [0., 0., obj_infos[1][1]]
            euler2 = [0, 0, 0]
        else:
            pos1 = [0., 0., 0.021957018536702427]
            euler1 = [0, 0, 0]
            pos2 = [0., 0., 0.0168]
            euler2 = [0, 0, 0]
        if bg_pos is None:
            bg_pos = [0, 0, -0.625]
        super().__init__(box_path, True, pos1, euler1, 
                         cup_path, True, pos2, euler2, 
                         renders=renders, image_output=image_output, mode=mode, width=width, fov = fov, loadmesh = loadmesh, bg_pos = bg_pos, bg_urdf=bg_urdf, 
                          task=2)
        self.success_count = 0
        self.grasp_count = 0
    def reward(self):
        blockPos, blockOrn, _, _ = self._object1.pos_and_rot()
        block2Pos, block2Orn, _, _ = self._object2.pos_and_rot()
        s = self._kuka.getObservation()
        finger_force = s[7]
        dist = np.sqrt((blockPos[0] - block2Pos[0]) ** 2 + (blockPos[1] - block2Pos[1]) ** 2)
        if not self.ok_grasp:
            if blockPos[2] > 0.1 and finger_force > 1:#0.076:
                # print("success:True")
                # print('success count:', self.success_count)
                self.grasp_count +=1
                if self.grasp_count > 5:
                    self.grasp_count = 0
                    self.ok_grasp = True
                    return False, 0.5
        if dist < 0.01 and blockPos[2] < 0.078*self.scale:#0.045 0.155:
            #print("success:True")
            # print('success count:', self.success_count)
            self.success_count +=1
            if self.success_count > 15:
                self.success_count = 0
                return True, 0.5
        else:
            self.success_count = 0
        
        return False, 0.0



class KukaCamEnv3(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=128, fov = 50.0, loadmesh = False,  bg_pos = None, bg_urdf = None, obj_infos = None):
        if loadmesh:
            if obj_infos is not None:
                box_path = obj_infos[0][0]
                cup_path = obj_infos[0][1]
            else:
                box_path = "mesh_models/cake_2.urdf" 
                cup_path = "mesh_models/small_cube.urdf"  
        else:
            box_path = "mesh_models/cake_2.urdf" 
            cup_path = "mesh_models/small_cube.urdf" 
        if obj_infos is not None:
            pos1 = [0., 0., obj_infos[1][0]]
            euler1 = [0, 0, 0]
            pos2 = [0., 0., obj_infos[1][1]]
            euler2 = [0, 0, 0]
        else:
            pos1 = [0., 0., 0.021957018536702427] 
            euler1 = [0, 0, 0]
            pos2 = [0., 0., 0.027]
            euler2 = [0, 0, 0] 
        if bg_pos is None:
            bg_pos = [0, 0, -0.625]
        super().__init__(box_path, True, pos1, euler1,
                         cup_path, True, pos2, euler2,
                         renders=renders, image_output=image_output, mode=mode, width=width, fov=fov, loadmesh=loadmesh,
                         bg_pos=bg_pos, bg_urdf=bg_urdf, task=3)
        self.success_count = 0
        self.grasp_count = 0
        

    def reward(self):
        blockPos, blockOrn, _, _ = self._object1.pos_and_rot()
        s = self._kuka.getObservation()
        finger_force = s[7]
        #print("force:", finger_force)
        #block2Pos, block2Orn, _, _ = self._object2.pos_and_rot()
        dist = np.sqrt((blockPos[0] - 0.4) ** 2 + (blockPos[1] - 0) ** 2)
        #print(blockPos[2])
        if not self.ok_grasp:
            if blockPos[2] > 0.1 and finger_force > 1:#0.076:
                # print("success:True")
                # print('success count:', self.success_count)
                self.grasp_count +=1
                if self.grasp_count > 8:
                    self.grasp_count = 0
                    self.ok_grasp = True
                    return False, 0.5
        if dist < 0.04 and blockPos[2] < 0.043 * self.scale:  # 0.076:
            # print("success:True")
            # print('success count:', self.success_count)
            self.success_count += 1
            if self.success_count > 15:
                self.success_count = 0
                return True, 0.5
        else:
            self.success_count = 0

    #     return False, 0.0
    def reward(self):
        blockPos, blockOrn, _, _ = self._object1.pos_and_rot()
        s = self._kuka.getObservation()
        finger_force = s[7]
        finger_angle = s[6]

        #block2Pos, block2Orn, _, _ = self._object2.pos_and_rot()
        dist = np.sqrt((blockPos[0] - 0.4) ** 2 + (blockPos[1] - 0) ** 2)

        if dist < 0.04 and blockPos[2] < 0.057 * self.scale:  # 0.043:
            # print("success:True")
            # print('success count:', self.success_count)
            self.success_count += 1
            if self.success_count > 15:
                self.success_count = 0
                return True, 1.0
        else:
            self.success_count = 0

        return False, 0.0
    

class KukaCamEnv4(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=512, fov = 50.0, loadmesh = False,  bg_pos = None, bg_urdf = None, obj_infos = None):
        if loadmesh:
            if obj_infos is not None:
                box_path = obj_infos[0][0]
                cup_path = obj_infos[0][1]
            else:
                box_path = "mesh_models/banana.urdf" 
                cup_path = "mesh_models/cake_2.urdf"
        else:
            box_path = "mesh_models/cake_2.urdf"
            cup_path = "models/box_green.urdf"
        ####TODO 改成对应物体urdf中质心高度坐标
        if obj_infos is not None:
            pos1 = [0., 0., obj_infos[1][0]]
            euler1 = [0, 0, 0]
            pos2 = [0., 0., obj_infos[1][1]]
            euler2 = [0, 0, 0]
        else:
            pos1 = [0., 0., 0.0245]
            euler1 = [0, 0, 0]
            pos2 = [0., 0., 0.021957018536702427]
            euler2 = [0, 0, 0]
        if bg_pos is None:
            bg_pos = [0, 0, -0.625]
        super().__init__(box_path, True, pos1, euler1, 
                         cup_path, True, pos2, euler2, 
                         renders=renders, image_output=image_output, mode=mode, width=width, fov = fov, loadmesh = loadmesh, bg_pos = bg_pos, bg_urdf = bg_urdf)
        self.success_count = 0
    def reward(self):
        blockPos, blockOrn, _, _ = self._object1.pos_and_rot()
        block2Pos, block2Orn, _, _ = self._object2.pos_and_rot()
        s = self._kuka.getObservation()
        finger_angle = s[6]
        finger_force = s[7]
        if blockPos[2] > 0.1 and finger_force > 1:#0.076:
            # print("success:True")
            # print('success count:', self.success_count)
            self.success_count +=1
            if self.success_count > 15:
                self.success_count = 0
                return True, 1.0
        else:
            self.success_count = 0
        
        return False, 0.0
