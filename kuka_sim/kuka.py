import pybullet as p
import numpy as np
import time


class Kuka:
    def __init__(self, urdfPath):
        # lower limits for null space
        # self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # # upper limits for null space
        # self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        self.ll = [-.967, -2, -2.96, -2.09, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.09, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # rest poses for null space
        self.rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]
        self.kukaUid = p.loadSDF(urdfPath)[0]
        self.numJoints = p.getNumJoints(self.kukaUid)
        self.endEffectorPos = [0.55, 0.0, 0.6]
        self.endEffectorAngle = 0
        self.fingerAngle = 0.0

        self.jointPositions = [0.0070825, 0.380528, - 0.009961, - 1.363558, 0.0037537, 1.397523, - 0.00280725,
                                np.pi, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0]

        self.motorIndices = []
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            qIndex = p.getJointInfo(self.kukaUid, jointIndex)[3]
            if qIndex > -1:
                self.motorIndices.append(jointIndex)

    def reset(self):
        self.endEffectorPos = [0.55, 0.0, 0.6]

        self.endEffectorAngle = 0
        self.fingerAngle = 0.0
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.kukaUid, 7)
        #state = p.getLinkState(self.kukaUid, 6)
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
        self.fingerAngle = min(max(self.fingerAngle + df, 0.0), 0.4)
        self.endEffectorAngle += da
        pos = self.endEffectorPos
        orn = p.getQuaternionFromEuler([np.pi, 0, np.pi-self.endEffectorAngle])
        joint_p = self.setInverseKine(pos, orn, self.fingerAngle)
        return joint_p


    def setInverseKine(self, pos, orn, fingerAngle):
        jointPoses = p.calculateInverseKinematics(self.kukaUid, 6, pos, orn,
                                                  self.ll, self.ul, self.jr, self.rp)
        # print("jointPoses",jointPoses)

        for i in range(7):
            p.setJointMotorControl2(bodyUniqueId=self.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=200,
                                    maxVelocity=1.0, positionGain=0.3, velocityGain=1)

        p.setJointMotorControl2(self.kukaUid, 8, p.POSITION_CONTROL,
                                targetPosition=-fingerAngle, force=2.5)
        p.setJointMotorControl2(self.kukaUid, 11, p.POSITION_CONTROL,
                                targetPosition=fingerAngle, force=2.5)
        p.setJointMotorControl2(self.kukaUid, 10, p.POSITION_CONTROL,
                                targetPosition=0, force=2)
        p.setJointMotorControl2(self.kukaUid, 13, p.POSITION_CONTROL,
                                targetPosition=0, force=2)
        return jointPoses


class Object:
    def __init__(self, urdfPath, block,num):
        self.id = p.loadURDF(urdfPath)
        self.half_height = 0.025 if block else 0.0745
        self.num = num

        self.block = block
    def reset(self):
        # random reset
        p.resetBasePositionAndOrientation(self.id,
                                          np.array([ np.random.uniform(0.4, 0.7),np.random.uniform(-0.15, 0.15),
                                                    self.half_height]),
                                          p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
        # if self.num==1:
        #     p.resetBasePositionAndOrientation(self.id,
        #                                  np.array([ 0.55,0.1,
        #                                            self.half_height]),
        #                                 p.getQuaternionFromEuler([0, 0,np.pi/6]))
        # else:
        #     p.resetBasePositionAndOrientation(self.id,
        #                                  np.array([ 0.5,-0.05,
        #                                            self.half_height]),
        #                                 p.getQuaternionFromEuler([0, 0,0]))

    def pos_and_euler(self):
        pos, quat = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(quat)
        return pos, euler


def check_pairwise_collisions(bodies):
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and \
                    len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0:
                return True
    return False


class KukaCamEnvBase:
    def __init__(self, object1_path, object1_shape, object2_path, object2_shape,
                 renders=False, image_output=True, mode='de', width=128):
        self._timeStep = 0.02
        self._renders = renders
        self._image_output = image_output
        self._mode = mode
        self._width = width
        self._height = self._width
        self._p = p

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
        p.loadURDF("models/floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF("models/table_collision/table.urdf", [0.6, 0, -0.625],p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True)
        self._kuka = Kuka("models/kuka_iiwa/kuka_with_gripper2.sdf")
        self._object1 = Object(object1_path, block=object1_shape,num=1)
        self._object2 = Object(object2_path, block=object2_shape,num=2)

    def reset(self):
        collision = True
        while collision:
            self._object1.reset()
            self._object2.reset()
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


            img_arr = p.getCameraImage(width=self._width, height=self._height,
                                       viewMatrix=viewMat, projectionMatrix=projMatrix)
            rgb = img_arr[2]
            observation[0] = rgb[:, :, 0]
            observation[1] = rgb[:, :, 1]
            observation[2] = rgb[:, :, 2]
            if self._mode == 'rgbd':
                depth_buffer = img_arr[3].reshape(self._height, self._width)
                observation[3] = np.round(255*farPlane*nearPlane / ((farPlane-(farPlane-nearPlane)*depth_buffer)*1.1))
            elif self._mode == 'de':
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)
                img_arr2 = p.getCameraImage(width=self._width, height=self._height,
                                            viewMatrix=viewMat2, projectionMatrix=projMatrix)
                rgb2 = img_arr2[2]
                observation[3] = rgb2[:, :, 0]
                observation[4] = rgb2[:, :, 1]
                observation[5] = rgb2[:, :, 2]
        additional_observation = self._kuka.getObservation()
        Pos1, Euler1 = self._object1.pos_and_euler()
        Pos2, Euler2 = self._object2.pos_and_euler()
        additional_observation.extend(list(Pos1))
        additional_observation.extend(list(Euler1))
        additional_observation.extend(list(Pos2))
        additional_observation.extend(list(Euler2))
        additional_observation = np.array(additional_observation, dtype=np.float32)
        return observation, additional_observation

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
            joint_p = self._kuka.applyAction(realAction)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        observation, additional_observation = self.getExtendedObservation()
        done, reward = self.reward()
        return observation, additional_observation, reward, done,joint_p

    def reward(self):
        raise NotImplementedError


class KukaCamEnv1(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=128):
        super().__init__("models/box_green.urdf", True,
                         "models/cup/cup.urdf", False,
                         renders=renders, image_output=image_output, mode=mode, width=width)

    def reward(self):
        blockPos, blockOrn = self._object1.pos_and_euler()
        cupPos, cupOrn = self._object2.pos_and_euler()
        if abs(cupOrn[0]) > 1 or abs(cupOrn[1]) > 1:
            return True, 0.0
        dist = np.sqrt((blockPos[0] - cupPos[0]) ** 2 + (blockPos[1] - cupPos[1]) ** 2)
        if dist < 0.01 and blockPos[2] - cupPos[2] < 0.05 and abs(cupOrn[0]) < 0.2 and abs(cupOrn[1]) < 0.2:
            return True, 1.0
        return False, 0.0


class KukaCamEnv2(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=128):
        super().__init__("models/box_green.urdf", True,
                         "models/box_purple.urdf", True,
                         renders=renders, image_output=image_output, mode=mode, width=width)

    def reward(self):
        blockPos, blockOrn = self._object1.pos_and_euler()
        block2Pos, block2Orn = self._object2.pos_and_euler()
        dist = np.sqrt((blockPos[0] - block2Pos[0]) ** 2 + (blockPos[1] - block2Pos[1]) ** 2)
        if dist < 0.02 and blockPos[2] < 0.076:
            return True, 1.0
        return False, 0.0


class KukaCamEnv3(KukaCamEnvBase):
    def __init__(self, renders=False, image_output=True, mode='de', width=128):
        super().__init__("models/cup/cup_small.urdf", False,
                         "models/cup/cup.urdf", False,
                         renders=renders, image_output=image_output, mode=mode, width=width)

    def reward(self):
        cupsPos, cupsOrn = self._object1.pos_and_euler()
        cupPos, cupOrn = self._object2.pos_and_euler()
        if abs(cupOrn[0]) > 1 or abs(cupOrn[1]) > 1 or abs(cupsOrn[0]) > 1 or abs(cupsOrn[1]) > 1:
            return True, 0.0
        dist = np.sqrt((cupsPos[0] - cupPos[0]) ** 2 + (cupsPos[1] - cupPos[1]) ** 2)
        if dist < 0.01 and cupsPos[2] - cupPos[2] < 0.08 and abs(cupOrn[0]) < 0.2 and abs(cupOrn[1]) < 0.2:
            return True, 1.0
        return False, 0.0


# import cv2
# env = KukaCamEnv2(renders=True, image_output=False, mode='de', width=128)
# _,state = env. getExtendedObservation()
#
# print(state)
# time.sleep(10000)
# cv2.waitKey(0)
# a = p.getQuaternionFromEuler([np.pi, 0, np.pi])
# print(a)
