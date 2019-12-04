import sys
from environment import *
import time
import random
import pygame
import csv
import os
import torch
from tools import SAC_Actor
from tools import getHeading, bool2num
import argparse

np.random.seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--vehicleNum', help="select specific vehicle to test, 1 (Audi A2), 2 (Audi Tt), 3 (Citroen C3), 4 (Truck, CarlaCola)", type=int)
args = parser.parse_args()
vehicleNum = args.vehicleNum

if __name__ == "__main__":

    pygame.init()
    pygame.font.init()
    if vehicleNum == 3:
        vehicleNum = 5
        vehicle_name = 'CitroenC3'
    elif vehicleNum == 1 or vehicleNum == 2 or vehicleNum == 4:
        vehicleNum = vehicleNum
        if vehicleNum == 1:
            vehicle_name = 'AudiA2'
        elif vehicleNum == 2:
            vehicle_name = 'AudiTt'
        else:
            vehicle_name = 'Truck_CarlaCola'
    else:
        print('Wrong VehicleNum Input ... 1 (Audi A2), 2 (Audi Tt), 3 (Citroen C3), 4 (Truck, CarlaCola)')
        exit()
    env = environment(traj_num=6,model='sac',vehicleNum=vehicleNum)


    action_dim = 2 # steer, throttle
    state = env.getState()
    state_dim = len(state)
    print('action_dimension:', action_dim, ' --- state_dimension:', state_dim)

    # Initializing the Agent for SAC and load the trained weights
    actor = SAC_Actor(state_dim=state_dim, action_dim = action_dim).to(device)
    actor.load_state_dict(torch.load( '../weights/sac-stg2/policy_net_1280.pth'))

    destinationFlag = False
    collisionFlag = False
    awayFlag = False
    carla_startFlag = False	

    

    # define the headers of the csv file to be saved
    headers = ['time','world_x','world_y','world_heading','local_vx','local_vy','total_v','slip_angle',
    'cte','cae','traj_index','reward','steer','throttle','collisionFlag','desitinationFlag','awayFlag']

    print('TESTING: vehicleNum: ', vehicleNum)
    os.makedirs('./test/sac-stg2/vehicles/',exist_ok=True)
    save_path = './test/sac-stg2/vehicles/'+ vehicle_name
    save_file = open(save_path + '.csv','w')
    writer = csv.writer(save_file)
    writer.writerow(headers)

    env.reset(traj_num=6, testFlag=True, test_friction=3.5, differentVehicles=True)

    t0 = time.time()
    first_step_pass = False

    while(True):
        env.render()

        # make sure the connection with carla is ok
        tmp_control = env.world.player.get_control()
        if tmp_control.throttle == 0 and carla_startFlag==False:
            tmp_control = carla.VehicleControl(
                            throttle = 0.5,
                            steer = 0,
                            brake = 0.0,
                            hand_brake = False,
                            reverse = False,
                            manual_gear_shift = False,
                            gear = 0)
            env.world.player.apply_control(tmp_control)
            continue
        carla_startFlag = True

        if time.time()-t0 < 0.5:
            # make sure the collision sensor is empty
            env.world.collision_sensor.history = []

        if time.time()-t0 > 0.5:
            
            if not first_step_pass:
                steer = 0.0
                throttle = 0.0
                hand_brake = False
            else:
                action = actor.test(tState)
                action = np.reshape(action, [1,2])

                steer = action[0,0]
                throttle = action[0,1]	

            next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer, throttle=throttle)
            next_state = np.reshape(next_state, [1, state_dim])
            
            tState = next_state

            # prepare the state information to be saved
            t = time.time() - t0
            location = env.world.player.get_location()
            wx = location.x
            wy = location.y
            course = getHeading(env)
            vx = env.velocity_local[0]
            vy = env.velocity_local[1]
            speed = np.sqrt(vx*vx + vy*vy)
            slip_angle = env.velocity_local[2]
            cte = tState[0,2]
            cae = tState[0,4]
            traj_index = env.traj_index
            steer = control.steer
            throttle = control.throttle
            cf = bool2num(collisionFlag)
            df = bool2num(destinationFlag)
            af = bool2num(awayFlag)
            
            # save to the csv file for further analysis
            print('time stamp: ', t)
            writer.writerow([t,wx,wy,course,vx,vy,speed,slip_angle,cte,cae,traj_index,reward,steer,throttle,cf,df,af])
            
            endFlag = collisionFlag or destinationFlag or awayFlag

            if endFlag:
                break
            
            first_step_pass = True 
        