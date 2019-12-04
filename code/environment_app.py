
from __future__ import print_function
import time
import collections
import datetime
import glob
import logging
import math
import os
import random
import math
import re
import sys
import weakref
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')
import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from tools_app import *
import argparse
from collections import deque
import pandas as pd

step_T_bound = (0.6,1)		# Boundary of throttle values
step_S_bound = (-0.8,0.8)	# Boundary of the steering angle values

def draw_waypoints(world, route):
	x0 = route[0,0]
	y0 = route[0,1]
	for k in range(1,route.shape[0]):
		r = route[k,:]
		x1 = r[0]
		y1 = r[1]
		dx = x1-x0
		dy = y1-y0
		if math.sqrt(dx*dx+dy*dy) > 15:
			x0 = x1
			y0 = y1
			begin = carla.Location(x = x1,y = y1, z = 0.1)
			
			world.debug.draw_point(begin, size = 0.05,life_time=7200, color=carla.Color(238,18, 137,0))


class environment():
	def __init__(self, throttleSize=4, steerSize=9, traj_num = 0, collectFlag = False,  vehicleNum=1):
		
		log_level = logging.INFO
		
		logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

		logging.info('listening to server %s:%s', '127.0.0.1', 2000)
		
		self.refreshRoute(traj_num)  # a series of caral.transform
		

		if not collectFlag:
			start_location = carla.Location(x = self.route[0,0], y = self.route[0,1], z = 0.1)
			start_rotation = carla.Rotation(pitch = 0, yaw = -90, roll = 0)
		else:
			start_location = carla.Location()
			start_rotation = carla.Rotation()
		
		self.start_point = carla.Transform(location = start_location, rotation = start_rotation)  # type : Transform (location, rotation)
		
		self.client = carla.Client('127.0.0.1', 2000)
		self.client.set_timeout(4.0)
		self.display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
		self.hud = HUD(1280, 720)
		self.world = World(self.client.get_world(), self.hud, 'vehicle.*', self.start_point, vehicleNum)
		self.clock = pygame.time.Clock()
		self.minDis = 0
		self.collectFlag = collectFlag
		self.traj_drawn_list = []
		

		self.control = carla.VehicleControl(
							throttle = 1,
							steer = 0.0,
							brake = 0.0,
							hand_brake = False,
							reverse = False,
							manual_gear_shift = False,
							gear = 0)
		
		self.destinationFlag = False
		self.away = False
		self.collisionFlag = False
		self.waypoints_ahead = [] 
		self.waypoints_neighbor = [] 
		self.steer_history = deque(maxlen=20)
		self.throttle_history = deque(maxlen=20)
		self.velocity_local = []

	

		self.e_heading = 0
		self.e_d_heading = 0
		self.e_dis = 0
		self.e_d_dis = 0
		self.e_slip = 0
		self.e_d_slip = 0
		self.e_vx = 0
		self.e_d_vx = 0
		self.e_vy = 0
		self.e_d_vy = 0


		self.tg = 0
		self.clock_history = 0 # pop the current location into self.waypoints_history every 0.2s

		self.k_heading = 0.1

		self.waypoints_ahead_local = []
		self.waypoints_history = deque(maxlen=5)
		self.waypoints_history_local = []
		self.waypoints_neighbor_local = []

		self.last_steer = 0.0
		self.last_throttle = 0.0

		self.tire_friction_array = np.arange(3,4.1,0.1) # [3,4], 11D
		self.mass_array = np.arange(1700,1910,50) # array([1700, 1750, 1800, 1850, 1900])

		self.ori_physics_control = self.world.player.get_physics_control()
		self.wheel_fl = self.ori_physics_control.wheels[0]
		self.wheel_fr = self.ori_physics_control.wheels[1]
		self.wheel_rl = self.ori_physics_control.wheels[2]
		self.wheel_rr = self.ori_physics_control.wheels[3]

		self.world.world.set_weather(carla.WeatherParameters.ClearNoon)

	def refreshRoute(self, traj_num):
		traj = pd.read_csv('waypoints/waypoints_' + str(traj_num) + '.csv')
		self.route = traj.values
		self.route_x = self.route[:,0]
		self.route_y = self.route[:,1]
		self.route_length = np.zeros(self.route.shape[0])
		for i in range(1, self.route.shape[0]):
			dx = self.route_x[i-1] - self.route_x[i]
			dy = self.route_y[i-1] - self.route_y[i]
		self.route_length[i] = self.route_length[i-1] + np.sqrt(dx * dx + dy * dy)
		


	def step(self, actionID = 4, steer = 0, throttle=0, manual_control = False):
		# apply the computed control commands, update endFlag and return state/reward
		if not manual_control:
			
			self.control = self.getAction(steer = steer,throttle = throttle)

			
			self.control.steer = 0.1*self.control.steer + 0.9*self.last_steer
			self.control.throttle = 0.3*self.control.throttle + 0.7*self.last_throttle
			

			self.last_steer = self.control.steer
			self.last_throttle = self.control.throttle

			self.world.player.apply_control(self.control)
			self.steer_history.append(self.control.steer)
			self.throttle_history.append(self.control.throttle)
			time.sleep(0.05)

		
		if manual_control and not self.collectFlag:
			control = self.world.player.get_control()
			self.steer_history.append(control.steer)
			self.throttle_history.append(control.throttle)
			time.sleep(0.05)
		
		newState = self.getState()

		if not self.collectFlag :
			reward = self.getReward(newState, self.steer_history, self.throttle_history)
		
			self.collisionFlag = self.collisionDetect()

			return newState, reward, self.collisionFlag, self.destinationFlag, self.away, self.control

		else:
			control = self.world.player.get_control()
			return newState, control
		
		


	def reset(self, traj_num = 0, collect_x = 0, collect_y = 0, collect_yaw = 0, randomPosition = False, testFlag = False, 
				test_friction = 3.5, test_mass = 1800.0, differentFriction=False, differentVehicles=False):
		# random change the tire friction and vehicle mass:
		if not testFlag:
			index_friction = np.random.randint(0,self.tire_friction_array.shape[0])
			index_mass = np.random.randint(0,self.mass_array.shape[0])


			self.tire_friction = self.tire_friction_array[index_friction]
			self.mass = self.mass_array[index_mass]
		else:
			self.tire_friction = test_friction
			self.mass = test_mass
		
		if not differentFriction:
			self.wheel_fl.tire_friction = self.tire_friction
			self.wheel_fr.tire_friction = self.tire_friction
			self.wheel_rl.tire_friction = self.tire_friction
			self.wheel_rr.tire_friction = self.tire_friction
		else:
			self.wheel_fl.tire_friction = 2.8
			self.wheel_fr.tire_friction = 2.8
			self.wheel_rl.tire_friction = 4.2
			self.wheel_rr.tire_friction = 4.2

		wheels = [self.wheel_fl, self.wheel_fr, self.wheel_rl, self.wheel_rr]

		self.ori_physics_control.wheels = wheels
		if not differentVehicles:
			self.ori_physics_control.mass = float(self.mass)
		
		
		self.world.player.apply_physics_control(self.ori_physics_control)
		time.sleep(0.5)

		# detect:
		physics = self.world.player.get_physics_control()
		print('firction: {}, mass: {}'.format(physics.wheels[0].tire_friction, physics.mass))
		print('center of mass: ', physics.center_of_mass.x, physics.center_of_mass.y, physics.center_of_mass.z)
		
		if not self.collectFlag:
			self.refreshRoute(traj_num)
			if not randomPosition:
				start_location = carla.Location(x = self.route[0,0], y = self.route[0,1], z = 0.1)
				start_rotation = carla.Rotation(pitch = 0, yaw = -90, roll = 0)
				velocity_local = [10,0]  # 5m/s
				angular_velocity = carla.Vector3D()
				
			else:
				k = np.random.randint(0,self.route.shape[0] - 100)
				start_location = carla.Location(x = self.route[k,0], y = self.route[k,1], z = 0.1)
				start_rotation = carla.Rotation(pitch = 0, yaw = self.route[k,2], roll = 0)
				velocity_local = [10, 0] 
				# angular_velocity = carla.Vector3D(z = self.route[k,6])
				angular_velocity = carla.Vector3D()
		else:
			start_location = carla.Location(x = collect_x, y=collect_y)
			start_rotation = carla.Rotation(yaw = collect_yaw)

		
		self.start_point = carla.Transform(location = start_location, rotation = start_rotation)  # type : Transform (location, rotation)
		ego_yaw = self.start_point.rotation.yaw

		if not self.collectFlag:
			if traj_num not in self.traj_drawn_list:
				self.drawPoints()
				self.traj_drawn_list.append(traj_num)

		
		ego_yaw = ego_yaw/180.0 * 3.141592653
		transformed_world_velocity = self.velocity_local2world(velocity_local, ego_yaw)

		self.world.player.set_transform(self.start_point)
		self.world.player.set_velocity(transformed_world_velocity)
		self.world.player.set_angular_velocity(angular_velocity)
		
		self.world.player.apply_control(carla.VehicleControl())

		self.world.collision_sensor.history = []
		self.away = False
		self.endFlag = False
		self.steer_history.clear()
		self.throttle_history.clear()
		self.waypoints_neighbor = []
		self.waypoints_neighbor_local = []
		self.waypoints_ahead = []

		self.waypoints_ahead_local = [] # carla.location 10pts
		self.waypoints_history.clear()  # carla.location  5pts
		self.waypoints_history_local = []
		self.destinationFlag = False

		self.last_steer = 0.0
		self.last_throttle = 0.0

		self.drived_distance = 0

		print('RESET!\n\n')
		
		return 0

	def getState(self):
		location = self.world.player.get_location()
		
		angular_velocity = self.world.player.get_angular_velocity()
		transform = self.world.player.get_transform()
		ego_yaw = transform.rotation.yaw
		if ego_yaw < 0:
			ego_yaw += 360
		if ego_yaw > 360:
			ego_yaw -= 360
		ego_yaw = ego_yaw/180.0 * 3.141592653

		self.getNearby() # will update self.minDis

		self.getLocalHistoryWay(location, ego_yaw)
		self.getLocalFutureWay(location, ego_yaw)
		self.getLocalNeighbor(location, ego_yaw)
		
		# print('history')
		# for his in self.waypoints_history_local:
		# 	print(his[:2])
		# print('future')
		# for fut in self.waypoints_ahead_local:
		# 	print(fut[:2])
		# print()

		self.velocity_world2local(ego_yaw) # will update self.velocity_local

		ego_yaw = ego_yaw/3.141592653 * 180
		if ego_yaw > 180:
			ego_yaw = -(360-ego_yaw)

		if self.collectFlag:
			state = [location.x, location.y, ego_yaw, self.velocity_local[0], self.velocity_local[1], self.velocity_local[2], angular_velocity.z]
			
			self.control = self.world.player.get_control()
			steer = self.control.steer
			ct = time.time()
			if ct - self.clock_history > 0.2:
				self.waypoints_history.append(np.array([location.x, location.y, steer, self.velocity_local[2]]))
				self.clock_history = ct

			return state
			
		else:
			dt = time.time() - self.tg
			self.e_d_dis = (self.minDis - self.e_dis) / dt
			self.e_dis = self.minDis

			if self.e_dis > 15:
				self.away = True

			# error of heading:
			this_index = self.nb_index
			theta = 0
			hdy = 0
			hdx = 0
			if self.nb_index != 0:
				hdy = self.waypoints_neighbor_local[this_index+1][0] - self.waypoints_neighbor_local[this_index-1][0]
				hdx = self.waypoints_neighbor_local[this_index+1][1] - self.waypoints_neighbor_local[this_index-1][1]
				theta = math.atan2(hdy,hdx)/3.1415926*180
				if hdy>0 or (hdy<0 and hdx>0):
					theta = 90-theta
				else:
					theta = -270-theta
			

			yaw = -theta/180.0 * 3.141592653
			nx = -hdx * math.cos(yaw) - hdy * math.sin(yaw)
			vgf_left = True
			if nx >0:
				vgf_left = False
			if vgf_left:
				theta = math.atan(self.k_heading * self.e_dis)/3.141592653*180 + theta
			else:
				theta = -math.atan(self.k_heading * self.e_dis)/3.141592653*180 + theta
			

			e_heading = theta
			if e_heading * self.e_heading > 0:
				if e_heading > 0:
					self.e_d_heading = (e_heading - self.e_heading)/dt
				else:
					self.e_d_heading = -(e_heading - self.e_heading)/dt
			else:
				self.e_d_heading = (abs(e_heading) - abs(self.e_heading)) / dt
				
			self.e_heading = e_heading
			
			

		
			# e_vx = self.velocity_local[0] - 30  ## GET VERY GOOD PERFORMANCE, EVEN BETTER THAN THE ONE FOLLOWING A HUMAN REF TRAJECTORY......
			e_vx = self.velocity_local[0] - 30.56
			self.e_d_vx = (e_vx - self.e_vx)/dt
			self.e_vx = e_vx

			self.control = self.world.player.get_control()

			steer = self.control.steer
			throttle = self.control.throttle
			
			ct = time.time()
			if ct - self.clock_history > 0.2:
				self.waypoints_history.append(np.array([location.x, location.y, steer, self.velocity_local[2]]))
				self.clock_history = ct

			vx = self.velocity_local[0]
			vy = self.velocity_local[1]
			e_d_slip = self.e_d_slip
			if math.sqrt(vx*vx + vy*vy) < 2: # if the speed is too small we ignore the error of slip angle
				e_slip = 0
				e_d_slip = 0

		
			state = [steer, throttle , self.e_dis, self.e_d_dis, self.e_heading, self.e_d_heading, 0, 0,
					self.e_vx, self.e_d_vx, 0, 0]
			state.extend([k[0] for k in self.waypoints_ahead_local]) #x
			state.extend([k[1] for k in self.waypoints_ahead_local]) #y
			state.extend([0,0,0,0,0,0,0,0,0,0]) #slip

			
			self.tg = time.time()


			# print(state)
			return state
	
	def getReward(self, state, steer_history, throttle_history):
		e_dis = state[2]
		e_slip = state[6]
		e_heading = state[4]
		std_steer = np.array(steer_history)
		std_steer = std_steer.std()

		std_throttle = np.array(throttle_history)
		std_throttle = std_throttle.std()

		r_dis = np.exp(-0.5*e_dis)

		if abs(e_heading)<90:
			r_heading = np.exp(-0.1*abs(e_heading))
		elif (e_heading)>= 90:
			r_heading = -np.exp(-0.1*(180-e_heading))
		else:
			r_heading = -np.exp(-0.1*(e_heading+180))

		if abs(e_slip)<90:
			r_slip = np.exp(-0.1*abs(e_slip))
		elif (e_slip)>= 90:
			r_slip = -np.exp(-0.1*(180-e_slip))
		else:
			r_slip = -np.exp(-0.1*(e_slip+180))

		r_std_steer = np.exp(-2*std_steer)
		r_std_throttle = np.exp(-2*std_throttle)

		vx = self.velocity_local[0]
		vy = self.velocity_local[1]
		v = math.sqrt(vx*vx + vy*vy)

		reward = v*(40*r_dis + 40*r_heading + 20*r_slip)

		if v < 6:
			reward  = reward / 2

		return reward

	def getNearby(self):

		self.waypoints_ahead = [] 
		# self.waypoints_neighbor = []
		egoLocation = self.world.player.get_location()
		dx_array = self.route_x - egoLocation.x
		dy_array = self.route_y - egoLocation.y
		dis_array = np.sqrt(dx_array * dx_array + dy_array * dy_array)
		self.minDis = np.amin(dis_array)
		_ = np.where(dis_array == self.minDis)
		index = _[0][0]  # index for the min distance to all waypoints.

		self.drived_distance = self.route_length[index]
		self.waypoints_ahead = self.route[index:,:]

		# if index >= 20:
		# 	index_st = index - 20
		# else:
		# 	index_st = 0
		# self.waypoints_neighbor = self.route[index_st:,:]
		self.traj_index = index


	def drawPoints(self):
		draw_waypoints(self.world.player.get_world(), self.route)


	def render(self):
		# show ROS client window by pygame
		self.world.tick(self.clock, self.e_dis, self.e_heading, self.velocity_local[2] )
		self.world.render(self.display)
		pygame.display.flip()


	def velocity_world2local(self,yaw):
		velocity_world = self.world.player.get_velocity()
		vx = velocity_world.x
		vy = velocity_world.y
		yaw = -yaw
		
		local_x = float(vx * math.cos(yaw) - vy * math.sin(yaw))
		local_y = float(vy * math.cos(yaw) + vx * math.sin(yaw))
		if local_x != 0:
			slip_angle = math.atan(local_y/local_x)/3.1415926*180
		else:
			slip_angle = 0
		
		self.velocity_local = [local_x,local_y,slip_angle]

	def velocity_local2world(self, velocity_local, yaw):
		vx = velocity_local[0]
		vy = velocity_local[1]

		world_x = vx * math.cos(yaw) - vy * math.sin(yaw)
		world_y = vy * math.cos(yaw) + vx * math.sin(yaw)

		return carla.Vector3D(world_x,world_y,0)

	def collisionDetect(self):
		if self.world.collision_sensor.history:
			return True
		else:
			return False

	def getAction(self,actionID=4,steer=0, throttle=0):
		
		self.control = carla.VehicleControl(
						throttle = throttle,
						steer = steer,
						brake = 0.0,
						hand_brake = False,
						reverse = False,
						manual_gear_shift = False,
						gear = 0)
		return self.control
		

	def getLocalFutureWay(self,egoLocation,yaw):
		# transfer the future waypoints (#10) to the local coordinate.
		# x, y, slip (degree)
		# ways = self.waypoints_ahead[0:-1:5,:]  # filter to 1m between way pts ### GET VERY GOOD PERFORMANCE, EVEN BETTER THAN THE ONE FOLLOWING A HUMAN REF TRAJECTORY......
		ways = self.waypoints_ahead[0:-1:4,:]

		if ways.shape[0] < 11:
			self.destinationFlag = True
		self.waypoints_ahead_local = []
		yaw = -yaw
		
		
		for w in ways[0:10]: 
		
			wx = w[0]
			wy = w[1]
			w_slip = 0
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y

			nx = dx * math.cos(yaw) - dy * math.sin(yaw)
			ny = dy * math.cos(yaw) + dx * math.sin(yaw)
			self.waypoints_ahead_local.append(np.array([nx,ny,w_slip]))
		
			
	def getLocalHistoryWay(self,egoLocation,yaw):
		# x, y, steer, slip (degree)
		ways = self.waypoints_history
		yaw = -yaw
		self.waypoints_history_local = []
		if len(ways) < 5:
			for i in range(5 - len(ways)):
				self.waypoints_history_local.append(np.array([0,0,0,0]))
		
		for w in ways:
			wx = w[0]
			wy = w[1]
			w_steer = w[2]
			w_slip = w[3]
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y

			nx = dx * math.cos(yaw) - dy * math.sin(yaw)
			ny = dy * math.cos(yaw) + dx * math.sin(yaw)
			self.waypoints_history_local.append(np.array([nx,ny,w_steer,w_slip]))


	def getLocalNeighbor(self,egoLocation,yaw):
		# x, y, steer, slip (degree)
		ways = self.route
		yaw = -yaw
		self.waypoints_neighbor_local = []
		
		index = self.traj_index
		if index >= 10:
			index_st = index - 10
		else:
			index_st = 0

		for w in ways[index_st:index+30,:]:
			wx = w[0]
			wy = w[1]
			
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y

			nx = dx * math.cos(yaw) - dy * math.sin(yaw)
			ny = dy * math.cos(yaw) + dx * math.sin(yaw)
			self.waypoints_neighbor_local.append([nx,ny])
		
		if index_st == 0:
			self.nb_index = index
		else:
			self.nb_index = 10
		

	def vgf_direction(self,egoLocation):
		way_x = self.waypoints_ahead[0,0]
		way_y = self.waypoints_ahead[0,1]
		yaw = -self.waypoints_ahead[0,2]/180.0 * 3.141592653
		
		dx = egoLocation.x - way_x
		dy = egoLocation.y - way_y

		nx = dx * math.cos(yaw) - dy * math.sin(yaw)
		ny = dy * math.cos(yaw) + dx * math.sin(yaw)

		if ny < 0:
			return True
		else:
			return False


	

