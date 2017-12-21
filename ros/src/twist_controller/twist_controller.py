from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

import time
import rospy
from math import fabs

GAS_DENSITY_KG_GAL = 2.858
GAS_DENSITY_KG_CUB_M = 755.00373
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, accel_limit, decel_limit, 
            brake_deadband, wheel_radius, fuel_capacity,
            wheel_base, steer_ratio, max_lat_accel, max_steer_angle,
            tau, ts,
            kp, ki, kd):

        # Initialize controller attributes
        self.vehicle_mass = vehicle_mass
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.brake_deadband = brake_deadband
        self.wheel_radius = wheel_radius
        self.fuel_capacity = fuel_capacity

        # Initialize itilities
        self.pid = PID(kp, ki, kd, self.decel_limit, self.accel_limit)
        self.low_pass_filter = LowPassFilter(tau, ts) #see if can pass real ts
	self.steer_lpf = LowPassFilter(tau=0.45, ts=ts)
        self.yaw_controller = YawController(wheel_base, steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)

        # Need time for throttle calculation?
        self.previous_time = None

    # Brake = Desired acceleration, weight of the vehicle, and wheel radius
    def control(self, current_velocity, twist_cmd):
        # If time = None, initialize
        if self.previous_time is None:
            self.previous_time =  time.time() 
        
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        
        linear_velocity = twist_cmd.twist.linear.x
        angular_velocity = twist_cmd.twist.angular.z
        current_vel = current_velocity.twist.linear.x
        
        # The param contains speed limit in kmph
        # Let's keep it 2 km below speed limit to prevent speed violations
        speed_limit = ((rospy.get_param('/waypoint_loader/velocity') - 2)
                        * 1000.) / (60. * 60.)
        
        error = linear_velocity - current_vel
        # When desired linear and current velocities are the same, 
        # the error is a very small negative number
        if fabs(error) < 0.0001:
            error=0.0
        # Calculate throttle using PID
        # Throttle values should be in the range -self.decel_limit (-5.0) to self.decel_limit (1.0)
        elapsed_time = time.time() - self.previous_time
        self.previous_time = time.time()
        #rospy.logwarn("***desired linear_velocity: {}".format(linear_velocity))
        #rospy.logwarn("***current_vel: {}".format(current_vel))
        #rospy.logwarn("***error: {}".format(error))
        #rospy.logwarn("***elapsed_time: {}".format(elapsed_time))
        throttle = self.pid.step(error, elapsed_time)
        throttle = min(self.accel_limit, throttle)
        #rospy.logwarn("throttle: {}".format(throttle))
    
        # Calculate brake
        # Brake values should be in units of torque (N*m)
        #https://discussions.udacity.com/t/what-is-the-range-for-the-brake-in-the-dbw-node/412339
        if throttle < -self.brake_deadband:
            brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY_KG_CUB_M) * fabs(throttle) * self.wheel_radius
            brake = self.low_pass_filter.filt(brake)
            throttle = 0.0
            #rospy.logwarn("brake: {}".format(brake))
        #See also https://carnd.slack.com/archives/C6NVDVAQ3/p1505233214000427?thread_ts=1505227052.000090&cid=C6NVDVAQ3
        elif throttle < 0:  # deadband between -0.2 and 0.0.
            throttle = 0.0
            brake = 0.0
        else: #positive throttle
            brake = 0.0  #redundant, just to make it clear.
        
        # Calculate steer
        #Good explanation on what to pass to get_steer function in forum:
        #https://discussions.udacity.com/t/no-able-to-keep-the-lane-with-yaw-controller/433887/5
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_vel)
	steer = self.steer_lpf.filt(steer)
        return throttle, brake, steer
