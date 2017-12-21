#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import distance
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

import tf
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        #rospy.logwarn("Inside Waypoint Updater")
        
        rospy.init_node('waypoint_updater')
        
        self.current_pose = None
        self.base_waypoints = None
        self.final_waypoints = []
        self.traffic_waypoint = None
        #self.obstacle_waypoint = None
        self.total_waypoints = 0
        self.last_closest_point = None
        self.current_velocity = None
        self.current_linear_speed = 0
        self.last_car_to_stop_distance = 0.0
        self.last_decrease_rate = 0.0
        
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        
        # TODO: Add other member variables you need below
        #rospy.spin()
        
        self.loop()

    def current_velocity_cb(self, current_velocity):
        #rospy.logwarn("Update current_velocity: {}".format(current_velocity))
        self.current_velocity = current_velocity

    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerate(self, waypoints, stop_wp, car_wp):   
        stop_relative_wp = stop_wp-car_wp
        current_velocity = self.current_velocity.twist.linear.x
        #total_distance = self.distance(self.base_waypoints.waypoints[stop_wp].pose.pose.position,
        #    self.base_waypoints.waypoints[car_wp].pose.pose.position)
        total_distance = self.distance_between_waypoints(self.base_waypoints.waypoints, car_wp, stop_wp)
        rospy.logwarn('DECELERATE! current_speed: %s, car_wp %s, stop_wp %s, distance %s', current_velocity, car_wp, stop_wp, total_distance)
        decrease_rate = abs(current_velocity) / total_distance

        # If the distance between car and stop point during last run is the same
        # as the distance during this run, the car is close to the same waypoint.
        # To make deceleration more efficient, we need to apply previously
        # calculated decrease rate
        if math.fabs(total_distance - self.last_car_to_stop_distance) < 0.0001:
            decrease_rate = self.last_decrease_rate
        self.last_car_to_stop_distance = total_distance
        self.last_decrease_rate = decrease_rate
        
        #If the car is stopped or almost stopped but we are still far away from the traffic light, we will
        #move forward at a slow speed
        if total_distance > 10 and current_velocity < 3.0:
            rospy.logwarn('DECEL-A. Getting close to the stop point slowly')
            for wp in waypoints:
               wp.twist.twist.linear.x = 2.0
        elif total_distance <= 5:
            if decrease_rate < 2.0:
                # if we are almost at the stop line we ensure to stop the car, if it safe.
                # (case of the car having approached slowly)
                rospy.logwarn('DECEL-B. Setting all speeds to zero')
                for wp in waypoints:
                    wp.twist.twist.linear.x = -5.0

        elif decrease_rate < 2.0:
            rospy.logwarn('DECEL-C. Standard deceleration')
            #We need to distinguish two cases:
            # case 1) the traffic light is before the end of the final waypoints
            if stop_relative_wp < len(waypoints):
                index_last = stop_relative_wp
                for wp in waypoints[index_last: ] :
                    wp.twist.twist.linear.x = -5.0
            # case 2) the traffic light is beyond the end of the final waypoints
            else:
                index_last = len(waypoints)

            i=0
            for j, wp in enumerate(waypoints[:index_last]):
                #dist = self.distance(wp.pose.pose.position, self.base_waypoints.waypoints[stop_wp].pose.pose.position)
                dist = self.distance_between_waypoints(self.base_waypoints.waypoints, car_wp+j, stop_wp)
                if dist < 5:
                    # To avoid throttle going up and have smooth stop.
                    vel = 0.9 * (decrease_rate * dist)
                else:
                    vel = decrease_rate * dist
                
                if i % 5 == 0:
                    rospy.logwarn('distance to [%s]: %s, decel_speed: %s',
                        i, dist, vel)
                i = i+1
                wp.twist.twist.linear.x = vel

        # If current_velocity is high and light changes when the car is at short distance to
        # the light, decrease_rate will be high. It is safer to keep on driving to avoid 
        # high jerk or stopping in the middle of intersection
        else:
            rospy.logwarn('DECEL-D. Not safe to decelerate!')


        rospy.logwarn('final_waypoints Speed samples [5]: %s [10]: %s, [20]: %s, [30] %s, [40] %s',
            waypoints[5].twist.twist.linear.x,
            waypoints[10].twist.twist.linear.x, waypoints[20].twist.twist.linear.x,
            waypoints[30].twist.twist.linear.x, waypoints[40].twist.twist.linear.x)

        return waypoints

    def current_pose_cb(self, msg):
        self.current_pose = msg


    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.current_pose is not None and self.base_waypoints is not None:
                #rospy.logwarn("Publishing from Waypoints Updater:")
        
                closest_point = self.find_next_waypoint()
                #rospy.logwarn("CLOSEST POINT {}".format(closest_point))
    
                self.final_waypoints = [] #Reinitialize each time
                for i in range(closest_point, closest_point+LOOKAHEAD_WPS+1):
                    if i >= self.total_waypoints:
                        i = i - self.total_waypoints
                    waypoint=self.base_waypoints.waypoints[i]
                    self.final_waypoints.append(waypoint)
    
                #rospy.logwarn("waypoints size: {}".format(len(self.final_waypoints)))
                #Linear decrease speeds of the next LOOKAHEAD_WPS waypoints
                if self.traffic_waypoint is not None \
                    and self.current_velocity is not None \
                    and self.traffic_waypoint != -1 \
                    and closest_point < self.traffic_waypoint.data:
                        self.final_waypoints = self.decelerate(self.final_waypoints, self.traffic_waypoint.data, closest_point)
                else:
                    for j, wp in enumerate(self.final_waypoints):
                        # The param contains speed limit in kmph
                        # Let's keep it 2 km below speed limit to prevent speed violations
                        speed_limit = ((rospy.get_param('/waypoint_loader/velocity') - 2)
                            * 1000.) / (60. * 60.)
                        self.set_waypoint_velocity(self.final_waypoints, j, speed_limit)
                        
    
                self.publish()
            rate.sleep()

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.total_waypoints=np.shape(waypoints.waypoints)[0]
        rospy.logwarn("Total waypoints: {}".format(self.total_waypoints))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_waypoint = msg


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def publish(self):
        #Publishing the Lane with final enpoints
        lane=Lane()
        lane.header=self.base_waypoints.header
        lane.waypoints=np.asarray(self.final_waypoints)
        self.final_waypoints_pub.publish(lane)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance_between_waypoints(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        j = wp1
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[j].pose.pose.position, waypoints[i].pose.pose.position)
            j = i
        return dist

    def find_next_waypoint(self):
        # Find one waypoint closest to current position of the car
        closest_point = 0
        closest_dist_so_far = 100000 #replace with highest float
        current_w_pos = self.current_pose.pose.position

        if self.last_closest_point is None:
            wp_search_list = list(range(0, self.total_waypoints))
        else:
            min_point = self.last_closest_point #assumes only forward movement
            max_point = (self.last_closest_point + 20) % self.total_waypoints
            if max_point > min_point:
                wp_search_list = list(range(min_point,max_point))
            else:
                wp_search_list = list(range(min_point, self.total_waypoints))
                wp_search_list.extend(list(range(0,max_point)))

        for i in wp_search_list:
            another_w_pos=self.base_waypoints.waypoints[i].pose.pose.position
            a = (current_w_pos.x, current_w_pos.y, current_w_pos.z)
            b = (another_w_pos.x, another_w_pos.y, another_w_pos.z)
            distance_between_wps = distance.euclidean(a, b)
            if(distance_between_wps<closest_dist_so_far):
                closest_dist_so_far=distance_between_wps
                closest_point = i

        # Check if the closest point is behind or in front of us.
        wp  = self.base_waypoints.waypoints[closest_point]
        _,_,yaw = tf.transformations.euler_from_quaternion([self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y,
                                                            self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w])
        angle = math.atan2(wp.pose.pose.position.y - current_w_pos.y, wp.pose.pose.position.x - current_w_pos.x)
        rel_angle = (angle - yaw) % (2 * math.pi)

        if (rel_angle > 0.5 * math.pi) & (rel_angle < 1.5 * math.pi):
            # The closes point was behind, so we return the next ahead
            closest_point = (closest_point + 1) % self.total_waypoints

        self.last_closest_point = closest_point
        return closest_point

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
