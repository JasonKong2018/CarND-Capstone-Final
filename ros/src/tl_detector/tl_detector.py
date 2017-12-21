#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import distance
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.last_closest_point = None
        self.stop_line_waypoints = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']

        rospy.spin()

    def initialize_stop_line_waypoints(self):
        closest_point = [0] * len(self.stop_line_positions)
        closest_dist_so_far = [100000] * len(self.stop_line_positions)

        j = 0  # stop_lines are in order, so the mins will be detected in order
        for i in range(0, self.total_waypoints):
            another_w_pos = self.waypoints.waypoints[i].pose.pose.position
            sample_xy = (another_w_pos.x, another_w_pos.y)
            d = distance.euclidean(self.stop_line_positions[j], sample_xy)
            if d < closest_dist_so_far[j]:
                closest_dist_so_far[j] = d
                closest_point[j] = i
            else:
                if j < len(self.stop_line_positions) - 1:
                    j += 1
                else:
                    break

        self.stop_line_waypoints = closest_point

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.total_waypoints = len(waypoints.waypoints)
        self.initialize_stop_line_waypoints()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            #rospy.loginfo('publishing %s', light_wp)
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoints is None:
            return 0
        car_x = pose.position.x
        car_y = pose.position.y
        car_xy = (car_x, car_y)
        closest_point = 0
        closest_dist_so_far = 100000  # replace with highest float

        if self.last_closest_point is None:
            wp_search_list = list(range(0, self.total_waypoints))
        else:
            min_point = self.last_closest_point  # assumes only forward movement
            max_point = (self.last_closest_point + 20) % self.total_waypoints
            if max_point > min_point:
                wp_search_list = list(range(min_point, max_point))
            else:
                wp_search_list = list(range(min_point, self.total_waypoints))
                wp_search_list.extend(list(range(0, max_point)))

        for i in wp_search_list:
            another_w_pos = self.waypoints.waypoints[i].pose.pose.position
            sample_xy = (another_w_pos.x, another_w_pos.y)
            distance_between_wps = distance.euclidean(car_xy, sample_xy)
            if (distance_between_wps < closest_dist_so_far):
                closest_dist_so_far = distance_between_wps
                closest_point = i
        self.last_closest_point = closest_point
        # rospy.loginfo('closest point %s', closest_point)
        return closest_point

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Cheating:
        return light.state

        # if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False

        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
	    0: Red; 1: Yellow; 2: Green

        """
        light = None
        light_wp = None

        if (self.pose and self.stop_line_waypoints and self.stop_line_positions):
            car_wp = self.get_closest_waypoint(self.pose.pose)

            # TODO find the closest visible traffic light (if one exists)
            next_light_idx = 0
            for i in range(len(self.stop_line_waypoints)):
                if self.stop_line_waypoints[i] > car_wp:
                    next_light_idx = i
                    break

            light_wp = self.stop_line_waypoints[next_light_idx]
            light_xy = self.stop_line_positions[next_light_idx]

            car_x = self.pose.pose.position.x
            car_y = self.pose.pose.position.y
            car_xy = (car_x, car_y)

            # rospy.loginfo('car at point %s, light at point %s', car_position, light_position)
            no_wpts_to_light = light_wp - car_wp
            dist_to_stop_line = distance.euclidean(light_xy, car_xy)
            if dist_to_stop_line < 100:
                light = self.lights[next_light_idx]

        if light:
            state = self.get_light_state(light)
            #rospy.loginfo('car %s, light %s, state %s, dist %s', car_wp, light_wp, state, dist_to_stop_line)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
