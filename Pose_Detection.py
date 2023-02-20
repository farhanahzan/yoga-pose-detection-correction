# import serial
import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
import mediapipe as mp
from numpy import array

MQTT_SERVER = "91.121.93.94"
MQTT_PORT = 1883
MQTT_TOPIC = "led"
MQTT_MESSAGE_ON = "1"
MQTT_MESSAGE_OFF = "0"

client = mqtt.Client()

client.connect(MQTT_SERVER, MQTT_PORT, 60)



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# current angle array
current_angle = []
first_wide_range_angle = []
cap = cv2.VideoCapture('Pose_Videos/correct yoga poses.mp4')
# cap = cv2.VideoCapture(1)

# find a way to loop a video
distance_pinky = 0
l_elbow_angle, l_hip_angle, l_knee_angle, r_elbow_angle, r_hip_angle, r_knee_angle, r_heel_angle, l_heel_angle, l_shoulder_angle, r_shoulder_angle = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
l_shoulder = l_elbow = l_wrist = l_hip = l_knee = l_ankle = l_heel = l_foot_index = l_pinky = r_pinky = r_shoulder = r_elbow = r_wrist = r_hip = r_knee = r_ankle = r_heel = r_foot_index = {}
# printed_statement_flag = False
global converted_image
global pose_label


def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)


def find_distance(a, b):
    point1 = np.array(a)  # first
    point2 = np.array(b)  # second

    diff_point1 = (point1[0] - point2[0]) ** 2
    diff_point2 = (point1[1] - point2[1]) ** 2
    return (diff_point1 + diff_point2) ** 0.5


def check_probability(angle_array):
    true_count = angle_array.count(True)
    # false_count = angle_array.count(False)
    if true_count > 0:
        success_prob = true_count * 100 / len(angle_array)

        return success_prob
    return 0


def add_text(text):
    return


def find_pose(current_angle_of_pose):
    global converted_image

    # r_elbow_angle, r_shoulder_angle,  r_knee_angle, r_heel_angle, l_elbow_angle, l_shoulder_angle, l_knee_angle, l_heel_angle
    warrior2_wide_range_left_angle = [[num in range(lo, hi)] for num, lo, hi in
                                      zip(current_angle_of_pose, [160, 70, 90, 55, 160, 70, 160, 155],
                                          [200, 110, 130, 95, 200, 110, 200, 195])]
    warrior2_wide_range_left_angle = [all(sublist) for sublist in warrior2_wide_range_left_angle]

    # warrior2_wide_range_right_angle = [[num in range(lo, hi)] for num, lo, hi in
    #                                   zip(current_angle_of_pose, [160, 70, 160, 155, 160, 70, 90, 55],
    #                                       [200, 110, 200, 195, 200, 110, 130, 95])]
    # warrior2_wide_range_right_angle = [all(sublist) for sublist in warrior2_wide_range_right_angle]

    tree_wide_range_angle = [[num in range(lo, hi)] for num, lo, hi in
                             zip(current_angle_of_pose, [35, 0, 155, 155, 35, 0, 30, 20],
                                 [75, 40, 195, 195, 75, 40, 70, 60])]
    tree_wide_range_angle = [all(sublist) for sublist in tree_wide_range_angle]

    tyoga_wide_range_angle = [[num in range(lo, hi)] for num, lo, hi in
                              zip(current_angle_of_pose, [160, 70, 155, 155, 160, 70, 155, 155],
                                  [200, 110, 195, 195, 200, 110, 195, 195])]

    tyoga_wide_range_angle = [all(sublist) for sublist in tyoga_wide_range_angle]

    pose_label = "UNKNOWN POSE"
    if check_probability(warrior2_wide_range_left_angle) > 85:
        pose_label = "warrior2"
        # time1 = time
        error_check(warrior2_flag(), 170, 80, 100, 65, 170, 80, 170, 165, 190, 110, 130, 85, 190, 110, 190, 185)

    # if check_probability(warrior2_wide_range_right_angle) > 85:
    #     pose_label = "warrior2"
    #     #time1 = time
    #     error_check(warrior2_flag(), 170, 80, 170, 165, 170, 80, 80, 65, 190, 110, 190, 185, 190, 110, 85, 190)

    if check_probability(tree_wide_range_angle) > 85:
        pose_label = "tree"
        error_check(tree_flag(), 45, 10, 165, 165, 45, 10, 40, 30, 65, 30, 185, 185, 65, 30, 60, 50)

    if check_probability(tyoga_wide_range_angle) > 85:
        pose_label = "T-Pose"
        error_check(tpose_flag(), 170, 80, 165, 165, 170, 80, 165, 165, 190, 100, 185, 185, 190, 100, 185, 185)
    return pose_label


def warrior2_flag():
    second_warrior2_wide_range_angle = [[num in range(lo + 10, hi - 10)] for num, lo, hi in
                                        zip(current_angle, [160, 70, 90, 55, 160, 70, 160, 155],
                                            [200, 120, 140, 95, 200, 120, 200, 195])]

    flag = [all(x) for x in zip(*second_warrior2_wide_range_angle)][0]
    return flag


def tree_flag():
    second_tree_wide_range_angle = [[num in range(lo + 10, hi - 10)] for num, lo, hi in
                                    zip(current_angle, [35, 0, 155, 155, 35, 0, 30, 20],
                                        [75, 40, 195, 195, 75, 40, 70, 60])]

    flag = [all(x) for x in zip(*second_tree_wide_range_angle)][0]
    return flag


def tpose_flag():
    second_tpose_wide_range_angle = [[num in range(lo + 10, hi - 10)] for num, lo, hi in
                                     zip(current_angle, [160, 70, 155, 155, 160, 70, 155, 155],
                                         [200, 110, 195, 195, 200, 110, 195, 195])]

    flag = [all(x) for x in zip(*second_tpose_wide_range_angle)][0]
    return flag


def print_statement(statement,landmark_position, printed_statement_flag=False):
    # global converted_image

    if not printed_statement_flag:
        print(statement)
        error_landmark()
        cv2.putText(converted_image, statement, (800, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        cv2.circle(converted_image,
                   tuple(np.multiply(landmark_position, [width, height]).astype(int)),
                   10, (0, 0, 255), -1
                   )


def error_check(flag, r_elbow_low, r_shoulder_low, r_knee_low, r_heel_low, l_elbow_low, l_shoulder_low, l_knee_low,
                l_heel_low, r_elbow_high, r_shoulder_high, r_knee_high, r_heel_high, l_elbow_high, l_shoulder_high,
                l_knee_high, l_heel_high):
    # r_elbow_angle, r_shoulder_angle,r_knee_angle, r_heel_angle, l_elbow_angle, l_shoulder_angle, l_knee_angle, l_heel_angle
    iteration_start_time = time.time()
    time.sleep(0.001)
    if not flag:

        client.publish(MQTT_TOPIC, MQTT_MESSAGE_ON)

        if r_elbow_angle < r_elbow_low:
            print_statement("Turn your right elbow in a clockwise direction.",r_elbow)

        elif r_elbow_angle > r_elbow_high:
            print_statement("Turn your right elbow in a counterclockwise direction.",r_elbow)

        elif l_elbow_angle < l_elbow_low:
            print_statement("Turn your left elbow in a clockwise direction.",l_elbow)

        elif l_elbow_angle > l_elbow_high:
            print_statement("Turn your left elbow in a counterclockwise direction.",l_elbow)

        elif r_shoulder_angle < r_shoulder_low:
            print_statement("Rotate your right shoulder in a clockwise direction.",r_shoulder)

        elif r_shoulder_angle > r_shoulder_high:
            print_statement("Rotate your right shoulder in a counterclockwise direction.",r_shoulder)

        elif l_shoulder_angle < l_shoulder_low:
            print_statement("Rotate your left shoulder in a clockwise direction.",l_shoulder)

        elif l_shoulder_angle > l_shoulder_high:
            print_statement("Rotate your left shoulder in a counterclockwise direction.",l_shoulder)

        elif r_knee_angle < r_knee_low:
            print_statement("Rotate your right knee in a clockwise direction.",r_knee)

        elif r_knee_angle > r_knee_high:
            print_statement("Rotate your right knee in a counterclockwise direction.",r_knee)

        elif l_knee_angle < l_knee_low:
            print_statement("Rotate your left knee in a clockwise direction.",l_knee)

        elif l_knee_angle > l_knee_high:
            print_statement("Rotate your left knee in a counterclockwise direction.",l_knee)

        elif r_heel_angle < r_heel_low:
            print_statement("Rotate your right heel in a clockwise direction.",r_heel)

        elif r_heel_angle > r_heel_high:
            print_statement("Rotate your right heel in a counterclockwise direction.",r_heel)

        elif l_heel_angle < l_heel_low:
            print_statement("Rotate your left heel in a clockwise direction.",l_heel)

        elif l_heel_angle > l_heel_high:
            print_statement("Rotate your left heel in a counterclockwise direction.",l_heel)

        client.publish(MQTT_TOPIC, MQTT_MESSAGE_OFF)

    else:

        print("Congratulations! You are performing yoga perfectly.")
        cv2.putText(image, "Congratulations! You are performing yoga perfectly.", (800, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        client.publish(MQTT_TOPIC, MQTT_MESSAGE_OFF)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=circle_color, thickness=8, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=landmark_color, thickness=1, circle_radius=1)
                                  )
    iteration_end_time = time.time()
    iteration_time_taken = iteration_end_time - iteration_start_time
    print("Time taken for this iteration: ", iteration_time_taken)

def error_landmark():
    circle_color = (0, 0, 255)
    return circle_color



with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        # global converted_image

        ret, frame = cap.read()

        if ret:
            if frame is not None and frame.shape != ():
                # Recolor image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # make detection
                results = pose.process(image)

                # Recolor image
                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                width, height = 1280, 720
                image = cv2.resize(image, (width, height))
                converted_image = image
                circle_color = (112, 188, 83)
                landmark_color = (255, 255, 255)
                # extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # calculate angle
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                    l_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    l_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]

                    r_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                    r_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                    distance_pinky = find_distance(tuple(np.multiply(l_pinky, [width, height]).astype(int)),
                                                   tuple(np.multiply(r_pinky, [width, height]).astype(int)))
                    # print(distance_pinky)

                    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                    l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    l_heel_angle = calculate_angle(l_ankle, l_heel, l_foot_index)
                    l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)

                    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
                    r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    r_heel_angle = calculate_angle(r_ankle, r_heel, r_foot_index)
                    r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

                    # current angle
                    # current_angle.extend(
                    #     [l_elbow_angle, l_hip_angle, l_knee_angle, l_heel_angle, l_shoulder_angle, r_elbow_angle, r_hip_angle,
                    #      r_knee_angle, r_heel_angle, r_shoulder_angle])
                    # for warrior2 doesnot need hip angle
                    if (
                            r_elbow_angle, r_shoulder_angle, r_knee_angle, r_heel_angle, l_elbow_angle,
                            l_shoulder_angle,
                            l_knee_angle, l_heel_angle > 0):
                        current_angle = array(
                            [r_elbow_angle, r_shoulder_angle,
                             r_knee_angle, r_heel_angle, l_elbow_angle, l_shoulder_angle, l_knee_angle, l_heel_angle])

                    # print(angle)
                    # visulize

                    cv2.putText(image, str(round(l_elbow_angle)),
                                tuple(np.multiply(l_elbow, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    # cv2.circle(image,
                    #             tuple(np.multiply(l_elbow, [width, height]).astype(int)),
                    #             7,(0,0,255),-1
                    #             )
                    cv2.putText(image, str(round(l_hip_angle)),
                                tuple(np.multiply(l_hip, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(l_knee_angle)),
                                tuple(np.multiply(l_knee, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(image, str(round(r_elbow_angle)),
                                tuple(np.multiply(r_elbow, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(r_hip_angle)),
                                tuple(np.multiply(r_hip, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(r_knee_angle)),
                                tuple(np.multiply(r_knee, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(l_heel_angle)),
                                tuple(np.multiply(l_heel, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(r_heel_angle)),
                                tuple(np.multiply(r_heel, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(image, str(round(l_shoulder_angle)),
                                tuple(np.multiply(l_shoulder, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(round(r_shoulder_angle)),
                                tuple(np.multiply(r_shoulder, [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                                )




                except:
                    pass

                cv2.putText(image, find_pose(current_angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2,
                            cv2.LINE_8)
                # cv2.putText(image, str(int(distance_pinky)), (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8)
                # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                # mp_pose.PoseLandmark.LEFT_SHOULDER.value
                # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                # print(calculate_angle(shoulder,elbow,wrist))
                # curl status box
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #                           mp_drawing.DrawingSpec(color=circle_color, thickness=8, circle_radius=2),
                #                           mp_drawing.DrawingSpec(color=landmark_color, thickness=1, circle_radius=1)
                #                           )


                cv2.imshow('Pose Detection', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                print("Frame is empty")
                break
        else:
            print("End of video")
            break
client.disconnect()
cap.release()
cv2.destroyAllWindows()
