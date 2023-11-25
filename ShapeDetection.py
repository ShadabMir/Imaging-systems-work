import cv2
import numpy as np


def values_are_similar(n1, n2, sensitivity):
    return abs(n1 - n2) < sensitivity * (abs(n1) + abs(n2)) / 2


def gradients_are_similar(n1, n2, sensitivity):
    return abs(np.arctan(n1) - np.arctan(n2)) < sensitivity


def get_shapes_in_image(img, minimum_area=0, maximum_area=float("inf"), approximation_method=cv2.CHAIN_APPROX_SIMPLE):
    contours = []
    contours, _ = cv2.findContours(
        img, cv2.RETR_LIST, approximation_method)
    contours_out = []
    area = []
    bounds = []
    for contour in contours:  # sort useful contours, save their bounds and areas
        b = cv2.boundingRect(contour)
        if maximum_area >= b[2] * b[3] >= minimum_area:
            contours_out.append(contour)
            bounds.append(b)
            area.append(cv2.contourArea(contour))

    return contours_out, bounds, area


def get_line_angle(point1, point2):  # Gets the angle of a line as a bearing from the bottom of the screen
    x, y = point2[0] - point1[0], point2[1] - point1[1]
    x_pos = x >= 0
    y_pos = y >= 0
    if x == 0:  # up or down
        return 0 if y_pos else np.pi
    elif y == 0:  # left or right
        return np.pi / 2 if x_pos else 3 * (np.pi / 2)
    elif x_pos and y_pos:  # 0-90
        return np.arctan(x / y)
    elif x_pos:  # 90-180
        return np.pi + np.arctan(x / y)
    elif y_pos:  # 270-360
        return 2 * np.pi + np.arctan(x / y)
    else:  # 180-270
        return np.pi + np.arctan(x / y)


def angles_are_similar(angle1, angle2, sensitivity):
    d_angle = abs(angle2 - angle1)
    if d_angle > np.pi:
        d_angle = 2 * np.pi - d_angle
    return d_angle < sensitivity


def rad2deg(rad):  # converts radians to degrees (multiplier pre-computed for efficiency, =180/pi)
    return int(rad * 57.29577951308232 + 0.5)  # +0.5 used to round to the nearest degree


def deg2rad(deg):  # converts degrees to radians (multiplier pre-computed for efficiency, =pi/180)
    return deg * 0.01745329252


def a_equals_b_plusminus_c(a, b, c):
    return a + c >= b >= a - c


def determine_shape_type_experimental(contour, img=False):
    # Some simplification is required in order to spread out points as close points follow pixels and
    # have high variation in angles as a result
    sensitivity = 0.004
    contour = cv2.approxPolyDP(contour, sensitivity * cv2.arcLength(contour, True), True)

    test_img = img.copy()
    last = contour[-1][0]
    angles = []  # Used to determine the gradients of lines
    diff_angles = []  # Used to determine the locations of corners
    diff2_angles = []  # Used to determine irregularities in the shape and better distinguish noise

    # High acceleration marks the start of a new zone e.g. curve to area of straight lines part-circles
    # A shape has regular angle size if acceleration is always low
    # A shape with high acceleration throughout is a star, a cross or noise
    # If a zone consists of a single point it can be considered a mistake and filtered out
    # Note to check if shape matches a cross or star before filtering these points

    # Velocities determine the exterior angles in a shape, measured in radians
    # Velocities below 45 degrees (use 30) belong to a curve

    # Angle shows the direction a line is facing measured in radians
    # Used to determine if lines are parallel to each other

    count = 0
    for cnt in contour:  # Find angles of all lines in contour
        count += 1
        current = cnt[0]
        angle = get_line_angle(last, current) % np.pi
        angles.append(angle)
        last = current
    for i in range(len(angles)):  # Find line angle velocity
        diff_angles.append(abs(angles[i] - angles[i - 1]))
        if diff_angles[i] > np.pi / 2:
            diff_angles[i] = np.pi - diff_angles[i]
    for i in range(len(angles)):  # Find line angle acceleration
        diff2_angles.append(abs(diff_angles[i] - abs(angles[i - 1] - angles[i - 2])))
        if diff2_angles[i] > np.pi / 2:
            diff2_angles[i] = np.pi - diff2_angles[i - 1]

        # Early debug work
        print(f"{i}: s={rad2deg(angles[i])}, v={rad2deg(diff_angles[i])}, a={rad2deg(diff2_angles[i])}")
        if abs(int(diff_angles[i] * 180 / np.pi)) >= 30:
            cv2.drawContours(test_img, [contour[i - 1]], 0, (255, 0, 0), 10)
        if abs(int(diff_angles[i] * 180 / np.pi)) < 30:
            cv2.drawContours(test_img, [contour[i]], 0, (0, 0, 255), 10)
        if abs(int(diff2_angles[i] * 180 / np.pi)) >= 30:
            cv2.drawContours(test_img, [contour[i - 1]], 0, (255, 255, 255), 5)

    # Split contour into segments at each major change in shape
    segments = []
    for i in range(len(diff2_angles)):
        if diff2_angles[i] > deg2rad(50):
            segments.append(i)
    # print(segments)

    # Find average velocity (exterior angle) of each contour segment
    segment_data = {}
    if len(segments) >= 1:
        for i in range(len(segments)):
            segment_length = len(angles) - segments[i] + segments[0] if i + 1 == len(segments) else segments[i + 1] - \
                                                                                                    segments[i]
            vel_list = []
            for j in range(segment_length):
                vel_list.append(diff_angles[(segments[i] + j) % len(diff_angles)])
            vel_list.sort()
            vel_median = vel_list[len(vel_list) // 2]
            segment_data[segments[i]] = (vel_median, segment_length)

    for i in segment_data.keys():
        print(f"{i}: {rad2deg(segment_data[i][0])} {segment_data[i][1]}")

    # ADD CROSS, STAR and Trapezium DETECTION HERE

    # ADD OUTLIER VERTEX SEGMENT REMOVER HERE

    if len(segment_data.keys()) == 1:
        if a_equals_b_plusminus_c(rad2deg(segment_data[i][0]), 60, 10):
            print("BOB FOUND A TRIANGLE MFS")
            return "Triangle"
        if a_equals_b_plusminus_c(rad2deg(segment_data[i][0]), 90, 10):
            print("BOB FOUND A QUAD MFS")
            return "Quad"
        # Holds rectangles circles and all regular shapes
    if len(segment_data.keys()) == 2:
        for i in segment_data.keys():
            if segment_data[i][1] == 2 and a_equals_b_plusminus_c(rad2deg(segment_data[i][0]), 90, 10):
                print("BOB FOUND A SEMICIRCLE MFS")
                return "Semicircle"
            elif segment_data[i][1] == 3 and a_equals_b_plusminus_c(rad2deg(segment_data[i][0]), 90, 10):
                print("BOB FOUND A QUARTER CIRCLE MFS")
                return "Quartercircle"

        # 2 sections holds quarter circles and semi circles

    # if no shape found return UNKNOWN

    cv2.imshow("Bob", test_img)
    cv2.waitKey(0)


def determine_shape_type(contour, img=False, sensitivity=0.004, deadzone=4):  # returns the contour's shape's name
    # Simplify contour to reduce edge count to true value

    # determine_shape_type_experimental(contour, img)

    approx = cv2.approxPolyDP(contour, sensitivity * cv2.arcLength(contour, True), True)

    temp = []
    last = [-1]
    for i in range(len(approx)):
        if last[0] != -1:
            dist = 1.4 * min(abs(approx[i][0][0] - last[0]), abs(approx[i][0][1] - last[1])) + \
                   abs(abs(approx[i][0][0] - last[0]) - abs(approx[i][0][1] - last[1]))
            if dist > deadzone:
                temp.append(approx[i])
        last = approx[i][0]
    dist = 1.4 * min(abs(approx[0][0][0] - last[0]), abs(approx[0][0][1] - last[1])) + \
           abs(abs(approx[0][0][0] - last[0]) - abs(approx[0][0][1] - last[1]))
    if dist > deadzone:
        temp.append(approx[0])

    approx = temp

    # Find outlier lengths, useful for semicircles and quarter circles

    last = [-1]
    distances = np.zeros(len(approx))
    for i in range(len(approx)):
        if last[0] != -1:
            dist = 1.4 * min(abs(approx[i][0][0] - last[0]), abs(approx[i][0][1] - last[1])) + \
                   abs(abs(approx[i][0][0] - last[0]) - abs(approx[i][0][1] - last[1]))
            distances[i] = dist
        last = approx[i][0]

    dist = 1.4 * min(abs(approx[0][0][0] - last[0]), abs(approx[0][0][1] - last[1])) + \
           abs(abs(approx[0][0][0] - last[0]) - abs(approx[0][0][1] - last[1]))
    distances[0] = dist
    sorted_distances = distances
    sorted_distances.sort()
    q1, q3 = sorted_distances[len(sorted_distances) // 4], sorted_distances[int(3 * (len(sorted_distances) / 4))]
    outliers = [x for x in sorted_distances if x > q3 + 3 * (q3 - q1)]  # find abnormally large lines in contour

    for cnt in approx:
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 10)

    if len(outliers) == 1:  # 1 far longer line than others, usually semicircles
        shape = "Semicircle"

    elif len(outliers) == 2:  # 2 far longer lines than others, usually quarter circles
        shape = "Quartercircle"

    elif len(approx) == 4:  # Many quadrilaterals used in competition, further processing required
        gradients = np.zeros(4)
        for i in range(3):
            if approx[i][0][1] > approx[i + 1][0][1]:
                gradients[i] = (approx[i][0][1] - approx[i + 1][0][1]) / (approx[i][0][0] - approx[i + 1][0][0] + 0.001)
            elif approx[i][0][1] == approx[i + 1][0][1]:
                if approx[i][0][0] > approx[i + 1][0][0]:
                    gradients[i] = (approx[i][0][1] - approx[i + 1][0][1]) / (
                                approx[i][0][0] - approx[i + 1][0][0] + 0.001)
                else:
                    gradients[i] = (approx[i][0][1] - approx[i + 1][0][1]) / (
                                approx[i + 1][0][0] - approx[i][0][0] + 0.001)
            else:
                gradients[i] = (approx[i + 1][0][1] - approx[i][0][1]) / (approx[i][0][0] - approx[i + 1][0][0] + 0.001)

        # Adds small decimal value to eliminate divide by 0 errors, coord values will be integers so /0 is impossible
        gradients[3] = (approx[-1][0][1] - approx[0][0][1]) / (approx[-1][0][0] - approx[0][0][0] + 0.001)
        if gradients_are_similar(gradients[0], gradients[2], 1) == gradients_are_similar(gradients[1], gradients[3], 1):
            shape = "Trapezium"
        else:
            if values_are_similar(sorted_distances[0], sorted_distances[-1], 0.1):
                shape = "Square"

            else:
                shape = "Rectangle"

    elif len(approx) == 3:  # Basic shapes, only regulars used in competition so angles are unnecessary
        shape = "Triangle"
    elif len(approx) == 5:
        shape = "Pentagon"
    elif len(approx) == 6:
        shape = "Hexagon"
    elif len(approx) == 7:
        shape = "Heptagon"
    elif len(approx) == 8:
        shape = "Octagon"

    else:  # non-quad/basic, requires further processing
        sum_x = 0
        sum_y = 0
        for point in approx:
            sum_x += point[0][0]
            sum_y += point[0][1]
        sum_x /= len(approx)
        sum_y /= len(approx)
        distances = np.zeros(len(approx))
        for i in range(len(approx)):
            dist = 1.4 * min(abs(approx[i][0][0] - sum_x), abs(approx[i][0][1] - sum_y)) + \
                   abs(abs(approx[i][0][0] - sum_x) - abs(approx[i][0][1] - sum_y))
            distances[i] = dist
        distances.sort()
        q1, q3 = distances[len(distances) // 4], distances[int(3 * (len(distances) / 4))]
        if values_are_similar(q1, q3, 0.2):  # if shape's vertices are evenly spread from the center assume circle
            shape = "Circle"
        else:
            if len(approx) == 10:
                shape = "Star"
            elif len(approx) == 12:
                shape = "Cross"
            else:
                shape = "UNKNOWN"

    return shape, approx