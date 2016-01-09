# -*- coding: utf-8 -*-
"""All road featuring tracking functionality

TODO: Comments and proper docstrings

"""

import collections
import cv2
import numpy as np
# import matplotlib.pyplot as plt


Point = collections.namedtuple('Point', 'x y')
Line = collections.namedtuple('Line', 'top bottom')  # Line segments

# TODO: Better solution
Extremes = collections.namedtuple('Extremes', 'min max')


class Lane(collections.namedtuple('Lane', 'left right')):

    def in_range(self, point):
        '''Check if point in range of the lane

        TODO: Make more accurate
        '''
        delta_x = self.left.top.x - self.left.bottom.x
        delta_y = self.left.top.y - self.left.bottom.y
        gradient = float(delta_y) / float(delta_x)

        # Calculate the left line segment position for given point
        left_x = self.left.top.x + (gradient * (point.y - self.left.top.y))

        delta_x = self.right.top.x - self.right.bottom.x
        delta_y = self.right.top.y - self.right.bottom.y
        gradient = float(delta_y) / float(delta_x)

        # Calculate the right line segment position for given point
        right_x = self.right.top.x + (gradient * (point.y - self.right.top.y))

        x = left_x < point.x < right_x
        y = self.left.top.y < point.y < self.left.bottom.y

        return x and y  # Has a small margin of error


class LaneDetection(object):
    """Keeps track of the current lane in the given image sequence.

    Attributes:
      __verbose (bool): Verbose mode outputs additional information
      __hough_threshold (int): Threshold used in the Hough Transform
      __hough_minLineLength (int): Line length used in the Hough Transform
      __hough_maxLineGap (int): The maximum gap used in the Hough Transform
      __roi_ratio (float): Determines which part of the image should be used
      __roi_offset (int): The height removed from the image
      __extremes_y_axis (dict): The min/max measured values of the Y-axis
      __extremes_x_axis (dict): The min/max measured values of the X-axis
      __left_segment (Line): The last known line on the left side of the car
      __right_segment (Line): The last known line on the right side of the car

    """

    def __init__(self, hough_threshold=50,
                 hough_minLinLength=50, hough_maxLineGap=40,
                 roi_ratio=0.5, verbose=False):
        """Initiates the left and right line storage.
        """
        # Verbose mode
        self.__verbose = verbose

        # Hough Settings
        self.__hough_threshold = hough_threshold
        self.__hough_minLinLength = hough_minLinLength
        self.__hough_maxLineGap = hough_maxLineGap

        self.__roi_ratio = int(1 // roi_ratio)
        self.__roi_offset = 0

        # We need to be able to set the min/max values individually
        self.__extremes_x_axis = {'min': None, 'max': None}
        self.__extremes_y_axis = {'min': None, 'max': None}

        # Last known lines
        self.__left_segment = None
        self.__right_segment = None

    def analyse_image(self, im):
        '''Takes an image and uses Hough Transform to find and save lane info.
        '''
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        height, width = im_gray.shape
        self.__roi_offset = height // self.__roi_ratio

        im_roi = im_gray[self.__roi_offset:height, :width]

        im_roi = cv2.GaussianBlur(im_roi, (3, 3), 0)

        threshold, img = cv2.threshold(
            np.array(im_roi, np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        roi_edges = cv2.Canny(im_roi, threshold * 0.5, threshold)

        if self.__verbose:
            cv2.imshow("Lane Detection - Edges - Verbose", roi_edges)

        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, self.__hough_threshold,
            self.__hough_minLinLength, self.__hough_maxLineGap)

        if self.__verbose:
            if lines is not None:
                for x1, y1, x2, y2 in lines[0]:
                    cv2.line(im, (x1, y1 + (height / 2)),
                             (x2, y2 + (height / 2)),
                             (0, 0, 255), 2, cv2.CV_AA)

        # Y-Intercepts for clustering optimization
        # y_intercepts, gradients = _find_y_intercepts(lines)

        # if not y_intercepts:
        #     return

        # if self.__verbose:
        #     plt.clf()
        #     plt.ylim(-2, 2)
        #     plt.xlim(-height / 2, height / 2)
        #     plt.xlabel('Y-Intercept')
        #     plt.ylabel('Gradient')
        #     plt.plot(y_intercepts, gradients, 'ro')
        #     plt.draw()
        #     plt.show(block=False)

        left, right = _classify_lane_lines(lines)

        if len(left) == 0 and len(right) == 0:
            # Nothing found, return early
            return

        self.__update_extremes(left + right)
        min_y = self.__extremes_y_axis['min']
        max_y = self.__extremes_y_axis['max']

        if len(left) > 0:
            left_line = _scale_line_y_axis(_get_mean_line(left), min_y, max_y)
            # Only accept x-coordinates that are actually left from the
            # right x-coordinates
            if self.__right_segment is None or \
                    left_line.top.x < self.__right_segment.top.x:
                self.__left_segment = left_line
        elif self.__left_segment is not None:
            # Refresh line
            self.__left_segment = _scale_line_y_axis(
                self.__left_segment, min_y, max_y)

        if len(right) > 0:
            right_line = _scale_line_y_axis(
                _get_mean_line(right), min_y, max_y)
            # Only accept x-coordinates that are actually right from the
            # left x-coordinates
            if self.__left_segment is None or \
                    right_line.top.x > self.__left_segment.top.x:
                self.__right_segment = right_line
        elif self.__right_segment is not None:
            self.__right_segment = _scale_line_y_axis(
                self.__right_segment, min_y, max_y)

    def have_lane(self):
        '''Simple check if both the left and right line is available
        '''
        return self.__left_segment and self.__right_segment

    def get_lane(self):
        '''Returns both the left and right line (lane information)
        '''
        left_line = _add_line_y_offset(self.__left_segment, self.__roi_offset)
        right_line = _add_line_y_offset(
            self.__right_segment, self.__roi_offset)
        return Lane(left_line, right_line)

    def get_center_line(self):
        '''Returns the average of the left and right line
        '''
        if self.have_lane():
            left_line = _add_line_y_offset(
                self.__left_segment, self.__roi_offset)
            right_line = _add_line_y_offset(
                self.__right_segment, self.__roi_offset)
            return _get_mean_line([left_line, right_line])

    def __update_extremes(self, lines):
        '''Finds the extremes given line points and saves them

        Args:
          lines (List): The lines to check of there are new extremes
        '''
        x_axis = [point.x for line in lines for point in line]
        y_axis = [point.y for line in lines for point in line]

        extremes_x_axis = Extremes(min(x_axis), max(x_axis))
        extremes_y_axis = Extremes(min(y_axis), max(y_axis))

        # Update extremes if they are lower/higher than the current extreme
        if self.__extremes_x_axis['min'] is None or \
                extremes_x_axis.min < self.__extremes_x_axis['min']:
            self.__extremes_x_axis['min'] = extremes_x_axis.min

        if self.__extremes_x_axis['max'] is None or \
                extremes_x_axis.max > self.__extremes_x_axis['max']:
            self.__extremes_x_axis['max'] = extremes_x_axis.max

        if self.__extremes_y_axis['min'] is None or \
                extremes_y_axis.min < self.__extremes_y_axis['min']:
            self.__extremes_y_axis['min'] = extremes_y_axis.min

        if self.__extremes_y_axis['max'] is None or \
                extremes_y_axis.max > self.__extremes_y_axis['max']:
            self.__extremes_y_axis['max'] = extremes_y_axis.max


class StopLineDetection(object):
    """Keeps track of stop lines

    Attributes:
      __verbose (bool): Verbose mode outputs additional information
      __hough_threshold (int): Threshold used in the Hough Transform
      __hough_minLineLength (int): Line length used in the Hough Transform
      __hough_maxLineGap (int): The maximum gap used in the Hough Transform
      __roi_ratio (float): Determines which part of the image should be used
      __roi_offset (int): The height removed from the image
      __lane (Lane): The last known lane
    """

    def __init__(self, hough_threshold=50,
                 hough_minLinLength=50, hough_maxLineGap=40,
                 roi_ratio=0.5, verbose=False):
        """Initiates the left and right line storage.
        """
        # Verbose mode
        self.__verbose = verbose

        # Hough Settings
        self.__hough_threshold = hough_threshold
        self.__hough_minLinLength = hough_minLinLength
        self.__hough_maxLineGap = hough_maxLineGap

        self.__roi_ratio = int(1 // roi_ratio)
        self.__roi_offset = 0

        self.__lane = None

    def analyse_image(self, im, lane=None):
        '''Takes an image and uses Hough Transform to find and save lane info.
        '''
        self.__lane = lane

        if self.__lane is None:
            return  # Useless if we don't have a lane (yet)

        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        height, width = im_gray.shape
        self.__roi_offset = height // self.__roi_ratio

        im_roi = im_gray[self.__roi_offset:height, :width]

        im_roi = cv2.GaussianBlur(im_roi, (3, 3), 0)

        threshold, img = cv2.threshold(
            np.array(im_roi, np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        roi_edges = cv2.Canny(im_roi, threshold * 0.5, threshold)

        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, self.__hough_threshold,
            self.__hough_minLinLength, self.__hough_maxLineGap)

        horizontal_lines = _classify_horizontal_lines(lines)

        stop_lines = self.__find_stop_lines(horizontal_lines)

        if stop_lines is not None:
            for line_segment in stop_lines:
                cv2.line(im, (line_segment.top.x, line_segment.top.y),
                         (line_segment.bottom.x, line_segment.bottom.y),
                         (255, 0, 0), 2, cv2.CV_AA)

        # TODO: Stop line tracking

    def __find_stop_lines(self, horizontal_lines):
        stop_lines = []
        for line_segment in horizontal_lines:
            top = Point(
                line_segment.top.x,
                line_segment.top.y + self.__roi_offset)
            bottom = Point(
                line_segment.bottom.x,
                line_segment.bottom.y + self.__roi_offset)
            top_in_range = self.__lane.in_range(top)
            bottom_in_range = self.__lane.in_range(bottom)
            if top_in_range and bottom_in_range:
                stop_lines.append(Line(top, bottom))
        return stop_lines


def _find_y_intercepts(lines):
    '''Finds the y_intercepts together with their gradients

    Args:
      lines (List): A list of the line segments to find intercepts of

    Returns:
      y_intercepts (List): List of all the y_intercepts
      gradients (List): List of all the gradients
    '''
    gradients = []
    y_intercepts = []
    if lines is not None:
        for x, y, a, b in lines[0]:
            delta_x = x - a
            delta_y = y - b

            if delta_x == 0:
                continue  # Vertical lines don't have x-intercept

            gradient = float(delta_y) / float(delta_x)

            y_intercept = y - (x * gradient)

            y_intercepts.append(y_intercept)
            gradients.append(gradient)

    return (y_intercepts, gradients)


def _classify_lane_lines(lines):
    '''Classifies the line segments

    Args:
      lines (List): A list of the line segments to classify
    '''
    left = []
    right = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:

            delta_x = x2 - x1
            delta_y = y2 - y1

            if delta_x == 0:
                continue  # Vertical line

            gradient = float(delta_y) / float(delta_x)

            # Gradient of lines on the right fall between these values
            if 0.8 < gradient < 1.5:
                """Right lines

                Top - y1 - left_line[0]
                Bottom - y2 - left_line[1]

                """
                right.append(Line(Point(x1, y1), Point(x2, y2)))

            # Gradient of lines on the left fall between these values
            elif -0.8 > gradient > -1.5:
                """Left line

                Top - y2 - left_line[1]
                Bottom - y1 - left_line[0]

                """

                left.append(Line(Point(x2, y2), Point(x1, y1)))

    # Reject outliers
    # TODO: Optimize
    if len(left) > 0:
        data = [line.top.x for line in left]
        mean = np.mean(data)
        stdev = np.std(data)
        for key, line in enumerate(left):
            if abs(line.top.x - mean) > mean * stdev:
                del left[key]

    if len(right) > 0:
        data = [line.top.x for line in right]
        mean = np.mean(data)
        stdev = np.std(data)
        for key, line in enumerate(right):
            if abs(line.top.x - mean) > mean * stdev:
                del right[key]

    return left, right


def _classify_horizontal_lines(lines):
    '''Classifies the line segments for stop lines

    Args:
      lines (List): A list of the line segments to classify
    '''
    classified = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            delta_y = y2 - y1
            if -10 < delta_y < 10:
                classified.append(Line(Point(x1, y1), Point(x2, y2)))
    return classified


def _add_line_y_offset(line, y_offset):
    '''Adds an offset to the y axis

    Args:
      line (Line): The line segment to add the offset together
      y_offset (int): The offset to add to the line segment

    Returns:
      The line with offset
    '''
    top = Point(line.top.x, line.top.y + y_offset)
    bottom = Point(line.bottom.x, line.bottom.y + y_offset)
    return Line(top, bottom)


def _get_mean_line(lines):
    '''Calculates the mean of all givens line segments

    Args:
      lines (List): A list of the line segments

    Returns:
      The mean line
    '''
    top_x_axis = np.array([line.top.x for line in lines])
    bottom_x_axis = np.array([line.bottom.x for line in lines])

    top_y_axis = np.array([line.top.y for line in lines])
    bottom_y_axis = np.array([line.bottom.y for line in lines])

    mean_top = Point(int(top_x_axis.mean()), int(top_y_axis.mean()))
    mean_bottom = Point(int(bottom_x_axis.mean()), int(bottom_y_axis.mean()))

    return Line(mean_top, mean_bottom)


def _scale_line_y_axis(line, top, bottom):
    '''Scales the y axis of a line segment to a new top and bottom

    Args:
        line (Line): The line segment to be scaled
        top (int): The new top of the line segment
        bottom (int): The new bottom of the line segment
    '''
    distance_to_top = line.top.y - top
    distance_to_bottom = line.top.y - bottom

    delta_x = line.top.x - line.bottom.x
    delta_y = line.top.y - line.bottom.y

    slope = float(delta_y) / float(delta_x)

    adjusted_x_top = line.top.x - int(round(distance_to_top / slope))
    adjusted_x_bottom = line.top.x - int(round(distance_to_bottom / slope))

    top_point = Point(adjusted_x_top, top)
    bottom_point = Point(adjusted_x_bottom, bottom)

    return Line(top_point, bottom_point)
