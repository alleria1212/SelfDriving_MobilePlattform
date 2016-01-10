#!/usr/bin/python
import cv2
import features.road
import glob
import operator


def main():
    files = glob.glob('image-sets/short/*.jpg')

    files_ordered = dict()
    for file in files:
        file_components = file.split('-')
        files_ordered[int(file_components[7])] = file

    sorted_files = sorted(
        files_ordered.iteritems(), key=operator.itemgetter(0))

    lane_detector = features.road.LaneDetection(verbose=True)
    stop_detector = features.road.StopLineDetection(verbose=True)

    for file in sorted_files:

        # Load image as grayscale
        im = cv2.imread(files_ordered[int(file[0])])
        lane_detector.analyse_image(im)

        if lane_detector.have_lane():
            Lane = lane_detector.get_lane()
            stop_detector.analyse_image(im, Lane)
            center = lane_detector.get_center_line()

            cv2.line(im, (center.top.x, center.top.y),
                     (center.bottom.x, center.bottom.y),
                     (12, 128, 232), 2, cv2.CV_AA)

            cv2.line(im, (Lane.left.top.x, Lane.left.top.y),
                     (Lane.left.bottom.x, Lane.left.bottom.y),
                     (0, 0, 255), 2, cv2.CV_AA)
            cv2.line(im, (Lane.right.top.x, Lane.right.top.y),
                     (Lane.right.bottom.x, Lane.right.bottom.y),
                     (255, 0, 0), 2, cv2.CV_AA)

        cv2.imshow('Processed', im)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
