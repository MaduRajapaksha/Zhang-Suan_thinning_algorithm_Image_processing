import cv2
import numpy as np

img = cv2.imread('image.jpg',0)
retval, orig_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
bin_thresh = (orig_thresh == 0).astype(int)

def pixel_is_black(arr, x, y):
    if arr[x, y] == 1:
        return True
    return False

# steps one and two, condition two
def pixel_has_2_to_6_black_neighbors(arr, x, y):
    # pixel values can only be 0 or 1, so simply check if sum of
    # neighbors is between 2 and 6
    if (2 <= arr[x, y - 1] + arr[x + 1, y - 1] + arr[x + 1, y] + arr[x + 1, y + 1] +
            arr[x, y + 1] + arr[x - 1, y + 1] + arr[x - 1, y] + arr[x - 1, y - 1] <= 6):
        return True
    return False


# steps one and two, condition three
def pixel_has_1_white_to_black_neighbor_transition(arr, x, y):
    # neighbors is a list of neighbor pixel values; neighbor P2 appears
    # twice since we will cycle around P1.
    neighbors = [arr[x, y - 1], arr[x + 1, y - 1], arr[x + 1, y], arr[x + 1, y + 1],
                 arr[x, y + 1], arr[x, y + 1], arr[x - 1, y], arr[x - 1, y - 1],
                 arr[x, y - 1]]
    # zip returns iterator of tuples composed of a neighbor and next neighbor
    # we then check if the neighbor and next neighbor is a 0 -> 1 transition
    # finally, we sum the transitions and return True if there is only one
    transitions = sum((a, b) == (0, 1) for a, b in zip(neighbors, neighbors[1:]))
    if transitions == 1:
        return True
    return False
# step one condition four
def at_least_one_of_P2_P4_P6_is_white(arr, x, y):
    # if at least one of P2, P4, or P6 is 0 (white), logic statement will
    # evaluate to false.
    if (arr[x, y - 1] and arr[x + 1, y] and arr[x, y + 1]) == False:
        return True
    return False
# step one condition five
def at_least_one_of_P4_P6_P8_is_white(arr, x, y):
    # if at least one of P4, P6, or P8 is 0 (white), logic statement will
    # evaluate to false.
    if (arr[x + 1, y] and arr[x, y + 1] and arr[x - 1, y]) == False:
        return True
    return False

# step two condition four
def at_least_one_of_P2_P4_P8_is_white(arr, x, y):
    # if at least one of P2, P4, or P8 is 0 (white), logic statement will
    # evaluate to false.
    if (arr[x, y - 1] and arr[x + 1, y] and arr[x - 1, y]) == False:
        return True
    return False
# step two condition five
def at_least_one_of_P2_P6_P8_is_white(arr, x, y):
    # if at least one of P2, P6, or P8 is 0 (white), logic statement will
    # evaluate to false.
    if (arr[x, y - 1] and arr[x, y + 1] and arr[x - 1, y]) == False:
        return True
    return False
# make a copy of the binary threshold array, upon which we will apply
# the thinning algorithm
thinned_thresh =  bin_thresh.copy()

# if the thinned threshold reaches a steady state, we'll break out of the loop
while 1:
    # make a copy of the thinned threshold array to check for changes
    thresh_copy = thinned_thresh.copy()
    # step one
    pixels_meeting_criteria = []
    # check all pixels except for border and corner pixels
    # if a pixel meets all criteria, add it to pixels_meeting_criteria list
    for i in range(1, thinned_thresh.shape[0] - 1):
        for j in range(1, thinned_thresh.shape[1] - 1):
            if (pixel_is_black(thinned_thresh, i, j) and
                    pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
                    pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
                    at_least_one_of_P2_P4_P6_is_white(thinned_thresh, i, j) and
                    at_least_one_of_P4_P6_P8_is_white(thinned_thresh, i, j)):
                pixels_meeting_criteria.append((i, j))

    # change noted pixels in thinned threshold array to 0 (white)
    for pixel in pixels_meeting_criteria:
        thinned_thresh[pixel] = 0

    # step two
    pixels_meeting_criteria = []
    # check all pixels except for border and corner pixels
    # if a pixel meets all criteria, add it to pixels_meeting_criteria list
    for i in range(1, thinned_thresh.shape[0] - 1):
        for j in range(1, thinned_thresh.shape[1] - 1):
            if (pixel_is_black(thinned_thresh, i, j) and
                    pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
                    pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
                    at_least_one_of_P2_P4_P8_is_white(thinned_thresh, i, j) and
                    at_least_one_of_P2_P6_P8_is_white(thinned_thresh, i, j)):
                pixels_meeting_criteria.append((i, j))

    # change noted pixels in thinned threshold array to 0 (white)
    for pixel in pixels_meeting_criteria:
        thinned_thresh[pixel] = 0

    # if the latest iteration didn't make any difference, exit loop
    if np.all(thresh_copy == thinned_thresh) == True:
        break
