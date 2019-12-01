#!/usr/bin/env python

"""
Usage: python carver.py <r/c> <scale> <image_in> <image_out>
Copyright 2018 Karthik Karanth, MIT License
"""

import sys

from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
import cv2
from skimage.segmentation import active_contour
from functools import cmp_to_key

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)
        # cv2.imshow('image',img)
        # cv2.waitKey(1)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def get_mask_from_snake(snake, shape):

    snake2 = np.round(snake).astype(int)
    snake_mask = np.zeros(shape, dtype=int)
    ##Earlier
    #snake_mask[snake[:,0], snake[:,1]] = 1

    def cmp(a,b):
        if a[0] > b[0]:
            return 1
        elif a[0] == b[0]:
            if a[1] > b[1]:
                return 1
            else:
                return -1
        else:
            return -1

    cmp_items_py3 = cmp_to_key(cmp)

    s = np.array(sorted(snake2, key=cmp_to_key(cmp)))

    # print(s)

    snake_new = []

    curr = 0
    curr_j = []
    last_e = None
    for e in s:
        if e[0] == curr:    ## counting multiple cols in same row
            curr_j += [e[1]]
        else:
            if len(curr_j) == 1:
                snake_mask[curr, curr_j[0]] = 1
                snake_new += [[curr, curr_j[0]]]
            elif len(curr_j) > 1:
                mean = sum(curr_j)/len(curr_j)
                snake_mask[curr, int(mean)] = 1
                snake_new += [[curr, int(mean)]]
            
            if e[0] - last_e[0] > 1: ## some row(s) skipped. last_e[0] = curr always
                for i in range(last_e[0]+1, e[0], 1):
                    p = last_e[1] + (((i-last_e[0])*(e[1]-last_e[1]))/(e[0]-last_e[0]))
                    snake_mask[i, int(p)] = 1
                    snake_new += [[i, int(p)]]

            curr = e[0]
            curr_j = [e[1]]

        last_e = e.copy()
    snake_mask[last_e[0],last_e[1]] = 1
    snake_new += [last_e]

    return snake_mask, np.array(snake_new)

def snake(mask, image):
    int_mask = (~mask).astype(int)
    r,c = np.nonzero(int_mask)
    init = np.array([r,c]).T

    img2 = image.copy()
    img2[r,c] *= 0
    cv2.imshow('mask',img2)
    cv2.waitKey(1)

    snake = active_contour(image, init, bc='fixed', alpha=0.005, beta=5, w_edge=-5, gamma=0.01, max_px_move=2)

    np.clip(snake[:,0], 0, image.shape[0]-1, out=snake[:,0])
    np.clip(snake[:,1], 0, image.shape[1]-1, out=snake[:,1])

    # print(snake)
    snake_mask, snake_new = get_mask_from_snake(snake, mask.shape)
    # print(snake_new)
    # snake = np.round(snake).astype(int)

    if len(np.unique(snake_new, axis=0)) != image.shape[0]:
        print(len(np.unique(snake_new, axis=0)))

    rows, row_counts = np.unique(snake_new[:,0], return_counts=True)
    # cols, col_counts = np.unique(snake_new[:,1], return_counts=True)
    if (len(rows[row_counts>1]) > 0):
        print("here")
        print(rows[row_counts>1])
        print(row_counts[row_counts>1])
        print(len(row_counts[row_counts>1]))
        print(row_counts[row_counts>1].sum())
        print(image.shape[0])
        print(len(rows))
        # print(cols[col_counts>1])
        # print(col_counts[col_counts>1])
        # cv2.waitKey(100)

    # print(snake_new)
    
    img3 = image.copy()
    img3[snake_new[:,0], snake_new[:,1]] *= 0
    cv2.imshow('snake',img3)
    cv2.waitKey(1)

    # print(snake_mask)
    # print(snake_mask.shape)

    return ~snake_mask.astype(bool)

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    # cv2.imshow('mask', mask.astype(np.uint8)*255)
    # print((~mask).astype(np.uint8)*255)
    # cv2.waitKey(10)

    mask = snake(mask, img)

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            # Handle right edge
            elif j == c-1:
                idx = np.argmin(M[i-1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]

            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def main():
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    img = imread(in_filename)
    # img = cv2.resize(img,None,fx=0.75,fy=0.75)

    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()