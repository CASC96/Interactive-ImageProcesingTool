# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 18:32:21 2022

@author: camil
"""
import argparse
import numpy as np


def parser():

    default_carpetas = [1]
    default_videos = [1]
    default_TVINT = [47]
    default_AFFINT = [50]
    default_rangesearch_radius = [6]
    default_limite = [50]

    default_fi = [1]
    default_ff = [50]

    parser = argparse.ArgumentParser()
    parser.add_argument('-folders', '--fo', dest='folders',
                        type=int, nargs=1, default=default_carpetas)
    parser.add_argument('-videos', type=int, nargs=1,
                        default=default_videos)

    parser.add_argument('-threshold', type=int,
                        nargs=1, default=default_TVINT)
    parser.add_argument('-filterArea', type=int,
                        nargs=1, default=default_AFFINT)

    parser.add_argument('-rangeSearch', type=int, nargs=1,
                        default=default_rangesearch_radius)
    parser.add_argument('-limit', type=int, nargs=1,
                        default=default_limite)

    parser.add_argument('-fi', type=int, nargs=1, default=default_fi)
    parser.add_argument('-ff', type=int, nargs=1, default=default_ff)

    args = parser.parse_args()

    # print(args)

    Threshold_value = np.divide(args.threshold, 100)
    areafilterFactor = np.divide(args.filterArea, 100)
    ncarpeta = np.array(args.folders)
    nvideo = np.array(args.videos)
    rangesearch_radius = np.array(args.rangeSearch)
    limite = np.array(args.limit)
    fi = np.array(args.fi)
    ff = np.array(args.ff)

    return (ncarpeta, nvideo, Threshold_value, areafilterFactor, rangesearch_radius, limite, fi, ff)
