#!/usr/bin/env python
#
# @file    varfeat.py
# @brief   Code for variability measures listed in Sokolovsky et al. (2016)
# @author  Matthew J. Graham
#
# <!---------------------------------------------------------------------------
# Copyright (C) 2017 by the California Institute of Technology.
# This software is part of the Caltech Time Series Analysis Suite.
# ------------------------------------------------------------------------- -->
#
# Methods assume that <data> is a Numpy array [3,N] with N data points and 
# dim = 0 is time (MJD), dim = 1 is magnitude and dim = 2 is magnitude error.
#

from __future__ import print_function

__authors__ = 'Matthew Graham <mjg@caltech.edu>'
__version__ = '20170603'

import numpy as np
import FATS
from utils import getErrorCorr, getData
import sys
import time


def chisq(data):
  # Reduced chi-square
  stat = 0.
  try:
    t, mag, err = data
    mbar = np.sum(mag / (err * err)) / np.sum(1. / (err * err))
    stat = np.sum(((mag  - mbar) / err) ** 2.)
  except Exception, e:
    print (e)
  return stat


def wstd(data):
  # Weighted standard deviation
  stat = 0.
  try:
    t, mag, err = data
    w = 1. / err
    mbar = np.mean(mag)
    stat = np.sum(w) * np.sum(w * (mag - mbar) ** 2.) / (np.sum(w) ** 2. - np.sum(w * w))
    stat = np.sqrt(stat)
  except Exception, e:
    print (e)
  return stat


def mad(data):
  # Median absolute deviation
  stat = 0.
  try:
    if len(data.shape) > 1:
      t, mag, err = data
    else:
      mag = data
    med = np.median(mag)
    stat = np.median(np.abs(mag - med))
  except Exception, e:
    print (e)
  return stat


def iqr(data):
  # Interquartile range
  stat = 0.
  try:
    t, mag, err = data
    stat = np.subtract(*np.percentile(mag, [75, 25]))
  except Exception, e:
    print (e)
  return stat


def roms(data):
  # Robust median statistic
  stat = 0.
  try:
    t, mag, err = data
    med = np.median(mag)
    stat = np.sum(np.abs(mag - med) / err) / (len(t) - 1)
  except Exception , e:
    print (e)
  return stat


def nev(data):
  # Normalized excess variance
  stat = 0.
  try:
    t, mag, err = data
    mbar = np.mean(mag)
    stat = np.sum((mag - mbar) ** 2. - err ** 2.) / (len(t) * mbar * mbar)
  except Exception, e:
    print (e)
  return stat


def p2pvar(data):
  # Peak-to-peak variability
  stat = 0.
  try:
    t, mag, err = data
    mmsig = np.max(mag - err)
    mpsig = np.min(mag + err)
    stat = (mmsig - mpsig) / (mmsig + mpsig)
  except Exception, e:
    print (e)
  return stat


def l1ac(data):
  # Lag-1 autocorrelation
  stat = 0.
  try:
    t, mag, err = data
    mbar = np.mean(mag)
    stat = np.sum((mag[:-1] - mbar) * (mag[1:] - mbar)) / np.sum((mag - mbar) ** 2.)
  except Exception, e:
    print (e)
  return stat


def stetj(data):
  # Stetson J
  stat = 0.
  try:
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['StetsonJ'])
    a = a.calculateFeature(lc)
    stat = a.result()
  except Exception, e:
    print (e)
  return stat


def stetk(data):
  # Stetson K
  stat = 0.
  try:
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['StetsonK'])
    a = a.calculateFeature(lc)
    stat = a.result()[0]
  except Exception, e:
    print (e)
  return stat


def stetl(data):
  # Stetson L
  stat = 0.
  try:
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['StetsonL'])
    a = a.calculateFeature(lc)
    stat = a.result()
  except Exception, e:
    print (e)
  return stat


def wstetj(data):
  # Stetson J with time-based weighting (TO BE COMPLETED)
  stat = 0.
  try:
    t, mag, err = data
    dt = np.median(np.diff(t))
    w = np.exp(-np.diff(t) / dt) 
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['StetsonL'])
    a = a.calculateFeature(lc)
    stat = a.result()
  except Exception, e:
    print (e)
  return stat


def wstetl(data):
  # Stetson L with time-based weighting (TO BE COMPLETED)
  stat = 0.
  try:
    t, mag, err = data
    dt = np.median(np.diff(t))
    w = np.exp(-np.diff(t) / dt) 
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['StetsonL'])
    a = a.calculateFeature(lc)
    stat = a.result()
  except Exception, e:
    print (e)
  return stat
  

def cssd(data, c = 3.):
  # Consecutive same-sign deviations from the mean magnitude
  stat = 0.
  try:
    t, mag, err = data
    med = np.median(mag)
    md = mad(data)
    fidx = np.where(np.abs(mag - med) > c * 1.4826 * md)[0]
    if len(fidx) > 0:
      grps = fidx[:-2] - fidx[2:] # Groups of 3 consecutive measurements
      gidx = np.where(grps == -2)[0]
      stat = len(gidx) / (len(t) - 2)
  except Exception, e:
    print (e)
  return stat


def ex(data, dscan = 30.):
  # Excursions
  stat = 0.
  try:
    t, mag, err = data
    bounds = np.where(np.diff(t) > dscan)[0] + 1
    if len(bounds) > 0:
      sum = 0. 
      mi = mag[:bounds[0]]
      for j in range(1, len(bounds) - 1):
        mj = mag[bounds[j]: bounds[j + 1]]
        if not(len(mi) == 1 and len(mj) == 1):
          sum += np.abs(np.median(mi) - np.median(mj)) / np.sqrt(mad(mi) ** 2. + mad(mj) ** 2.)
      for i in range(len(bounds) - 2):
        mi = mag[bounds[i]: bounds[i + 1]]
        for j in range(i + 1, len(bounds) - 1):
          mj = mag[bounds[j]: bounds[j + 1]]
          if not(len(mi) == 1 and len(mj) == 1):
            sum += np.abs(np.median(mi) - np.median(mj)) / np.sqrt(mad(mi) ** 2. + mad(mj) ** 2.)
      mi = mag[bounds[-2]: bounds[-1]]
      mj = mag[bounds[-1]:]
      if not(len(mi) == 1 and len(mj) == 1):
        sum += np.abs(np.median(mi) - np.median(mj)) / np.sqrt(mad(mi) ** 2. + mad(mj) ** 2.)           
      stat = 2. * sum / (1.4826 * len(bounds) * (len(bounds) - 1))
  except Exception, e:
    print (e)
  return stat
      

def neu(data):
  # Time weighted von Neumann ratio
  stat = 0.
  try:
    lc = np.array([data[1], data[0], data[2]])
    a = FATS.FeatureSpace(featureList = ['Eta_e'])
    a = a.calculateFeature(lc)
    stat = a.result()
  except Exception, e:
    print (e)
  return stat


def sb(data):
  # S_B variability statistic
  stat = 0.
  try:
    t, mag, err = data
    res = mag - np.mean(mag)
    idx = np.where(res[:-1] * res[1:] < 0)[0] + 1
    sum = np.sum((res[:idx[0]] / err[:idx[0]]) ** 2.)
    for i in range(len(idx) - 1):
      sum += np.sum((res[idx[i]: idx[i + 1]] / err[idx[i]: idx[i + 1]]) ** 2.)
    sum += np.sum((res[idx[-1]:] / err[idx[-1]:]) ** 2.)
    stat = sum / (len(t) * len(idx))
  except Exception, e:
    print (e)
  return stat

