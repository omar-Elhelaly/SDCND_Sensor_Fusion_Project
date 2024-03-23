# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # time increment
        self.q= params.q # process noise variable for Kalman filter Q


    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        return np.matrix([[1, 0, 0, self.dt, 0, 0],
                        [0, 1, 0, 0, self.dt, 0],
                        [0, 0, 1, 0, 0, self.dt],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])

        ############
        # END student code
        ############

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        q1 = ((self.dt**3)/3) * self.q
        q2 = ((self.dt**2)/2) * self.q
        q3 = self.dt * self.q
        return np.matrix([[q1, 0, 0, q2, 0, 0],
                        [0, q1, 0, 0, q2, 0],
                        [0, 0, q1, 0, 0, q2],
                        [q2, 0, 0, q3, 0, 0],
                        [0, q2, 0, 0, q3, 0],
                        [0, 0, q2, 0, 0, q3]])

        ############
        # END student code
        ############

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        track.set_x(F*track.x) # state prediction
        track.set_P(F*track.P*F.transpose() + self.Q()) # covariance prediction

        ############
        # END student code
        ############

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        K = track.P * meas.sensor.get_H(track.x).transpose() * np.linalg.inv(self.S(track, meas, meas.sensor.get_H(track.x))) # Kalman gain
        x = track.x + K*self.gamma(track, meas) # state update
        I = np.identity(self.dim_state)
        P = (I - K * meas.sensor.get_H(track.x)) * track.P

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############
        track.update_attributes(meas)

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        return meas.z - meas.sensor.get_hx(track.x)

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H * track.P * H.transpose() + meas.R

        ############
        # END student code
        ############
