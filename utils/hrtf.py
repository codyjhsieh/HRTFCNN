import os
import sys
import logging
import numpy as np
import h5py
import scipy.io

class HRTF(object):

    def __init__(self, nbChannels, samplingRate, maxLength=None):
        self.nbChannels = nbChannels
        self.elevations = None
        self.azimuths = None
        self.distances = None
        self.impulses = None
        self.channels = None
        self.samplingRate = samplingRate

class CipicHRTF(HRTF):
    def __init__(self, filename, samplingRate):

        super(CipicHRTF, self).__init__(nbChannels=2,
                                        samplingRate=44100.0)

        self.filename = filename

        if self.filename.split('.')[-1] == 'mat':

            try:
                elevations_vals = np.linspace(-45, 230.625, num=50)
                azimuths_vals = np.concatenate(
                    ([-80, -65, -55], np.linspace(-45, 45, num=19), [55, 65, 80]))
                self.elevations = []
                self.azimuths = []
                for i in range(len(azimuths_vals)):
                    for j in range(len(elevations_vals)):
                        self.azimuths.append(azimuths_vals[i])
                        self.elevations.append(elevations_vals[j])
                self.elevations = np.array(self.elevations)
                self.azimuths = np.array(self.azimuths)
                self.impulses = self._loadImpulsesFromFileMat()
                self.channels = ['left', 'right']
            except FileNotFoundError as err:
                print('File ' + filename + ' not found')
                print(err)

        elif self.filename.split('.')[-1] == 'sofa':
            try:
                self.impulses = self._loadImpulsesFromFileSofa()
                self.channels = ['left', 'right']
                self.elevations, self.azimuths, self.distances = self._loadPositionsFromFileSofa()
                self.elevations, self.azimuths = np.round(verticalPolarToInterauralPolarCoordinates(self.elevations, self.azimuths), 3)

            except FileNotFoundError as err:
                print(err)
                print('File ' + filename + ' not found')
        
        else:
            print('File',self.filename.split('.')[-1],'not supported. Only mat and sofa are supported.')

    def _loadImpulsesFromFileMat(self):
        elevations_vals = np.linspace(-45, 230.625, num=50)
        azimuths_vals = np.concatenate(
            ([-80, -65, -55], np.linspace(-45, 45, num=19), [55, 65, 80]))
        # Load CIPIC HRTF data
        cipic = scipy.io.loadmat(self.filename)
        hrirLeft = np.transpose(cipic['hrir_l'], [2, 0, 1])
        hrirRight = np.transpose(cipic['hrir_r'], [2, 0, 1])

        # Store impulse responses in time domain
        N = len(hrirLeft[:, 0, 0])
        impulses = np.zeros((len(azimuths_vals)*len(
            elevations_vals), self.nbChannels, N))
        count = 0
        for i in range(len(azimuths_vals)):
            for j in range(len(elevations_vals)):
                impulses[count, 0, :] = hrirLeft[:, i, j]
                impulses[count, 1, :] = hrirRight[:, i, j]
                count += 1

        return impulses

    def _loadImpulsesFromFileSofa(self):

        # Load CIPIC HRTF data
        impulses = np.array(h5py.File(self.filename,'r')["Data.IR"].value.tolist())

        return impulses
    
    def _loadPositionsFromFileSofa(self):

        # Load CIPIC HRTF data
        positions = np.array(h5py.File(self.filename,'r')["SourcePosition"].value.tolist())
        azimuths = positions[:,0]
        elevations = positions[:,1]
        distance = positions[:,2]

        return elevations, azimuths, distance

    def setFileImpulses(self, impulses):
        try:
            hrtf = h5py.File(self.filename,'a')
            hrtf["Data.IR"][:] = impulses[:]
            hrtf.close()
        except FileNotFoundError as err:
            print(err)
            

    def setFilePositions(self, elevations, azimuths):
        try:
            hrtf = h5py.File(self.filename,'a')
            hrtf["SourcePosition"][:,0] = azimuths
            hrtf["SourcePosition"][:,1] = elevations
            hrtf.close()
        except FileNotFoundError as err:
            print(err)

def interauralPolarToVerticalPolarCoordinates(elevations, azimuths):

    elevations = np.atleast_1d(elevations)
    azimuths = np.atleast_1d(azimuths)

    # Convert interaural-polar coordinates to 3D cartesian coordinates on the
    # unit sphere
    x = np.cos(azimuths * np.pi / 180.0) * np.cos(elevations * np.pi / 180.0)
    y = np.sin(azimuths * np.pi / 180.0) * -1.0
    z = np.cos(azimuths * np.pi / 180.0) * np.sin(elevations * np.pi / 180.0)
    assert np.allclose(x**2 + y**2 + z**2, np.ones_like(elevations))

    # Convert 3D cartesian coordinates on the unit sphere to vertical-polar
    # coordinates
    azimuths = np.arctan2(-y, x) * 180.0 / np.pi
    elevations = np.arcsin(z) * 180.0 / np.pi

    azimuths[azimuths < -46] += 360.0

    return elevations, azimuths


def verticalPolarToInterauralPolarCoordinates(elevation, azimuths):

    # Convert vertical-polar coordinates to 3D cartesian coordinates on the
    # unit sphere
    x = np.cos(elevation * np.pi / 180.0) * np.sin(azimuths * np.pi / 180.0)
    y = np.cos(elevation * np.pi / 180.0) * np.cos(azimuths * np.pi / 180.0)
    z = np.sin(elevation * np.pi / 180.0) * 1.0
    assert np.allclose(x**2 + y**2 + z**2, np.ones_like(elevation))

    # Convert 3D cartesian coordinates on the unit sphere to interaural-polar
    # coordinates
    azimuths = np.arcsin(x) * 180.0 / np.pi
    elevation = np.arctan2(z, y) * 180.0 / np.pi

    elevation[elevation < -46] += 360.0

    return elevation, azimuths


def verticalPolarToCipicCoordinates(elevation, azimut):

    elevation, azimut = verticalPolarToInterauralPolarCoordinates(
        elevation, azimut)

    elevation = np.arctan2(np.sin(elevation * np.pi / 180),
                           np.cos(elevation * np.pi / 180)) * 180.0 / np.pi

    if isinstance(elevation, np.ndarray):
        elevation[elevation < -90.0] += 360.0
    else:
        if elevation < -90:
            elevation += 360.0

    return elevation, azimut

def get_hrtf_mat(hrtf_folder, num):
    num_str = str(num)
    if num < 100:
        num_str = '0' + num_str
    if num < 10:
        num_str = '0' + num_str
    return CipicHRTF(hrtf_folder + '/subject_' + str(num_str) + '/hrir_final.mat', 44100.0)

def get_hrtf_sofa(hrtf_folder, num):
    num_str = str(num)
    if num < 100:
        num_str = '0' + num_str
    if num < 10:
        num_str = '0' + num_str
    return CipicHRTF(hrtf_folder + '/subject_' + str(num_str) + '.sofa', 44100.0)

def create_cipic_hrtf(template_filename, filename, impulses, elevations, azimuths):
    try:
        reference = h5py.File(template_filename,'r')
        hrtf = h5py.File(filename,'w')
        
        for key in list(reference.keys()):
            reference.copy(key, hrtf)

        elevations, azimuths = interauralPolarToVerticalPolarCoordinates(elevations, azimuths)
        hrtf["Data.IR"][:] = impulses[:]
        hrtf["SourcePosition"][:,0] = azimuths
        hrtf["SourcePosition"][:,1] = elevations
        hrtf.close()

    except FileNotFoundError as err:
        print(err)
