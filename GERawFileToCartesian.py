# -*- coding: utf-8 -*-
"""
GERawFileToCartesian Created on Thu Jun 21 14:41:18 2012

@author: gordon

class to display and convert a 2D GE .raw file into a numpy array. Provides helper methods to
print image attributes as well as to display the converted image.

"""

import tables as tables
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import SimpleITK as sitk

class GERawFileToCartesian:

    image = []
    image_x = []
    image_y = []
    has_doppler = False
    has_waveform = False

    def __init__(self, args1):
        # idiot check that file is valid
        if tables.is_hdf5_file(args1) == False:
            print ("File is Not HDF5 format - abort.")
            return

        self.raw_file = tables.open_file(args1, mode='r')
        self.node_bmode_data = self.raw_file.get_node(
            "/MovieGroup1/AcqTissue/RawData")
        self.node_bmode_attributes = self.raw_file.get_node(
            "/MovieGroup1/AcqTissue/AcquisitionTissueCF")
        self.node_viewer_attributes = self.raw_file.get_node(
            "/MovieGroup1/ViewerTissue")

        self.pd_scan_end_cfm = None
        self.pd_scan_end_raw = None
        self.pd_scan_end_volbox = None
        self.pd_scan_end_viewer = None

        self.pd_scan_start_cfm = None
        self.pd_scan_start_raw = None
        self.pd_scan_start_volbox = None
        self.pd_scan_start_viewer = None

        # cannot load data that isn't there!
        # first check if the Doppler data is there
        if self.raw_file.__contains__('/MovieGroup1/ViewerTissueCF'):
            self.node_pd_viewer_attributes = self.raw_file.get_node(
                "/MovieGroup1/ViewerTissueCF")

        if self.raw_file.__contains__("/MovieGroup1/AcqColorFlow/RawData"):
            self.has_doppler = True
            self.node_powerdoppler_data = self.raw_file.get_node(
                "/MovieGroup1/AcqColorFlow/RawData")

        if self.raw_file.__contains__('/MovieGroup1/GraphicViewerTissueCF'):
            self.node_graphicviewercf = self.raw_file.get_node(
                '/MovieGroup1/GraphicViewerTissueCF')

        if self.raw_file.__contains__(
                "/MovieGroup1/AcqColorFlow/AcquisitionTissueCF"):
            self.has_doppler = True
            self.node_powerdoppler_attributes = self.raw_file.get_node(
                "/MovieGroup1/AcqColorFlow/AcquisitionTissueCF")

        if self.raw_file.__contains__("/MovieGroup2/AcqPWCW/RawData/"):
            self.has_waveform = True
            self.node_waveform_data = self.raw_file.get_node(
                "/MovieGroup2/AcqPWCW/RawData/")

        if self.raw_file.__contains__("/MovieGroup2/AcqPWCW/RawData/"):
            self.has_waveform = True
            self.node_waveform_attributes = self.raw_file.get_node(
                "/MovieGroup2/AcqPWCW/AcquisitionPWCW/")

        if self.containsDoppler():
            pass
            #self.getBModeBounds()
            #self.getPowerDopplerBounds()
            #self.createBModeCartesian()
            # self.printBModeAttributes()
            # self.printViewerAttributes()
            # self.printPDAttributes()
            #self.createPDCartesian()

        if self.has_waveform:
            self.getWaveformBounds()

    def __del__(self):
        self.raw_file.close()

    def getBModeBounds(self):
        # returns the dimensions of a cartesian sweep based
        # on the polar data in the file

        # number of samples
        self.bmode_samples_per_line = self.node_bmode_attributes.nr_of_samples_per_line[
            0]
        # distance
        self.bmode_samples_perM = self.node_bmode_attributes.samples_per_mUS[0]
        self.bmode_number_of_bytes_per_line = self.node_bmode_attributes.nr_of_bytes_per_line[
            0]

        # offset
        self.bmode_offset_M = self.node_bmode_attributes.first_sample_depth_meter[
            0]

        # number of lines
        self.bmode_lines = self.node_bmode_attributes.nr_of_lines[0]

        # start sweep
        #self.bmode_scan_start = self.node_bmode_attributes.hrs_scan_start[0]
        #self.bmode_scan_end = self.node_bmode_attributes.hrs_scan_end[0]
        self.bmode_scan_start = self.node_viewer_attributes.hrs_scan_start[0]
        self.bmode_scan_end = self.node_viewer_attributes.hrs_scan_end[0]

        self.image_x = self.node_viewer_attributes.hrs_rect_right[
            0] - self.node_viewer_attributes.hrs_rect_left[0]
        self.image_y = self.node_viewer_attributes.hrs_rect_bottom[
            0] - self.node_viewer_attributes.hrs_rect_top[0]

        self.pixels_per_mUS = self.node_viewer_attributes.pixel_per_mUS[0]
        self.bmode_depth_start_m = self.node_viewer_attributes.start_depth[0]
        self.bmode_depth_end_m = self.node_viewer_attributes.end_depth[0]

        # distance in line depth in pixels
        self.distance_in_pixels = (
            self.bmode_depth_end_m - self.bmode_depth_start_m) * self.pixels_per_mUS
        self.bmode_offset_pixels = self.bmode_depth_start_m * \
            self.pixels_per_mUS
        # print self.dist_to_start_pixels

        # 3D array containg the image data.
        self.rawdata = self.node_bmode_data.RawDataUnit
        self.rawdata = np.array(self.rawdata)
        # print np.shape(self.rawdata)
        # convert to 2D - 1st ele contains 2D array

        self.bmode_lines *= np.min(np.shape(self.rawdata))

        # actually - this looks like the real data
        self.fulldata = np.zeros(
            [(self.bmode_lines), (self.bmode_samples_per_line)])
        # print (np.shape(self.fulldata))

        # TA Version matters for B-mode too!
        self.ta_version = self.node_viewer_attributes.TA_Version[0]

        # multi-line adaptation.
        for r in range(0, int(self.bmode_lines), np.min(
                np.shape(self.rawdata))):
            for i in range(np.min(np.shape(self.rawdata))):
                self.fulldata[
                    r +
                    i] = self.rawdata[i][
                    r /
                    np.min(
                        np.shape(
                            self.rawdata))]

    def getPowerDopplerBounds(self, angle=0):
        # gets the bounds of the power Doppler in the file
        self.pd_samples_per_line = self.node_powerdoppler_attributes.nr_of_samples_per_line[
            0]

        # distance
        self.pd_samples_perM = self.node_powerdoppler_attributes.samples_per_mUS[
            0]

        # offset
        self.pd_offset_M = self.node_powerdoppler_attributes.first_sample_depth_meter[
            0]

        # number of lines - may need to adjust this for the final data
        self.pd_lines = self.node_powerdoppler_attributes.nr_of_lines[0]

        # number of bytes per line
        self.pd_number_of_bytes_per_line = self.node_powerdoppler_attributes.nr_of_bytes_per_line[
            0]

        # start sweep, end sweep
        if self.node_pd_viewer_attributes.__contains__('hrs_scan_start'):
            self.pd_scan_start = self.node_pd_viewer_attributes.hrs_scan_start[
                0]
            self.pd_scan_end = self.node_pd_viewer_attributes.hrs_scan_end[0]
            self.pd_scan_start_raw = self.pd_scan_start
            self.pd_scan_end_raw = self.pd_scan_end
        elif self.node_pd_viewer_attributes.__contains__('hrsScan_Start'):
            self.pd_scan_start = self.node_pd_viewer_attributes.hrsScan_Start[
                0]
            self.pd_scan_end = self.node_pd_viewer_attributes.hrsScan_End[0]
            self.pd_scan_start_raw = self.pd_scan_start
            self.pd_scan_end_raw = self.pd_scan_end
        #start and depth in metres
        self.pd_depth_start_m = self.node_pd_viewer_attributes.hrsDepth_Start[
            0]
        self.pd_depth_end_m = self.node_pd_viewer_attributes.hrsDepth_End[0]

        # need to check if we're in a Volbox
        self.pd_volbox_active = 0
        if self.node_graphicviewercf.__contains__(
                'bVolumeBoxOn') and self.node_graphicviewercf.bVolumeBoxOn[0]:
            self.pd_volbox_active = 1
        if self.node_graphicviewercf.__contains__('dVolboxStartScan'):
            self.pd_scan_start_volbox = self.node_graphicviewercf.dVolboxStartScan[
                0]
            self.pd_scan_end_volbox = self.node_graphicviewercf.dVolboxEndScan[
                0]
        # other mystery boxes
        self.pd_imagebox_active = 0
        if self.node_graphicviewercf.__contains__(
                'bImageBoxOn') and self.node_graphicviewercf.bImageBoxOn[0]:
            self.pd_imagebox_active = 1

        if self.node_graphicviewercf.__contains__('dStartScan'):
            self.pd_scan_start_viewer = self.node_graphicviewercf.dStartScan[0]
            self.pd_scan_end_viewer = self.node_graphicviewercf.dEndScan[0]

        self.pd_cfmbox_active = 0
        if self.node_graphicviewercf.__contains__(
                'bCFMBoxOn') and self.node_graphicviewercf.bCFMBoxOn[0]:
            self.pd_cfmbox_active = 1
        if self.node_graphicviewercf.__contains__('dCFMStartScan'):
            self.pd_scan_start_cfm = self.node_graphicviewercf.dCFMStartScan[0]
        if self.node_graphicviewercf.__contains__('dCFMEndScan'):
            self.pd_scan_end_cfm = self.node_graphicviewercf.dCFMEndScan[
                0]  # np.max([self.pd_scan_end,])

        if angle == 1:
            self.pd_scan_start = self.pd_scan_start_raw
            self.pd_scan_end = self.pd_scan_end_raw
        if angle == 2:
            if (self.pd_scan_start_volbox is not None) or (
                    self.pd_scan_start_volbox is not None):
                self.pd_scan_start = self.pd_scan_start_volbox
                self.pd_scan_end = self.pd_scan_end_volbox
        if angle == 3:
            if (self.pd_scan_start_cfm is not None) or (
                    self.pd_scan_start_cfm is not None):
                self.pd_scan_start = self.pd_scan_start_cfm
                self.pd_scan_end = self.pd_scan_end_cfm
        if angle == 4:
            if(self.pd_scan_start_viewer is not None) or (self.pd_scan_end_viewer is not None):
                self.pd_scan_start = self.pd_scan_start_viewer
                self.pd_scan_end = self.pd_scan_end_viewer
                
        if self.pd_scan_start == None:
            self.pd_scan_start = self.pd_scan_start_raw
            self.pd_scan_end = self.pd_scan_end_raw

        # array containg the image data.
        self.pd_rawdata = self.node_powerdoppler_data.RawDataUnit

        # got the 2D array
        self.pd_rawdata = np.array(self.pd_rawdata)[0]

        # the number of bytes
        self.num_bytes = np.max(
            np.shape(
                self.pd_rawdata)) / self.pd_number_of_bytes_per_line

        # datatype for conversion
        self.datatype = {1: 'uint8',
                         2: 'uint16',
                         4: 'uint32'
                         }
        self.pd_ta_version = self.node_powerdoppler_attributes.TA_Version[0]

        # calculate the conversion based upon the dimensionality of the raw data
        # known that the y axis should be the number of elements?

        self.pd_multilines = np.max(np.shape(self.pd_rawdata)) / self.num_bytes
        
        # have length of line in pixels - need to recover this from
        # the depth in metres first and then from that
        # recover the distance in pixels.
        self.pd_sample_depths = np.linspace(
            self.pd_depth_start_m,
            self.pd_depth_end_m,
            self.pd_samples_per_line)

        # data held in the following - extract
        # new data - reshape to the size of the test data
        # each row contains n number of scans need to re-order

        if len(self.pd_rawdata.shape) == 2:
        
            if self.pd_ta_version == 3 or self.pd_ta_version == 2:
                self.pd_testdata = self.pd_rawdata[
                    :,
                    range(
                        0,
                        np.shape(
                            self.pd_rawdata)[1],
                        4)]
                self.pd_fulldata = self.pd_rawdata[
                    :,
                    range(
                        0,
                        np.shape(
                            self.pd_rawdata)[1],
                        4)]

            elif self.pd_ta_version == 1:
                self.pd_testdata = self.pd_rawdata[
                    :,
                    range(
                        0,
                        np.shape(
                            self.pd_rawdata)[1],
                        4)]
                self.pd_fulldata = np.zeros(
                    (self.pd_lines, self.pd_samples_per_line))

                ln = 0
                l = 0
                start_inds = range(
                    0,
                    np.shape(
                        self.pd_testdata)[1],
                    self.pd_samples_per_line)
                for r in range(self.pd_lines):
                     # 0 ---> 4
                    for s in range(4):
                        if ln >= self.pd_lines - 1:
                            break
                        self.pd_fulldata[
                            ln,
                            :] = self.pd_testdata[
                            l,
                            start_inds[s]:start_inds[s] +
                            self.pd_samples_per_line]
                        ln += 1
                        if s == 3:
                            l += 1
                            
        # assume the angles are evenly spaced?
        self.pd_angles = np.linspace(
            self.pd_scan_start,
            self.pd_scan_end,
            np.shape(
                self.pd_fulldata)[0])



    def getWaveformBounds(self):
        self.wf_frequency = self.node_waveform_attributes.dma_pw_line_frq[0]
        self.line_period_correction_factor = self.node_waveform_attributes.line_peroid_correction_fact[
            0]

        self.wf_rawdata = np.array(self.node_waveform_data.RawDataUnit)
        self.wf_rawdata = np.roll(
            self.wf_rawdata,
            np.min(
                np.size(
                    self.wf_rawdata)) /
            2,
            1)
        self.wf_rawdata_log = np.log(self.wf_rawdata)

    def printBModeAttributes(self):
        print ("######## B-Mode Attributes ########")
        print ("Samples per line = {}".format(self.bmode_samples_per_line))
        print ("Samples per metre = {}".format(self.bmode_samples_perM))

        print ("Depth - start (m) = {}".format(self.bmode_depth_start_m))
        print ("Depth - end (m) = {}".format(self.bmode_depth_end_m))

        print ("Offset = {}".format(self.bmode_offset_M))
        print ("Bytes Per Line = {}".format(self.bmode_number_of_bytes_per_line))
        print ("Number of Lines = {}".format(self.bmode_lines))
        print ("Samples Per Line = {}".format(self.bmode_samples_per_line))

        print ("Scan start theta (rad) = {}".format(self.bmode_scan_start))
        print ("Scan end theta (rad) = {}".format(self.bmode_scan_end))
        print ("###################################")

    def printPDAttributes(self):
        print ("######## PD Attributes ########")
        print ("Samples per line = {}".format(self.pd_samples_per_line))
        print ("Samples per metre = {}".format(self.pd_samples_perM))

        print ("Depth - start (m) = {}".format(self.pd_depth_start_m))
        print ("Depth - end (m) = {}".format(self.pd_depth_end_m))

        print ("Offset = {}".format(self.pd_offset_M))
        print ("Bytes Per Line = {}".format(self.pd_number_of_bytes_per_line))
        print ("Number of Lines = {}".format(self.pd_lines))

        print ("Scan start theta (rad) = {}".format(self.pd_scan_start))
        print ("Scan end theta (rad) = {}".format(self.pd_scan_end))

        print ("TA Version = {}".format(self.pd_ta_version))
        print ("RawData (lines, samples) = {},{}".format(np.shape(self.pd_rawdata)[0], np.shape(self.pd_rawdata)[1]))
        print ("###################################")

    def printViewerAttributes(self):
        print ("######## Viewer Attributes ########")
        print ("Pixels per Metre US = {}".format(self.pixels_per_mUS))
        print ("X Dimension = {}".format(self.image_x))
        print ("Y Dimension = {}".format(self.image_y))

        print ("Start Depth = {}".format(self.bmode_depth_start_m))
        print ("End depth = {}".format(self.bmode_depth_end_m))
        print ("###################################")

    def printWaveformAttributes(self):
        print ("######## Waveform Attributes ########")
        print ("Waveform Frequency (per line) = {}".format(self.wf_frequency))
        print ("Line period correction factor = {}".format(self.line_period_correction_factor))
        print ("###################################")

    def showPolarGreyScale(self):
        # show raw image from file - in weird format
        plt.imshow(self.fulldata, interpolation='bicubic', cmap=cm.gray)
        plt.ylabel("Angle ($^{o}$)")
        plt.xlabel("Depth (mm)")
        dmy, ylabels = plt.yticks()
        deg_lbls = np.arange(
            np.rad2deg(
                self.bmode_scan_start),
            np.rad2deg(
                self.bmode_scan_end),
            ((np.rad2deg(
                self.bmode_scan_end) -
                np.rad2deg(
                    self.bmode_scan_start)) /
             len(ylabels)))
        deg_lbls.tolist()
        deg_lbls = ["{:.0f}".format(f - 90.0) for f in deg_lbls]

        dmy2, xlabels = plt.xticks()
        depth_lbls = np.arange(
            self.bmode_depth_start_m,
            self.bmode_depth_end_m,
            (self.bmode_depth_end_m -
             self.bmode_depth_start_m) /
            len(xlabels))
        depth_lbls.tolist()
        depth_lbls = ["{:.1f}".format(f * 1000) for f in depth_lbls]

        ax = plt.gca()
        ax.set_yticklabels(deg_lbls)
        ax.set_xticklabels(depth_lbls)
        plt.show()

    def showPolarPowerDoppler(self, clrmap=cm.jet):
        # show raw image from file - in weird format
        plt.imshow(self.pd_fulldata, interpolation='nearest', cmap=clrmap)
        plt.ylabel("Angle ($^{o}$)")
        plt.xlabel("Depth mm")

        dmy, ylabels = plt.yticks()
        deg_lbls = np.arange(
            np.rad2deg(
                self.pd_scan_start),
            np.rad2deg(
                self.pd_scan_end),
            ((np.rad2deg(
                self.pd_scan_end) -
                np.rad2deg(
                    self.pd_scan_start)) /
             len(ylabels)))
        deg_lbls.tolist()
        deg_lbls = ["{:.0f}".format(f - 90.0) for f in deg_lbls]

        dmy2, xlabels = plt.xticks()
        depth_lbls = np.arange(
            self.pd_depth_start_m,
            self.pd_depth_end_m,
            (self.pd_depth_end_m -
             self.pd_depth_start_m) /
            len(xlabels))
        depth_lbls.tolist()
        depth_lbls = ["{:.1f}".format(f * 1000) for f in depth_lbls]

        ax = plt.gca()
        ax.set_yticklabels(deg_lbls)
        ax.set_xticklabels(depth_lbls)
        plt.show()

    def createBModeCartesian(self):
        # assume the angles are evenly spaced?
        self.bmode_angles = np.linspace(
            self.bmode_scan_start,
            self.bmode_scan_end,
            self.bmode_lines)

        # have length of line in pixels - need to recover this from
        # the depth in metres first and then from that
        # recover the distance in pixels.
        self.bmode_sample_depths = np.linspace(
            self.bmode_depth_start_m,
            self.bmode_depth_end_m,
            self.bmode_samples_per_line)

        # find bounds of data
        self.width_bound_1 = self.bmode_depth_end_m * \
            np.cos(self.bmode_angles[len(self.bmode_angles) - 1])
        self.width_bound_2 = self.bmode_depth_end_m * \
            np.cos(self.bmode_angles[0])

        # print self.width_bound_1, self.width_bound_2
        if self.containsDoppler():
            self.pd_width_bound_1 = self.pd_depth_end_m * \
                np.cos(self.pd_angles[len(self.pd_angles) - 1])
            self.pd_width_bound_2 = self.pd_depth_end_m * \
                np.cos(self.pd_angles[0])
            #self.width_bound_1 = np.max(self.width_bound_1, pd_bound1)
            #self.width_bound_2 = np.max(self.width_bound_2, pd_bound2)

        self.axial_bound_1 = self.bmode_sample_depths[
            0] * np.max(np.sin(self.bmode_angles[0]), np.sin(self.bmode_angles[len(self.bmode_angles) - 1]))
        self.axial_bound_2 = np.max(self.bmode_sample_depths)

        if self.containsDoppler():
            self.pd_axial_bound_1 = self.pd_sample_depths[
                0] * np.max(np.sin(self.pd_angles[0]), np.sin(self.pd_angles[len(self.pd_angles) - 1]))
            self.pd_axial_bound_2 = np.max(self.pd_sample_depths)
            #self.axial_bound_1 = np.max(self.axial_bound_1, pd_bound1)
            #self.axial_bound_2 = np.max(self.axial_bound_2, pd_bound2)

        # get widths at maximum of polar data
        self.width = self.width_bound_2 - self.width_bound_1
        self.width = np.ceil(self.width * self.pixels_per_mUS) + 1

        # get lowest pixel
        self.height = np.ceil(
            (self.axial_bound_2 -
             self.axial_bound_1) *
            self.pixels_per_mUS)
        #(np.max(self.bmode_sample_depths) *self.pixels_per_mUS) -
        self.height_ceil = (
            np.min(
                self.bmode_sample_depths) *
            np.sin(
                self.bmode_angles[0]) *
            self.pixels_per_mUS)
        # print width,height

        # nudge width/heights
        while (self.width % 4) != 0:
            self.width += 1
        self.height += np.ceil(self.bmode_offset_M * self.pixels_per_mUS) + 1
        while (self.height % 4) != 0:
            self.height += 1
        self.probe_origin = self.width / 2
        # create a b-mode cartesian volume.
        self.image = np.zeros([self.width, self.height])

        # for each line in the raw data.
        for i in range(len(self.bmode_angles)):
            # get the angle
            theta = self.bmode_angles[i]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # for each sample in the line of the raw data
            for j in range(self.bmode_samples_per_line):
                # get the ray distance in metres
                r = self.bmode_sample_depths[j]
                x = r * cos_t
                y = r * sin_t

                x = x * self.pixels_per_mUS
                y = y * self.pixels_per_mUS
                # put that element into the image
                self.image[x + self.probe_origin, y] = self.fulldata[i, j]

        # waveform_data -needs flipped.
        if(self.has_waveform):
            self.image = np.flipud(self.image)

        #self.image = np.roll(self.image,int(self.width/2),0)

        # get image mask by mathematical dilation
        mask = ndimage.binary_erosion(
            ndimage.binary_dilation(
                self.image,
                iterations=3),
            iterations=3)
        mask[np.where(mask > 0)] = 1

        # interpolate image
        points = np.transpose(np.nonzero(self.image))
        vals = self.image[points[:, 0], points[:, 1]]
        grid_x, grid_y = np.mgrid[0:self.width, 0:self.height]
        fab = interpolate.griddata(
            points,
            vals,
            (grid_x,
             grid_y),
            method='linear',
            fill_value=0.0)

        # make mask zero
        self.image = fab * mask

    def createPDCartesian(self):

        # create a PD cartesian image. - this is the same bounds for the PD
        self.pd_image = np.zeros([self.width, self.height])
        #nudge = np.floor((self.width - self.pd_width)/4)
        # for each line in the raw data.
        for i in range(len(self.pd_angles)):
            # get the angle
            theta = self.pd_angles[i]
            # for each sample in the line of the raw data
            for j in range(self.pd_samples_per_line):
                # get the ray distance in metres
                r = self.pd_sample_depths[j]
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                x = x * self.pixels_per_mUS
                y = y * self.pixels_per_mUS

                # put that element into the image
                self.pd_image[
                    x +
                    self.probe_origin,
                    y] = self.pd_fulldata[
                    i,
                    j]

        # get image mask by mathematical dilation
        mask = ndimage.grey_erosion(
            ndimage.grey_dilation(
                self.pd_image, size=(
                    50, 50)), size=(
                50, 50))
        mask[np.where(mask > 0)] = 1

        x = np.array([np.arange(np.shape(self.pd_image)[0])])
        y = np.array([np.arange(np.shape(self.pd_image)[1])])

        np.where(self.pd_image == 0)
        #test = self.pd_image > 0
        points = np.transpose(np.nonzero(self.pd_image))

        vals = self.pd_image[points[:, 0], points[:, 1]]

        grid_x, grid_y = np.mgrid[0:self.width, 0:self.height]

        fab = interpolate.griddata(
            points,
            vals,
            (grid_x,
             grid_y),
            method='linear',
            fill_value=0.0)

        self.pd_image = fab * mask
        # self.showPDImage()

    def showBModeImage(self):
        plt.imshow(self.image.T, cmap=cm.gray)
        plt.show()

    def showPDImage(self, clrmap=cm.jet):
        plt.imshow(self.pd_image.T, cmap=clrmap)
        plt.show()

    def saveBModeImage(self, path):
        sitk.WriteImage(sitk.GetImageFromArray(self.image.T), path)

    def savePDImage(self, path):
        sitk.WriteImage(sitk.GetImageFromArray(self.pd_image.T), path)

    def saveWVImage(self, path):
        sitk.WriteImage(sitk.GetImageFromArray(self.wf_rawdata), path)

    def showWFImage(self):
        plt.imshow(self.wf_rawdata_log.T, cmap=cm.gray)
        plt.show()

    def getBModeCartesian(self):
        return self.image

    def getPowerDopplerCartesian(self):
        return self.pd_image

    def getWaveformCartesian(self):
        return self.wf_rawdata

    def containsDoppler(self):
        return self.has_doppler

    def containsWaveform(self):
        return self.has_waveform

    def showImageAll(self):

        # overlay the doppler on top of the B-Mode
        plt.imshow(np.fliplr(self.image.T), cmap=cm.gray)
        plt.hold(True)
        plt.imshow(np.fliplr(self.pd_image.T), cmap=self.getPDCMap())
        plt.show()

    def getPDCMap(self):
        # cmap Power Doppler
        cdict_pd = {'red': (
            (0.0, 0.0, 0.0),
            (0.0666666666667, 0.0, 0.0),
            (0.133333333333, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.266666666667, 0.131218560723, 0.131218560723),
            (0.333333333333, 0.205564110461, 0.205564110461),
            (0.4, 0.279909660199, 0.279909660199),
            (0.466666666667, 0.354255209937, 0.354255209937),
            (0.533333333333, 0.428600759676, 0.428600759676),
            (0.6, 0.502946309414, 0.502946309414),
            (0.666666666667, 0.577291859152, 0.577291859152),
            (0.733333333333, 0.65163740889, 0.65163740889),
            (0.8, 0.725982958628, 0.725982958628),
            (0.866666666667, 0.800328508367, 0.800328508367),
            (0.933333333333, 0.874674058105, 0.874674058105),
            (1.0, 0.949019607843, 0.949019607843),
        ),
            'green': (
            (0.0, 0.0, 0.0),
            (0.0666666666667, 0.0, 0.0),
            (0.133333333333, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.266666666667, 0.0514582591069, 0.0514582591069),
            (0.333333333333, 0.0806133766513, 0.0806133766513),
            (0.4, 0.109768494196, 0.109768494196),
            (0.466666666667, 0.13892361174, 0.13892361174),
            (0.533333333333, 0.168078729285, 0.168078729285),
            (0.6, 0.200390439801, 0.200390439801),
            (0.666666666667, 0.266551223094, 0.266551223094),
            (0.733333333333, 0.342123247206, 0.342123247206),
            (0.8, 0.427106512137, 0.427106512137),
            (0.866666666667, 0.521501017888, 0.521501017888),
            (0.933333333333, 0.625306764457, 0.625306764457),
            (1.0, 0.738523751846, 0.738523751846),
        ),
            'blue': (
            (0.0, 0.0, 0.0),
            (0.0666666666667, 0.0, 0.0),
            (0.133333333333, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.266666666667, 0.0921613825904, 0.0921613825904),
            (0.333333333333, 0.131367038482, 0.131367038482),
            (0.4, 0.161161453554, 0.161161453554),
            (0.466666666667, 0.181544627808, 0.181544627808),
            (0.533333333333, 0.192516561242, 0.192516561242),
            (0.6, 0.197233846829, 0.197233846829),
            (0.666666666667, 0.226388964373, 0.226388964373),
            (0.733333333333, 0.255544081918, 0.255544081918),
            (0.8, 0.284699199462, 0.284699199462),
            (0.866666666667, 0.313854317007, 0.313854317007),
            (0.933333333333, 0.343009434551, 0.343009434551),
            (1.0, 0.372164552095, 0.372164552095),
        )}

        cdict_pd['alpha'] = ((0.0, 0.0, 0.0),
                             (0.247, 0.0, 1.0),
                             (1.0, 1.0, 1.0))
        pd_map = colors.LinearSegmentedColormap('PD_CMap', cdict_pd)
        return pd_map


#ge = GERawFileToCartesian("C:\\Users\\z3485348\\Dropbox\\Tim_testdata\\19.10.13_2_1_1.raw")
#ge = GERawFileToCartesian("C:\\Users\\z3485348\\Dropbox\\Tim_testdata\\28.3.12_1_1.raw")
#ge.showBModeImage()
#ge.showPDImage()
#ge.showPolarGreyScale()
#plt.imshow(ge.rawdata)
