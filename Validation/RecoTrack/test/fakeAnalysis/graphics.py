#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import ROOT
from array import array
from copy import copy
from collections import OrderedDict

from Validation.RecoTrack.plotting.ntuple import *
import analysis

from math import sqrt, copysign, sin, cos, pi
import six

class EventPlotter(object):
    '''
    This class plots histograms and graphical objects with ROOT.
    The 3D and 2D graphics objects (except histograms) are stored in member lists plots_2D and plots_3D
    and all of them can be drawn with Draw method.
    '''
    def __init__(self):
        self.plots_2D = []
	self.plots_3D = []
	c = ROOT.TColor()
        self.colors_G = [c.GetColor(0,255,0), c.GetColor(0,185,0), c.GetColor(50,255,50), \
	               c.GetColor(0,100,0), c.GetColor(50,155,0), c.GetColor(0,70,155), \
		       c.GetColor(0,255,0), c.GetColor(0,255,0), c.GetColor(0,255,0)]
	self.colors_B = [c.GetColor(0,0,255), c.GetColor(0,0,155), c.GetColor(50,50,255), \
	               c.GetColor(0,0,80), c.GetColor(50,0,155), c.GetColor(0,70,155), \
		       c.GetColor(0,0,255), c.GetColor(0,0,255), c.GetColor(0,0,255)]
        
    def Reset(self):
        self.plots_2D = []
	self.plots_3D = []
 
    ###### OLD UNNECESSARY FUNCTIONS ######
    
    def PlotEvent3DHits(self, event, flag="PSG"):
	'''
	Plots the 3D hits of an event.
	flag is an string which defines which hit types to plot
	(p for pixel, s for strip, g for glued)
	'''	
	if('p' in flag or 'P' in flag):
	    pixel_hits = event.pixelHits()
	    pix_coords = []
	    for hit in pixel_hits:
		pix_coords.append(hit.z())
		pix_coords.append(hit.x())
		pix_coords.append(hit.y())
	    pix_plot = ROOT.TPolyMarker3D(len(pix_coords)/3, array('f', pix_coords), 1)
	    pix_plot.SetMarkerColor(4)

	if('s' in flag or 'S' in flag):
	    strip_hits = event.stripHits()
	    str_coords = []
	    for hit in strip_hits:
		str_coords.append(hit.z())
		str_coords.append(hit.x())
		str_coords.append(hit.y())
	    str_plot = ROOT.TPolyMarker3D(len(str_coords)/3, array('f', str_coords), 1)
	    str_plot.SetMarkerColor(2)

	if('g' in flag or 'G' in flag):
	    glued_hits = event.gluedHits()
	    glu_coords = []
	    for hit in glued_hits:
		glu_coords.append(hit.z())
		glu_coords.append(hit.x())
		glu_coords.append(hit.y())
	    glu_plot = ROOT.TPolyMarker3D(len(glu_coords)/3, array('f', glu_coords), 1)
	    glu_plot.SetMarkerColor(3)        
	    
	if('p' in flag or 'P' in flag): self.plots_3D.append(pix_plot)
	if('s' in flag or 'S' in flag): self.plots_3D.append(str_plot)
	if('g' in flag or 'G' in flag): self.plots_3D.append(glu_plot)	

    def PlotXY(self, event, limits=[-1000,1000], flag="PSG"):
	'''
	Plots the hits of an event in an XY plane.
	flag is an string which defines which hit types to plot
	(p for pixel, s for strip, g for glued)
	'''
	
	if('p' in flag or 'P' in flag):
	    pixel_hits = event.pixelHits()
	    pix_x = []
	    pix_y = []
	    for hit in pixel_hits:
		if(limits[0] < hit.z() < limits[1]):
		    pix_x.append(hit.x())
		    pix_y.append(hit.y())
	    pix_plot = ROOT.TGraph(len(pix_x), array('f', pix_x), array('f', pix_y))
	    pix_plot.SetMarkerColor(4)

	if('s' in flag or 'S' in flag):
	    strip_hits = event.stripHits()
	    str_x = []
	    str_y = []
	    for hit in strip_hits:
		if(limits[0] < hit.z() < limits[1]):
		    str_x.append(hit.x())
		    str_y.append(hit.y())
	    str_plot = ROOT.TGraph(len(str_x), array('f', str_x), array('f', str_y))
	    str_plot.SetMarkerColor(2)

	if('g' in flag or 'G' in flag):
	    glued_hits = event.gluedHits()
	    glu_x = []
	    glu_y = []
	    for hit in glued_hits:
		if(limits[0] < hit.z() < limits[1]):
		    glu_x.append(hit.x())
		    glu_y.append(hit.y())
	    glu_plot = ROOT.TGraph(len(glu_x), array('f', glu_x), array('f', glu_y))
	    glu_plot.SetMarkerColor(3)        

	plot = ROOT.TMultiGraph()

	if('p' in flag or 'P' in flag): plot.Add(pix_plot,"P")
	if('s' in flag or 'S' in flag): plot.Add(str_plot,"P")
	if('g' in flag or 'G' in flag): plot.Add(glu_plot,"P")
	self.plots_2D.append(plot)

    def PlotZY(self, event, limits=[-1000,1000], flag="PSG"):
	'''
	Plots the hits of an event in an ZR plane.
	flag is an string which defines which hit types to plot
	(p for pixel, s for strip, g for glued)
	'''
	
	if('p' in flag or 'P' in flag):
	    pixel_hits = event.pixelHits()
	    pix_z = []
	    pix_y = []
	    for hit in pixel_hits:
		if(limits[0] < hit.z() < limits[1]):
		    pix_z.append(hit.z())
		    pix_y.append(hit.y())
	    pix_plot = ROOT.TGraph(len(pix_z), array('f', pix_z), array('f', pix_y))
	    pix_plot.SetMarkerColor(4)

	if('s' in flag or 'S' in flag):
	    strip_hits = event.stripHits()
	    str_z = []
	    str_y = []
	    for hit in strip_hits:
		if(limits[0] < hit.z() < limits[1]):
		    str_z.append(hit.z())
		    str_y.append(hit.y())
	    str_plot = ROOT.TGraph(len(str_z), array('f', str_z), array('f', str_y))
	    str_plot.SetMarkerColor(2)

	if('g' in flag or 'G' in flag):
	    glued_hits = event.gluedHits()
	    glu_z = []
	    glu_y = []
	    for hit in glued_hits:
		if(limits[0] < hit.z() < limits[1]):
		    glu_z.append(hit.z())
		    glu_y.append(hit.y())
	    glu_plot = ROOT.TGraph(len(glu_z), array('f', glu_z), array('f', glu_y))
	    glu_plot.SetMarkerColor(3)        

	plot = ROOT.TMultiGraph()

	if('p' in flag or 'P' in flag): plot.Add(pix_plot,"P")
	if('s' in flag or 'S' in flag): plot.Add(str_plot,"P")
	if('g' in flag or 'G' in flag): plot.Add(glu_plot,"P")
	self.plots_2D.append(plot)

    def PlotTracksXY(self, tracks):
	''' Plots tracks like polyline graphs in the XY plane. tracks is an iterable for tracks. '''
	plot = ROOT.TMultiGraph()

	for track in tracks:
	    X = []; Y = [];
	    for hit in track.hits():
		if(hit.isValidHit()):
		    X.append(hit.x())
		    Y.append(hit.y())
	    plot.Add(ROOT.TGraph(len(X),array("f",X),array("f",Y)),"L")

	self.plots_2D.append(plot)

    def PlotTracksZY(self, tracks):
	''' Plots tracks like polyline graphs in the ZY plane. tracks is an iterable for tracks. '''
	plot = ROOT.TMultiGraph()

	for track in tracks:
	    Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValidHit()):
		    Y.append(hit.y())
		    Z.append(hit.z())
	    plot.Add(ROOT.TGraph(len(Z),array("f",Z),array("f",Y)),"L")

	self.plots_2D.append(plot)    
   
    def PlotTracks3D(self, tracks): 
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValidHit()):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())	
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y)))
	
    def PlotPixelTracks3D(self, tracks): 
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.pixelHits():
		if(hit.isValidHit()):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y)))

    def PlotPixelGluedTracks3D(self, tracks):
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValidHit() and hit.hitType != 1):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y)))

    def PlotTrack3D(self, track, color = 1): 
	'''Plots a single track as a polyline and prints track info'''
	# Not so hasardous experimental edit:
	#hits = sorted([hit for hit in track.hits()], key = lambda hit: hit.index())
	#print [hit.index() for hit in hits]
	X = []; Y = []; Z = [];
	for hit in track.hits(): #hits: #track.hits():
	    if(hit.isValidHit()):
		X.append(hit.x())
		Y.append(hit.y())
		Z.append(hit.z())	
	if(not X):
	    print("Track has no valid points")
	    return
	plot = ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y))
	plot.SetLineColor(color)
	self.plots_3D.append(plot)

	''' Uncomment to print track info
	print "Track parameters:"
	print "px  : " + str(track.px())
	print "py  : " + str(track.py())
	print "pz  : " + str(track.pz())
	print "pt  : " + str(track.pt())
	print "eta : " + str(track.eta())
	print "phi : " + str(track.phi())
	print "dxy : " + str(track.dxy())
	print "dz  : " + str(track.dz())
	print "q   : " + str(track.q())
	'''
    
    ###### METHODS USED BY PLOTFAKES ######

    def TrackHelix(self, track, color = 1, style = 0):
	'''
	Creates a THelix object which can be plotted with Draw() method.

	NOTE: The helixes are better drawn with the vertical z-axis.
	Even then the helixes are not drawn exactly correct.
	'''
	if isinstance(track, TrackingParticle):
	    phi = track.pca_phi()
	    dxy = track.pca_dxy()
	    dz = track.pca_dz()
	else:
	    phi = track.phi()
	    dxy = track.dxy()
	    dz = track.dz()
	xyz = array("d", [-dxy*ROOT.TMath.Sin(phi), dxy*ROOT.TMath.Cos(phi), dz])# [dz, -dxy*ROOT.TMath.Sin(phi), dxy*ROOT.TMath.Cos(phi)])
	v = array("d", [track.py(), track.px(), track.pz()]) #[track.px(), track.py(), track.pz()]) #[track.pz(), track.px(), track.py()])
	w = 0.3*3.8*track.q()*0.01 #Angular frequency = 0.3*B*q*hattuvakio, close enough
	z_last = dz
	for hit in track.hits(): 
	    if(hit.isValidHit()): z_last = hit.z()
	
	helix = ROOT.THelix(xyz, v, w)#, array("d", [dz, z_last]))#, ROOT.kHelixX, array("d", [1,0,0]))
	helix.SetAxis(array("d", [-1,0,0]))
	helix.SetRange(z_last, dz, ROOT.kHelixX)
	helix.SetLineColor(color)
	if style == 1: helix.SetLineStyle(9)
	if style == 2: helix.SetLineStyle(7)
	return helix

    def Plot3DHelixes(self, tracks, color = 1, style = 0):	
	''' Plots 3D helixes from a track iterable '''
	for track in tracks:
	    if(track.hits()):
		self.plots_3D.append(self.TrackHelix(track, color, style))	

    def Plot3DHelix(self, track, color = 1, style = 0):	
	''' Plots a single track helix '''
	if(track.hits()):
	    self.plots_3D.append(self.TrackHelix(track, color, style))

    def Plot3DHits(self, track, color = 1, style = 0):
	'''
	Plots the 3D hits from a track.
	'''	
	pix_coords = []
	for hit in track.pixelHits():    
	    if hit.isValidHit():
		pix_coords.append(hit.z())
		pix_coords.append(hit.x())
		pix_coords.append(hit.y())
	    if pix_coords:
		pix_plot = ROOT.TPolyMarker3D(len(pix_coords)/3, array('f', pix_coords), 2)
		pix_plot.SetMarkerColor(color)
		if style == 1: pix_plot.SetMarkerStyle(5)
		if style == 2: pix_plot.SetMarkerStyle(4)
		self.plots_3D.append(pix_plot)
	
	for hit in track.gluedHits():
	    if hit.isValidHit():
		x = hit.x(); y = hit.y(); z = hit.z()
		if hit.isBarrel():
		    X = [x, x]
		    Y = [y, y]
		    Z = [z - sqrt(hit.zz()), z + sqrt(hit.zz())]
		else:
		    X = [x - copysign(sqrt(hit.xx()),x), x + copysign(sqrt(hit.xx()),x)]
		    Y = [y - copysign(sqrt(hit.yy()),y), y + copysign(sqrt(hit.yy()),y)]
		    Z = [hit.z(), hit.z()]
		glu_plot = ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y))
		#glu_plot.SetLineStyle(2)    
		if style == 1: glu_plot.SetLineStyle(2)
		if style == 2: glu_plot.SetLineStyle(3)
		glu_plot.SetLineColor(color) 
		self.plots_3D.append(glu_plot)

	for hit in track.stripHits():
	    if hit.isValidHit():
		x = hit.x(); y = hit.y(); z = hit.z()
		if hit.isBarrel():
		    X = [x, x]
		    Y = [y, y]
		    Z = [z - 1.5*sqrt(hit.zz()), z + 1.5*sqrt(hit.zz())]
		else:
		    X = [x - 1.5*copysign(sqrt(hit.xx()),x), x + 1.5*copysign(sqrt(hit.xx()),x)]
		    Y = [y - 1.5*copysign(sqrt(hit.yy()),y), y + 1.5*copysign(sqrt(hit.yy()),y)]
		    Z = [hit.z(), hit.z()]
		str_plot = ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y))
		if style == 1: str_plot.SetLineStyle(2)
		if style == 2: str_plot.SetLineStyle(3)
		str_plot.SetLineColor(color) 
		self.plots_3D.append(str_plot)

    def PlotVertex3D(self, vertex, color=1):
	''' Plots a single vertex object '''
	plot = ROOT.TPolyMarker3D(1, array('f', [vertex.z(), vertex.x(), vertex.y()]),3)
	plot.SetMarkerColor(color)
        self.plots_3D.append(plot)

    def PlotPoint3D(self, point, color=1):
        ''' Plots a single 3D point from a 3-element list '''
	plot = ROOT.TPolyMarker3D(1, array('f', [point[2], point[0], point[1]]),3)
	plot.SetMarkerColor(color)
        self.plots_3D.append(plot)

    def PlotTrackingFail(self, match):
        '''
	Plots the end of tracking hits for a fake.
	This is drawn with thick red lines from last shared hit to following fake and particle hits,
	and the particle hit is indicated with a purple star marker.
	'''
	X = array('f', [match.fake_end_loc[0], match.last_loc[0], match.particle_end_loc[0]])
	Y = array('f', [match.fake_end_loc[1], match.last_loc[1], match.particle_end_loc[1]])
	Z = array('f', [match.fake_end_loc[2], match.last_loc[2], match.particle_end_loc[2]])
        plot = ROOT.TPolyLine3D(3, Z, X, Y)
	plot.SetLineWidth(3)
	plot.SetLineColor(2) 
        self.plots_3D.append(plot)
	self.PlotPoint3D(match.last_loc, 2)
	self.PlotPoint3D(match.particle_end_loc, 6)

    def PlotFakes(self, event, reconstructed = 1, fake_filter = None, end_filter = None, last_filter = None, particle_end_filter = None):
	'''
        This is the main function to plot fakes in 3D with related TrackingParticles and tracks.
	Fake track is drawn as red, TrackingParticles as green and reconstructed tracks as blue.
	The detector scheme is also drawn with black circles.
	The decay vertices (yellow) and reconstructed collision vertices (colour of the track) are also drawn.
	Alongside with the fake plotting, the track and particle informations are printed.

	Set reconstructed to false for not drawing other reconstructed tracks related to a fake,
	use filters (lists of class indexes or detector layer strings) to filter out everything else from plotting.
	'''
	iterative = 1 # simple edits for not needing to refactorise code	

        fakes = analysis.FindFakes(event)
	if iterative:
            # Plot fakes one by one
	    for fake in fakes:
		fake_info = analysis.FakeInfo(fake)
                # Check for filter
		if fake_filter and fake_info.fake_class not in fake_filter:
			continue
	        if last_filter or particle_end_filter:
		    found_flag = False
		    for match in fake_info.matches:
			if (not end_filter or match.end_class in end_filter) and (not last_filter or last_filter in match.last_str) and (not particle_end_filter or particle_end_filter in match.particle_end_str):
			    found_flag = True
	            if not found_flag:
			continue 

		self.Reset()
		self.Plot3DHelixes([fake],2)
		self.Plot3DHits(fake, 2)
		#self.PlotVertex3D(vertex,2)
		analysis.PrintTrackInfo(fake, None, 0, fake_info)
                if fake_info.matches:
		    for match in fake_info.matches:
			self.PlotTrackingFail(match)
			#self.PlotPoint3D(fake_info.end_loc, 2)

                # Find real particle tracks which include fake tracks hits
		icol = 0
		particle_inds = []
		particles = []
		reco_inds = []
		recos = []
		for hit in fake.hits():
		    if hit.isValidHit() and hit.nSimHits() >= 0:
			for simHit in hit.simHits():
			    particle = simHit.trackingParticle()
			    if particle.index() not in particle_inds:
				particle_inds.append(particle.index())
				particles.append(particle)
			    if reconstructed and particle.nMatchedTracks() > 0:
				for info in particle.matchedTrackInfos():
				    track = info.track()
				    if track.index() not in reco_inds:
					reco_inds.append(track.index())
					recos.append(track)


		    # Find reconstructed tracks included in fakes hits 
		    if hit.isValidHit() and reconstructed and hit.ntracks() > 0:
			for track in hit.tracks():
			    #track = info.track()
			    if (track.index() not in reco_inds) and track.index() != fake.index():
				reco_inds.append(track.index())
				recos.append(track)
 
                # Plot the particles and reconstructed tracks
		icol =  0
		self.ParticleTest(particles, draw=False)
                for particle in particles:
		    self.Plot3DHelix(particle, self.colors_G[icol], 1)
		    self.plots_3D[-1].SetLineStyle(5)
		    self.Plot3DHits(particle, self.colors_G[icol], 1) 
		    self.PlotVertex3D(particle.parentVertex(),self.colors_G[icol])
                    # EXPERIMENTAL LINE:
                    #self.PlotTrack3D(particle, self.colors_G[icol])

		    for decay in particle.decayVertices():
			self.PlotVertex3D(decay, 5)
		    analysis.PrintTrackInfo(particle, fake)
		    icol += 1
		icol = 0
		for track in recos:
		    self.Plot3DHelix(track,self.colors_B[icol],2)
		    self.Plot3DHits(track,self.colors_B[icol],2)
		    #self.PlotVertex3D(vertex,self.colors_B[icol])
		    analysis.PrintTrackInfo(track, fake)
		    icol += 1
		    
		if hit.isValidHit() and hit.z() >= 0: self.PlotDetectorRange("p",4)
		elif hit.isValidHit() and hit.z() < 0: self.PlotDetectorRange("n",4)
		else: self.PlotDetectorRange("b",4)

                print("****************************\n")
	        self.Draw()
            return

    def DrawTrackTest(self, track): 
	self.PlotDetectorRange("b", 4)	
	self.PlotTrack3D(track)	

    def ParticleTest(self, particles, draw=False):
	for particle in particles:
	    print("\nPARTICLE " + str(particle.index()))
	    for hit in particle.hits():
		tof = -1
		for simHit in hit.simHits():
		    if simHit.trackingParticle().index() == particle.index():
			#if len(info.tof()): 
			tof = simHit.tof()
		print("Index: " + str(hit.index()) + ", Layer: " + str(hit.layerStr()) + ", TOF: " + str(tof) +\
	        "     XY distance: " + str(sqrt(hit.x()**2 + hit.y()**2)) + ", Z: " + str(hit.z()))
	    self.DrawTrackTest(particle)
	    if draw:
		self.Draw()
		self.Reset()
 
    def PlotEndOfTrackingPoints3D(self, last, fail, fake_classes = [], color = ROOT.kBlue):
	alpha = 0.01
        for i in range(len(last)):
            if fail and last[i] != fail[i]:
		X = array('f', [last[i][0], fail[i][0]])
		Y = array('f', [last[i][1], fail[i][1]])
		Z = array('f', [last[i][2], fail[i][2]])
		line = ROOT.TPolyLine3D(2, Z, X, Y)
		line.SetLineWidth(1)
		line.SetLineColorAlpha(color, alpha)
		self.plots_3D.append(line)
	    else:
		point = ROOT.TPolyMarker3D(1, array('f', [last[i][2], last[i][0], last[i][1]]))
		point.SetMarkerColorAlpha(color, alpha)
		self.plots_3D.append(point)

    def PlotEndOfTracking3D(self, last, fake_ends, particle_ends, end_classes = [], fake_classes = [], end_mask = [], fake_mask = []):
	'''
	Plots the end of tracking hits in 3D
	'''
	alpha = 0.01
        for i in range(len(last)):
	    if (not end_mask or end_classes[i] in end_mask) and (not fake_mask or fake_classes[i] in fake_mask):
		point = ROOT.TPolyMarker3D(1, array('f', [last[i][2], last[i][0], last[i][1]]))
		point.SetMarkerColorAlpha(4, alpha)
		self.plots_3D.append(point)

		point = ROOT.TPolyMarker3D(1, array('f', [fake_ends[i][2], fake_ends[i][0], fake_ends[i][1]]))
		point.SetMarkerColorAlpha(2, alpha)
		self.plots_3D.append(point)

		point = ROOT.TPolyMarker3D(1, array('f', [particle_ends[i][2], particle_ends[i][0], particle_ends[i][1]]))
		point.SetMarkerColorAlpha(3, alpha)
		self.plots_3D.append(point)

    def PlotEndOfTrackingPoints2D(self, last, fail, classes = [], color = ROOT.kBlue):
	alpha = 0.5
	plot = ROOT.TMultiGraph()

        for i in range(len(last)):
            if fail and last[i] != fail[i]:
		#c = ROOT.TCanvas("s","S",1000,1000)
		R = array('f', [sqrt(last[i][0]**2 + last[i][1]**2), sqrt(fail[i][0]**2 + fail[i][1]**2)])
		Z = array('f', [last[i][2], fail[i][2]])
		#line = ROOT.TPolyLine(2, Z, R, "L")	
		line = ROOT.TGraph(2, Z, R)
		line.SetLineWidth(1)
		line.SetLineColorAlpha(color, alpha)
		plot.Add(line, "L")
		#self.plots_2D.append(line)
	    else:
		#point = ROOT.TMarker(array("f", [last[i][2]]), array("f", [sqrt(last[i][0]**2 + last[i][1]**2)]), 1)
		#point = ROOT.TMarker(last[i][2], sqrt(last[i][0]**2 + last[i][1]**2), 1)
		point = ROOT.TGraph(1, array("f", [last[i][2]]), array("f", [sqrt(last[i][0]**2 + last[i][1]**2)]))
		#point.SetDrawOption("*")
		point.SetMarkerColorAlpha(color, alpha)
		plot.Add(point, "P")
		#self.plots_2D.append(point)
	self.plots_2D.append(plot)
	

    def PlotDetectorRange(self, area="b", plates = 6):
	'''
	Plots the detector schematic layout.
	Area  means which part to plot (p = positive side, n = negative side, b = both)
	Plates means how many detector circle schemes to plot per side.
	'''
        r = [17, 17, 57, 57, 112, 112]
	z = [30, 70, 70, 120, 120, 280]
	for i in range(plates):
	    X = []; Y = []; Z = []; ZN = []
	    for t in range(0,101):
		X.append(r[i]*cos(2*pi*t/100))
		Y.append(r[i]*sin(2*pi*t/100))
		Z.append(z[i]*1.0)
		ZN.append(-1.0*z[i])
	    plot = ROOT.TPolyLine3D(len(X),array("f",Z),array("f",X),array("f",Y),"S")
	    nplot = ROOT.TPolyLine3D(len(X),array("f",ZN),array("f",X),array("f",Y),"S")
	    plot.SetLineStyle(3)
	    nplot.SetLineStyle(3)
	    if area == "p": self.plots_3D.append(plot)
	    if area == "n": self.plots_3D.append(nplot)
	    if area == "b":
		self.plots_3D.append(plot)
		self.plots_3D.append(nplot)

    ###### HISTOGRAM PLOTTING METHODS ######

    def ClassHistogram(self, data, name = "Histogram", labels = False, ready = False, logy = False, xmin = None, xmax = None, nbin = None, small = False):
        '''
        A method for drawing histograms with several types of data.
	Data can be in two different forms:
	    A dictionary (used for labeled data)
	    Raw numerical data
        
	In the case of the dictionary, labels (ordered dictionary) should contain class indexes with class names.
	Also if ready = True, the observations for classes are already calculated.	
	'''
	if ready:
	    if labels:
		keys = [key for key in data]
		hist_labels = [labels[key] for key in keys]
		hist_data = [data[key] for key in keys]
	    else:
		hist_labels = [key for key in data]
		hist_data = [data[key] for key in hist_labels]
	
	if small: c1 = ROOT.TCanvas(name, name, 500, 500)
	else: c1 = ROOT.TCanvas(name, name, 700, 500)
	c1.SetGrid()
	#c1.SetTopMargin(0.15)
	if ready: hist = ROOT.TH1F(name, name, 30,0,30)#, 3,0,3)
	else:
	    if xmin != None and xmax != None and nbin != None:
		hist = ROOT.TH1F(name, name, nbin, xmin, xmax)
	    elif "fraction" in name or isinstance(data[0], float):
		hist = ROOT.TH1F(name, name, 41,0,1.025)#100,0,5)#max(data)+0.0001)#11,0,1.1)#21,0,1.05)#41*(max(data)+1)/max(data))#, max(data)+1,-0.5,max(data)+0.5) #### EXPERIMENT!!!111	
	    elif isinstance(data[0], int): hist = ROOT.TH1F(name, name, max(data)-min(data)+1, min(data)-0.5,max(data)+0.5)
	    else: hist = ROOT.TH1F(name, name, 3,0,3)
	    #c1.SetLogy()
	    #hist.SetBinsLength(1)
	if logy: c1.SetLogy()
	
        hist.SetStats(0);
        hist.SetFillColor(38);
	hist.SetCanExtend(ROOT.TH1.kAllAxes);
	if ready: # cumulative results already calculated in data
	    for i in range(len(hist_labels)):
		#hist.Fill(hist_labels[i],hist_data[i])
		[hist.Fill(hist_labels[i],1) for j in range(hist_data[i])]
	elif not ready and labels:
	    for entry in data: hist.Fill(labels[entry],1)
	    hist.LabelsOption(">")
	else: # data is in samples, not in cumulative results
	    for entry in data: hist.Fill(entry,1)
	    hist.LabelsOption("a")
	hist.LabelsDeflate()
	if name == "Classification of all fakes": pass
	#hist.SetLabelSize(0.05)
	#hist.GetXaxis().SetRange(0,10)
	hist.Draw()

	input("Press any key to continue")

    def EndOfTrackingHistogram(self, end_list, hist_type="end_class", end_mask = [], fake_mask = [], normalised = False, BPix3mask = False):
	''' Plots several types of histograms related to end of tracking analysis '''
	data = []
        if hist_type == "end_class": # Classification of end of trackings
	    for end in end_list:
		if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
		    data.append(end.end_class)
	    self.ClassHistogram(data, "End of tracking classes", analysis.end_class_names, False)
	elif hist_type == "particle_pdgId":
	    ids = OrderedDict()
	    for end in end_list:
		if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
		    data.append(end.particle_pdgId)
		    if str(end.particle_pdgId) not in ids:
			ids[end.particle_pdgId] = str(end.particle_pdgId)
	    self.ClassHistogram(data, "Particle pdgId", ids, False)
	elif "invalid" not in hist_type:
	    for end in end_list:
		if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
		    if hist_type == "last_layer": data.append(end.last_str.split()[0]) # split()[0] for taking only the name of the layer, skipping reason for invalid hit
	            elif hist_type == "fake_end_layer": data.append(end.fake_end_str.split()[0])
		    elif hist_type == "particle_end_layer": data.append(end.particle_end_str.split()[0])

            data_dict = copy(analysis.layer_data_tmp)
	    for entry in data: 
		#print entry
		data_dict[analysis.layer_names_rev[entry]] += 1

	    if normalised:
		norm_cff = analysis.Get_Normalisation_Coefficients()
		for i, v in six.iteritems(data_dict):
		    data_dict[i] = int(round(v*norm_cff[i]))
            
	    name = ""
	    if hist_type == "last_layer": name = "End of tracking layers"
	    elif hist_type == "fake_end_layer": name = "Fake end of tracking layers"
	    elif hist_type == "particle_end_layer": name = "Particle end of tracking layers"
	    if normalised: name += " normalised"

	    self.ClassHistogram(data_dict, name, analysis.layer_names , True)
            
	else:
	    if hist_type == "invalid_reason_fake":
		data_dict = copy(analysis.invalid_types_tmp)
		for end in end_list:
		    if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
                        if "missing" in end.fake_end_str: data_dict[1] += 1; #print end.fake_end_str
			elif "inactive" in end.fake_end_str: data_dict[2] += 1; #print end.fake_end_str
			elif "bad" in end.fake_end_str: data_dict[3] += 1; #print end.fake_end_str
			elif "missing_inner" in end.fake_end_str: data_dict[4] += 1; #print end.fake_end_str
			elif "missing_outer" in end.fake_end_str: data_dict[5] += 1; #print end.fake_end_str
			else: data_dict[0] += 1 # valid hit

		self.ClassHistogram(data_dict, "Invalid hit reasons in fake EOT", analysis.invalid_types, True)
	    elif hist_type == "invalid_reason_last":
		data_dict = copy(analysis.invalid_types_tmp)
		for end in end_list:
		    if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
                        if "missing" in end.last_str: data_dict[1] += 1
			elif "inactive" in end.last_str: data_dict[2] += 1
			elif "bad" in end.last_str: data_dict[3] += 1
			elif "missing_inner" in end.last_str: data_dict[4] += 1
			elif "missing_outer" in end.last_str: data_dict[5] += 1
			else: data_dict[0] += 1 # valid hit

		self.ClassHistogram(data_dict, "Invalid hit reasons in EOT last shared hit", analysis.invalid_types, True)
	    elif hist_type == "invalid_reason_particle":
		data_dict = copy(analysis.invalid_types_tmp)
		for end in end_list:
		    if (not end_mask or end.end_class in end_mask) and (not fake_mask or end.fake_class in fake_mask) and (not BPix3mask or "BPix3" in end.last_str):
                        if "missing" in end.particle_end_str: data_dict[1] += 1
			elif "inactive" in end.particle_end_str: data_dict[2] += 1
			elif "bad" in end.particle_end_str: data_dict[3] += 1
			elif "missing_inner" in end.particle_end_str: data_dict[4] += 1
			elif "missing_outer" in end.particle_end_str: data_dict[5] += 1
			else: data_dict[0] += 1 # valid hit

		self.ClassHistogram(data_dict, "Invalid hit reasons in particle EOT", analysis.invalid_types, True)

    def ResolutionHistograms(self, res_data, tracks = "fake", fake_mask = [], draw = True, skip = True, normalised = False):
	'''
	Returns (and possibly draws) resolution histograms for the ResolutionData object.
        '''
	if tracks == "fake":
	    if fake_mask:
		res_tmp = analysis.ResolutionData()
		for i in range(len(res_data)):
		    if res_data.fake_class in fake_mask:
			res_tmp.pt.append(res_data.pt[i])
			res_tmp.eta.append(res_data.eta[i])
			res_tmp.Lambda.append(res_data.Lambda[i])
			res_tmp.cotTheta.append(res_data.cotTheta[i])
			res_tmp.phi.append(res_data.phi[i])
			res_tmp.dxy.append(res_data.dxy[i])
			res_tmp.dz.append(res_data.dz[i])
			res_tmp.fake_class.append(res_data.fake_class[i])

		res_data = res_tmp 
	    color = ROOT.kRed
	elif tracks == "true":
	    if normalised: color = ROOT.kBlue
	    else: color = 38	
	else:
	    print("Unspecified data type")
	    return
	
	c1 = ROOT.TCanvas("Resolution_histograms","Resolution histograms", 1000, 900)
	c1.Divide(3,3)
	c1.SetGrid()
	#c1.SetLogy()
        histograms = []

	c1.cd(1)
	histograms.append( self.Histogram1D([res.pt_delta/res.pt_rec for res in res_data], "Resolution of #Delta p_{t}/p_{t,rec} in " + tracks + " tracks", 100, -0.15, 0.15, "p_{t}/p_{t,rec}", color, draw, normalised) )
	c1.GetPad(1).SetLogy()
	c1.cd(2)
	if not skip: histograms.append( self.Histogram1D(res_data.eta, "Resolution of #eta in " + tracks + " tracks", 100, -0.02, 0.02, "#eta", color, draw, normalised) )
	c1.GetPad(2).SetLogy()
	c1.cd(3)
	if not skip: histograms.append( self.Histogram1D(res_data.Lambda, "Resolution of #lambda in " + tracks + " tracks", 100, -0.015, 0.015, "#lambda", color, draw, normalised) )
	c1.GetPad(3).SetLogy()
	c1.cd(4)
	histograms.append( self.Histogram1D(res_data.cotTheta, "Resolution of cot(#theta) in " + tracks + " tracks", 100, -0.03, 0.03, "cot(#theta)", color, draw, normalised) )
	c1.GetPad(4).SetLogy()
	c1.cd(5)
	histograms.append( self.Histogram1D(res_data.phi, "Resolution of #phi in " + tracks + " tracks", 100, -0.03, 0.03, "#phi", color, draw, normalised) )
	c1.GetPad(5).SetLogy()
	c1.cd(6)
	histograms.append( self.Histogram1D(res_data.dxy, "Resolution of dxy in " + tracks + " tracks", 100, -0.15, 0.15, "dxy", color, draw, normalised) )
	c1.GetPad(6).SetLogy()
	c1.cd(7)
	histograms.append( self.Histogram1D(res_data.dz, "Resolution of dz in " + tracks + " tracks", 100, -0.15, 0.15, "dz", color, draw, normalised) )
	c1.GetPad(7).SetLogy()
	#c1.SetLogy()
	if draw:
	    c1.Draw()
	    input("Press enter to continue")
	
	return histograms

    def ResolutionHistogramsCombine(self, res_data_trues, res_data_fakes, fake_mask = [], draw = True):
	'''
        Combines resolutionhistograms for true and matched fake tracks to a single set of histograms.
	'''
	true_hists = self.ResolutionHistograms(res_data_trues, "true", [], False, True)
	fake_hists = self.ResolutionHistograms(res_data_fakes, "fake", fake_mask, False, True)

	c1 = ROOT.TCanvas("Resolution_histograms","Resolution histograms", 700, 900)
	c1.Divide(2,3)
	c1.SetGrid()
        histograms = []

	for i in range(len(true_hists)):
	    c1.cd(i+1)
	    histograms.append( self.Histogram1DStacked(true_hists[i], fake_hists[i], true_hists[i].GetName() + " stacked") )
	    c1.GetPad(i+1).SetLogy()

	
	c1.cd(i+2)
        leg = ROOT.TLegend(0.,0.2,0.99,0.8)
	leg.AddEntry(true_hists[0], "True tracks", "F")
	leg.AddEntry(fake_hists[0], "Fake tracks", "F")
	leg.Draw()
	

	if draw:
	    c1.Draw()
	    input("Press enter to continue")
	
	return histograms

    def ResolutionHistogramsNormalised(self, res_data_trues, res_data_fakes, fake_mask = [], draw = True):
	'''
        Plots the resolution histograms of true tracks and matched fake tracks, normalised to have an area of 1.
	'''
	true_hists = self.ResolutionHistograms(res_data_trues, "true", [], False, True, True)
	fake_hists = self.ResolutionHistograms(res_data_fakes, "fake", fake_mask, False, True, True)

	c1 = ROOT.TCanvas("Resolution_histograms","Resolution histograms", 1100, 700) #700, 900)
	c1.Divide(3,2)
	c1.SetGrid()
        #histograms = []

	for i in range(len(true_hists)):
	    c1.cd(i+1)
	    #true_hists[i].SetTitle("#splitline{True tracks: " + true_hists[i].GetTitle() + "}{Fake tracks: " + fake_hists[i].GetTitle() + "}")
	    true_hists[i].SetTitle("Resolution of " + fake_hists[i].GetXaxis().GetTitle())
	    true_hists[i].GetXaxis().SetTitle("#Delta" + true_hists[i].GetXaxis().GetTitle())
	    #true_hists[i].SetTitleOffset(1.2)
	    #c1.SetTopMargin(0.9)
	    true_hists[i].DrawNormalized("")
	    fake_hists[i].DrawNormalized("Same")
	    #histograms.append( self.Histogram1DStacked(true_hists[i], fake_hists[i], true_hists[i].GetName() + " stacked") )
	    #c1.GetPad(i+1).SetLogy()

	
	c1.cd(i+2)
        leg = ROOT.TLegend(0.1,0.3,0.9,0.7)
	leg.AddEntry(true_hists[0], "True tracks", "L")
	leg.AddEntry(fake_hists[0], "Fake tracks", "L")
	leg.Draw()
	

	if draw:
	    c1.Draw()
	    input("Press enter to continue")
	
	#return histograms
    
    def ResolutionHistogramsNormalisedCumulative(self, res_data_trues, res_data_fakes, fake_mask = [], draw = True):
	'''
        Plots the cumulative resolution histograms of true tracks and matched fake tracks from the normalised histograms from the previous method.
	'''
	true_hists = self.ResolutionHistograms(res_data_trues, "true", [], False, True, True)
	fake_hists = self.ResolutionHistograms(res_data_fakes, "fake", fake_mask, False, True, True)

	c1 = ROOT.TCanvas("Resolution_histograms","Resolution histograms", 1100, 700) #700, 900)
	c1.Divide(3,2)
	c1.SetGrid()
        #histograms = []

	for i in range(len(true_hists)):
	    c1.cd(i+1)
	    true_hists[i].SetTitle("Cumulative resolution of " + fake_hists[i].GetXaxis().GetTitle())
	    true_hists[i].GetXaxis().SetTitle("#Delta" + true_hists[i].GetXaxis().GetTitle())
	    #true_hists[i].SetTitleOffset(1.2)
	    #c1.SetTopMargin(0.9)
	    for res in res_data_trues:
		if i == 0 and res.pt_delta/res.pt_rec < -0.15: true_hists[i].Fill(-0.15)
		if i == 1 and res.cotTheta < -0.03: true_hists[i].Fill(-0.03)
		if i == 2 and res.phi < -0.03: true_hists[i].Fill(-0.03)
		if i == 3 and res.dxy < -0.15: true_hists[i].Fill(-0.15)
		if i == 4 and res.dz < -0.15: true_hists[i].Fill(-0.15)
	    for res in res_data_fakes:
		if i == 0 and res.pt_delta/res.pt_rec < -0.15: fake_hists[i].Fill(-0.15)
		if i == 1 and res.cotTheta < -0.03: fake_hists[i].Fill(-0.03)
		if i == 2 and res.phi < -0.03: fake_hists[i].Fill(-0.03)
		if i == 3 and res.dxy < -0.15: fake_hists[i].Fill(-0.15)
		if i == 4 and res.dz < -0.15: fake_hists[i].Fill(-0.15)
	    true_hists[i].Scale(1.0/len(res_data_trues.pt_rec))
	    fake_hists[i].Scale(1.0/len(res_data_fakes.pt_rec))
	    #c1.GetPad(i+1).Clear()
	    true_hists[i].GetCumulative().Draw("")
	    fake_hists[i].GetCumulative().Draw("Same")
	
	c1.cd(i+2)
        leg = ROOT.TLegend(0.1,0.3,0.9,0.7)
	leg.AddEntry(true_hists[0], "True tracks", "L")
	leg.AddEntry(fake_hists[0], "Fake tracks", "L")
	leg.Draw()
	

	if draw:
	    c1.Draw()
	    input("Press enter to continue")

    def Histogram1D(self, data, name = "Histogram", nbins = 100, xmin = -1, xmax = 1, xlabel = "", color = 38, draw = True, normalised = False):
	''' Creates a single histogram of resolutions for one parameter list '''
	
	hist = ROOT.TH1F(name, name, nbins, xmin, xmax)	
	
	hist.SetStats(0);
	if normalised:
	    hist.SetLineColor(color)
	    hist.SetLineWidth(1)
	else:
	    hist.SetFillColor(color);
	hist.GetXaxis().SetTitle(xlabel)
	#hist.SetCanExtend(ROOT.TH1.kAllAxes);
	for entry in data: hist.Fill(entry,1)

	f = ROOT.TF1("f1", "gaus", xmin, xmax)
	hist.Fit(f,"0")

	hist.SetTitle("#sigma(#Delta" + xlabel + ") = " + str(f.GetParameter("Sigma"))[0:8])

	if draw: hist.Draw()
	return hist

	#input("Press any key to continue")

    def HistogramSigmaPt(self, data, name = "Sigma(Pt) vs Pt"): # Old function
        
	c1 = ROOT.TCanvas(name, name, 700, 500)
	c1.SetLogx()
	c1.SetLogy()
	
	hist = ROOT.TH2F(name, name, 10, array("d",[10**(i/10.0) for i in range(-10, 12, 2)]), 100, -0.15, 0.15)
	
	for res in data: 
	    hist.Fill(res.pt_sim, res.pt_delta/res.pt_rec)
	
	hist.FitSlicesY()
	hist_2 = ROOT.gDirectory.Get(name + "_2")	

	hist_2.GetXaxis().SetTitle("p_{t,sim}")
	hist_2.GetYaxis().SetTitle("#sigma(#Delta p_{t}/p_{t,res})")

	hist_2.Draw()

	input("Press enter to continue")

    def HistogramSigmaPtCombined(self, data1, data2, name = "Sigma(Pt) vs Pt"): # Old function
        
	c1 = ROOT.TCanvas(name, name, 700, 500)
	c1.SetLogx()
	c1.SetLogy()
	
	hist = ROOT.TH2F(name, name, 10, array("d",[10**(i/10.0) for i in range(-10, 12, 2)]), 100, -0.15, 0.15)
	
	for res in data1:
	    hist.Fill(res.pt_sim, res.pt_delta/res.pt_rec)
	for res in data2:
	    hist.Fill(res.pt_sim, res.pt_delta/res.pt_rec)
	
	hist.FitSlicesY()
	hist_2 = ROOT.gDirectory.Get(name + "_2")	

	hist_2.GetXaxis().SetTitle("p_{t,sim}")
	hist_2.GetYaxis().SetTitle("#sigma(#Delta p_{t}/p_{t,rec})")

	hist_2.Draw()

	input("Press enter to continue")
    
    def HistogramSigmaPtDual(self, data1, data2, parameter = "pt", pad = None, name = "Sigma(parameter) vs Pt"):
	'''
	Plots the sigma parameter from Gaussian fits to a single parameter resolution with respect to simulated particle pt.

	data1 = True tracks, data2 = Fake track
	'''
	
        
	if not pad:
	    c1 = ROOT.TCanvas(name, name, 700, 500)
	    c1.SetLogx()
	    c1.SetLogy()
	else:
	    pad.SetLogx()
	    pad.SetLogy()	
	
	hist1 = ROOT.TH2F(name, name, 10, array("d",[10**(i/10.0) for i in range(-10, 12, 2)]), 100, -0.15, 0.15)
	hist2 = ROOT.TH2F(name + "_h2", name + "_h2", 10, array("d",[10**(i/10.0) for i in range(-10, 12, 2)]), 100, -0.15, 0.15)
	hist3 = ROOT.TH2F(name + "_h3", name + "_h3", 10, array("d",[10**(i/10.0) for i in range(-10, 12, 2)]), 100, -0.15, 0.15)

	#print min(data.pt_sim)
	for res in data1:
	    if parameter == "pt": hist1.Fill(res.pt_sim, res.pt_delta/res.pt_rec)
	    if parameter == "cotTheta": hist1.Fill(res.pt_sim, res.cotTheta)
	    if parameter == "phi": hist1.Fill(res.pt_sim, res.phi)
	    if parameter == "dxy": hist1.Fill(res.pt_sim, res.dxy)
	    if parameter == "dz": hist1.Fill(res.pt_sim, res.dz)
	for res in data2:
	    if parameter == "pt": hist2.Fill(res.pt_sim, res.pt_delta/res.pt_rec)
	    if parameter == "cotTheta": hist2.Fill(res.pt_sim, res.cotTheta)
	    if parameter == "phi": hist2.Fill(res.pt_sim, res.phi)
	    if parameter == "dxy": hist2.Fill(res.pt_sim, res.dxy)
	    if parameter == "dz": hist2.Fill(res.pt_sim, res.dz)

        hist3.Add(hist1)
	hist3.Add(hist2)

	hist1.FitSlicesY()
	hist1_2 = ROOT.gDirectory.Get(name + "_2")
        hist1_2.SetLineColor(ROOT.kBlue)
	#hist1_2.GetYaxis().SetRange(0.01, 1)

	hist2.FitSlicesY()
	hist2_2 = ROOT.gDirectory.Get(name + "_h2" + "_2")
	hist2_2.SetLineColor(ROOT.kRed)

	hist2_2.GetXaxis().SetTitle("p_{t,sim}")
	hist2_2.SetTitleSize(0.04,"xy")
	if parameter == "pt":
	    hist2_2.GetYaxis().SetTitle("#sigma(#Delta p_{t}/p_{t,rec})")
	    hist2_2.SetMaximum(1.5)
	    hist2_2.SetMinimum(0.003)
	if parameter == "cotTheta":
	    hist2_2.GetYaxis().SetTitle("#sigma(#Delta cot(#theta))")
	    hist2_2.SetMaximum(1.0)
	    hist2_2.SetMinimum(0.0005)
	if parameter == "phi":
	    hist2_2.GetYaxis().SetTitle("#sigma(#Delta #phi)")
	    hist2_2.SetMaximum(1.0)
	    hist2_2.SetMinimum(0.0005)
	if parameter == "dxy":
	    hist2_2.GetYaxis().SetTitle("#sigma(#Delta dxy)")
	    hist2_2.SetMaximum(1.3)
	    hist2_2.SetMinimum(0.0005)
	if parameter == "dz":
	    hist2_2.GetYaxis().SetTitle("#sigma(#Delta dz)")
	    hist2_2.SetMaximum(1.5)
	    hist2_2.SetMinimum(0.0005)


	hist2_2.SetStats(0)
	hist2_2.SetTitle("")

	hist3.FitSlicesY()
	hist3_2 = ROOT.gDirectory.Get(name + "_h3" + "_2")
	hist3_2.SetLineColor(ROOT.kViolet)

	hist2_2.Draw()
	hist3_2.Draw("Same")
	hist1_2.Draw("Same")
        
	'''
	leg = ROOT.TLegend(0.25,0.75,0.45,0.9)
	leg.AddEntry(hist1_2, "True tracks", "L")
	leg.AddEntry(hist2_2, "Fake tracks", "L")
	leg.AddEntry(hist3_2, "Both", "L")
	leg.Draw()
	
	input("Press enter to continue")
	'''
	return [hist1_2, hist2_2, hist3_2]

    def ResolutionsSigmaVsPt(self, res_data_trues, res_data_fakes, fake_mask = [], draw = True):
        '''
        Plots the sigma parameter from Gaussian fits to all parameter resolutions with respect to simulated particle pt.
	'''

	c1 = ROOT.TCanvas("Resolution_sigma_vs_pt","Resolution_sigma_vs_pt", 1100, 700) #700, 900)
	c1.Divide(3,2)
	c1.SetGrid()
        hists_tmp = []

	pad = c1.cd(1)
	hists_tmp.append(self.HistogramSigmaPtDual(res_data_trues, res_data_fakes, "pt", pad, "pt"))
	pad = c1.cd(2)
	hists_tmp.append(self.HistogramSigmaPtDual(res_data_trues, res_data_fakes, "cotTheta", pad, "cotTheta"))
	pad = c1.cd(3)
	hists_tmp.append(self.HistogramSigmaPtDual(res_data_trues, res_data_fakes, "phi", pad, "phi"))
	pad = c1.cd(4)
	hists_tmp.append(self.HistogramSigmaPtDual(res_data_trues, res_data_fakes, "dxy", pad, "dxy"))
	pad = c1.cd(5)
	hists = self.HistogramSigmaPtDual(res_data_trues, res_data_fakes, "dz", pad, "dz")
	hists_tmp.append(hists)

	c1.cd(6)
        leg = ROOT.TLegend(0.1,0.3,0.9,0.7)
	leg.AddEntry(hists[0], "True tracks", "L")
	leg.AddEntry(hists[1], "Fake tracks", "L")
	leg.AddEntry(hists[2], "Both", "L")
	leg.Draw()
	

	if draw:
	    c1.Draw()
	    input("Press enter to continue")
	


    def Histogram1DStacked(self, hist1, hist2, name): # Old function
        hist_stack = ROOT.THStack(name, name)
	hist_stack.Add(hist2)
	hist_stack.Add(hist1)

	hist = ROOT.TH1F(hist1)
	hist.Add(hist2)
	f = ROOT.TF1("f1", "gaus", hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax())
	hist.Fit(f,"0")
	hist_stack.SetTitle("#sigma(" + hist.GetXaxis().GetTitle() + ") = " + str(f.GetParameter("Sigma")))
	hist_stack.Draw()#("nostack")	
	hist_stack.GetXaxis().SetTitle(hist.GetXaxis().GetTitle())

	#hist_stack.Draw("nostack")
	return hist_stack
    
    def Histogram1D_Draw(self, data, name = "Histogram", nbins = 10): # Old function
	c1 = ROOT.TCanvas(name, name, 700, 500)
	#if "pt " in name: c1.SetLogx()
	c1.SetGrid()
	
	hist = ROOT.TH1F(name, name, nbins, min(data),max(data))
	c1.SetLogy()
	
	hist.SetStats(0);
        hist.SetFillColor(38);
	#hist.SetCanExtend(ROOT.TH1.kAllAxes);
	for entry in data: hist.Fill(entry,1)
	hist.Draw()

    def Draw(self):
        ''' Draws everything else except histograms. '''
	if self.plots_3D:
	    c3 = ROOT.TCanvas("3D Plot","3D Plot", 1200, 1200)
	    axis = ROOT.TAxis3D()
	    axis.SetXTitle("z")
	    axis.SetYTitle("x")
	    axis.SetZTitle("y")
	    #axis.GetZaxis().RotateTitle()

	    for plot in self.plots_3D:
		if plot: plot.Draw()
	    #axis.SetAxisRange(-278,278,"Z")
	    axis.Draw()
	    axis.ToggleZoom()

	if self.plots_2D:
	    c2 = ROOT.TCanvas("2D Plot","2D Plot", 1200, 1200)
	    #axis = ROOT.TAxis()

	    for plot in self.plots_2D:
		plot.Draw("A")
	    #axis.Draw()

	    '''
	    plots = ROOT.TMultiGraph()
	    for plot in self.plots_2D:
		plots.Add(plot) #plot.Draw("A")
            plots.Draw("A")
	    '''

        input("Press any key to continue")


