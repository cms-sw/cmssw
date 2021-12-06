#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import ROOT
from array import array

from Validation.RecoTrack.plotting.ntuple import *
import analysis

from math import sqrt, copysign, sin, cos, pi

class EventPlotter(object):

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
		pix_coords.append(hit.x())
		pix_coords.append(hit.y())
		pix_coords.append(hit.z())
	    pix_plot = ROOT.TPolyMarker3D(len(pix_coords)/3, array('f', pix_coords), 1)
	    pix_plot.SetMarkerColor(4)

	if('s' in flag or 'S' in flag):
	    strip_hits = event.stripHits()
	    str_coords = []
	    for hit in strip_hits:
		str_coords.append(hit.x())
		str_coords.append(hit.y())
		str_coords.append(hit.z())
	    str_plot = ROOT.TPolyMarker3D(len(str_coords)/3, array('f', str_coords), 1)
	    str_plot.SetMarkerColor(2)

	if('g' in flag or 'G' in flag):
	    glued_hits = event.gluedHits()
	    glu_coords = []
	    for hit in glued_hits:
		glu_coords.append(hit.x())
		glu_coords.append(hit.y())
		glu_coords.append(hit.z())
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
	#ntuple.tree().GetBranch("pix_x")
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
	#ntuple.tree().GetBranch("pix_x")
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
	plot = ROOT.TMultiGraph()

	for track in tracks:
	    X = []; Y = [];
	    for hit in track.hits():
		if(hit.isValid()):
		    X.append(hit.x())
		    Y.append(hit.y())
	    plot.Add(ROOT.TGraph(len(X),array("f",X),array("f",Y)),"L")

	self.plots_2D.append(plot)

    def PlotTracksZY(self, tracks):
	plot = ROOT.TMultiGraph()

	for track in tracks:
	    Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValid()):
		    Y.append(hit.y())
		    Z.append(hit.z())
	    plot.Add(ROOT.TGraph(len(Z),array("f",Z),array("f",Y)),"L")

	self.plots_2D.append(plot)    
   
    def PlotTracks3D(self, tracks): 
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValid()):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())	
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z)))
	
    def PlotPixelTracks3D(self, tracks): 
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.pixelHits():
		if(hit.isValid()):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z)))

    def PlotPixelGluedTracks3D(self, tracks):
	for track in tracks:
	    X = []; Y = []; Z = [];
	    for hit in track.hits():
		if(hit.isValid() and hit.hitType != 1):
		    X.append(hit.x())
		    Y.append(hit.y())
		    Z.append(hit.z())
	    if(X):
		self.plots_3D.append(ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z)))

    def PlotTrack3D(self, track, color = 1): 
	'''Plots a single track and prints track info'''
	# Not so hasardous experimental edit:
	#hits = sorted([hit for hit in track.hits()], key = lambda hit: hit.index())
	#print [hit.index() for hit in hits]
	X = []; Y = []; Z = [];
	for hit in track.hits(): #hits: #track.hits():
	    if(hit.isValid()):
		X.append(hit.x())
		Y.append(hit.y())
		Z.append(hit.z())	
	if(not X):
	    print("Track has no valid points")
	    return
	plot = ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z))
	plot.SetLineColor(color)
	self.plots_3D.append(plot)

	'''
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

    def TrackHelix(self, track, color = 1, style = 0):
	'''Creates a THelix object which can be plotted with Draw() method.'''
	if isinstance(track, TrackingParticle):
	    phi = track.pca_phi()
	    dxy = track.pca_dxy()
	    dz = track.pca_dz()
	else:
	    phi = track.phi()
	    dxy = track.dxy()
	    dz = track.dz()
	xyz = array("d", [-dxy*ROOT.TMath.Sin(phi), dxy*ROOT.TMath.Cos(phi), dz])
	v = array("d", [track.px(), track.py(), track.pz()])
	w = 0.3*3.8*track.q()*0.01 #Angular frequency = 0.3*B*q*hattuvakio, close enough
	z_last = dz
	for hit in track.hits(): 
	    if(hit.isValidHit()): z_last = hit.z()
	
	helix = ROOT.THelix(xyz, v, w, array("d", [dz, z_last]))
	helix.SetLineColor(color)
	if style == 1: helix.SetLineStyle(9)
	if style == 2: helix.SetLineStyle(7)
	return helix

    def Plot3DHelixes(self, tracks, color = 1, style = 0):	
	for track in tracks:
	    if(track.hits()):
		self.plots_3D.append(self.TrackHelix(track, color, style))	

    def Plot3DHelix(self, track, color = 1, style = 0):	
	if(track.hits()):
	    self.plots_3D.append(self.TrackHelix(track, color, style))

    def Plot3DHits(self, track, color = 1, style = 0):
	'''
	Plots the 3D hits from a track.
	'''	
	pix_coords = []
	for hit in track.pixelHits():    
	    if hit.isValid():
		pix_coords.append(hit.x())
		pix_coords.append(hit.y())
		pix_coords.append(hit.z())
	    if pix_coords:
		pix_plot = ROOT.TPolyMarker3D(len(pix_coords)/3, array('f', pix_coords), 2)
		pix_plot.SetMarkerColor(color)
		if style == 1: pix_plot.SetMarkerStyle(5)
		if style == 2: pix_plot.SetMarkerStyle(4)
		self.plots_3D.append(pix_plot)
	
	for hit in track.gluedHits():
	    if hit.isValid():
		x = hit.x(); y = hit.y(); z = hit.z()
		if hit.isBarrel():
		    X = [x, x]
		    Y = [y, y]
		    Z = [z - sqrt(hit.zz()), z + sqrt(hit.zz())]
		else:
		    X = [x - copysign(sqrt(hit.xx()),x), x + copysign(sqrt(hit.xx()),x)]
		    Y = [y - copysign(sqrt(hit.yy()),y), y + copysign(sqrt(hit.yy()),y)]
		    Z = [hit.z(), hit.z()]
		glu_plot = ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z))
		#glu_plot.SetLineStyle(2)    
		if style == 1: glu_plot.SetLineStyle(2)
		if style == 2: glu_plot.SetLineStyle(3)
		glu_plot.SetLineColor(color) 
		self.plots_3D.append(glu_plot)

	for hit in track.stripHits():
	    if hit.isValid():
		x = hit.x(); y = hit.y(); z = hit.z()
		if hit.isBarrel():
		    X = [x, x]
		    Y = [y, y]
		    Z = [z - 1.5*sqrt(hit.zz()), z + 1.5*sqrt(hit.zz())]
		else:
		    X = [x - 1.5*copysign(sqrt(hit.xx()),x), x + 1.5*copysign(sqrt(hit.xx()),x)]
		    Y = [y - 1.5*copysign(sqrt(hit.yy()),y), y + 1.5*copysign(sqrt(hit.yy()),y)]
		    Z = [hit.z(), hit.z()]
		str_plot = ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z))
		if style == 1: str_plot.SetLineStyle(2)
		if style == 2: str_plot.SetLineStyle(3)
		str_plot.SetLineColor(color) 
		self.plots_3D.append(str_plot)

    def PlotVertex3D(self, vertex, color=1):
	plot = ROOT.TPolyMarker3D(1, array('f', [vertex.x(), vertex.y(), vertex.z()]),3)
	plot.SetMarkerColor(color)
        self.plots_3D.append(plot)

    def PlotPoint3D(self, point, color=1):
	plot = ROOT.TPolyMarker3D(1, array('f', [point[0], point[1], point[2]]),3)
	plot.SetMarkerColor(color)
        self.plots_3D.append(plot)

    def PlotTrackingFail(self, match):
	X = array('f', [match.last_loc[0], match.fail_loc[0]])
	Y = array('f', [match.last_loc[1], match.fail_loc[1]])
	Z = array('f', [match.last_loc[2], match.fail_loc[2]])
        plot = ROOT.TPolyLine3D(2, X, Y, Z)
	plot.SetLineWidth(3)
	plot.SetLineColor(2) 
        self.plots_3D.append(plot)
	self.PlotPoint3D(match.last_loc, 2)

    def PlotFakes_MatchedRecos(self, event, iterative = 1, reconstructed = 0):
        fakes = analysis.FindFakes(event)
	if iterative:
            # Plot fakes one by one
	    for fake in fakes:
		self.Reset()
		self.Plot3DHelixes([fake],2)
		self.Plot3DHits(fake, 2)

                # Plot real particle tracks which include fake tracks hits
		icol = 0
		particle_inds = []
		particles = []
		reco_inds = []
		recos = []
		for hit in fake.hits():
		    if hit.isValid() and hit.nSimHits() >= 0:
			for simHit in hit.simHits():
			    particle = simHit.trackingParticle()
			    if particle.index() not in particle_inds:
				particle_inds.append(particle.index())
				particles.append(particle)

			    '''
			    self.Plot3DHelix(particle, ROOT.TColor().GetColor(0,255,0), 1) # kAzure color, maybe ;)
			    self.Plot3DHits(particle, 3+icol, 1) 
			    icol += 1

			    print "Number of matched tracks to real particle: " + str(particle.nMatchedTracks())
			    '''
                            # Plot reconstructed tracks of these real particle tracks
			    if reconstructed and particle.nMatchedTracks() > 0:
				for info in particle.matchedTrackInfos():
				    track = info.track()
				    if track.index() not in reco_inds:
					reco_inds.append(track.index())
					recos.append(track)

				    '''
				    if particle.nMatchedTracks() == 1:
					self.Plot3DHelix(track,1,2)
					self.Plot3DHits(track,1,2)
				    else:
					self.Plot3DHelix(track,5,2)
					self.Plot3DHits(track,5,2)
				    '''
		icol =  0
                for particle in particles:
		    self.Plot3DHelix(particle, self.colors_G[icol], 1) # kAzure color, maybe ;)
		    self.Plot3DHits(particle, self.colors_G[icol], 1) 
		    icol += 1
		for track in recos:
		    self.Plot3DHelix(track,1,2)
		    self.Plot3DHits(track,1,2)
		    '''
		    if track.trackingParticle().nMatchedTracks() == 1:
			self.Plot3DHelix(track,1,2)
			self.Plot3DHits(track,1,2)
		    else:
			self.Plot3DHelix(track,5,2)
			self.Plot3DHits(track,5,2)
		    '''

	        self.Draw()
            return
        # the following is useless by now
        # Plot all fakes at once (messy picture)
	'''
	self.Plot3DHelixes(fakes,2)
	for fake in fakes:
	    self.Plot3DHits(fake, 2)
            # Plot real particle tracks which include fake tracks hits
	    for hit in fake.hits():
		if hit.isValid() and hit.nMatchedTrackingParticles() >= 0:
		    for info in hit.matchedTrackingParticleInfos():
			particle = info.trackingParticle()
			self.Plot3DHelix(particle,3,1)
			self.Plot3DHits(particle,3,1)
 
			print "Number of matched tracks to real particle: " + str(particle.nMatchedTracks())
			# Plot reconstructed tracks of these real particle tracks
			if reconstructed and particle.nMatchedTracks() > 0:
			    for info in particle.matchedTrackInfos():
				track = info.track()	
				if particle.nMatchedTracks() == 1:
				    self.Plot3DHelix(track,1,2)
				    self.Plot3DHits(track,1,2)
				else:
				    self.Plot3DHelix(track,5,2)
				    self.Plot3DHits(track,5,2)
        '''

    def PlotFakes(self, event, reconstructed = 1, fake_filter = None, end_filter = None, last_filter = None, particle_end_filter = None):
	iterative = 1

        fakes = analysis.FindFakes(event)
	if iterative:
            # Plot fakes one by one
	    for fake in fakes:
                # Check for filter
		if fake_filter:
		    info = analysis.FakeInfo(fake)
		    if info.fake_class not in fake_filter:
			continue

		self.Reset()
		self.Plot3DHelixes([fake],2)
		self.Plot3DHits(fake, 2)
		#self.PlotVertex3D(vertex,2)
		fake_info = analysis.FakeInfo(fake)
		analysis.PrintTrackInfo(fake, None, 0, fake_info)
                if fake_info.matches:
		    for match in fake_info.matches:
			continue #self.PlotTrackingFail(match)
			#self.PlotPoint3D(fake_info.end_loc, 2)

                # Find real particle tracks which include fake tracks hits
		icol = 0
		particle_inds = []
		particles = []
		reco_inds = []
		recos = []
		for hit in fake.hits():
		    if hit.isValid() and hit.nMatchedTrackingParticles() >= 0:	
			for info in hit.matchedTrackingParticleInfos():
			    particle = info.trackingParticle()
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
		    if hit.isValid() and reconstructed and hit.ntracks() > 0:
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
		    
		if hit.isValid() and hit.z() >= 0: self.PlotDetectorRange("p",4)
		elif hit.isValid() and hit.z() < 0: self.PlotDetectorRange("n",4)
		else: self.PlotDetectorRange("b",4)

                print("****************************\n")
	        self.Draw()
            return

    def DrawTrackTest_old(self, track):
        c3 = ROOT.TCanvas("3D Plot","3D Plot", 1200, 1200)
	self.PlotDetectorRange("b", 4)
	for i in range(8): self.plots_3D[i].Draw()
	axis = ROOT.TAxis3D()
	axis.Draw()
	for hit in track.hits():
	    if hit.isValid(): 
		point = ROOT.TPolyMarker3D(1, array('f', [hit.x(), hit.y(), hit.z()]), 2)
		point.Draw()
		input("continue")
	#axis.SetAxisRange(-278,278,"Z")
	
	axis.ToggleZoom()

    def DrawTrackTest(self, track): 
	self.PlotDetectorRange("b", 4)	
	self.PlotTrack3D(track)
	#self.Draw()
	'''
	for hit in track.hits():
	    if hit.isValid(): 
		
		point = ROOT.TPolyMarker3D(1, array('f', [hit.x(), hit.y(), hit.z()]), 2)
		self.plots_3D.append(point)	
		self.Draw()
		'''

    def ParticleTest(self, particles, draw=False):
	for particle in particles:
	    print("\nPARTICLE " + str(particle.index()))
	    for hit in particle.hits():
		tof = -1
		for info in hit.matchedTrackingParticleInfos():
		    if info.trackingParticle().index() == particle.index():
			#if len(info.tof()): 
			tof = info.tof()
		print("Index: " + str(hit.index()) + ", Layer: " + str(hit.layerStr()) + ", TOF: " + str(tof) +\
	        "     XY distance: " + str(sqrt(hit.x()**2 + hit.y()**2)) + ", Z: " + str(hit.z()))
	    self.DrawTrackTest(particle)
	    if draw:
		self.Draw()
		self.Reset()
 
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
	    plot = ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",Z),"S")
	    nplot = ROOT.TPolyLine3D(len(X),array("f",X),array("f",Y),array("f",ZN),"S")
	    plot.SetLineStyle(3)
	    nplot.SetLineStyle(3)
	    if area == "p": self.plots_3D.append(plot)
	    if area == "n": self.plots_3D.append(nplot)
	    if area == "b":
		self.plots_3D.append(plot)
		self.plots_3D.append(nplot)

    def ClassHistogram(self, data, name = "Histogram", labels = False, ready = False):
        '''
	Draws histogram with data. Data is in form of a dictionary,
	labels assigned to amount of hits. If labels is defined, it
	defines the labels assigned to data.
	'''
	if ready:
	    if labels:
		keys = [key for key in data]
		hist_labels = [labels[key] for key in keys]
		hist_data = [data[key] for key in keys]
	    else:
		hist_labels = [key for key in data]
		hist_data = [data[key] for key in hist_labels]
	
	c1 = ROOT.TCanvas(name, name, 700, 500)
	c1.SetGrid()
	#c1.SetTopMargin(0.15)
	hist = ROOT.TH1F(name, name, 30,0,30)#, 3,0,3)
        hist.SetStats(0);
        hist.SetFillColor(38);
	hist.SetCanExtend(ROOT.TH1.kAllAxes);
	if ready: # cumulative results already calculated in data
	    for i in range(len(hist_labels)):
		#hist.Fill(hist_labels[i],hist_data[i])
		[hist.Fill(hist_labels[i],1) for j in range(hist_data[i])]
	else: # data is in samples, not in cumulative results
	    for entry in data: hist.Fill(entry,1)
	hist.LabelsDeflate()
	#hist.SetLabelSize(0.05)
	hist.Draw()

	input("Press any key to continue")


    def Draw(self):
	if self.plots_3D:
	    c3 = ROOT.TCanvas("3D Plot","3D Plot", 1200, 1200)
	    view = ROOT.TView3D()
	    axis = ROOT.TAxis3D()
	    axis.SetXTitle("x")
	    axis.SetYTitle("y")
	    axis.SetZTitle("z")
	    #axis.GetZaxis().RotateTitle()

	    for plot in self.plots_3D:
		if plot: plot.Draw()
	    
	    #view.DefineViewDirection(array("d",[1,1,1]),array("d",[0,0,0]),1.0,0.0,0.0,1.0,0.0,1.0)#,array("d",[-1,-1,-1]),array("d",[1,1,1]))
	    axis.Draw()
	    #view.SetAxisNDC(array("d",[0,0,0]),array("d",[0,1,0]),array("d",[0,0,0]),array("d",[0,0,1]),array("d",[0,0,0]),array("d",[1,0,0]))
    
	    #view.SetPsi(30)
	    #view.SetView(0,0,pi/2,ROOT.Long())
	    #view.DefineViewDirection(0,0,pi/2)
	    #view.RotateView(pi/2,pi/2)
	    axis.ToggleZoom()

	if self.plots_2D:
	    c2 = ROOT.TCanvas("2D Plot","2D Plot", 1200, 1200)

	    for plot in self.plots_2D: plot.Draw("A") 

	input("Press any key to continue")


