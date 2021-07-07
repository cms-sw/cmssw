#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import ROOT
from array import array
from collections import OrderedDict
from copy import copy
import pickle
from math import sqrt, copysign, sin, cos, pi

from Validation.RecoTrack.plotting.ntuple import *
import six

##### GLOBAL VARIABLES #####

''' These are different class labels, which are mainly used for plotting histograms. '''

# Classification of fake tracks:
classes = {-1: "UNCLASSIFIED",
	    0: "UNMATCHED",
	   21: "MERGED DECAY TRACK MATCHES",
	   22: "MULTIPLE ALL RECONSTRUCTED MATCHES",
	   23: "MULTIPLE PARTIALLY RECONSTRUCTED MATCHES",
	   20: "MULTIPLE UNRECONSTRUCTED MATCHES",
           11: "RECONSTRUCTED AND DECAYED MATCH",
           12: "RECONSTRUCTED MATCH",
	   13: "DECAYED MATCH",
	   10: "MATCH"}

# Classification of different types of the end of tracking between a fake and a particle track
end_class_names = {0: "Fake and particle tracks differ",
                   1: "Tracking ended correctly",
		   2: "Fake ended before particle",
		   3: "Particle ended before fake",#}#,
		   4: "Particle ended within two hits in same layer"}

# Detector layer names
layer_names_t = ((1, "BPix1"),
               (2, "BPix2"),
               (3, "BPix3"),
               (4, "FPix1"),
               (5, "FPix2"),
	       (11, "TIB1"),
	       (12, "TIB2"),
	       (13, "TIB3"),
	       (14, "TIB4"),
	       (21, "TID1"),
	       (22, "TID2"),
	       (23, "TID3"),
	       (31, "TOB1"),
	       (32, "TOB2"),
	       (33, "TOB3"),
	       (34, "TOB4"),
	       (35, "TOB5"),
	       (36, "TOB6"),
	       (41, "TEC1"),
	       (42, "TEC2"),
	       (43, "TEC3"),
	       (44, "TEC4"),
	       (45, "TEC5"),
	       (46, "TEC6"),
	       (47, "TEC7"),
	       (48, "TEC8"),
	       (49, "TEC9"))
layer_names = OrderedDict(layer_names_t)

layer_names_rev_t = (("BPix1", 1),
               ("BPix2", 2),
               ("BPix3", 3),
               ("FPix1", 4),
               ("FPix2", 5),
	       ("TIB1", 11),
	       ("TIB2", 12),
	       ("TIB3", 13),
	       ("TIB4", 14),
	       ("TID1", 21),
	       ("TID2", 22),
	       ("TID3", 23),
	       ("TOB1", 31),
	       ("TOB2", 32),
	       ("TOB3", 33),
	       ("TOB4", 34),
	       ("TOB5", 35),
	       ("TOB6", 36),
	       ("TEC1", 41),
	       ("TEC2", 42),
	       ("TEC3", 43),
	       ("TEC4", 44),
	       ("TEC5", 45),
	       ("TEC6", 46),
	       ("TEC7", 47),
	       ("TEC8", 48),
	       ("TEC9", 49))
layer_names_rev = OrderedDict(layer_names_rev_t)

# The following is used as a template to store data with respect to detector layers
layer_data_tmp_t = ((1,0),(2,0),(3,0),(4,0),(5,0),
                    (11,0),(12,0),(13,0),(14,0),
		    (21,0),(22,0),(23,0),
		    (31,0),(32,0),(33,0),(34,0),(35,0),(36,0),
		    (41,0),(42,0),(43,0),(44,0),(45,0),(46,0),(47,0),(48,0),(49,0))
layer_data_tmp = OrderedDict(layer_data_tmp_t)

# classes for different invalid hit types
invalid_types_t = ((0, "valid"),
               (1, "missing"),
               (2, "inactive"),
	       (3, "bad"),
	       (4, "missing_inner"),
	       (5, "missing_outer"))
invalid_types = OrderedDict(invalid_types_t)

# The following is used as a template to store data with respect invalid hit types
invalid_types_tmp_t = ((0,0),(1,0),(2,0),(3,0),(4,0),(5,0))
invalid_types_tmp = OrderedDict(invalid_types_tmp_t)


##### ANALYSIS TOOLS #####

def FindFakes(event):
    '''Returns fake tracks of an event in a list'''
    fakes = []
    for track in event.tracks():
	if track.nMatchedTrackingParticles() == 0:
	    fakes.append(track)
    print("Event: " + str(event.entry()+1) + " Number of fake tracks: " + str(len(fakes)))
    return fakes

def FindTrues(event):
    '''Returns true tracks of an event in a list'''
    trues = []
    for track in event.tracks():
	if track.nMatchedTrackingParticles() >= 0:
	    trues.append(track)
    print("Event: " + str(event.entry()+1) + " Number of true tracks: " + str(len(trues)))
    return trues

def Distance(x1,x2):
    '''Returns Euclidean distance between two iterable vectors.'''
    d = 0
    for i in range(len(x1)):
	d += abs(x1[i]-x2[i])**2
    d = sqrt(d)
    return d

def FindUntrackedParticles(event):
    '''Returns untracked TrackingParticles of an event, according to MTFEfficiencySelector.'''
    untracked = []
    #print len(event.trackingParticles())
    for particle in event.trackingParticles():
	if (particle.isValid() and particle.nMatchedTracks() == 0 and MTFEfficiencySelector(particle)):
	    untracked.append(particle)
    print("Number of untracked particles: " + str(len(untracked)))
    return untracked

def MTFEfficiencySelector(particle):
    '''
    A selector to analyse MultiTrackFinder efficiency rate.
    Used to analyse untracked particles.
    Returns True if particle fulfills the criterion for an "interesting" particle,
    which could have been detected precisely.
    '''
    if(particle.pt() > 0.9 and abs(particle.eta()) < 2.5 
       and (particle.parentVertex().x()**2 + particle.parentVertex().y()**2 < 3.5**2)
       and abs(particle.parentVertex().z()) < 30 and particle.q() != 0 and particle.event() == 0):
	return True
    return False

def EfficiencyRate(ntuple, N):
    '''
    Calculates the efficiency rate of N first events in the ntuple class.
    Efficiency rate is the fraction between tracked particles and all particles.
    Prints output.
    '''
    tracked = 0
    untracked = 0
    i = 0
    for event in ntuple:
	if (i >= N): break
	for particle in event.trackingParticles():
	    if (MTFEfficiencySelector(particle)):
		if(particle.nMatchedTracks() == 0): untracked += 1
		else: tracked += 1 
	i += 1
    print("In " + str(N) + " events there are:")
    print("Tracked particles:   " + str(tracked))
    print("Untracked particles: " + str(untracked))
    print("Efficiency rate:     " + str(float(tracked)/(tracked+untracked)))

def SharedHitFrac(track, particle, frac = 0):
    '''
    Shared hits or hit fractions between a Track and a TrackingParticle.
    If frac = 0, returns number of shared hits, number of hits in the Track and number of hits in the TrackingParticle.
    Otherwise returns the shared hit fraction between the Track and the TrackingParticle.
    '''
    particle_hits = [hit.index() for hit in particle.hits() if hit.isValidHit()]
    shared_hits = 0
    ntrack_hits = 0
    for hit in track.hits():
	if hit.isValidHit() and hit.index() in particle_hits:
            shared_hits += 1
	ntrack_hits += 1
    if frac:
	return shared_hits, ntrack_hits, len(particle_hits)
    else:
	return 1.0*shared_hits/ntrack_hits

def SharedHitsFromBeginning(track, particle, tolerance=0):
    '''
    Checks the shared hits with a particle from the beginning of the track.
    Tolerance is the amount of allowed differences (errors) between tracks.
    Returns an list including the amount of shared hits from the beginning
    as the function of tolerance (index).

    Example:
    Output: [3, 3, 4, 4, 4]
    Means: first 3 particle hits are shared with track,
           then 1 unshared particle hit,
	   then 1 more shared particle hit,
	   then 2 unshared particle hits
	   until tolerance < 0 (or particle track ended)

    NOTE: Often this is called with a TrackingParticle as parameter "track" and a Track as the parameter "particle",
          which is because it was later needed to analyse the hits which are consecutive with respect to the Track.
	  Sorry about inconvenience.
    '''
    particle_hits = [hit.index() for hit in particle.hits() if hit.isValidHit()]
    track_hits = [hit.index() for hit in track.hits() if hit.isValidHit()]
    #print track_hits
    #print particle_hits
    count = 0
    i = 0
    result = []
    while tolerance >= 0 and i < len(particle_hits) and count < len(track_hits):
	if particle_hits[i] in track_hits:
	    count += 1
	    i += 1
	else:
	    tolerance -= 1
	    result.append(count)
	    i += 1
    if tolerance >= 0:
	result.append(count)
    return result

def SharedHits(track, particle):
    '''Returns shared hits between a Track and a TrackingParticle in a list'''
    res_hits = []
    particle_hit_indexes = [hit.index() for hit in particle.hits() if hit.isValidHit()]
    track_hits = [hit for hit in track.hits() if hit.isValidHit()]   

    for hit in track_hits:
	if hit.index() in particle_hit_indexes:
	    res_hits.append(hit)
	    
    return res_hits	

def FindAssociatedParticles(track):
    '''Returns TrackingParticles in a list that have any shared hits with the given Track'''
    particle_inds = []
    particles = []
    for hit in track.hits():
	if hit.isValidHit() and hit.nSimHits() >= 0:
	    for simHit in hit.simHits():
		particle = simHit.trackingParticle()
		if particle.index() not in particle_inds:
		    particle_inds.append(particle.index())
		    particles.append(particle)
    return particles 

def MatchedParticles(fake, real_criterion = ["consecutive", 3]):
    '''
    Find the possible matched real particle of a fake track.
    Currently three possible criterions for evaluating a possible match:
    consecutive: has a given amount of hits which are consecutive in a particle track (default with 3 hit limit)
    fraction: a given fraction of fake hits is included in a particle track
    NLay: a given number of pixel / strip layers are included in the shared hits 

    Parameters: fake track, criterion type and limit value in a list
    Returns: a list of matched particles
    '''
    CRITERION = real_criterion[0]
    LIMIT = real_criterion[1]
 
    particles = FindAssociatedParticles(fake)
    matches = []
    for particle in particles:
	if CRITERION == "consecutive":
	    tolerance_mask = SharedHitsFromBeginning(particle, fake, particle.nValid())
	    diff = [abs(tolerance_mask[i+1] - tolerance_mask[i]) for i in range(len(tolerance_mask)-1)]
	    if tolerance_mask[0] >= LIMIT or (diff and max(diff) >= LIMIT):
		matches.append(particle)
	if CRITERION == "fraction":
	    if SharedHitFrac(fake, particle, 0) >= LIMIT:
		matches.append(particle)
	if CRITERION == "NLay":
	    hits = SharedHits(fake, particle)
	    nPix = 0
	    nStr = 0
	    tracked_layers = []
	    for hit in hits:	
		layer = hit.layerStr()
		if layer not in tracked_layers:
		    tracked_layers.append(layer)
		    if "Pix" in layer: nPix += 1
		    else: nStr += 1
	    if 2*nPix + nStr >= LIMIT: # LIMIT default should be 6
		matches.append(particle)
    return matches
	
def IsUnmatched(fake, unmatch_criterion = ["nShared", 2]):
    '''
    Returns True if the the particle is unmatched to any particle with respect
    to the criterion. Criterion is by default that if there are n <= 2 hits
    shared between any track an the fake, the fake is unmatched.
    '''
    CRITERION = unmatch_criterion[0]
    LIMIT = unmatch_criterion[1]
 
    for particle in FindAssociatedParticles(fake):
	if CRITERION == "nShared":
	    shared, track_hits, particle_hits = SharedHitFrac(fake, particle, 1)
	    if shared > LIMIT:
		return False
    return True

def FindEndOfTracking(fake, particle, end_criterion = ["nMissing", 2], real_criterion = ["consecutive", 3], only_valid = False):
    '''
    Finds the point where the fake does not track the particle anymore, according to
    end_criterion, which is 2 consecutive missing layers in particle hits by default.
    Returns: last: the last shared hit between the fake and the particle, which is the first hit in particle trajectory after which tracks separate by the end criterion (or end)
             fake_end: the fake track hit following the last shared hit
	     particle_end: the particle hit following the last shared hit
    fake_end and particle_end might be the same as the last shared hit, if the fake track or the particle track (or both) end
    '''
    CRITERION = end_criterion[0]
    LIMIT = end_criterion[1]
    REAL_CRITERION = real_criterion[0]
    REAL_LIMIT = real_criterion[1]

    if CRITERION == "nMissing" and REAL_CRITERION == "consecutive":
	if only_valid:
	    particle_hits = [hit for hit in particle.hits() if hit.isValidHit()]
	    particle_hit_indexes = [hit.index() for hit in particle.hits() if hit.isValidHit()]
	    track_hits = [hit for hit in fake.hits() if hit.isValidHit()]
	    track_hit_indexes = [hit.index() for hit in fake.hits() if hit.isValidHit()]
	else:
	    particle_hits = [hit for hit in particle.hits()]
	    particle_hit_indexes = [hit.index() for hit in particle.hits()]
	    track_hits = [hit for hit in fake.hits()]
	    track_hit_indexes = [hit.index() for hit in fake.hits()]

	#print particle_hit_indexes
	#print track_hit_indexes
	tolerance = LIMIT
	i = 0
	start_tolerance = 0
	last = particle_hits[0]
	particle_end = particle_hits[0]
	fake_end = particle_hits[0]
	# FIND THE END OF THE MATCHED 3 POINTS
	while i < len(track_hits):
	    #print track_hits[i].index()
	    if track_hit_indexes[i] in particle_hit_indexes:
		start_tolerance += 1
		#print start_tolerance
		if start_tolerance >= REAL_LIMIT:
		    #print "STARTED"
		    tolerance = LIMIT
		    i = particle_hit_indexes.index(track_hit_indexes[i])
		    last = particle_hits[i]
		    particle_end = particle_hits[min([i+1, len(particle_hits)-1])]
		    fake_end = track_hits[min(track_hit_indexes.index(particle_hit_indexes[i])+1, len(track_hits)-1)]
		    #fake_end = [hit for hit in track_hits if hit.index() == particle_hits[i].index()][0]
		    break
		i += 1    
	    else:
		start_tolerance = 0
		i += 1	
        # FIND THE END OF TRACKING AFTER MATCHED POINTS
	while tolerance >= 0 and i < len(particle_hits):
	    #print particle_hits[i].index()
	    #print i
	    if particle_hit_indexes[i] in track_hit_indexes:	
		tolerance = LIMIT
		last = particle_hits[i]
		particle_end = particle_hits[min([i+1, len(particle_hits)-1])]
		fake_end = track_hits[min(track_hit_indexes.index(particle_hit_indexes[i])+1, len(track_hits)-1)]
		#fake_end = [hit for hit in track_hits if hit.index() == particle_hits[i].index()][0]
	    elif not (particle_hits[i-1].layerStr() in particle_hits[i].layerStr() or particle_hits[i].layerStr() in particle_hits[i-1].layerStr()): # only missing layers are considered # double condition for invalid hits
		tolerance -= 1 
	    i += 1
	end_class = 0
	if last.index() == fake_end.index() and last.index() == particle_end.index():
	    end_class = 1
	elif last.index() == fake_end.index(): end_class = 2
	elif last.index() == particle_end.index(): end_class = 3
	elif last.layerStr() == particle_hits[-1].layerStr() and (len(particle_hits)-1 - i < 4): end_class = 3 #4 #3 # particle_end.layerStr()
	'''
	if tolerance >= LIMIT: # If the end of the particle was reached, last and fail are the same
	    last = particle_end
	    fake_end = particle_end	
	    end_class = 1
	
	print [[hit.index(), hit.layerStr()] for hit in track_hits]
	print [[hit.index(), hit.layerStr()] for hit in particle_hits]
	print i
	print last.index()
	print fake_end.index()
	print particle_end.index()
	print end_class
	print "*****"
	input()
	'''
	return last, fake_end, particle_end, end_class

def MatchPixelHits(fake, particle, real_criterion = ["consecutive", 3]):
    '''
    Finds how many shared pixelhits fake has with a particle, starting at the beginning
    of shared hits.
    '''
    CRITERION = real_criterion[0]
    LIMIT = real_criterion[1]

    if CRITERION == "consecutive":
	particle_hits = [hit for hit in particle.hits() if hit.isValidHit()]
	track_hits = [hit for hit in fake.hits() if hit.isValidHit()]
	particle_hit_indexes = [hit.index() for hit in particle.hits() if hit.isValidHit()]
	#print particle_hits
	#print track_hits
	i = 0
	start_tolerance = 0
	hit_candidates = []
	start_flag = False

	layer_strs = []
	nPix = 0

	while i <= len(track_hits)-1: 
	    if track_hits[i].index() in particle_hit_indexes:
		start_tolerance += 1
		hit_candidates.append(track_hits[i])
		if start_tolerance >= LIMIT:
		    start_flag = True 
		i += 1 
	    elif start_flag:
	        # End the iteration	
		break
	    else:
		hit_candidates = []
		start_tolerance = 0
		i += 1

        # Analyse the results, end the iteration
	for hit in hit_candidates:
	    if "Pix" in hit.layerStr():
		if hit.layerStr() not in layer_strs:
		    layer_strs.append(hit.layerStr())
		nPix += 1
	    else:
		break
	nPixLayers = len(layer_strs)
	''' Uncomment to analyse fakes having >= 4 pixelhits
	if nPixLayers >= 4: #print [hit.layerStr() for hit in hit_candidates]
	    if 'BPix1' in layer_strs and 'BPix2' in layer_strs:
		if 'BPix3' in layer_strs and 'FPix1' in layer_strs:
		    print "B3-F1"
		elif 'FPix1' in layer_strs and 'FPix2' in layer_strs:
		    print "F1-F2"
		else:
		    print "B1-B2 Other"
            else:
		print "Other"
	'''

	if start_tolerance == 0: # The end of the particle was reached
	    print("Match is not a real match :\\")
	if len(hit_candidates)<3 or not start_flag:
	    print("No hit candidates from a match")
	    print([hit.index() for hit in fake.hits() if hit.isValidHit()])
	    print(particle_hit_indexes)
	    print([hit.index() for hit in hit_candidates])
	    print(start_tolerance)
	    print(start_flag)
	    return -1, -1
	return nPix, nPixLayers

    if CRITERION == "NLay":
	particle_hits = [hit for hit in particle.hits() if hit.isValidHit()]
	track_hits = [hit for hit in fake.hits() if hit.isValidHit()]
	particle_hit_indexes = [hit.index() for hit in particle.hits() if hit.isValidHit()]
	#print particle_hits
	#print track_hits
	i = 0
	hit_candidates = []
	start_flag = False

	layer_strs = []
	nPix = 0

	while i <= len(track_hits)-1: 
	    if track_hits[i].index() in particle_hit_indexes:
		hit_candidates.append(track_hits[i])
		if "Pix" in hit.layerStr():
		    start_flag = True 
		i += 1 
	    elif start_flag:
	        # End the iteration	
		break
	    else:
		i += 1

        # Analyse the results, end the iteration
	for hit in hit_candidates:
	    if "Pix" in hit.layerStr():
		if hit.layerStr() not in layer_strs:
		    layer_strs.append(hit.layerStr())
		nPix += 1
	    else:
		break
	nPixLayers = len(layer_strs)
	
	return nPix, nPixLayers

    return -1, -1 # Something failed, unknown match criterion


def MaxSharedHits(fake, fraction = False):
   ''' Returns the maximum amount of shared hits which fake shares with some simulated particle. '''
   max_shared = 0
   max_frac = 0
   for particle in FindAssociatedParticles(fake): 
       shared, nTrack, nParticle = SharedHitFrac(fake, particle, 1)
       frac = 1.0*shared/nParticle
       if shared > max_shared:
	   max_shared = shared
	   max_frac = frac
   if fraction: return max_shared, max_frac
   return max_shared


##### MONITORING FUNCTIONS #####

def StopReason(track):
    ''' Converts track stop reason index to string '''
    reason = track.stopReason()
    if reason == 0:
	return "UNINITIALIZED"
    if reason == 1:
	return "MAX_HITS"
    if reason == 2:
	return "MAX_LOST_HITS"
    if reason == 3:
	return "MAX_CONSECUTIVE_LOST_HITS"
    if reason == 4:
	return "LOST_HIT_FRACTION"
    if reason == 5:
	return "MIN_PT"
    if reason == 6:
	return "CHARGE_SIGNIFICANCE"
    if reason == 7:
	return "LOOPER"
    if reason == 8:
	return "MAX_CCC_LOST_HITS"
    if reason == 9:
	return "NO_SEGMENTS_FOR_VALID_LAYERS"
    if reason == 10:
	return "SEED_EXTENSION"
    if reason == 255:
	return "NOT_STOPPED"
    else:
	return "UNDEFINED STOPPING REASON"


def PrintTrackInfo(track, fake = None, frac = 0, fake_info = None):
    ''' Prints info on the track. Called from PlotFakes method in graphics.py.  '''
    if isinstance(track, Track):
	if track.nMatchedTrackingParticles() == 0: # FAKE
	    print(str(track.index()) + ": FAKE \nSTOP REASON: " + StopReason(track))
	    print("Has " + str(track.nValid()) + " valid hits")
	    if fake_info:
		fake_info.Print()
	else: # RECONSTRUCTED
	    reco_str = str(track.index()) + ": RECOVERED "
	    for info in track.matchedTrackingParticleInfos():
		reco_str += str(info.trackingParticle().index()) + " " + str(info.shareFrac()) + "\nSTOP REASON: " + StopReason(track) # sharefrac()[0] old version
	    print(reco_str)
    else: # REAL
	print(str(track.index()) + ": REAL")
	if track.nMatchedTracks() == 0: print("NOT RECOVERED")
	elif track.nMatchedTracks() == 1: print("RECOVERED")
	else: print("RECOVERED " + str(track.nMatchedTracks()) + " TIMES")
	decaycount = 0
	for decay in track.decayVertices(): decaycount += 1
	if decaycount: print("DECAYED " + str(decaycount) + " TIMES")
    
    if fake: # tell the shared hit fraction compared to fake
	if frac:
	    num, div, npar = SharedHitFrac(fake, track, 1)
	    print("Fake share fraction: " + str(num) + " / " + str(div) + ", track has " + str(npar) + " hits")
	else: 
	    dec = SharedHitFrac(fake, track, 0)
	    print("Fake shares " + str(dec) + " fraction of hits with track")
	print("Shared hits from beginning: " + str(SharedHitsFromBeginning(track, fake, 10)))

    if isinstance(track, TrackingParticle):
	print("Parameters:")
	print("px  : " + str(track.px()) + "  py  : " + str(track.py()) + "  pz  : " + str(track.pz()))
	print("pt  : " + str(track.pca_pt()) + "  eta : " + str(track.pca_eta()) + "  phi : " + str(track.pca_phi()))
	print("dxy : " + str(track.pca_dxy()) + "  dz  : " + str(track.pca_dz()) + "  q   : " + str(track.q()) + "\n")
    else:
	print("Parameters:")
	print("px  : " + str(track.px()) + "  py  : " + str(track.py()) + "  pz  : " + str(track.pz()))
	print("pt  : " + str(track.pt()) + "  eta : " + str(track.eta()) + "  phi : " + str(track.phi()))
	print("dxy : " + str(track.dxy()) + "  dz  : " + str(track.dz()) + "  q   : " + str(track.q()) + "\n")


##### CLASSIFICATION #####

class FakeInfo(object):
    '''
    A storage and analysis class for a fake track.
    Construct this object with a fake track as a parameter to perform analysis on that fake track.
    The results can then be obtained from object attributes.
    '''
    def __init__(self, fake, real_criterion = ["consecutive", 3], end_criterion = ["nMissing", 1]):
	self.fake = fake
        self.index = fake.index()
	self.nHits = fake.nValid()
	self.nMatches = 0
	self.matches = []
	self.nReconstructed = 0 # Number of reconstructed matched particles
	self.nDecays = 0 # Number of decayed matched particles

	start = next(iter(fake.hits()))
	self.start_loc = [start.x(), start.y(), start.z()]

	self.stop_reason = fake.stopReason()
        # Classify the fake
	self.FindMatches(fake, real_criterion, end_criterion)
	self.fake_class, self.fake_class_str = self.Classify()

    def FindMatches(self, fake, real_criterion, end_criterion):
	''' Finds matches for the fake track. '''
	matched_particles = MatchedParticles(fake, real_criterion)
	self.nMatches = len(matched_particles)
        self.nIncludedDecayParticles = 0
	self.decayLinks = []

	if matched_particles:
	    for particle in matched_particles:
		self.matches.append(MatchInfo(particle, fake, real_criterion, end_criterion))
	    for match in self.matches:
		if match.nReconstructed > 0:
		    self.nReconstructed += 1
		if match.nDecays > 0:
		    self.nDecays += 1
		if match.nDaughters > 0:
		    for another_match in self.matches:
			if another_match.parentVtxIndex in match.decayVtxIndexes:
			    self.nIncludedDecayParticles += 1
			    self.decayLinks.append([match.index, another_match.index])
			
            # The following 3 lines: define how many hits does an unmatched fake have with any particles at maximum (-1 for having matches)
	    self.unmatched_max_shared = -1
	else:
	    self.unmatched_max_shared = MaxSharedHits(fake)
	self.max_shared, self.max_frac = MaxSharedHits(fake, True)

	if not self.nMatches and not IsUnmatched(fake):
	    self.nMatches = -1 # UNCLASSIFIED

    def Classify(self):
        ''' Classify the fake after analysis '''
	if self.nMatches == -1:
	    return -1, "UNCLASSIFIED"
	if self.nMatches == 0:
	    return 0, "UNMATCHED" 
	if self.nMatches >= 2:
	    if self.decayLinks:
		return 21, "MERGED DECAY TRACK MATCHES"
	    elif self.nReconstructed >= self.nMatches:
	        return 22, "MULTIPLE ALL RECONSTRUCTED MATCHES"
	    elif self.nReconstructed > 0:
	        return 23, "MULTIPLE PARTIALLY RECONSTRUCTED MATCHES"
	    else:
		return 20, "MULTIPLE UNRECONSTRUCTED MATCHES"
	if self.nMatches == 1:
	    if self.nReconstructed > 0 and self.nDecays > 0:
		return 11, "RECONSTRUCTED AND DECAYED MATCH"
	    if self.nReconstructed > 0:
		return 12, "RECONSTRUCTED MATCH"
	    if self.nDecays > 0:
		return 13, "DECAYED MATCH"
	    else:
		return 10, "MATCH"

    def Print(self):
	''' Prints fake track classification info with matched particle infos. '''
        print("CLASS: " + str(self.fake_class) + " WITH " + str(self.nMatches) + " MATCHES")
	print("Has " + str(self.nIncludedDecayParticles) + " included decay particles, with links: " + str(self.decayLinks))
	for match in self.matches: match.Print()

class MatchInfo(object):
    ''' A storage and analysis class for TrackingParticles matched to a fake track. '''
    def __init__(self, match, fake, real_criterion = ["consecutive", 3], end_criterion = ["nMissing", 2]):
	''' match = the matched TrackingParticle '''
	self.index = match.index()
	self.particle = match

        # Find out the daughter particles
	self.decayVertices = [[vtx.x(), vtx.y(), vtx.z()] for vtx in match.decayVertices()]
	self.nDecays = len(self.decayVertices)
	self.decayVtxIndexes = [vtx.index() for vtx in match.decayVertices()]
        
	self.shared_hits, un, needed = SharedHitFrac(fake, match, 1)

	self.daughterIndexes = []
	for vtx in match.decayVertices():
	    for particle in vtx.daughterTrackingParticles():
		self.daughterIndexes.append(particle.index())
	self.nDaughters = len(self.daughterIndexes)
	vtx = match.parentVertex()
	self.parentVtx = [vtx.x(), vtx.y(), vtx.z()]
	self.parentVtxIndex = vtx.index()

        self.nReconstructed = match.nMatchedTracks()
        # more reconstruction analysis here

        # Check how many pixelhits or pixellayers in match
	self.nPix, self.nPixLayers = MatchPixelHits(fake, match, real_criterion)

	# Check where tracking ended	
	last, fake_end, particle_end, self.end_class = FindEndOfTracking(fake, match, end_criterion)
	if last.isValidHit(): self.last_loc = [last.x(), last.y(), last.z()]
	else: self.last_loc = [0,0,0]
	if fake_end.isValidHit(): self.fake_end_loc = [fake_end.x(), fake_end.y(), fake_end.z()]
	else: self.fake_end_loc = [0,0,0]
	if particle_end.isValidHit(): self.particle_end_loc = [particle_end.x(), particle_end.y(), particle_end.z()]
        else: self.particle_end_loc = [0,0,0]

	self.last_detId = last.detId()
	self.fake_end_detId = fake_end.detId()
	self.particle_end_detId = particle_end.detId()	

        self.particle_pdgId = match.pdgId()
	self.particle_pt = match.pt()

	if isinstance(last, GluedHit):
	    self.last_str = last.monoHit().layerStr()
	else:
	    self.last_str = last.layerStr()
	if isinstance(fake_end, GluedHit):
	    self.fake_end_str = fake_end.monoHit().layerStr()
	else:
	    self.fake_end_str = fake_end.layerStr()
	if isinstance(particle_end, GluedHit):
	    self.particle_end_str = particle_end.monoHit().layerStr()
	else:
	    self.particle_end_str = particle_end.layerStr()


    def Print(self):
	''' Prints match info. '''
        print("Match " + str(self.index) + ": nReconstructed " + str(self.nReconstructed) +\
	 ", daughters: " + str(self.daughterIndexes) +\
	  ", tracking failed in " + self.last_str + ", to " + self.particle_end_str + ", wrong hit in " + self.fake_end_str)

##### STORAGE CLASSES #####

class EndInfo(object):
    ''' Storage class for end of tracking information for a matched fake '''
    def __init__(self):
	self.last = []
	self.fake_end = []
	self.particle_end = []
	self.last_str = ""
	self.fake_end_str = ""
	self.particle_end_str = ""
	self.fake_class = -1
	self.end_class = -1
	self.last_detid = -1
	self.fake_end_detid = -1
	self.particle_end_detid = -1
	self.particle_pdgId = -1
	self.particle_pt = -1
 
class ResolutionData(object):
    ''' Storage class for matched fake track resolutions '''
    def __init__(self):
	self.pt_rec = []
	self.pt_sim = []
        self.pt_delta = []
	self.eta = []
	self.Lambda = []
	self.cotTheta = []
	self.phi = []
	self.dxy = []
	self.dz = []
	self.fake_class = []
    
    def __iter__(self):
        for i in range(len(self.pt_rec)):
	    yield ResolutionItem(self.pt_rec[i], self.pt_sim[i], self.pt_delta[i], self.eta[i], self.Lambda[i], self.cotTheta[i], self.phi[i], self.dxy[i], self.dz[i], self.fake_class[i])

class ResolutionItem(object):
    ''' A storage class for ResolutionData iteration '''
    def __init__(self, pt_rec, pt_sim, pt_delta, eta, Lambda, cotTheta, phi, dxy, dz, fake_class):
	self.pt_rec = pt_rec
	self.pt_sim = pt_sim
        self.pt_delta = pt_delta
	self.eta = eta
	self.Lambda = Lambda
	self.cotTheta = cotTheta
	self.phi = phi
	self.dxy = dxy
	self.dz = dz
	self.fake_class = fake_class

##### ANALYSIS WORKROUNDS #####

def ClassifyEventFakes(ntuple_file, nEvents = 100, return_fakes = False, real_criterion = ["consecutive", 3]):
    '''
    Classifies all fakes in the first nEvents events in the ntuple_file (TrackingNtuple object).
    Returns a dictionary of class items, with class index as a key and number of fakes in the class as the value.
    '''
    i = 0
    results = {class_item: 0 for class_item in classes} # This line has issues with the python version, worked with Python 2.17.12. Comment something to compile with older version
    fake_list = []
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake, real_criterion)
	    results[info.fake_class] += 1

	    if return_fakes:
		fake_list.append(info)
        i += 1
	if i >= nEvents:
	    break
    
    if return_fakes:
	return results, fake_list
    return results

def Calculate_MaxMatchedHits(ntuple_file, nEvents = 100):
    '''
    Calculates the maximum number of shared hits for the fakes in the first nEvents events in the ntuple_file (TrackingNtuple object).
    Returns a list of these maximum numbers for each analysed fake track.
    '''
    i = 0
    results = []
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    res_temp = 0
	    for particle in FindAssociatedParticles(fake):
		shared, par, nAll = SharedHitFrac(fake, particle, 1)
	        if shared > res_temp:
		    res_temp = shared
	    results.append(res_temp)

	i += 1
	if i >= nEvents:
	    break

    return results
    

def Calculate_MaxMatchedConsecutiveHits(ntuple_file, nEvents = 100, frac = False):
    '''
    Calculates the maximum number of CONSECUTIVE (with respect to fake track) shared hits for the fakes in the first nEvents events in the ntuple_file (TrackingNtuple object).
    Returns a list of these maximum numbers for each analysed fake track.
    '''
    i = 0
    results = []
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    res_temp = 0
	    #fake_temp = fake
	    for particle in FindAssociatedParticles(fake):
		tolerance_mask = SharedHitsFromBeginning(particle, fake, fake.nValid())
		diff = [abs(tolerance_mask[j+1] - tolerance_mask[j]) for j in range(len(tolerance_mask)-1)]	        
		if frac:
		    if diff and 1.0*max(diff)/fake.nValid() > res_temp:
			res_temp = 1.0*max(diff)/fake.nValid()
		    elif 1.0*tolerance_mask[0]/fake.nValid() > res_temp:
			res_temp = 1.0*tolerance_mask[0]/fake.nValid()
		else:
		    if diff and max(diff) > res_temp:
			res_temp = max(diff)
		    elif tolerance_mask[0] > res_temp:
			res_temp = tolerance_mask[0] 
	    ''' Uncomment to debug
	    if frac:	
		if 1.0*res_temp/fake.nValid() > 0.75:
		    print 1.0*res_temp/fake.nValid()
		    print res_temp
		    print [hit.index() for hit in fake.hits()]
		    print [hit.index() for hit in particle.hits()]
		    print tolerance_mask
		    print diff
		    print fake.nValid()
		    print particle.nValid()
		
		results.append(1.0*res_temp/fake.nValid())
	    '''
	    results.append(res_temp)

	i += 1
	if i >= nEvents:
	    break

    return results

def Calculate_MaxMatchedHits_RealTracks(ntuple_file, nEvents = 100):
    '''
    Similar as Calculate_MaxMatchedHits, but for true tracks.
    '''
    i = 0
    results = []
    for event in ntuple_file:
	for track in event.tracks():
	    if track.nMatchedTrackingParticles() >= 1:
		res_temp = 0
		for info in track.matchedTrackingParticleInfos():
		    particle = info.trackingParticle()
		    shared, par, nAll = SharedHitFrac(track, particle, 1)
		    if shared > res_temp:
			res_temp = shared
		results.append(res_temp)

	i += 1
	if i >= nEvents:
	    break

    return results   

def Calculate_MaxMatchedConsecutiveHits_RealTracks(ntuple_file, nEvents = 100):
    '''
    Similar as Calculate_MaxMatchedConsecutiveHits, but for true tracks.
    '''
    i = 0
    results = []
    for event in ntuple_file:
	for track in event.tracks():
	    if track.nMatchedTrackingParticles() >= 1:
		res_temp = 0
		for info in track.matchedTrackingParticleInfos():
		    particle = info.trackingParticle()
		    tolerance_mask = SharedHitsFromBeginning(particle, track, track.nValid())
		    diff = [abs(tolerance_mask[j+1] - tolerance_mask[j]) for j in range(len(tolerance_mask)-1)]	        
		    if diff and max(diff) > res_temp:
			res_temp = max(diff)
		    elif tolerance_mask[0] > res_temp:
			res_temp = tolerance_mask[0]
		    #print res_temp
		results.append(res_temp)

	i += 1
	if i >= nEvents:
	    break

    return results

def Calculate_MatchPixelHits(ntuple_file, nEvents = 100, fake_mask = []):
    '''
    Calculate the amount of pixelhits and pixellayers in the first consectutive shared fake hits, starting from the matched three hits.
    Returns a list including numbers of pixelhits (for each fake) in nPix_data, and a list including numbers of pixellayers (for each fake) in nLay_data
    '''
    i = 0
    nPix_data = []
    nLay_data = []
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake)
	    if not fake_mask or info.fake_class in fake_mask:
		for match in info.matches:
		    nPix_data.append(match.nPix)
		    nLay_data.append(match.nPixLayers)
 
        i += 1
	if i >= nEvents:
	    break
    
    return nPix_data, nLay_data

def Calculate_UnmatchedSharedHits(ntuple_file, nEvents = 100, fake_mask = [0, -1]):
    '''
    Calculates the amount of shared hits between fake tracks and some TrackingParticles for the "no match" and "loose match" classes.
    Returns a list of the results for each fake.
    '''
    results = []
    i = 0
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake)
	    if info.fake_class in fake_mask:
		results.append(info.unmatched_max_shared)
	i += 1
	if i >= nEvents:
	    break
    
    return results

def Calculate_SharedHits(ntuple_file, nEvents = 100, mask = []):
    '''
    Calculates the amount of shared hits between fake tracks and some TrackingParticle.
    For filtering only some of the classes to the data, put the class indexes inside the mask list.
    Returns a list of the results for each fake.
    '''
    hits_shared = []
    hit_fractions = []
    i = 0
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake)
	    if not mask or info.fake_class in mask:
		hits_shared.append(info.max_shared)
		hit_fractions.append(info.max_frac)
	i += 1
	if i >= nEvents:
	    break
    
    return hits_shared, hit_fractions

def Calculate_SharedHits_RealTracks(ntuple_file, nEvents = 100):
    '''
    Calculates the amount of shared hits between true tracks and associated TrackingParticles.
    Returns a list of the results for each true track.
    '''
    hits_shared = []
    hit_fractions = []
    i = 0
    for event in ntuple_file:	
	for track in event.tracks():
	    if track.nMatchedTrackingParticles() >= 1:
		info = FakeInfo(track)	
		hits_shared.append(info.max_shared)
		hit_fractions.append(info.max_frac)
	i += 1
	if i >= nEvents:
	    break
    
    return hits_shared, hit_fractions

def Calculate_IndludedDecayHitFractions(ntuple_file, nEvents = 100):
    '''
    Calculates the hit fractions (from fake) for the daughter and parent particles from a decay,
    to analyse the multiple matches class "fake includes a decay interaction".
    Returns: daughter_frac = fraction of daughter particle hits from fake
             parent_frac = fraction of parent particle hits from fake
	     total_frac = the sum of these daughter and parent fractions
    '''
    daughter_frac = []
    parent_frac = []
    total_frac = []
    i = 0
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake)
	    if info.fake_class == 21:
		if len(info.decayLinks) >= 2: print("Double or more decays!!!11")
		for link in info.decayLinks:
		    par_ind = link[0]
		    dau_ind = link[1]
		    for match in info.matches:
			if match.index == par_ind: par_hits = match.shared_hits
			if match.index == dau_ind: dau_hits = match.shared_hits
		    fake_hits = info.nHits

		    parent_frac.append(1.0*par_hits/fake_hits)
		    daughter_frac.append(1.0*dau_hits/fake_hits)
		    total_frac.append(1.0*(dau_hits + par_hits)/fake_hits)
	i += 1
	if i >= nEvents:
	    break
    
    return daughter_frac, parent_frac, total_frac

def Get_EndOfTrackingPoints(ntuple_file, nEvents = 100, mask = []):
    '''
    Performs analysis and returns a list of EndInfos containing information on end of tracking for each fake track.
    '''
    end_infos = []
    i = 0
    for event in ntuple_file:
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake)
	    if not mask or info.fake_class in mask:
		for match in info.matches:
		    end = EndInfo()
		    end.last = match.last_loc
		    end.fake_end = match.fake_end_loc
		    end.particle_end = match.particle_end_loc
		    end.end_class = match.end_class
		    end.fake_class = info.fake_class
		    end.last_str = match.last_str
		    end.fake_end_str = match.fake_end_str
		    end.particle_end_str = match.particle_end_str
		    end.last_detId = match.last_detId
		    end.fake_end_detId = match.fake_end_detId
		    end.particle_end_detId = match.particle_end_detId
		    end.particle_pdgId = match.particle_pdgId
		    end.particle_pt = match.particle_pt

		    end_infos.append(end)
	i += 1
	if i >= nEvents:
	    break
    
    return end_infos

def Get_EndOfTrackingPointsReal(ntuple_file, nEvents = 100):
    '''
    Performs analysis and returns a list of EndInfos containing information on end of tracking for each TRUE track.
    '''
    end_infos = []
    i = 0
    for event in ntuple_file:
	trues = FindTrues(event)
	for true in trues:
	    for info in true.matchedTrackingParticleInfos():
		particle = info.trackingParticle()
		last, track_end, particle_end, end_class = FindEndOfTracking(true, particle)
		end = EndInfo()
		if last.isValidHit(): end.last = [last.x(), last.y(), last.z()]
		if track_end.isValidHit(): end.fake_end = [track_end.x(), track_end.y(), track_end.z()]
		if particle_end.isValidHit(): end.particle_end = [particle_end.x(), particle_end.y(), particle_end.z()]
		end.end_class = end_class
		end.fake_class = -1
		end.last_str = last.layerStr()
		end.fake_end_str = track_end.layerStr()
		end.particle_end_str = particle_end.layerStr()
		end.last_detId = last.detId()
		end.fake_end_detId = track_end.detId()
		end.particle_end_detId = particle_end.detId()
		end.particle_pdgId = particle.pdgId()
		end.particle_pt = particle.pt()

		end_infos.append(end)
	i += 1
	if i >= nEvents:
	    break
    
    return end_infos

def Save_Normalisation_Coefficients(ntuple_file):
    '''
    Calculates normalisation coefficients for detector layers, which would normalise the amount of
    hits per layer with respect to the total amount of TrackingParticle hits in the layer.
    Saves the results in a dictionary with the layer indexes as the keys and the normalised coefficients as the values.
    The resulted dictionary is saved to file "normalisation_coefficients.dmp" with pickle library.
    '''
    norm_c = copy(layer_data_tmp)

    print(sum([val for ind, val in six.iteritems(norm_c)]))
    for event in ntuple_file:
	print(event.entry()+1)
	for particle in event.trackingParticles():
	    for hit in particle.hits():
		if hit.isValidHit():
		    norm_c[layer_names_rev[hit.layerStr()]] += 1
    norm_sum = sum([val for ind, val in six.iteritems(norm_c)])
    print(norm_sum)
    print(norm_c)
    for i, c in six.iteritems(norm_c):
	norm_c[i] = 1.0*c/norm_sum
    #normalisation = [1.0*c/norm_sum for c in norm_c]
    print("normalisation_coefficients.dmp")
    print(norm_c)
    
    norm_file = open("normalisation_coefficients.dmp",'w')
    pickle.dump(norm_c, norm_file)
    norm_file.close()

def Get_Normalisation_Coefficients():
    '''
    Loads the detector layer normalisation coefficients from a file created by the previous Save_Normalisation_Coefficients function.
    '''
    norm_file = open("normalisation_coefficients.dmp",'r')
    coefficients = pickle.load(norm_file)
    norm_file.close()
    return coefficients

def Resolution_Analysis_Fakes(ntuple_file, nEvents = 100, real_criterion = ["consecutive", 3]):
    '''
    Performs analysis and returns a ResolutionData object containing the track parameter resolutions for fakes and matched particles.
    '''
    res = ResolutionData()
    i = 0
    for event in ntuple_file:
	#print event.entry() + 1
	fakes = FindFakes(event)
	for fake in fakes:
	    info = FakeInfo(fake, real_criterion)
	    for match in info.matches:
		par = match.particle
		if par.pca_pt() == 0: continue

		res.pt_rec.append(fake.pt())
		res.pt_sim.append(par.pca_pt())
		res.pt_delta.append(fake.pt() - par.pca_pt())
		res.eta.append(fake.eta() - par.pca_eta())
		res.Lambda.append(getattr(fake, 'lambda')() - par.pca_lambda())
		res.cotTheta.append(fake.cotTheta() - par.pca_cotTheta())
		res.phi.append(fake.phi() - par.pca_phi())
		res.dxy.append(fake.dxy() - par.pca_dxy())
		res.dz.append(fake.dz() - par.pca_dz())

		res.fake_class.append(info.fake_class)
	
	i += 1
	if i >= nEvents:
	    break
    return res

def Resolution_Analysis_Trues(ntuple_file, nEvents = 100):
    '''
    Performs analysis and returns a ResolutionData object containing the track parameter resolutions for true tracks and associated particles.
    '''
    res = ResolutionData()
    i = 0
    for event in ntuple_file:
	#print event.entry() + 1
	trues = FindTrues(event)
	for true in trues:
	    for info in true.matchedTrackingParticleInfos():
		par = info.trackingParticle()
		if par.pca_pt() == 0: continue

		res.pt_rec.append(true.pt())
		res.pt_sim.append(par.pca_pt())
		res.pt_delta.append(true.pt() - par.pca_pt())
		res.eta.append(true.eta() - par.pca_eta())
		res.Lambda.append(getattr(true, 'lambda')() - par.pca_lambda())
		res.cotTheta.append(true.cotTheta() - par.pca_cotTheta())
		res.phi.append(true.phi() - par.pca_phi())
		res.dxy.append(true.dxy() - par.pca_dxy())
		res.dz.append(true.dz() - par.pca_dz())

		res.fake_class.append(-1)
	
	i += 1
	if i >= nEvents:
	    break
    return res
    
def Calculate_AmountOfFakes(ntuple_file):
    ''' Calculated the amount of fakes in the data. '''
    n = 0
    for event in ntuple_file:
	n += len(FindFakes(event))
    return n

def Analyse_EOT_Error(end_list, bpix3_mask = False, detId_mask = "all", pt_mask = False):
    '''
    Analyses the distance between the fake and particle hits following the last shared hit in the end of tracking.
    bpix3_mask filters out all fakes except those which have the last shared hit in BPix3 layer and the following particle hit in TIB1 layer.
    detId_mask filters with respect to the fact if the following fake and particle hits have the same or different detector ID.
    pt_mask filters out the fakes with the particle pt < 0.7.

    Returns a list of the errors for each fake.
    '''
    error = []
    for end in end_list:
	if end.fake_end != [0,0,0] and end.particle_end != [0,0,0]:
	    if pt_mask and end.particle_pt < 0.7: continue

	    if bpix3_mask and end.end_class == 0 and end.last_str == "BPix3" and end.particle_end_str == "TIB1":
		if detId_mask == "same" and end.fake_end_detId == end.particle_end_detId:
		    error.append(Distance(end.fake_end[0:2], end.particle_end[0:2]))
		elif detId_mask == "different" and end.fake_end_detId != end.particle_end_detId:
		    error.append(Distance(end.fake_end, end.particle_end))
		elif detId_mask == "all":
		    error.append(Distance(end.fake_end, end.particle_end))
		if error and error[-1] == 0.0: print("Error is 0.0?!")
		if error and error[-1] > 10: print(str(end.fake_end_detId) + " and " + str(end.particle_end_detId) + ": " + str(error[-1]) + " z: " + str(end.fake_end[2]) + " " + str(end.particle_end[2]))
	    elif not bpix3_mask:
		if detId_mask == "same" and end.fake_end_detId == end.particle_end_detId:
		    error.append(Distance(end.fake_end[0:2], end.particle_end[0:2]))
		elif detId_mask == "different" and end.fake_end_detId != end.particle_end_detId:
		    error.append(Distance(end.fake_end, end.particle_end))
		elif detId_mask == "all":
		    error.append(Distance(end.fake_end, end.particle_end))

    print(sum(error)/len(error))
    return error

def EndOfTrackingDetectorInfo(end_list, end_mask = [0], BPix3mask = False):
    '''
    Prints info about how many end of trackings have the particle hit and the fake hit in the same detector module.
    BPix3_mask filters out all fakes except those which have the last shared hit in BPix3 layer and the following particle hit in TIB1 layer.
    '''
    data = []
    for end in end_list:
	if (not end_mask or end.end_class in end_mask) and (not BPix3mask or (end.last_str == "BPix3" and end.particle_end_str == "TIB1")):
	    if end.particle_end_detId == end.fake_end_detId:
		data.append(1)
	    else:
		data.append(0)
    print("Same detector id between fake end and particle end: " + str(sum(data)))
    print("Different detector id: " + str(len(data)-sum(data)))

def Analyse_EOT_ParticleDoubleHit(end_list, layer = "BPix3", end_mask = [0,4]):
    '''
    Prints info on how many fakes have an end of tracking with both last shared hit and the following particle hit in the same detector layer.
    On default, this is analysed on BPix3 layer and the fakes with the end classes 0 or 4 (4 is merged to 3 in some analysis).
    '''
    doubles = 0
    all_particles = 0
    for end in end_list:
	if (not end_mask or end.end_class in end_mask) and (layer in end.last_str):
	    if end.last_str == end.particle_end_str:
		doubles += 1
		#print end.end_class
	    all_particles += 1
    
    print("In layer " + layer + " there are " + str(doubles) + " end of trackings out of " + str(all_particles) + " (" + str(100.0*doubles/all_particles) + ") which have a double hit in the EOT layer")

def TrackPt(ntuple_file, nEvents = 100):
    '''
    Returns two lists on fake track and true track pt values.
    '''
    fake_pts = []
    true_pts = []
    
    i = 0
    for event in ntuple_file:	
	print("Event: " + str(i))
	for track in event.tracks():
	    if track.nMatchedTrackingParticles() == 0:
		fake_pts.append(track.pt())
            else:
		true_pts.append(track.pt())

	i += 1
	if i >= nEvents:
	    break
    
    return fake_pts, true_pts

