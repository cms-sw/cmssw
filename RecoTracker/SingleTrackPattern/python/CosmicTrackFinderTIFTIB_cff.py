import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderTIFTIB = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderTIFTIB.cosmicSeeds = 'cosmicseedfinderTIFTIB'
cosmictrackfinderTIFTIB.MinHits = 3

