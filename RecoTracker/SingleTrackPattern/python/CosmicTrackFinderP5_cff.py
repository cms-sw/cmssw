import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderP5 = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderP5.GeometricStructure = 'STANDARD'
cosmictrackfinderP5.cosmicSeeds = 'cosmicseedfinderP5'

