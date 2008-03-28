import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderTIFTOBTEC = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderTIFTOBTEC.cosmicSeeds = 'cosmicseedfinderTIFTOBTEC'

