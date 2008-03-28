import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderTIFTOB = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderTIFTOB.cosmicSeeds = 'cosmicseedfinderTIFTOB'

