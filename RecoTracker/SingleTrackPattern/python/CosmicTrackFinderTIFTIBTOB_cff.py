import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderTIFTIBTOB = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderTIFTIBTOB.cosmicSeeds = 'cosmicseedfinderTIFTIBTOB'

