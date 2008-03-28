import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicTrackFinder_cfi import *
cosmictrackfinderTIF = copy.deepcopy(cosmictrackfinder)
cosmictrackfinderTIF.cosmicSeeds = 'cosmicseedfinderTIF'

