import FWCore.ParameterSet.Config as cms
import copy

from Configuration.StandardSequences.MagneticField_cff import *

import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi

# Track quality parameters
trackQuality = cms.PSet(
	hitAssociator = cms.PSet(
		associatePixel = cms.bool(True),
		associateStrip = cms.bool(True),
		associateRecoTracks = cms.bool(True),
		ROUList = copy.deepcopy(SimTracker.TrackAssociation.TrackAssociatorByHits_cfi.TrackAssociatorByHits.ROUList)
	)
)
