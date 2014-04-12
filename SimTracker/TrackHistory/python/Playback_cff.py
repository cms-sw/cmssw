# Playback file
import FWCore.ParameterSet.Config as cms

# Playback
from Configuration.StandardSequences.Services_cff import *
del RandomNumberGeneratorService.generator
RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
from SimGeneral.MixingModule.mixNoPU_cfi import *
mix.playback = cms.untracked.bool(True)

# TrackingTruth
from Configuration.StandardSequences.Digi_cff import *
from Configuration.StandardSequences.RawToDigi_cff import *
from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *

# Tracking truth and SiStrip(Pixel)DigiSimLinks
trackingTruth = cms.Sequence(mix * doAllDigi * trackingParticles)

# Reconstruction
playback = cms.Sequence(RawToDigi * trackingTruth)

