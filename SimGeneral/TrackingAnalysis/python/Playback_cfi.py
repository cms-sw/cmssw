import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Services_cff import *
del RandomNumberGeneratorService.generator
RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
# from SimGeneral.MixingModule.mixNoPU_cfi import *
from Configuration.StandardSequences.Mixing156BxLumiPileUp_cff import *
mix.playback = cms.untracked.bool(True)

