import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Services_cff import *
del RandomNumberGeneratorService.generator
RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
# from SimGeneral.MixingModule.mixNoPU_cfi import *
from SimGeneral.MixingModule.StageA156Bx_cfi import *
mix.playback = cms.untracked.bool(True)

