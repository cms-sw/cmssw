import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCSimProducers.hgcDigiProducers_cff import *

hgcDigiSequence = cms.Sequence( simHGCEEdigis + simHGCHEfrontDigis + simHGCHEbackDigis )

