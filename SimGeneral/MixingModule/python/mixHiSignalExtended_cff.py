import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixHiSignal_cff import *
hiSignalG4SimHits.Generator.HepMCProductLabel = 'hiSignalLHCTransport' 

##################################################################################
# Transport to forward detectors
from SimTransport.HectorProducer.HectorTransportZDC_cfi import *
hiSignalLHCTransport = LHCTransport.clone()
hiSignalLHCTransport.HepMCProductLabel = 'hiSignal'

hiSignalSequence = cms.Sequence(cms.SequencePlaceholder("hiSignal")*matchVtx*hiGenParticles*hiSignalLHCTransport*hiSignalG4SimHits)
