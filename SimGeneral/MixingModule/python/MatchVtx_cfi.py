import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
matchVtx = cms.EDProducer("MixEvtVtxGenerator",
                        signalLabel = cms.InputTag("hiSignal"), 
                        heavyIonLabel = cms.InputTag("generator","unsmeared")
                        )
# foo bar baz
# P8kSOlLyN72py
# s6d2VzGs2hLHS
