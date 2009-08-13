import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
matchVtx = cms.EDFilter("MixEvtVtxGenerator",
                        signalLabel = cms.InputTag("hiSignal"), 
                        heavyIonLabel = cms.InputTag("generator")
                        )
