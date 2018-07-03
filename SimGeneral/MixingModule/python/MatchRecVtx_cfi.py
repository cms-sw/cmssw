# match GEN vertex to RECO vertex for DataMixer, including pixel offset:
# https://hypernews.cern.ch/HyperNews/CMS/get/physics-validation/301.html

import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
matchRecVtx = cms.EDProducer("MixEvtVtxGenerator",
                             signalLabel = cms.InputTag("generator","unsmeared"),
                             heavyIonLabel = cms.InputTag("offlinePrimaryVertices"),
                             useRecVertex = cms.bool(True),
                             vtxOffset = cms.vdouble(0.1475, 0.3782, 0.4847)
                             )
