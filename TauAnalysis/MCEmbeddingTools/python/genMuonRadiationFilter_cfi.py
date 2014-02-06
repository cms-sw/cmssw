# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms

genMuonRadiationFilter = cms.EDFilter("GenMuonRadiationFilter",
    srcGenParticles = cms.InputTag('genParticles::SIM'),
    # CV: set Pt thresholds and cone sizes to values
    #     matching muon -> muon + photon radiation filter running on reconstruction level
    #    (cf. TauAnalysis/MCEmbeddingTools/python/muonRadiationFilter_cfi.py)                                
    minPtLow = cms.double(2.),
    dRlowPt = cms.double(0.07),
    minPtHigh = cms.double(4.),
    dRhighPt = cms.double(0.5),
    invert = cms.bool(False),
    filter = cms.bool(False),
    verbosity = cms.int32(0)
)
