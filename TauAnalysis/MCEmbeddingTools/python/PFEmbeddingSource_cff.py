# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
  
from Configuration.Generator.PythiaUESettings_cfi import *

TauolaNoPolar = cms.PSet(
  UseTauolaPolarization = cms.bool(False)
)
TauolaPolar = cms.PSet(
  UseTauolaPolarization = cms.bool(True)
)

from TauAnalysis.MCEmbeddingTools.MCParticleReplacer_cfi import *
generator.algorithm = "Ztautau"
generator.pluginType = "ParticleReplacerZtautau"
generator.src = cms.InputTag("") # CV: replaced in embeddingCustomizeAll.py
generator.Ztautau.TauolaOptions.InputCards.mdtau = cms.int32(0)
generator.Ztautau.minVisibleTransverseMomentum = cms.untracked.string("")

filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
  target = cms.untracked.int32(1),
  src = cms.untracked.InputTag("generator", "", "HLT2")
)

cleanedGeneralTracks = cms.EDProducer("MuonTrackCleaner",
    selectedMuons = cms.InputTag(""), # CV: replaced in embeddingCustomizeAll.py
    tracks = cms.VInputTag("generalTracks"),
    dRmatch = cms.double(3.e-1),
    removeDuplicates = cms.bool(True),
    type = cms.string("inner tracks"),
    verbosity = cms.int32(0)                                           
)
cleanedParticleFlow = cms.EDProducer("MuonPFCandidateCleaner",
    selectedMuons = cms.InputTag(""), # CV: replaced in embeddingCustomizeAll.py
    pfCands = cms.InputTag("particleFlow"),
    dRmatch = cms.double(3.e-1),
    removeDuplicates = cms.bool(True),                          
    verbosity = cms.int32(0)                                           
)
 
ProductionFilterSequence = cms.Sequence(
  cleanedGeneralTracks
 + cleanedParticleFlow
 + generator
 + filterEmptyEv
)
