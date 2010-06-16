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
newSource.algorithm = "ZTauTau"
newSource.ZTauTau.TauolaOptions.InputCards.mdtau = cms.int32(216)
newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.double(0)


source = cms.Source("PoolSource",
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring('file:/tmp/fruboes/Zmumu/patLayer1_fromAOD_PF2PAT_full.root')
)

filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
    minEvents = cms.untracked.int32(1),
    target =  cms.untracked.int32(1) 
)

adaptedMuonsFromDiTauCands = cms.EDProducer("CompositePtrCandidateT1T2MEtAdapter",
    diTau  = cms.untracked.InputTag("zMuMuCandsMuEta"),
    pfCands = cms.untracked.InputTag("particleFlow","")
)

dimuonsGlobal = cms.EDProducer('ZmumuPFEmbedder',
    tracks = cms.InputTag("generalTracks"),
    selectedMuons = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted"),
    keepMuonTrack = cms.bool(False)
)

generator = newSource.clone()
generator.src = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted")

ProductionFilterSequence = cms.Sequence(adaptedMuonsFromDiTauCands*dimuonsGlobal*generator*filterEmptyEv)
