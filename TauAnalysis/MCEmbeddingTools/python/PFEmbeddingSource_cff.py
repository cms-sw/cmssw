# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
import os
  
from Configuration.Generator.PythiaUESettings_cfi import *

TauolaNoPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(False)
)
TauolaPolar = cms.PSet(
   UseTauolaPolarization = cms.bool(True)
)


from TauAnalysis.MCEmbeddingTools.MCParticleReplacer_cfi import *
newSource.algorithm = "ZTauTau"
newSource.ZTauTau.TauolaOptions.InputCards.mdtau = cms.int32(0)
newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.double(0)


source = cms.Source("PoolSource",
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring('file:/tmp/fruboes/Zmumu/patLayer1_fromAOD_PF2PAT_full.root')
)
if os.path.exists("/storage/6/zeise/temp/goldenZmumuEvents_RAW_RECO_9_1_EzB.root"):
	source.fileNames=cms.untracked.vstring("file:/storage/6/zeise/temp/goldenZmumuEvents_RAW_RECO_9_1_EzB.root")

filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
    minEvents = cms.untracked.int32(1),
    target =  cms.untracked.int32(1) 
)

#adaptedMuonsFromDiTauCands = cms.EDProducer("CompositePtrCandidateT1T2MEtAdapter",
#    diTau  = cms.untracked.InputTag("zMuMuCandsMuEta"),
#    pfCands = cms.untracked.InputTag("particleFlow","")
#)

#inputColl = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted")
inputColl = cms.InputTag("goldenZmumuCandidatesGe1IsoMuons")

dimuonsGlobal = cms.EDProducer('ZmumuPFEmbedder',
    tracks = cms.InputTag("generalTracks"),
    selectedMuons = inputColl,
    keepMuonTrack = cms.bool(False),
    useCombinedCandidate = cms.untracked.bool(True),
)

generator = newSource.clone()
generator.src = inputColl

#ProductionFilterSequence = cms.Sequence(adaptedMuonsFromDiTauCands*dimuonsGlobal*generator*filterEmptyEv)
ProductionFilterSequence = cms.Sequence(dimuonsGlobal*generator*filterEmptyEv)
