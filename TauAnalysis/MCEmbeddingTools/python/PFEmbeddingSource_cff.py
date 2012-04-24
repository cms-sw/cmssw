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
newSource.ZTauTau.minVisibleTransverseMomentum = cms.untracked.string("")



#source = cms.Source("EmptySource")

source = cms.Source("PoolSource",
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring('file:/tmp/fruboes/Zmumu/patLayer1_fromAOD_PF2PAT_full.root')
)


if os.path.exists("/storage/6/zeise/temp/goldenZmumuEvents_RAW_RECO_9_1_EzB.root"):
	source.fileNames=cms.untracked.vstring("file:/storage/6/zeise/temp/goldenZmumuEvents_RAW_RECO_9_1_EzB.root")
if os.path.exists("/scratch/scratch0/tfruboes/2011.04.Embedding/CMSSW_4_1_4/DATA/goldenZmumu500.root"):
	source.fileNames=cms.untracked.vstring("file:/scratch/scratch0/tfruboes/2011.04.Embedding/CMSSW_4_1_4/DATA/goldenZmumu500.root")

filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
    target = cms.untracked.int32(1),
    src = cms.untracked.InputTag("generator","","HLT2")
)

#adaptedMuonsFromDiTauCands = cms.EDProducer("CompositePtrCandidateT1T2MEtAdapter",
#    diTau  = cms.untracked.InputTag("zMuMuCandsMuEta"),
#    pfCands = cms.untracked.InputTag("particleFlow","")
#)

#inputColl = cms.InputTag("adaptedMuonsFromDiTauCands","zMusExtracted")
#inputColl = cms.InputTag("goldenZmumuCandidatesGe2IsoMuons")
inputColl = cms.InputTag("goldenZmumuCandidatesGe0IsoMuons")

# Removes input muons from tracks and PF candidate collections
removedInputMuons = cms.EDProducer('ZmumuPFEmbedder',
    tracks = cms.InputTag("generalTracks"),
    selectedMuons = inputColl,
    keepMuonTrack = cms.bool(False),
    useCombinedCandidate = cms.untracked.bool(True),
)

generator = newSource.clone()
generator.src = inputColl

#ProductionFilterSequence = cms.Sequence(adaptedMuonsFromDiTauCands*removedInputMuons*generator*filterEmptyEv)
# MuonCaloCleaner

#
from TrackingTools.TrackAssociator.default_cfi import *
anaDeposits = cms.EDProducer('MuonCaloCleaner',
   TrackAssociatorParameterBlock,
   selectedMuons = inputColl,
   useCombinedCandidate = cms.untracked.bool(True)
)
 

ProductionFilterSequence = cms.Sequence(removedInputMuons*generator*filterEmptyEv*anaDeposits)


#ProductionFilterSequence = cms.Sequence(generator)
