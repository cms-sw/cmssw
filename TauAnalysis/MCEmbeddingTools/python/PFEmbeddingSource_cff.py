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
generator.algorithm = "ZTauTau"
generator.src = cms.InputTag("") # CV: replaced in embeddingCustomizeAll.py
generator.ZTauTau.TauolaOptions.InputCards.mdtau = cms.int32(0)
generator.ZTauTau.minVisibleTransverseMomentum = cms.untracked.string("")

filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
  target = cms.untracked.int32(1),
  src = cms.untracked.InputTag("generator", "", "HLT2")
)

# Removes input muons from tracks and PF candidate collections
removedInputMuons = cms.EDProducer('ZmumuPFEmbedder',
  tracks = cms.InputTag("generalTracks"),
  trajectories = cms.InputTag("generalTracks"),                                 
  pfCands = cms.InputTag("particleFlow"),				   
  selectedMuons = cms.InputTag("") # CV: replaced in embeddingCustomizeAll.py
)

from TrackingTools.TrackAssociator.default_cfi import *
anaDeposits = cms.EDProducer('MuonCaloCleaner',
  TrackAssociatorParameterBlock,
  selectedMuons = cms.InputTag(""), # CV: replaced in embeddingCustomizeAll.py
  useCombinedCandidate = cms.untracked.bool(True)
)
 
ProductionFilterSequence = cms.Sequence(
  removedInputMuons
 * generator
 * filterEmptyEv
 * anaDeposits
)
