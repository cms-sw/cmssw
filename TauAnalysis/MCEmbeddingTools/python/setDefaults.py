# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

def setDefaults(process):
 
  # Define configuration parameter default values
  if not hasattr(process, "customization_options"):
    process.customization_options = cms.PSet(
      parseCommandLine             = cms.bool(False),          # enable reading of configuration parameter values by parsing command-line
      ZmumuCollection              = cms.InputTag('goldenZmumuCandidatesGe2IsoMuons'), # collection of selected Z->mumu candidates
      inputProcess                 = cms.string("RECO"),       # instanceLabel to be used for retrieving collections of objects reconstructed in original Z->mumu event
                                                               # CV: use'HLT' for test samples privately producer with cmsDriver, 'RECO' for samples from official production
      cleaningMode                 = cms.string("PF"),         # option for muon calo. cleaning: 'DEDX'=muon energy loss expected on average, 'PF'=actual energy deposits associated to PFMuon
      mdtau                        = cms.int32(0),             # mdtau value passed to TAUOLA: 0=no tau decay mode selection, 
      transformationMode           = cms.untracked.int32(1),   # transformation mode: 0=mumu->mumu, 1=mumu->tautau
      embeddingMode                = cms.string("RH"),         # embedding mode: 'PF'=particle flow embedding, 'RH'=recHit embedding
      replaceGenOrRecMuonMomenta   = cms.string("rec"),        # take momenta of generated tau leptons from: 'rec'=reconstructed muons, 'gen'=generator level muons 
      minVisibleTransverseMomentum = cms.untracked.string(""), # generator level cut on visible transverse momentum (typeN:pT,[...];[...])
      useJson                      = cms.bool(False),          # should I enable event selection by JSON file ?
      overrideBeamSpot             = cms.bool(False),          # should I override beamspot in globaltag ?
      applyZmumuSkim               = cms.bool(True),           # should I apply the Z->mumu event selection cuts ?
      applyMuonRadiationFilter     = cms.bool(True)            # should I apply the filter to reject events with muon -> muon + photon radiation ?
    )
