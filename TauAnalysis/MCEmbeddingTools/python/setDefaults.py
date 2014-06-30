# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

def setDefaults(process):
 
  # Define configuration parameter default values
  if not hasattr(process, "customization_options"):
    process.customization_options = cms.PSet(
      parseCommandLine             = cms.bool(False),    # enable reading of configuration parameter values by parsing command-line
      isMC                         = cms.bool(True),     # set to true for MC/false for data
      ZmumuCollection              = cms.InputTag('goldenZmumuCandidatesGe2IsoMuons'), # collection of selected Z->mumu candidates
      inputProcessRECO             = cms.string("RECO"), # instanceLabel to be used for retrieving collections of reconstructed objects reconstructed in original Z->mumu event
      inputProcessSIM              = cms.string("HLT"),  # instanceLabel to be used for retrieving collections of generator level objects in original Z->mumu event
                                                         # CV: use inputProcessRECO = inputProcessSIM = 'HLT' for test samples privately producer with cmsDriver;
                                                         #     use inputProcessRECO = 'RECO', inputProcessSIM = 'SIM' for samples from official production
      cleaningMode                 = cms.string("DEDX"), # option for muon calo. cleaning: 'DEDX'=muon energy loss expected on average, 'PF'=actual energy deposits associated to PFMuon
      muonCaloCleaningSF           = cms.double(1.0),    # option for subtracting too much (muonCaloSF > 1.0) or too few (muonCaloSF < 1.0) calorimeter energy around muon,
                                                         # too be used for studies of systematic uncertainties
      muonTrackCleaningMode        = cms.int32(2),       # option for muon track cleaning: 1=remove at most one track/charged PFCandidate matching muon,
                                                         # 2=remove all tracks/charged PFCandidates matched to muon in dR
      mdtau                        = cms.int32(0),       # mdtau value passed to TAUOLA: 0=no tau decay mode selection,
      useTauolaPolarization        = cms.bool(False),    # disable tau polarization effects in TAUOLA, weight events by weights computed by TauSpinner instead
      transformationMode           = cms.int32(1),       # transformation mode: 0=mumu->mumu, 1=mumu->tautau
      rfRotationAngle              = cms.double(0.),     # rotation angle around Z-boson direction, used when replacing muons by simulated taus
      rfMirror                     = cms.bool(True),  # mirror at the Z-boson / proton plane
      embeddingMode                = cms.string("RH"),   # embedding mode: 'PF'=particle flow embedding, 'RH'=recHit embedding
      replaceGenOrRecMuonMomenta   = cms.string("rec"),  # take momenta of generated tau leptons from: 'rec'=reconstructed muons, 'gen'=generator level muons
      applyMuonRadiationCorrection = cms.string(""),     # should I correct the momementa of replaced muons for muon -> muon + photon radiation ?
                                                         # (""=no correction, "pythia"/"photos"=correction is applied using PYTHIA/PHOTOS)
      minVisibleTransverseMomentum = cms.string(""),     # generator level cut on visible transverse momentum (typeN:pT,[...];[...])
      useJson                      = cms.bool(False),    # should I enable event selection by JSON file ?
      overrideBeamSpot             = cms.bool(False),    # should I override beamspot in globaltag ?
      applyZmumuSkim               = cms.bool(False),     # should I apply the Z->mumu event selection cuts ?
      applyMuonRadiationFilter     = cms.bool(False),    # should I apply the filter to reject events with muon -> muon + photon radiation ?
      disableCaloNoise             = cms.bool(True),     # should I disable the simulation of calorimeter noise when simulating the detector response for the embedded taus ?
      applyRochesterMuonCorr       = cms.bool(True),     # should I apply muon momentum corrections determined by the Rochester group (documented in AN-12/298) ?
      skipCaloRecHitMixing         = cms.bool(False),    # disable mixing of calorimeter recHit collections
                                                         # WARNING: needs to be set to false for production samples !!
      muonMixingMode               = cms.int32(1)        # option for mixing hits and tracks in muon detectors: 1=mix recHits, run muon track segment and track reconstruction on mixed recHit collection;
                                                         # mix recHits, but mix tracks instead of rerunning track reconstruction on mixed recHit collection; 3=mix tracks, do not mix recHits
                                                         # WARNING: options 2 and 3 not thoroughly tested yet !!
    )
