# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

import CommonTools.ParticleFlow.pfPileUp_cfi as config
pfPileUpForMuonRadiationFilter = config.pfPileUp.clone(
    PFCandidates = cms.InputTag('particleFlow')
)    

import CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi as config
pfNoPileUpForMuonRadiationFilter = config.pfNoPileUp.clone(
    topCollection = cms.InputTag('pfPileUpForMuonRadiationFilter'),
    bottomCollection = pfPileUpForMuonRadiationFilter.PFCandidates
)

muonRadiationFilter = cms.EDFilter("MuonRadiationFilter",
    srcSelectedMuons = cms.InputTag(''), # CV: replaced in embeddingCustomizeAll.py
    srcPFCandsNoPU = cms.InputTag('pfNoPileUpForMuonRadiationFilter'),
    srcPFCandsPU = cms.InputTag('pfPileUpForMuonRadiationFilter'),
    # CV: the following configuration parameter values
    #     are taken from the Higgs -> ZZ -> 4l analysis (kindly provided by Mike Bachtis)
    minPtLow = cms.double(3.),
    dRlowPt = cms.double(0.07),
    addCaloEnECALlowPt = cms.bool(True),
    applyMassWindowSelectionLowPt = cms.bool(True),
    minPtHigh = cms.double(5.),
    dRhighPt = cms.double(0.5),
    addCaloEnECALhighPt = cms.bool(True),
    applyMassWindowSelectionHighPt = cms.bool(True),
    dRvetoCone = cms.double(1.e-3),
    dRisoCone = cms.double(0.4),
    maxRelIso = cms.double(1.0),
    maxMass = cms.double(105.),
    invert = cms.bool(False),                                     
    filter = cms.bool(True),
    verbosity = cms.int32(0)
)

muonRadiationFilterSequence = cms.Sequence(
    pfPileUpForMuonRadiationFilter
   * pfNoPileUpForMuonRadiationFilter
   * muonRadiationFilter
)
