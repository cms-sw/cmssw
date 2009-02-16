import FWCore.ParameterSet.Config as cms

# File: RecHits.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for ECAL and HCAL RecHits.
ECALAnalyzer = cms.EDAnalyzer(
    "ECALRecHitAnalyzer",
    OutputFile = cms.untracked.string('ECALRecHitAnalyzer_data.root'),
    EBRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    DumpGeometry = cms.bool(False),
    EERecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    Debug = cms.bool(False),
    GeometryFile = cms.untracked.string('ECALRecHitAnalyzer_geometry.dat'),
    FineBinning = cms.untracked.bool(False)
)

HCALAnalyzer = cms.EDAnalyzer(
    "HCALRecHitAnalyzer",
    OutputFile = cms.untracked.string('HCALRecHitAnalyzer_data.root'),
    HORecHitsLabel = cms.InputTag("horeco"),
    DumpGeometry = cms.bool(False),
    HBHERecHitsLabel = cms.InputTag("hbhereco"),
    Debug = cms.bool(False),
    HFRecHitsLabel = cms.InputTag("hfreco"),
    GeometryFile = cms.untracked.string('HCALRecHitAnalyzer_geometry.dat'),
    FineBinning = cms.untracked.bool(False)                          
)


