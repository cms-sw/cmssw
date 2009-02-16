import FWCore.ParameterSet.Config as cms

# File: caloTowers.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for caloTowers.
towerSchemeBAnalyzer = cms.EDAnalyzer(
    "CaloTowerAnalyzer",
    Debug = cms.bool(False),
    #CaloTowersLabel = cms.InputTag("caloTowers"),
    CaloTowersLabel = cms.InputTag("towerMaker"),
    OutputFile = cms.untracked.string('CaloTowerAnalyzer_SchemeB.root'),
    DumpGeometry = cms.bool(False),
    GeometryFile = cms.untracked.string('CaloTowerAnalyzer_geometry.dat'),
    FineBinning = cms.untracked.bool(False)
    )

towerOptAnalyzer = cms.EDAnalyzer(
    "CaloTowerAnalyzer",
    Debug = cms.bool(False),
    # CaloTowersLabel = cms.InputTag("caloTowersOpt"),
    CaloTowersLabel = cms.InputTag("calotoweroptmaker"),
    OutputFile = cms.untracked.string('CaloTowerAnalyzer_Opt.root'),
    DumpGeometry = cms.bool(False),
    GeometryFile = cms.untracked.string('CaloTowerAnalyzer_geometry.dat'),
    FineBinning = cms.untracked.bool(False)
)

