import FWCore.ParameterSet.Config as cms

generalTpSelectorBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(False),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5)
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    generalTpSelectorBlock.stableOnly = True

TpSelectorForEfficiencyVsEtaBlock = generalTpSelectorBlock.clone()
TpSelectorForEfficiencyVsPhiBlock = generalTpSelectorBlock.clone()
TpSelectorForEfficiencyVsPtBlock = generalTpSelectorBlock.clone(ptMin = 0.050 )
TpSelectorForEfficiencyVsVTXRBlock = generalTpSelectorBlock.clone(tip = 60.0)
TpSelectorForEfficiencyVsVTXZBlock = generalTpSelectorBlock.clone()
