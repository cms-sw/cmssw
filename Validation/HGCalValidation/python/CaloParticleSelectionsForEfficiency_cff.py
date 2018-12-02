import FWCore.ParameterSet.Config as cms

generalCpSelectorBlock = cms.PSet(
    ptMinCP = cms.double(0.005),
    ptMaxCP = cms.double(1e100),
    minRapidityCP = cms.double(-4.5),
    maxRapidityCP = cms.double(4.5),
    chargedOnlyCP = cms.bool(True),
    stableOnlyCP = cms.bool(False),
    pdgIdCP = cms.vint32(11, -11, 13, -13, 22, 111, 211, -211, 321, -321),
    #--signal only means no PU particles
    signalOnlyCP = cms.bool(True),
    #--intime only means no OOT PU particles
    intimeOnlyCP = cms.bool(True),
    #The total number of rechits
    minHitCP = cms.int32(0)
)

CpSelectorForEfficiencyVsEtaBlock = generalCpSelectorBlock.clone()
CpSelectorForEfficiencyVsPhiBlock = generalCpSelectorBlock.clone()
CpSelectorForEfficiencyVsPtBlock = generalCpSelectorBlock.clone(ptMin = 0.050 )

