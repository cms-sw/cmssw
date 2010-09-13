import FWCore.ParameterSet.Config as cms
from Validation.RecoTrack.TrackingParticleSelectionsForEfficiency_cff import *

MTVHistoProducerAlgoForTrackerBlock = cms.PSet(
    ComponentName = cms.string('MTVHistoProducerAlgoForTracker'),

    ### tp selectors for efficiency
    generalTpSelector             = generalTpSelectorBlock,
    TpSelectorForEfficiencyVsEta  = TpSelectorForEfficiencyVsEtaBlock,
    TpSelectorForEfficiencyVsPhi  = TpSelectorForEfficiencyVsPhiBlock,
    TpSelectorForEfficiencyVsPt   = TpSelectorForEfficiencyVsPtBlock,
    TpSelectorForEfficiencyVsVTXR = TpSelectorForEfficiencyVsVTXRBlock,
    TpSelectorForEfficiencyVsVTXZ = TpSelectorForEfficiencyVsVTXZBlock,

    # to be added here all the other histogram settings

    #
    minEta = cms.double(-2.5),
    maxEta = cms.double(2.5),
    nintEta = cms.int32(50),
    useFabsEta = cms.bool(False),
    #
    minPt = cms.double(0.1),
    maxPt = cms.double(1000),
    nintPt = cms.int32(40),
    useInvPt = cms.bool(False),
    useLogPt=cms.untracked.bool(True),
    #                               
    minHit = cms.double(-0.5),                            
    maxHit = cms.double(34.5),
    nintHit = cms.int32(35),
    #
    minPhi = cms.double(-3.15),
    maxPhi = cms.double(3.15),
    nintPhi = cms.int32(36),
    #
    minDxy = cms.double(-40),
    maxDxy = cms.double(40),
    nintDxy = cms.int32(80),
    #
    minDz = cms.double(-30),
    maxDz = cms.double(30),
    nintDz = cms.int32(60),
    #
    # dE/dx
    minDeDx = cms.double(0.),
    maxDeDx = cms.double(10.),
    nintDeDx = cms.int32(40),
    #
    # TP originating vertical position
    minVertpos = cms.double(0),
    maxVertpos = cms.double(80),
    nintVertpos = cms.int32(40),
    #
    # TP originating z position
    minZpos = cms.double(-30),
    maxZpos = cms.double(30),
    nintZpos = cms.int32(60),                               

    #parameters for resolution plots
    ptRes_rangeMin = cms.double(-0.1),
    ptRes_rangeMax = cms.double(0.1),
    ptRes_nbin = cms.int32(100),                                   

    phiRes_rangeMin = cms.double(-0.01),
    phiRes_rangeMax = cms.double(0.01),
    phiRes_nbin = cms.int32(300),                                   

    cotThetaRes_rangeMin = cms.double(-0.02),
    cotThetaRes_rangeMax = cms.double(+0.02),
    cotThetaRes_nbin = cms.int32(300),                                   

    dxyRes_rangeMin = cms.double(-0.1),
    dxyRes_rangeMax = cms.double(0.1),
    dxyRes_nbin = cms.int32(500),                                   

    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double(+0.05),
    dzRes_nbin = cms.int32(150),                                   

)
