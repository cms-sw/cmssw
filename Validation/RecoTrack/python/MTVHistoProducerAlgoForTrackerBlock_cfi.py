import FWCore.ParameterSet.Config as cms
from Validation.RecoTrack.TrackingParticleSelectionsForEfficiency_cff import *
from Validation.RecoTrack.GenParticleSelectionsForEfficiency_cff import *

MTVHistoProducerAlgoForTrackerBlock = cms.PSet(
    ### tp selectors for efficiency
    generalTpSelector             = generalTpSelectorBlock,
    TpSelectorForEfficiencyVsEta  = TpSelectorForEfficiencyVsEtaBlock,
    TpSelectorForEfficiencyVsPhi  = TpSelectorForEfficiencyVsPhiBlock,
    TpSelectorForEfficiencyVsPt   = TpSelectorForEfficiencyVsPtBlock,
    TpSelectorForEfficiencyVsVTXR = TpSelectorForEfficiencyVsVTXRBlock,
    TpSelectorForEfficiencyVsVTXZ = TpSelectorForEfficiencyVsVTXZBlock,

    ### gp selectors for efficiency
    generalGpSelector             = generalGpSelectorBlock,
    GpSelectorForEfficiencyVsEta  = GpSelectorForEfficiencyVsEtaBlock,
    GpSelectorForEfficiencyVsPhi  = GpSelectorForEfficiencyVsPhiBlock,
    GpSelectorForEfficiencyVsPt   = GpSelectorForEfficiencyVsPtBlock,
    GpSelectorForEfficiencyVsVTXR = GpSelectorForEfficiencyVsVTXRBlock,
    GpSelectorForEfficiencyVsVTXZ = GpSelectorForEfficiencyVsVTXZBlock,

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
    maxHit = cms.double(40.5),
    nintHit = cms.int32(41),
    #                               
    minPu = cms.double(-0.5),                            
    maxPu = cms.double(199.5),
    nintPu = cms.int32(100),
    #
    minLayers = cms.double(-0.5),                            
    maxLayers = cms.double(15.5),
    nintLayers = cms.int32(16),
    #
    minPhi = cms.double(-3.1416),
    maxPhi = cms.double(3.1416),
    nintPhi = cms.int32(36),
    #
    minDxy = cms.double(-25),
    maxDxy = cms.double(25),
    nintDxy = cms.int32(100),
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
    maxVertpos = cms.double(60),
    nintVertpos = cms.int32(60),
    #
    # TP originating z position
    minZpos = cms.double(-30),
    maxZpos = cms.double(30),
    nintZpos = cms.int32(60),                               
    #
    # dR
    mindr = cms.double(0.001),
    maxdr = cms.double(1),
    nintdr = cms.int32(100),
    #
    # chi2/ndof
    minChi2 = cms.double(0),
    maxChi2 = cms.double(20),
    nintChi2 = cms.int32(40),

    # Pileup vertices
    minVertcount = cms.double(-0.5),
    maxVertcount = cms.double(160.5),
    nintVertcount = cms.int32(161),

    minTracks = cms.double(0),
    maxTracks = cms.double(2000),
    nintTracks = cms.int32(100),
    #
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


    maxDzpvCumulative = cms.double(0.6),
    nintDzpvCumulative = cms.int32(240),

    maxDzpvsigCumulative = cms.double(10),
    nintDzpvsigCumulative = cms.int32(200),
)
