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
    maxHit = cms.double(80.5),
    nintHit = cms.int32(81),
    #
    minPu = cms.double(-0.5),
    maxPu = cms.double(259.5),
    nintPu = cms.int32(130),
    #
    minLayers = cms.double(-0.5),
    maxLayers = cms.double(25.5),
    nintLayers = cms.int32(26),
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
    dxyDzZoom = cms.double(25),
    #
    # dE/dx
    minDeDx = cms.double(0.),
    maxDeDx = cms.double(10.),
    nintDeDx = cms.int32(40),
    #
    # TP originating vertical position
    minVertpos = cms.double(1e-2),
    maxVertpos = cms.double(100),
    nintVertpos = cms.int32(40),
    useLogVertpos = cms.untracked.bool(True),
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
    # dR_jet
    mindrj = cms.double(0.001),
    maxdrj = cms.double(0.1),
    nintdrj = cms.int32(50),
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
    nintTracks = cms.int32(200),

    # PV z coordinate (to be kept in synch with PrimaryVertexAnalyzer4PUSlimmed)
    minPVz = cms.double(-60),
    maxPVz = cms.double(60),
    nintPVz = cms.int32(120),

    # MVA distributions
    minMVA = cms.double(-1),
    maxMVA = cms.double(1),
    nintMVA = cms.int32(100),

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

    seedingLayerSets = cms.vstring(),

    doMTDPlots = cms.untracked.bool(False), # meant to be switch on in Phase2 workflows
)

def _modifyForPhase1(pset):
    pset.minEta = -3
    pset.maxEta = 3
    pset.nintEta = 60
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
#phase1Pixel.toModify(MTVHistoProducerAlgoForTrackerBlock, dict(minEta = -3, maxEta = 3, nintEta = 60) )
phase1Pixel.toModify(MTVHistoProducerAlgoForTrackerBlock, _modifyForPhase1)

def _modifyForPhase2(pset):
    pset.minEta = -4.5
    pset.maxEta = 4.5
    pset.nintEta = 90
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
#phase2_tracker.toModify(MTVHistoProducerAlgoForTrackerBlock, dict(minEta = -4.5, maxEta = 4.5, nintEta = 90) )
phase2_tracker.toModify(MTVHistoProducerAlgoForTrackerBlock, _modifyForPhase2)

def _modifyForPhase2wMTD(pset):
    pset.doMTDPlots = True
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
#phase2_timing_layer.toModify(MTVHistoProducerAlgoForTrackerBlock, dict(doMTDPlots = True) )
phase2_timing_layer.toModify(MTVHistoProducerAlgoForTrackerBlock, _modifyForPhase2wMTD)

