import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.CaloParticleSelectionForEfficiency_cfi import *
from Validation.HGCalValidation.HGVHistoProducerAlgoBlock_cfi import *

hgcalValidator = cms.EDAnalyzer(
    "HGCalValidator",

    ### general settings ###
    # selection of CP for evaluation of efficiency #
    CaloParticleSelectionForEfficiency,

    ### reco input configuration ###
    #2dlayerclusters, pfclusters, multiclusters
    #label = cms.VInputTag(cms.InputTag("hgcalLayerClusters"), cms.InputTag("particleFlowClusterHGCal"), cms.InputTag("hgcalMultiClusters") ),
    label = cms.VInputTag(cms.InputTag("hgcalLayerClusters")),
    
    #CaloParticle related plots
    doCaloParticlePlots = cms.untracked.bool(True),
    dolayerclustersPlots = cms.untracked.bool(True),

    ### sim input configuration ###
    label_cp_effic = cms.InputTag("mix","MergedCaloTruth"),
    label_cp_fake = cms.InputTag("mix","MergedCaloTruth"),

    simVertices = cms.InputTag("g4SimHits"),
    
    #Total number of layers of HGCal that we want to monitor
    #Could get this also from HGCalImagingAlgo::maxlayer but better to get it from here
    totallayers_to_monitor = cms.int32(52),
    #Thicknesses we want to monitor. -1 is for scintillator
    thicknesses_to_monitor = cms.vint32(120,200,300,-1),

    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoProducerAlgoBlock = HGVHistoProducerAlgoBlock,

    ### output configuration
    dirName = cms.string('HGCAL/HGCalValidator/')

)

