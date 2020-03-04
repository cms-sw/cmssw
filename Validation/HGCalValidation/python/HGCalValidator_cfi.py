import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.CaloParticleSelectionForEfficiency_cfi import *
from Validation.HGCalValidation.HGVHistoProducerAlgoBlock_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hgcalValidator = DQMEDAnalyzer(
    "HGCalValidator",

    ### general settings ###
    # selection of CP for evaluation of efficiency #
    CaloParticleSelectionForEfficiency,

    ### reco input configuration ###
    #2dlayerclusters, pfclusters, multiclusters
    label_lcl = cms.InputTag("hgcalLayerClusters"),
    label_mcl = cms.VInputTag(
      cms.InputTag("multiClustersFromTrackstersTrk", "TrkMultiClustersFromTracksterByCA"),
      cms.InputTag("multiClustersFromTrackstersEM", "MultiClustersFromTracksterByCA"),
      cms.InputTag("multiClustersFromTrackstersHAD", "MultiClustersFromTracksterByCA"),
      cms.InputTag("multiClustersFromTrackstersMerge", "MultiClustersFromTracksterByCA")),

    #General info on layers etc.
    SaveGeneralInfo = cms.untracked.bool(True),
    #CaloParticle related plots
    doCaloParticlePlots = cms.untracked.bool(True),
    #Layer Cluster related plots
    dolayerclustersPlots = cms.untracked.bool(True),
    #Multi Cluster related plots
    domulticlustersPlots = cms.untracked.bool(False),

    #The cumulative material budget in front of each layer. To be more specific, it
    #is the material budget just in front of the active material (not including it).
    #This file is created using the official material budget code.
    cummatbudinxo = cms.FileInPath('Validation/HGCalValidation/data/D41.cumulative.xo'),

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

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalValidator,
    label_cp_effic = "mixData:MergedCaloTruth",
    label_cp_fake = "mixData:MergedCaloTruth"
)

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(hgcalValidator, totallayers_to_monitor = cms.int32(50))
