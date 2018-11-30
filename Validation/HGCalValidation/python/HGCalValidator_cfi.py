import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGVHistoProducerAlgoBlock_cfi import *

hgcalValidator = cms.EDAnalyzer(
    "HGCalValidator",

    ### reco input configuration ###
    #2dlayerclusters, pfclusters, multiclusters
    #label = cms.VInputTag(cms.InputTag("hgcalLayerClusters"), cms.InputTag("particleFlowClusterHGCal"), cms.InputTag("hgcalMultiClusters") ),
    label = cms.VInputTag(cms.InputTag("hgcalLayerClusters")),

    dolayerclustersPlots = cms.untracked.bool(True),

    ### sim input configuration ###
    label_cp_effic = cms.InputTag("mix","MergedCaloTruth"),
    label_cp_fake = cms.InputTag("mix","MergedCaloTruth"),

    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoProducerAlgoBlock = HGVHistoProducerAlgoBlock,

    ### output configuration
    dirName = cms.string('HGCAL/HGCalValidator/')

)

