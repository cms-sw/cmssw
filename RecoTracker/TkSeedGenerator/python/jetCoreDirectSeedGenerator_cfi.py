import FWCore.ParameterSet.Config as cms

jetCoreDirectSeedGenerator = cms.EDProducer("JetCoreDirectSeedGenerator",
    vertices=    cms.InputTag("offlinePrimaryVertices"),
    pixelClusters=    cms.InputTag("siPixelClustersPreSplitting"),
    cores= cms.InputTag("jetsForCoreTracking"),
    ptMin= cms.double(300),
    deltaR= cms.double(0.1),
    chargeFractionMin= cms.double(18000.0),
    centralMIPCharge= cms.double(2),
    pixelCPE= cms.string( "PixelCPEGeneric" ),
    weightFile= cms.FileInPath("RecoTracker/TkSeedGenerator/data/JetCoreDirectSeedGenerator_TrainedModel.pb"),
    inputTensorName= cms.vstring(["input_1","input_2","input_3"]),
    outputTensorName= cms.vstring(["output_node0","output_node1"]),
    nThreads= cms.uint32(1),
    singleThreadPool=  cms.string("no_threads"),
    probThr = cms.double(0.99),
)
