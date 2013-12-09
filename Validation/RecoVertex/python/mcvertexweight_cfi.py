import FWCore.ParameterSet.Config as cms

mcvertexweight = cms.EDFilter("MCVerticesWeight",
                              pileupSummaryCollection = cms.InputTag("addPileupInfo"),
                              mcTruthCollection = cms.InputTag("generator"),
                              weighterConfig = cms.PSet(
    initSigma = cms.double(6.26),
    initMean = cms.double(0.4145),
    finalSigma = cms.double(5.2),
    useMainVertex = cms.bool(True)
    )
                              )

