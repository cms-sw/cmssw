import FWCore.ParameterSet.Config as cms

pileupVtxDigitizer = cms.PSet(
    accumulatorType = cms.string("PileupVertexAccumulator"),
    hitsProducer = cms.string('generator'),
    vtxTag = cms.InputTag("generatorSmeared"),
    vtxFallbackTag = cms.InputTag("generator"),
    makeDigiSimLinks = cms.untracked.bool(False))

