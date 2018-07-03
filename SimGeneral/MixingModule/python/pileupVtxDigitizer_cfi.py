import FWCore.ParameterSet.Config as cms

pileupVtxDigitizer = cms.PSet(
    accumulatorType = cms.string("PileupVertexAccumulator"),
    hitsProducer = cms.string('generator'),
    vtxTag = cms.InputTag("generatorSmeared"),
    vtxFallbackTag = cms.InputTag("generator"),
    makeDigiSimLinks = cms.untracked.bool(False),
    saveVtxTimes = cms.bool(False))

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( pileupVtxDigitizer, saveVtxTimes = cms.bool(True) )
