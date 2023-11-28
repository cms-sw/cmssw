import FWCore.ParameterSet.Config as cms

mtdTruth = cms.PSet(
    accumulatorType = cms.string('MtdTruthAccumulator'),
    MinEnergy = cms.double(0.5),
    MaxPseudoRapidity = cms.double(5.0),
    premixStage1 = cms.bool(False),
    maximumPreviousBunchCrossing = cms.uint32(0),
    maximumSubsequentBunchCrossing = cms.uint32(0),
    bunchspace = cms.uint32(25), #ns

    simHitCollections = cms.PSet(
        mtdCollections = cms.VInputTag(
           cms.InputTag('g4SimHits','FastTimerHitsBarrel'),
           cms.InputTag('g4SimHits','FastTimerHitsEndcap')
       ),
    ),
    simTrackCollection = cms.InputTag('g4SimHits'),
    simVertexCollection = cms.InputTag('g4SimHits'),
    genParticleCollection = cms.InputTag('genParticles'),
    allowDifferentSimHitProcesses = cms.bool(False),
    HepMCProductLabel = cms.InputTag('generatorSmeared'),
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(mtdTruth, premixStage1 = True)
