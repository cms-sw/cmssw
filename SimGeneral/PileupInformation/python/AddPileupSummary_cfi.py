import FWCore.ParameterSet.Config as cms

#pileupSummary = cms.EDProducer("PileupInformation",
addPileupInfo = cms.EDProducer("PileupInformation",
    isPreMixed = cms.bool(False),
    TrackingParticlesLabel = cms.InputTag('mergedtruth'),
    PileupMixingLabel = cms.InputTag('mix'),
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    volumeZ = cms.double(3000.0),
    pTcut_1 = cms.double(0.1),
    pTcut_2 = cms.double(0.5),                               
    doTrackTruth = cms.untracked.bool(False),
    saveVtxTimes = cms.bool(False)
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( addPileupInfo, saveVtxTimes = cms.bool(True) )

#addPileupInfo = cms.Sequence(pileupSummary)

# I'd like to move the contents of AddPileupSummaryPremixed_cfi here,
# but first I have to figure out what to do with the import of that
# cfi in DataMixerDataOnSim_cff (without any apparent connection to
# premxing)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
from SimGeneral.PileupInformation.AddPileupSummaryPreMixed_cfi import addPileupInfo as _addPileupInfoPreMixed
premix_stage2.toReplaceWith(addPileupInfo, _addPileupInfoPreMixed)
