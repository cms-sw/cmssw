import FWCore.ParameterSet.Config as cms

process = cms.Process("PremixStage1")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(10)
)

process.load("IOMC.RandomEngine.IOMC_cff")

from SimGeneral.MixingModule.mix_flat_0_10_cfi import mix as _mix
process.mix = _mix.clone(
    input = dict(
        nbPileupEvents = dict(
            probFunctionVariable = [0,1,2,3],
            probValue = [0.25, 0.25, 0.25, 0.25]
        ),
        fileNames = ["file:testFakeMinBias.root"]
    ),
    digitizers = cms.PSet(),
    mixObjects = cms.PSet(),
    minBunch = -1,
    maxBunch = 1,
)

process.load("SimGeneral.PileupInformation.AddPileupSummary_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testPremixStage1.root'),
)

process.t = cms.Task(
    process.mix,
    process.addPileupInfo
)
process.p = cms.Path(process.t)
process.e = cms.EndPath(process.out)
