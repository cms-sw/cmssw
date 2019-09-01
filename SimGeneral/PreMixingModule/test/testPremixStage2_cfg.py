import FWCore.ParameterSet.Config as cms

process = cms.Process("PremixStage1")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(3)
)

process.load("IOMC.RandomEngine.IOMC_cff")

from SimGeneral.PreMixingModule.mixOne_premix_on_sim_cfi import mixData as _mixData
process.mixData1 = _mixData.clone(
    input = dict(fileNames = ["file:testPremixStage1.root"]),
    workers = cms.PSet(
        pileup = _mixData.workers.pileup.clone(
            GenPUProtonsInputTags = []
        )
    ),
    adjustPileupDistribution = [
        cms.PSet(
            firstRun = cms.uint32(1),
            firstBinPileup = cms.uint32(0),
            pileupProbabilities = cms.vdouble(0,1,1,0)
        ),
    ],
    minBunch = -1,
    maxBunch = 1,
)
process.RandomNumberGeneratorService.mixData1 = process.RandomNumberGeneratorService.mixData.clone()
process.testMixData1 = cms.EDAnalyzer("TestPreMixingPileupAnalyzer",
    src = cms.untracked.InputTag("mixData1"),
    allowedPileups = cms.untracked.vuint32(1,2)
)

process.mixData2 = process.mixData1.clone(
    adjustPileupDistribution = {0: dict(pileupProbabilities = [0.8,0,0,0.5])}
)
process.RandomNumberGeneratorService.mixData2 = process.RandomNumberGeneratorService.mixData.clone()
process.testMixData2 = process.testMixData1.clone(
    src = "mixData2",
    allowedPileups = [0,3]
)

process.t = cms.Task(
    process.mixData1,
    process.mixData2,
)
process.s = cms.Sequence(
    process.testMixData1+
    process.testMixData2
)
process.p = cms.Path(process.s, process.t)
