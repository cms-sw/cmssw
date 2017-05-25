import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
)

process.source = cms.Source('EmptySource')

process.load('SimRomanPot.CTPPSOpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimRomanPot.CTPPSOpticsParameterisation.ctppsOpticsParameterisation_cfi')

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

process.RandomNumberGeneratorService.lhcBeamProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(1),
    #engineName = cms.untracked.string('TRandom3'),
)

 
process.p = cms.Path(
    process.lhcBeamProducer
    * process.ctppsOpticsParameterisation
)

process.e = cms.EndPath(process.out)
