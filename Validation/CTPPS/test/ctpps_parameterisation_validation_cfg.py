import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)
#process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('EmptySource')

process.load('SimRomanPot.CTPPSOpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimRomanPot.CTPPSOpticsParameterisation.ctppsOpticsParameterisation_cfi')
process.load('SimRomanPot.CTPPSOpticsParameterisation.ctppsOpticsReconstruction_cfi')
process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

process.RandomNumberGeneratorService.lhcBeamProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(1),
    #engineName = cms.untracked.string('TRandom3'),
)
# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsOpticsParameterisation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('output.root'),
    closeFileFast = cms.untracked.bool(True),
)

process.Timing = cms.Service('Timing',
    summaryOnly = cms.untracked.bool(True),
    useJobReport = cms.untracked.bool(True),
)

#process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
#    ignoreTotal = cms.untracked.int32(1),
#)
 
process.p = cms.Path(
    process.lhcBeamProducer
    * process.ctppsOpticsParameterisation
    * process.ctppsOpticsReconstruction
    * process.ctppsParameterisationValidation
)

process.e = cms.EndPath(process.out)
