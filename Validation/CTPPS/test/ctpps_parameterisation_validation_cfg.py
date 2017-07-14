import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)
#process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('EmptySource')

process.load('SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

# load the reconstruction
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.load('RecoCTPPS.ProtonReconstruction.ctppsOpticsReconstruction_cfi')

process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ctppsFastProtonSimulation")
#process.totemRPUVPatternFinder.verbosity = cms.untracked.uint32(10)
process.ctppsLocalTrackLiteProducer.doNothing = cms.bool(False)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

process.RandomNumberGeneratorService.lhcBeamProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(1),
    #engineName = cms.untracked.string('TRandom3'),
)
# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

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
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
    * process.ctppsOpticsReconstruction
    * process.ctppsParameterisationValidation
)

process.e = cms.EndPath(process.out)
