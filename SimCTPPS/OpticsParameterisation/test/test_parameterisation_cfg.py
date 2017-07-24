import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('HLT', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticCrossingAngleCollision2016_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')

process.source = cms.Source('EmptySource')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

# load the simulation part
process.load('SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

process.lhcBeamProducer.MinXi = cms.double(0.05)
process.lhcBeamProducer.MaxXi = cms.double(0.010)
process.ctppsFastProtonSimulation.beamParticlesTag = cms.InputTag('lhcBeamProducer', 'unsmeared')

# load the reconstruction part
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulation')
process.totemRPUVPatternFinder.verbosity = cms.untracked.uint32(10)
#process.ctppsLocalTrackLiteProducer.doNothing = cms.bool(False)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

# for initial particles' position/angular smearing
process.RandomNumberGeneratorService.lhcBeamProducer = cms.PSet( initialSeed = cms.untracked.uint32(1) )
# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1) )

process.Timing = cms.Service('Timing',
    summaryOnly = cms.untracked.bool(True),
    useJobReport = cms.untracked.bool(True),
)

#process.geomInfo = cms.EDAnalyzer("GeometryInfoModule")
#process.eca = cms.EDAnalyzer("EventContentAnalyzer")
 
process.p = cms.Path(
    process.lhcBeamProducer
    #* process.geomInfo * process.eca
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)

process.e = cms.EndPath(process.out)
