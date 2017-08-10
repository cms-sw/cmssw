import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('HLT', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticCrossingAngleCollision2016_cfi')
process.load('Configuration.StandardSequences.Generator_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')

process.source = cms.Source('EmptySource')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

# load the simulation part
from SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi import lhcBeamProducer
process.generator = lhcBeamProducer.clone(
    MinXi = cms.double(0.03),
    MaxXi = cms.double(0.15),
)

process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

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

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1) )

process.Timing = cms.Service('Timing',
    summaryOnly = cms.untracked.bool(True),
    useJobReport = cms.untracked.bool(True),
)

#process.geomInfo = cms.EDAnalyzer("GeometryInfoModule")
#process.eca = cms.EDAnalyzer("EventContentAnalyzer")
 
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(
    process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)
process.outpath = cms.EndPath(process.out)
    #* process.geomInfo * process.eca

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.outpath
)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq

