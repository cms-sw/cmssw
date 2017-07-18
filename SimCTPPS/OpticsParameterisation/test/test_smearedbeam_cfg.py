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

# load the simulation part
from SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi import lhcBeamProducer
process.generator = lhcBeamProducer.clone(
    MinXi = cms.double(0.03),
    MaxXi = cms.double(0.15),
)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_VtxSmeared_*_*',
        'keep *_generator*_*_*',
    ),
)

# for initial particles' position/angular smearing
process.RandomNumberGeneratorService.lhcBeamProducer = cms.PSet( initialSeed = cms.untracked.uint32(1) )

process.Timing = cms.Service('Timing',
    summaryOnly = cms.untracked.bool(True),
    useJobReport = cms.untracked.bool(True),
)
 
process.generation_step = cms.Path(process.pgen)
process.outpath = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.generation_step,
    process.outpath
)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq
