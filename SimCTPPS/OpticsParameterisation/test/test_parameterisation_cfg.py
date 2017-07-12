import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000),
)
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

process.source = cms.Source('EmptySource')

# load the geometry
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')

# load the simulation part
process.load('SimRomanPot.CTPPSOpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimRomanPot.CTPPSOpticsParameterisation.ctppsFastProtonSimulation_cfi')

# load the reconstruction part
#from RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi import *
#from RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi import *
process.load("RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi")
process.load("RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi")

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ctppsFastProtonSimulation")
process.totemRPUVPatternFinder.verbosity = cms.untracked.uint32(10)

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
    closeFileFast = cms.untracked.bool(True)
)
 
process.p = cms.Path(
    process.lhcBeamProducer
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
)

process.e = cms.EndPath(process.out)
