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

#from Configuration.Generator.Pythia8CommonSettings_cfi import *
#from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
#from Configuration.Generator.Pythia8aMCatNLOSettings_cfi import *

process.source = cms.Source('EmptySource')

# load the geometry
process.load('SimCTPPS.OpticsParameterisation.simGeometryRP_cfi')

process.generator = cms.EDFilter('Pythia8GeneratorFilter',
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8_proc = cms.vstring(
            'SoftQCD:nonDiffractive = off',
            'SoftQCD:singleDiffractive = on',
            'SoftQCD:doubleDiffractive = off',
            'Tune:pp 5', #4C
        ),
        parameterSets = cms.vstring('pythia8_proc')
    )
)

process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')
#process.ctppsFastProtonSimulation.yOffsetSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.yOffsetSector56 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector45 = cms.double(0.0)
#process.ctppsFastProtonSimulation.beamConditions.halfCrossingAngleSector56 = cms.double(0.0)

# load the reconstruction part
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulation')

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root')
)

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1), )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(
    process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)
process.outpath = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.outpath
)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq

