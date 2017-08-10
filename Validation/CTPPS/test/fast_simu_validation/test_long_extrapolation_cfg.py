import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('CTPPSFastSimulation', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticCrossingAngleCollision2016_cfi')
process.load('Configuration.StandardSequences.Generator_cff')

# number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000),
)

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

process.source = cms.Source("EmptySource")

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import totemGeomXMLFiles, ctppsDiamondGeomXMLFiles

process.XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = totemGeomXMLFiles+ctppsDiamondGeomXMLFiles+["Validation/CTPPS/test/fast_simu_with_phys_generator/qgsjet/global/RP_Dist_Beam_Cent.xml"],
    rootNodeName = cms.string('cms:CMSE'),
)

# particle generator
from SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi import lhcBeamProducer
process.generator = lhcBeamProducer.clone(
    MinXi = cms.double(0.0),
    MaxXi = cms.double(0.1),
)

process.TotemRPGeometryESModule = cms.ESProducer("TotemRPGeometryESModule")

# alternative association between RPs and optics approximators
from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import genericStripsPackage
detectorPackagesLong = cms.VPSet(
    #----- sector 45
    genericStripsPackage.clone(
        potId = cms.uint32(0x76100000), # 002
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb2'),
        zPosition = cms.double(-203.826),
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x76180000), # 003
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb2'),
        zPosition = cms.double(-203.826),
    ),

    #----- sector 56
    genericStripsPackage.clone(
        potId = cms.uint32(0x77100000), # 102
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb1'),
        zPosition = cms.double(+203.826),
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x77180000), # 103
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb1'),
        zPosition = cms.double(+203.826),
    ),
)

# fast simulation
from SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi import ctppsFastProtonSimulation
ctppsFastProtonSimulation.checkApertures = True
ctppsFastProtonSimulation.produceHitsRelativeToBeam = False
ctppsFastProtonSimulation.stripsRecHitsParams.roundToPitch = False

process.ctppsFastProtonSimulationStd = ctppsFastProtonSimulation.clone(
    produceScoringPlaneHits = cms.bool(True),
    produceRecHits = cms.bool(False)
)

process.ctppsFastProtonSimulationLong = ctppsFastProtonSimulation.clone(
    produceScoringPlaneHits = cms.bool(False),
    produceRecHits = cms.bool(True),
    detectorPackages = detectorPackagesLong
)

# strips reco: pattern recognition
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulationLong')

# strips reco: track fitting
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')

# common reco: lite track production
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

# distribution plotter
process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')
process.scoringPlaneValidation.spTracksTag = cms.InputTag('ctppsFastProtonSimulationStd', 'scoringPlane')
process.scoringPlaneValidation.recoTracksTag = cms.InputTag('ctppsLocalTrackLiteProducer')

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulationStd = cms.PSet( initialSeed = cms.untracked.uint32(1) )
process.RandomNumberGeneratorService.ctppsFastProtonSimulationLong = cms.PSet( initialSeed = cms.untracked.uint32(1) )

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('output_long_extrapolation.root'),
    closeFileFast = cms.untracked.bool(True),
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(
    process.ctppsFastProtonSimulationStd
    * process.ctppsFastProtonSimulationLong
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)
process.validation_step = cms.Path(
    process.scoringPlaneValidation
)
#process.out = cms.OutputModule('PoolOutputModule', fileName = cms.untracked.string('test.root') )
#process.outpath = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.validation_step
    #process.outpath
)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq

