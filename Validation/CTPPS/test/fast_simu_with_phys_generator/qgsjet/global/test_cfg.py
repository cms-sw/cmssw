import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('CTPPSFastSimulation', eras.ctpps_2016)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealisticCrossingAngleCollision2016_cfi')
process.load('Configuration.StandardSequences.Generator_cff')

# number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

process.source = cms.Source('EmptySource')

# load the geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import totemGeomXMLFiles, ctppsDiamondGeomXMLFiles

process.XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = totemGeomXMLFiles+ctppsDiamondGeomXMLFiles+["Validation/CTPPS/test/fast_simu_with_phys_generator/qgsjet/global/RP_Dist_Beam_Cent.xml"],
    rootNodeName = cms.string('cms:CMSE'),
)

process.TotemRPGeometryESModule = cms.ESProducer("TotemRPGeometryESModule")

# QGSJET-II-04
process.generator = cms.EDFilter("ReggeGribovPartonMCGeneratorFilter",
    beamid = cms.int32(1),
    targetid = cms.int32(1),
    model = cms.int32(7),
    targetmomentum = cms.double(-6500),
    beammomentum = cms.double(6500),
    bmin = cms.double(0),
    bmax = cms.double(10000),
    paramFileName = cms.untracked.string("Configuration/Generator/data/ReggeGribovPartonMC.param"),
    skipNuclFrag = cms.bool(True)
)

# fast simulation
process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')
process.ctppsFastProtonSimulation.produceHitsRelativeToBeam = False

# strips reco: pattern recognition
process.load('RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi')
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag('ctppsFastProtonSimulation')

# strips reco: track fitting
process.load('RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi')

# common reco: lite track production
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')
process.load('RecoCTPPS.ProtonReconstruction.ctppsProtonReconstruction_cfi')
# distribution plotters
process.load('Validation.CTPPS.ctppsParameterisationValidation_cfi')

# for detectors resolution smearing
process.RandomNumberGeneratorService.ctppsFastProtonSimulation = cms.PSet( initialSeed = cms.untracked.uint32(1) )

# prepare the output file
process.TFileService = cms.Service('TFileService',
    fileName = cms.string("reco_hit_distributions.root"),
    closeFileFast = cms.untracked.bool(True),
)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('ctppsSim.root'),
)

# processing path
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(
    process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)
process.validation_step = cms.Path(
    process.ctppsProtonReconstruction
    # distribution plotter
    * process.paramValidation
)
process.outpath = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.validation_step,
    process.outpath
)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq

