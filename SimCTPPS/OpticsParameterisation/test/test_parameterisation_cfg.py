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
from Geometry.VeryForwardGeometry.geometryRP_cfi import totemGeomXMLFiles, ctppsDiamondGeomXMLFiles
process.XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = totemGeomXMLFiles+ctppsDiamondGeomXMLFiles+["SimCTPPS/OpticsParameterisation/test/RP_Dist_Beam_Cent.xml"],
    rootNodeName = cms.string('cms:CMSE'),
)
process.TotemRPGeometryESModule = cms.ESProducer("TotemRPGeometryESModule")

# load the simulation part
process.load('SimCTPPS.OpticsParameterisation.lhcBeamProducer_cfi')
process.load('SimCTPPS.OpticsParameterisation.ctppsFastProtonSimulation_cfi')

# load the reconstruction part
#from RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi import *
#from RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi import *
process.load("RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi")
process.load("RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi")
process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')

process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ctppsFastProtonSimulation")
process.totemRPUVPatternFinder.verbosity = cms.untracked.uint32(10)
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
    closeFileFast = cms.untracked.bool(True)
)
 
process.p = cms.Path(
    process.lhcBeamProducer
    * process.ctppsFastProtonSimulation
    * process.totemRPUVPatternFinder
    * process.totemRPLocalTrackFitter
    * process.ctppsLocalTrackLiteProducer
)

process.e = cms.EndPath(process.out)
