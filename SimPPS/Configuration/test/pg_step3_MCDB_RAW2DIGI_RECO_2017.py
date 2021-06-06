# Auto generated configuration file
# using: 
# Revision: 1.19 
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('RECO',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")                       # totem frame
#process.load("Geometry.VeryForwardGeometry.geometryPPS_CMSxz_fromDD_2017_cfi")           # CMS frame
"""
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("GeometryFileRcd"),
             tag = cms.string("XMLFILE_Geometry_110YV3_Extended2017CTPPS_mc"),
             connect = cms.string("sqlite_file:PPS_geometry_DB.db")
    )
)
process.XMLFromDBSource.label=''
"""

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step2_DIGI_DIGI2RAW2017.root'),
    secondaryFileNames = cms.untracked.vstring()
)

# Track memory leaks
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Output definition

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('step3_RAW2DIGI_RECO2017.root'),
    outputCommands = cms.untracked.vstring("drop *","keep SimVertexs_g4SimHits_*_*","keep PSimHits*_*_*_*","keep CTPPS*_*_*_*","keep *_*RP*_*_*",'keep *_LHCTransport_*_*')
)


# Additional output definition
# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

"""
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('CTPPSPixelGainCalibrationsRcd'),
        tag = cms.string("CTPPSPixelGainCalibrations_mc"),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
    cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("CTPPSPixelAnalysisMask_mc"),
        label = cms.untracked.string(""),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
    cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string("CTPPSPixelDAQMapping_mc"),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        )
)
"""
# modify Totem 2017 mapping
process.load('CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi')
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
    cms.PSet(
      validityRange = cms.EventRange("1:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2017.xml"),
      maskFileNames = cms.vstring()
    )   
)

# modify CTPPS 2017 raw-to-digi modules
process.load('Configuration.StandardSequences.RawToDigi_cff')
             
# do not make testID for simulation - keeping the frame
from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi
totemRPRawToDigi.RawToDigi.testID = cms.uint32(1)

from RecoPPS.Local.totemRPLocalReconstruction_cff import totemRPLocalReconstruction
process.load('RecoPPS.Local.totemRPLocalReconstruction_cff')
from RecoPPS.Local.ctppsPixelLocalReconstruction_cff import ctppsPixelLocalReconstruction
process.load('RecoPPS.Local.ctppsPixelLocalReconstruction_cff')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.ctppsRawToDigi)
process.reco_step = cms.Path(process.totemRPLocalReconstruction*process.ctppsPixelLocalReconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reco_step,process.endjob_step,process.output_step)

# filter all path with the production filter sequence
for path in process.paths:
    #  getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
    getattr(process,path)._seq = getattr(process,path)._seq

