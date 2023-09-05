import FWCore.ParameterSet.Config as cms

process = cms.Process('Anal')

# import of standard configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB181Oct1XML_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBNumberingInitialization_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Geometry.HcalCommonData.caloSimulationParameters_cff')
process.load('Geometry.CaloEventSetup.HGCalTopology_cfi')
process.load('Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBCheckGunPosition_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCSim=dict()

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:TBGenSim181Oct1.root',
                            )
                        )

process.options = cms.untracked.PSet(
)


# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('TBAnalOct181.root')
                                   )

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


process.HGCalTBAnalyzer.useFH       = True
process.HGCalTBAnalyzer.useBH       = True
process.HGCalTBAnalyzer.useBeam     = True
process.HGCalTBAnalyzer.zFrontEE    = 1110.0
process.HGCalTBAnalyzer.zFrontFH    = 1176.5
process.HGCalTBAnalyzer.zFrontFH    = 1307.5
process.HGCalTBAnalyzer.maxDepth    = 39
process.HGCalTBAnalyzer.deltaZ      = 26.2
process.HGCalTBAnalyzer.zFirst      = 22.8

# Path and EndPath definitions
process.gunfilter_step  = cms.Path(process.HGCalTBCheckGunPostion)
process.analysis_step = cms.Path(process.HGCalTBAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.gunfilter_step,
				process.analysis_step,
				process.endjob_step,
				)


