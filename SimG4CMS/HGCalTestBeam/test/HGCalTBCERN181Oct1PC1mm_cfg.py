import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB181Oct1PC1mmXML_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBCheckGunPosition_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#if 'MessageLogger' in process.__dict__:
#    process.MessageLogger.HGCalGeom=dict()
#    process.MessageLogger.SimG4CoreGeometry=dict()
#    process.MessageLogger.HGCSim=dict()
#    process.MessageLogger.HcalSim=dict()
#    process.MessageLogger.HcalTB06BeamSD=dict()

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleMuonE200_cfi nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:TBGenSim181Oct1.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('TBGenSim.root')
                                   )

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        MinE = cms.double(99.99),
        MaxE = cms.double(100.01),
        MinTheta = cms.double(0.0),
        MaxTheta = cms.double(0.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        PartID = cms.vint32(11)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single muon E 100')
)
process.VtxSmeared.MinZ                 = -800.0
process.VtxSmeared.MaxZ                 = -800.0
process.VtxSmeared.MinX                 = -7.5
process.VtxSmeared.MaxX                 =  7.5
process.VtxSmeared.MinY                 = -7.5
process.VtxSmeared.MaxY                 =  7.5
process.g4SimHits.HGCSD.RejectMouseBite = True
process.g4SimHits.HGCSD.RotatedWafer    = True
process.g4SimHits.OnlySDs = ['HGCalSensitiveDetector', 
                             'AHcalSensitiveDetector',
                             'CaloTrkProcessing',
                             'HGCSensitiveDetector',
                             'EcalTBH4BeamDetector',
                             'HGCalTB1601SensitiveDetector',
                             'HcalTB02SensitiveDetector',
                             'HcalTB06BeamDetector',
                             'EcalSensitiveDetector',
                             'HcalSensitiveDetector']
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
		HGCPassive = cms.PSet(
			LVNames = cms.vstring('HGCalEE','HGCalHE','HGCalAH', 'HGCalBeam', 'CMSE'),
			MotherName = cms.string('OCMS'),
			),
		type = cms.string('HGCPassive'),
		)
				       )
process.HGCalTBAnalyzer.doDigis         = False
process.HGCalTBAnalyzer.doRecHits       = False
process.HGCalTBAnalyzer.useFH           = True
process.HGCalTBAnalyzer.useBH           = True
process.HGCalTBAnalyzer.useBeam         = True
process.HGCalTBAnalyzer.zFrontEE        = 1110.0
process.HGCalTBAnalyzer.zFrontFH        = 1176.5
process.HGCalTBAnalyzer.zFrontFH        = 1307.5
process.HGCalTBAnalyzer.maxDepth        = 39
process.HGCalTBAnalyzer.deltaZ          = 26.2
process.HGCalTBAnalyzer.zFirst          = 22.8
process.HGCalTBAnalyzer.doPassive       = True
process.HGCalTBAnalyzer.doPassiveEE     = True
process.HGCalTBAnalyzer.doPassiveHE     = True
process.HGCalTBAnalyzer.doPassiveBH     = True

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.gunfilter_step  = cms.Path(process.HGCalTBCheckGunPostion)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.analysis_step = cms.Path(process.HGCalTBAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
				process.genfiltersummary_step,
				process.simulation_step,
				process.gunfilter_step,
				process.analysis_step,
				process.endjob_step,
                                process.FEVTDEBUGoutput_step
				)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


