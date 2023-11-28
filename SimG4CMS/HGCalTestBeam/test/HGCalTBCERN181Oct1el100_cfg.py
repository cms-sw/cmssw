import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_hgcaltb_cff import hgcaltb

process = cms.Process('SIM', hgcaltb)

# import of standard configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB181Oct1XML_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBNumberingInitialization_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBCheckGunPosition_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

if 'MessageLogger' in process.__dict__:
     process.MessageLogger.BeamMomentumGun=dict()
     process.MessageLogger.HGCSim=dict()
#    process.MessageLogger.HcalSim=dict()

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

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:gensim.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
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

process.generator = cms.EDProducer("BeamMomentumGunProducer",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        FileName = cms.FileInPath('SimG4CMS/HGCalTestBeam/data/HGCTBeamProfTree_PosE100.root'),
        MinTheta = cms.double(.012138),
        MaxTheta = cms.double(.012138),
        MinPhi = cms.double(3.638332),
        MaxPhi = cms.double(3.638332),
        XOffset = cms.double(0.0),
        YOffset = cms.double(0.0),
        ZPosition = cms.double(111.0),
        PartID = cms.vint32(11)
    ),
    Verbosity = cms.untracked.int32(1),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single muon E 100')
)

process.VtxSmeared.MeanX  = 0
process.VtxSmeared.SigmaX = 0
process.VtxSmeared.MeanY  = 0
process.VtxSmeared.SigmaY = 0
process.VtxSmeared.MeanZ  = 0
process.VtxSmeared.SigmaZ = 0
process.g4SimHits.HGCSD.RejectMouseBite = True
process.g4SimHits.HGCSD.RotatedWafer    = True

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
		HGCPassive = cms.PSet(
                     LVNames = cms.vstring('HGCalEE','HGCalHE','HGCalAH', 'HGCalBeam', 'CMSE'),
                     MotherName = cms.string('OCMS'),
                     IfDD4hep = cms.bool(False),
                ),
		type = cms.string('HGCPassive'),
		)
				       )
process.HGCalTBAnalyzer.doDigis         = False
process.HGCalTBAnalyzer.doRecHits       = False
process.HGCalTBAnalyzer.useFH           = True
process.HGCalTBAnalyzer.useBH           = True
process.HGCalTBAnalyzer.useBeam         = True
process.HGCalTBAnalyzer.addP            = True
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
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)


process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMN'

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
				process.genfiltersummary_step,
				process.simulation_step,
#				process.gunfilter_step,
				process.analysis_step,
				process.endjob_step,
#				process.RAWSIMoutput_step,
				)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


#print process.dumpPython()
