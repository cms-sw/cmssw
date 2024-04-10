import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_hgcaltb_cff import hgcaltb

process = cms.Process('SIM', hgcaltb)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTB161TimingLayerXML_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBNumberingInitialization_cfi')
process.load('Geometry.HGCalTBCommonData.hgcalTBParametersInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
#process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTimingAnalyzer_cfi')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        HGCSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleElectronE4_cfi'),
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

process.generator = cms.EDProducer("GaussRandomPThetaGunProducer",
				   AddAntiParticle = cms.bool(False),
				   PGunParameters = cms.PSet(
		MeanP    = cms.double(100.0),
		SigmaP   = cms.double(0.108),
		MinTheta = cms.double(0.0),
		MaxTheta = cms.double(0.0),
		MinPhi = cms.double(-3.14159265359),
		MaxPhi = cms.double(3.14159265359),
		PartID = cms.vint32(11)
		),
				   Verbosity = cms.untracked.int32(0),
				   firstRun = cms.untracked.uint32(1),
				   psethack = cms.string('single electron E 250')
				   )


process.VtxSmeared.MinZ = -800.0
process.VtxSmeared.MaxZ = -800.0
process.VtxSmeared.MinX = -7.5
process.VtxSmeared.MaxX =  7.5
process.VtxSmeared.MinY = -7.5
process.VtxSmeared.MaxY =  7.5
process.g4SimHits.HGCSD.StoreAllG4Hits = True
process.HGCalTimingAnalyzer.GroupHits = False

process.g4SimHits.HGCSD.Detectors = 1
process.g4SimHits.HGCSD.UseDetector = 1

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.analysis_step = cms.Path(process.HGCalTimingAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.analysis_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


