import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023HGCalV6MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalV6Muon_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(5)
	)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet()

# Production Info
process.configurationMetadata = cms.untracked.PSet(
	version    = cms.untracked.string('$Revision: 1.20 $'),
	annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
	name       = cms.untracked.string('Applications')
	)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
					      splitLevel                   = cms.untracked.int32(0),
					      eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
					      outputCommands               = process.FEVTDEBUGHLTEventContent.outputCommands,
					      fileName                     = cms.untracked.string('file:digi_input_5000evts.root'),
					      dataset                      = cms.untracked.PSet(
	filterName = cms.untracked.string(''),
	dataTier   = cms.untracked.string('GEN-SIM-DIGI-RAW')
	),
					      SelectEvents                 = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))
					      )

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
   PGunParameters = cms.PSet(
	MaxPt  = cms.double(10.01),
	MinPt  = cms.double(9.99),
	PartID = cms.vint32(13),
	MaxEta = cms.double(2.50),
	MaxPhi = cms.double(3.14159265359),
	MinEta = cms.double(1.75),
	MinPhi = cms.double(-3.14159265359)
	),
   Verbosity       = cms.untracked.int32(0),
   psethack        = cms.string('single electron pt 10'),
   AddAntiParticle = cms.bool(True),
   firstRun        = cms.untracked.uint32(1)
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Output definition
process.ValidationOutput = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('file:output_digiVal_test.root'),
)

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.load("Validation.HGCalValidation.digiValidationV6_cff")

## path and endpath deffinition 
process.p1 = cms.Path(process.hgcalDigiValidationEE+
		      process.hgcalDigiValidationHEF+
		      process.hgcalDigiValidationHEB)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.output_step = cms.EndPath(process.ValidationOutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,
				process.simulation_step,process.digitisation_step,
				process.L1simulation_step,process.digi2raw_step,
				process.endjob_step,process.p1,process.p2,process.output_step)

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
	
	# customisation of the process.
	# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
	from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023HGCalV6Muon
	
#call to customisation function cust_2023HGCalMuon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023HGCalV6Muon(process)

