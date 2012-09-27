# Auto generated configuration file
# using: 
# Revision: 1.303.2.3 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Configuration/Generator/python/SinglePiE30HCAL_cfi.py -s GEN,SIM --conditions START311_V2::All --beamspot Realistic7TeV2011Collision --datatier GEN-SIM --eventcontent RAWSIM -n 10000 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIMVAL')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic7TeV2011Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Validation.HcalHits.SimHitsValidationHcal_cfi')
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource",
			    firstRun = cms.untracked.uint32(2)
)

process.options = cms.untracked.PSet(

)



# Production Info
process.configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('$Revision: 1.1 $'),
	annotation = cms.untracked.string('Configuration/Generator/python/SinglePiE30HCAL_cfi.py nevts:10000'),
	name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.ValidationOutput = cms.OutputModule("PoolOutputModule",
					    outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
					    fileName = cms.untracked.string('output_seed2.root'),
)

# Additional output definition

# Other statements
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_60_V5::All')

process.generator = cms.EDProducer("FlatRandomEGunProducer",
				   PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(-5.0),
        MaxEta = cms.double(5.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE = cms.double(29.99),
        MaxE = cms.double(30.01)
	),
				   Verbosity = cms.untracked.int32(0),
				   psethack = cms.string('single pi E 30 HCAL'),
				   AddAntiParticle = cms.bool(True)
)

process.RandomNumberGeneratorService.generator.initialSeed = 1988504

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.p1 = cms.Path(process.simHitsValidationHcal)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.output_step = cms.EndPath(process.ValidationOutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.p1,process.p2,process.output_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 
