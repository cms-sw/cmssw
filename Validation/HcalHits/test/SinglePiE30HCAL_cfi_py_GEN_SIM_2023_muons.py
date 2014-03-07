# Auto generated configuration file
# using: 
# Revision: 1.events0 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/Generator/python/SinglePiE30HCAL_cfi.py -s GEN,SIM --conditions STAR17_61_V1A::All --geometry Extended2017 --datatier GEN-SIM -n 10 --eventcontent FEVTDEBUG --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017 --fileout file:step1P1_0.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
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
			    firstRun = cms.untracked.uint32(1)
			    )

process.options = cms.untracked.PSet(

)

process.simHitsValidationHcal.TestNumber = True

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('Configuration/Generator/python/SinglePiE30HCAL_cfi.py nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition


process.ValidationOutput = cms.OutputModule("PoolOutputModule",
					    outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
					    fileName = cms.untracked.string('file:2023_run1.root'),
					    )




# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'DES17_62_V8::All', '')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgradePLS3']

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
				 pythiaPylistVerbosity = cms.untracked.int32(0),
				 filterEfficiency = cms.untracked.double(1.0),
				 pythiaHepMCVerbosity = cms.untracked.bool(False),
				 comEnergy = cms.double(8000.0),
				 maxEventsToPrint = cms.untracked.int32(0),
				 PythiaParameters = cms.PSet(
	pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function',
				       'MSTJ(22)=2     ! Decay those unstable particles',
				       'PARJ(71)=10 .  ! for which ctau  10 mm',
				       'MSTP(2)=1      ! which order running alphaS',
				       'MSTP(33)=0     ! no K factors in hard cross sections',
				       'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
				       'MSTP(52)=2     ! work with LHAPDF',
				       'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default',
				       'MSTP(82)=4     ! Defines the multi-parton model',
				       'MSTU(21)=1     ! Check on possible errors during program execution',
				       'PARP(82)=1.8387   ! pt cutoff for multiparton interactions',
				       'PARP(89)=1960. ! sqrts for which PARP82 is set',
				       'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter',
				       'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter',
				       'PARP(90)=0.16  ! Multiple interactions: rescaling power',
				       'PARP(67)=2.5    ! amount of initial-state radiation',
				       'PARP(85)=1.0  ! gluon prod. mechanism in MI',
				       'PARP(86)=1.0  ! gluon prod. mechanism in MI',
				       'PARP(62)=1.25   ! ',
				       'PARP(64)=0.2    ! ',
				       'MSTP(91)=1      !',
				       'PARP(91)=2.1   ! kt distribution',
				       'PARP(93)=15.0  ! '),
	processParameters = cms.vstring('MSEL=0         ! User defined processes',
					'MSUB(11)=1     ! Min bias process',
					'MSUB(12)=1     ! Min bias process',
					'MSUB(13)=1     ! Min bias process',
					'MSUB(28)=1     ! Min bias process',
					'MSUB(53)=1     ! Min bias process',
					'MSUB(68)=1     ! Min bias process',
					'MSUB(92)=1     ! Min bias process, single diffractive',
					'MSUB(93)=1     ! Min bias process, single diffractive',
					'MSUB(94)=1     ! Min bias process, double diffractive',
					'MSUB(95)=1     ! Min bias process'),
		        parameterSets = cms.vstring('pythiaUESettings',
						    'processParameters')
	)
				 )





process.RandomNumberGeneratorService.generator.initialSeed = 1876638

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
#process.endjob_step = cms.EndPath(process.endOfProcess)


process.p1 = cms.Path(process.simHitsValidationHcal)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.output_step = cms.EndPath(process.ValidationOutput)



# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.p1,process.p2,process.output_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019 

#call to customisation function cust_2017 imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
#process = cust_2019(process)

# End of customisation functions
processDumpFile = open('processsimhits.dump', 'w')
print >> processDumpFile, process.dumpPython()
