# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/FourteenTeV/Hadronizer_TuneZ2star_14TeV_madgraph_tauola_cff.py --filein file:essai_GEN.root --mc --eventcontent FEVTDEBUG --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023TTI,Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM --conditions DES23_62_V1::All --beamspot HLLHC --step GEN,SIM --magField 38T_PostLS1 --geometry Extended2023TTI --python_filename forGENSIM_cfg.py --no_exec -n 1000
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')


################################################################################
# Basic running parameters (modify to your needs)
index=2

#   random seed
theSeedValue = int(9283983+1000*index)
print 'Seed value: '+str(theSeedValue)

################################################################################

    

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

nevents = 100
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nevents)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring("/store/group/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/DiHiggs/GEN/dih_gluglufusion_GEN.root") ,
    skipEvents=cms.untracked.uint32( 100 )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('Configuration/GenProduction/python/FourteenTeV/Hadronizer_TuneZ2star_14TeV_madgraph_tauola_cff.py nevts:1000'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string( "dih_gluglufusion_SIM_2.root" ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'DES23_62_V1::All', '')

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            UseTauolaPolarization = cms.bool(True),
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak2 = cms.int32(0),
                pjak1 = cms.int32(0)
            )
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    maxEventsToPrint = cms.untracked.int32(10),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(14000.0),
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTU(21)=1     ! Check on possible errors during program execution', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10 .  ! for which ctau  10 mm', 
            'MSTP(33)=0     ! no K factors in hard cross sections', 
            'MSTP(2)=1      ! which order running alphaS', 
            'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
            'MSTP(52)=2     ! work with LHAPDF', 
            'PARP(82)=1.921 ! pt cutoff for multiparton interactions', 
            'PARP(89)=1800. ! sqrts for which PARP82 is set', 
            'PARP(90)=0.227 ! Multiple interactions: rescaling power', 
            'MSTP(95)=6     ! CR (color reconnection parameters)', 
            'PARP(77)=1.016 ! CR', 
            'PARP(78)=0.538 ! CR', 
            'PARP(80)=0.1   ! Prob. colored parton from BBR', 
            'PARP(83)=0.356 ! Multiple interactions: matter distribution parameter', 
            'PARP(84)=0.651 ! Multiple interactions: matter distribution parameter', 
            'PARP(62)=1.025 ! ISR cutoff', 
            'MSTP(91)=1     ! Gaussian primordial kT', 
            'PARP(93)=10.0  ! primordial kT-max', 
            'MSTP(81)=21    ! multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4     ! Defines the multi-parton model'),
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
            'PMAS(5,1)=4.8   ! b quark mass', 
            'PMAS(6,1)=172.5 ! t quark mass', 
            'MSTJ(1)=1       ! Fragmentation/hadronization on or off', 
            'MSTP(61)=1      ! Parton showering on or off',
            'MDME(210,1)=0    ! Higgs decay into dd',
            'MDME(211,1)=0    ! Higgs decay into uu',
            'MDME(212,1)=0    ! Higgs decay into ss',
            'MDME(213,1)=0    ! Higgs decay into cc',
            'MDME(214,1)=1    ! Higgs decay into bb',
            'MDME(215,1)=0    ! Higgs decay into tt',
            'MDME(216,1)=0    ! Higgs decay into bbbar prime',
            'MDME(217,1)=0    ! Higgs decay into ttbar prime',
            'MDME(218,1)=0    ! Higgs decay into e e',
            'MDME(219,1)=0    ! Higgs decay into mu mu',
            'MDME(220,1)=0    ! Higgs decay into tau tau',
            'MDME(221,1)=0    ! Higgs decay into tau tau prime',
            'MDME(222,1)=0    ! Higgs decay into g g',
            'MDME(223,1)=0    ! Higgs decay into gam gam',
            'MDME(224,1)=0    ! Higgs decay into gam Z',
            'MDME(225,1)=0    ! Higgs decay into Z Z',
            'MDME(226,1)=0    ! Higgs decay into W W'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'DES23_62_V1::All', '')

process.load('IOMC/RandomEngine/IOMC_cff')
process.RandomNumberGeneratorService.generator.initialSeed  = theSeedValue+111 ###cms.untracked.uint32(1136201) #########
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = theSeedValue+222 ###cms.untracked.uint32(2132001)
process.RandomNumberGeneratorService.g4SimHits.initialSeed  = theSeedValue+333 ###cms.untracked.uint32(3136201)


# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023TTI 

#call to customisation function cust_2023TTI imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023TTI(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions
