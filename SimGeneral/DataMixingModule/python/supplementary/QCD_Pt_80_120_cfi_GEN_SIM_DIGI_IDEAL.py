# Auto generated configuration file
# using: 
# Revision: 1.119 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: QCD_Pt_80_120.cfi -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT -n 10 --conditions FrontierConditions_GlobalTag,IDEAL_31X::All --relval 9000,25 --datatier GEN-SIM-RAW --eventcontent RAWSIM --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/Digi_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('QCD_Pt_80_120.cfi nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("EmptySource")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('QCD_Pt_80_120_cfi_GEN_SIM_DIGI.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RAW'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition
process.RAWSIMEventContent.outputCommands.append('keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*')
process.RAWSIMEventContent.outputCommands.append('keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*')
process.RAWSIMEventContent.outputCommands.append('keep CSCDetIdCSCComparatorDigiMuonDigiCollection_*_*_*')
process.RAWSIMEventContent.outputCommands.append('keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*')
process.RAWSIMEventContent.outputCommands.append('keep DTLayerIdDTDigiMuonDigiCollection_*_*_*')
process.RAWSIMEventContent.outputCommands.append('keep EBDigiCollection_simEcalUnsuppressedDigis_*_*')
process.RAWSIMEventContent.outputCommands.append('keep EEDigiCollection_simEcalUnsuppressedDigis_*_*')
process.RAWSIMEventContent.outputCommands.append('keep EBDigiCollection_simEcalDigis_*_*')
process.RAWSIMEventContent.outputCommands.append('keep EEDigiCollection_simEcalDigis_*_*')
process.RAWSIMEventContent.outputCommands.append('keep ESDataFramesSorted_simEcalPreshowerDigis_*_*')
process.RAWSIMEventContent.outputCommands.append('keep SiStripDigiedmDetSetVector_simSiStripDigis_ZeroSuppressed_*')
process.RAWSIMEventContent.outputCommands.append('keep PixelDigiedmDetSetVector_*_*_*')

#process.RAWSIMEventContent.outputCommands.append('keep *')

# Other statements
process.GlobalTag.globaltag = 'IDEAL_31X::All'
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(10000.0),
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
        processParameters = cms.vstring('MSEL=1               ! QCD hight pT processes', 
            'CKIN(3)=80.          ! minimum pt hat for hard interactions', 
            'CKIN(4)=120.         ! maximum pt hat for hard interactions'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.digitisation_step)
#process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.out_step])
# special treatment in case of production filter sequence  
for path in process.paths: 
    getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq
