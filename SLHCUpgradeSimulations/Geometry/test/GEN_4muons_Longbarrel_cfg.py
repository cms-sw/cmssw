# Auto generated configuration file
# using: 
# Revision: 1.303 
# Source: /cvs/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SLHCUpgradeSimulations/Configuration/python/FourMuPt_1_50_cfi.py -s GEN,SIM --conditions DESIGN42_V17::All --eventcontent FEVTDEBUG --beamspot Gauss --slhc Phase1_R39F16 --datatier GEN-SIM --python_filename GEN_4muons_mod_cfg.py --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Longbarrel_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
process.Timing =  cms.Service("Timing")

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('SLHCUpgradeSimulations/Configuration/python/FourMuPt_1_50_cfi.py nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('FourMuPt_1_50_cfi_py_GEN_SIM.root'),
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
process.GlobalTag.globaltag = 'DESIGN42_V17::All'

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(50.0),
        MinPt = cms.double(0.9),
        PartID = cms.vint32(-13, -13),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('Four mu pt 1 to 50'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

#process.generator = cms.EDFilter("Pythia6GeneratorFilter",
#    ExternalDecays = cms.PSet(
#        Tauola = cms.untracked.PSet(
#            UseTauolaPolarization = cms.bool(True),
#            InputCards = cms.PSet(
#                mdtau = cms.int32(0),
#                pjak2 = cms.int32(0),
#                pjak1 = cms.int32(0)
#            )
#        ),
#        parameterSets = cms.vstring('Tauola')
#    ),
#    pythiaPylistVerbosity = cms.untracked.int32(0),
#    filterEfficiency = cms.untracked.double(1.0),
#    pythiaHepMCVerbosity = cms.untracked.bool(False),
#    comEnergy = cms.double(14000.0),
#    maxEventsToPrint = cms.untracked.int32(0),
#    PythiaParameters = cms.PSet(
#        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function',
#            'MSTJ(22)=2     ! Decay those unstable particles',
#            'PARJ(71)=10 .  ! for which ctau  10 mm',
#            'MSTP(2)=1      ! which order running alphaS',
#            'MSTP(33)=0     ! no K factors in hard cross sections',
#            'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
#            'MSTP(52)=2     ! work with LHAPDF',
#            'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default',
#            'MSTP(82)=4     ! Defines the multi-parton model',
#            'MSTU(21)=1     ! Check on possible errors during program execution',
#            'PARP(82)=1.8387   ! pt cutoff for multiparton interactions',
#            'PARP(89)=1960. ! sqrts for which PARP82 is set',
#            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter',
#            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter',
#            'PARP(90)=0.16  ! Multiple interactions: rescaling power',
#            'PARP(67)=2.5    ! amount of initial-state radiation',
#            'PARP(85)=1.0  ! gluon prod. mechanism in MI',
#            'PARP(86)=1.0  ! gluon prod. mechanism in MI',
#            'PARP(62)=1.25   ! ',
#            'PARP(64)=0.2    ! ',
#            'MSTP(91)=1      !',
#            'PARP(91)=2.1   ! kt distribution',
#            'PARP(93)=15.0  ! '),
#        processParameters = cms.vstring('MSEL      = 0     ! User defined processes',
#            'MSUB(81)  = 1     ! qqbar to QQbar',
#            'MSUB(82)  = 1     ! gg to QQbar',
#            'MSTP(7)   = 6     ! flavour = top',
#            'PMAS(6,1) = 175.  ! top quark mass'),
#        parameterSets = cms.vstring('pythiaUESettings',
#            'processParameters')
#    )
#)



# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 
