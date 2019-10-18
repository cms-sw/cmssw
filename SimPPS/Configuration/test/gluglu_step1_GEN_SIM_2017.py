# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: GluGluTo2Jets_M_100_7TeV_exhume_cff.py --mc --eventcontent FEVTDEBUG --datatier GEN-SIM --conditions 80X_mcRun2_asymptotic_2016_v2 --step GEN,SIM --era Run2_25ns --geometry Extended2017dev --processName=CTPPS --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('SIM',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2017_CTPPS_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2017Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')

process.load('SimG4Core.Application.g4SimHits_cfi')
process.g4SimHits.Physics.DefaultCutValue = 100.
process.g4SimHits.UseMagneticField              = False
process.g4SimHits.Generator.ApplyPCuts          = False
process.g4SimHits.Generator.ApplyPhiCuts        = False
process.g4SimHits.Generator.ApplyEtaCuts        = False
process.g4SimHits.Generator.HepMCProductLabel   = 'LHCTransport'
process.g4SimHits.Generator.MinEtaCut        = -13.0
process.g4SimHits.Generator.MaxEtaCut        = 13.0
process.g4SimHits.Generator.Verbosity        = 0
process.g4SimHits.Generator.EtaCutForHector  = 7.0

process.g4SimHits.G4TrackingManagerVerbosity = cms.untracked.int32(3)
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)

process.common_maximum_timex = cms.PSet( # need to be localy redefined
   MaxTrackTime  = cms.double(2000.0),  # need to be localy redefined
   MaxTimeNames  = cms.vstring('ZDCRegion'), # need to be localy redefined
   MaxTrackTimes = cms.vdouble(10000.0),  # need to be localy redefined
   DeadRegions = cms.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('GluGluTo2Jets_M_100_7TeV_exhume_cff.py nevts:1'),
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
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('GluGlu_GEN_SIM_2017.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

process.load('PhysicsTools.HepMCCandAlgos.genParticles_cfi')
process.generator = cms.EDFilter("ExhumeGeneratorFilter",
    ExhumeParameters = cms.PSet(
        AlphaEw = cms.double(0.0072974),
        B = cms.double(4.0),
        BottomMass = cms.double(4.6),
        CharmMass = cms.double(1.42),
        HiggsMass = cms.double(120.0),
        HiggsVev = cms.double(246.0),
        LambdaQCD = cms.double(80.0),
        MinQt2 = cms.double(0.64),
        MuonMass = cms.double(0.1057),
        PDF = cms.double(11000),
        Rg = cms.double(1.2),
        StrangeMass = cms.double(0.19),
        Survive = cms.double(0.03),
        TauMass = cms.double(1.77),
        TopMass = cms.double(175.0),
        WMass = cms.double(80.33),
        ZMass = cms.double(91.187)
    ),
    ExhumeProcess = cms.PSet(
        MassRangeHigh = cms.double(2000.0),
        MassRangeLow = cms.double(300.0),
        ProcessType = cms.string('GG'),
        ThetaMin = cms.double(0.3)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.g4Simhits_step = cms.Path(process.g4SimHits)

process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)


process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.g4Simhits_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 


from SimPPS.PPSSimTrackProducer.SimTrackProducerForFullSim_cff import customise
process = customise(process)
