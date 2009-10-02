import FWCore.ParameterSet.Config as cms

process = cms.Process("HecFP420Test")


process.load("Configuration.StandardSequences.Generator_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimTransport.HectorProducer.HectorTransport_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
# Input source
#process.load('Configuration/Generator/MinBias_cfi')

# Input source
process.source = cms.Source("EmptySource")

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        myParameters = cms.vstring(),
        parameterSets = cms.vstring('pythiaMinBias', 
            'myParameters'),
        pythiaMinBias = cms.vstring('MSEL=0         ! User defined processes', 
            'MSUB(11)=1     ! Min bias process', 
            'MSUB(12)=1     ! Min bias process', 
            'MSUB(13)=1     ! Min bias process', 
            'MSUB(28)=1     ! Min bias process', 
            'MSUB(53)=1     ! Min bias process', 
            'MSUB(68)=1     ! Min bias process', 
            'MSUB(91)=0     ! Min bias process, elastic scattering', 
            'MSUB(92)=1     ! Min bias process, single diffractive', 
            'MSUB(93)=1     ! Min bias process, single diffractive', 
            'MSUB(94)=1     ! Min bias process, double diffractive', 
            'MSUB(95)=1     ! Min bias process', 
            'MSTJ(11)=3     ! Choice of the fragmentation function', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10.   ! for which ctau  10 mm', 
            'MSTP(2)=1      ! which order running alphaS', 
            'MSTP(33)=0     ! no K factors in hard cross sections', 
            'MSTP(51)=7     ! structure function chosen', 
            'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4     ! Defines the multi-parton model', 
            'MSTU(21)=1     ! Check on possible errors during program execution', 
            'PARP(82)=1.9409   ! pt cutoff for multiparton interactions', 
            'PARP(89)=1960. ! sqrts for which PARP82 is set', 
            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
            'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
            'PARP(67)=2.5    ! amount of initial-state radiation', 
            'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(62)=1.25   ! ', 
            'PARP(64)=0.2    ! ', 
            'MSTP(91)=1     !', 
            'PARP(91)=2.1   ! kt distribution', 
            'PARP(93)=15.0  ! '),
        pythiaDefaultBlock = cms.PSet(
            pythiaDefault = cms.vstring('PMAS(5,1)=4.8 ! b quark mass', 
                'PMAS(6,1)=172.3 ! t quark mass')
        )
    )
)

#ProductionFilterSequence = cms.Sequence(process.generator)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*',
        'keep SimTracks_*_*_*',
        'keep SimVertexs_*_*_*',
        'keep PSimHitCrossingFrame_mix_FP420SI_*', 
        'keep PSimHits_*_FP420SI_*'),
    fileName = cms.untracked.string('HecMBEvent.root')
)

process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer")
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.LHCTransport*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)
process.g4SimHits.Physics.DefaultCutValue =  1000.
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Generator.HepMCProductLabel = 'LHCTransport'
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)
#rocess.FP420Digi.ApplyTofCut = cms.bool(False)
