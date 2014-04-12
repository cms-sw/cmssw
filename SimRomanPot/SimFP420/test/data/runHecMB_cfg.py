import FWCore.ParameterSet.Config as cms

process = cms.Process("HecFP420Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimTransport.HectorProducer.HectorTransport_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimRomanPot.SimFP420.FP420Digi_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        PythiaSource = cms.untracked.uint32(12345),
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
#   input = cms.untracked.int32(20)
)
process.source = cms.Source("PythiaSource",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
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
            'MSUB(92)=1     ! Min bias process', 
            'MSUB(93)=1     ! Min bias process', 
            'MSUB(94)=1     ! Min bias process', 
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
            'PARP(82)=1.9   ! pt cutoff for multiparton interactions', 
            'PARP(89)=1000. ! sqrts for which PARP82 is set', 
            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
            'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
            'PARP(67)=1.    ! amount of initial-state radiation', 
            'PARP(85)=0.33  ! gluon prod. mechanism in MI', 
            'PARP(86)=0.66  ! gluon prod. mechanism in MI', 
            'PARP(87)=0.7   ! ', 
            'PARP(88)=0.5   ! ', 
            'PARP(91)=1.0   ! kt distribution'),
        pythiaDefaultBlock = cms.PSet(
            pythiaDefault = cms.vstring('PMAS(5,1)=4.8 ! b quark mass', 
                'PMAS(6,1)=172.3 ! t quark mass')
        )
    )
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*', 
        'keep SimTracks_*_*_*', 
        'keep SimVertexs_*_*_*', 
        'keep PSimHitCrossingFrame_mix_FP420SI_*', 
        'keep PSimHits_*_FP420SI_*', 
        'keep DigiCollectionFP420_*_*_*'),
    fileName = cms.untracked.string('HecMBEvent.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.LHCTransport*process.g4SimHits*process.mix*process.FP420Digi)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)
process.g4SimHits.Physics.DefaultCutValue =  cms.double(1000.)
process.g4SimHits.UseMagneticField = cms.bool(False)
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.Generator.ApplyEtaCuts = cms.bool(False)
process.g4SimHits.Generator.HepMCProductLabel = cms.string('LHCTransport')
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)


