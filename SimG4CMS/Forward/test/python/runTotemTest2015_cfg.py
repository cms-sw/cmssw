import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedFlat_cfi")
process.load("Geometry.ForwardCommonData.totemTest2015_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

process.VtxSmeared.MinZ = -10.5
process.VtxSmeared.MaxZ = -9.5

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ForwardSim = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

#process.generator = cms.EDProducer("FlatRandomPtGunProducer",
#    PGunParameters = cms.PSet(
#        PartID = cms.vint32(13),
#        MinEta = cms.double(-6.5),
#        MaxEta = cms.double(-2.5),
#        MinPhi = cms.double(-3.14159265359),
#        MaxPhi = cms.double(3.14159265359),
#        MinPt  = cms.double(20.),
#        MaxPt  = cms.double(20.)
#    ),
#    Verbosity       = cms.untracked.int32(0),
#    AddAntiParticle = cms.bool(False)
#)
#process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("Configuration.Generator.MinBias_13TeV_pythia8_cff")

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('GenSim.root')
                                   )

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load('SimG4CMS.Forward.SimG4FluxAnalyzer_cfi')

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step = cms.Path(process.SimG4FluxAnalyzer)
#process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
                SimG4FluxProducer = cms.PSet(
                        LVNames = cms.untracked.vstring('TotemT1Part1','TotemT1Part2','TotemT1Part3','TotemT2Part1','TotemT2Part2','TotemT2Part3'),
			LVTypes = cms.untracked.vint32(0,0,1,0,0,1)
                        ),
                type = cms.string('SimG4FluxProducer'),
))

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
				process.analysis_step,
#                                process.out_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

