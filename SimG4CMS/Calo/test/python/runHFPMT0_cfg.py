import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
#process.load("SimG4CMS.Calo.testGeometryPMTXML_cfi")
process.load("Geometry.CMSCommonData.cmsHFPMTAverageXML_cfi")
#process.load("SimG4CMS.Calo.cmsHFPMTXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.HFPMTHitAnalyzer_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff"
)
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('CaloSim', 
        'EcalSim', 'G4cerr', 'G4cout',
        'HcalSim', 'HFShower'),
#    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HFShower = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(3.5),
        MaxEta = cms.double(3.5),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(100.00),
        MaxE   = cms.double(100.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

# Event output
process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('HFPMT0_3.5.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.hfPMTHitAnalyzer)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.DefaultCutValue   = 0.1
process.g4SimHits.HCalSD.UseShowerLibrary   = True
process.g4SimHits.HCalSD.UseParametrize     = False
process.g4SimHits.HCalSD.UsePMTHits         = True
process.g4SimHits.HFShower.UseShowerLibrary = True
process.g4SimHits.HFShower.UseHFGflash      = False
process.g4SimHits.HFShower.TrackEM          = False
process.g4SimHits.HFShower.OnlyLong         = True
process.g4SimHits.HFShower.EminLibrary      = 0.0
process.g4SimHits.HFShower.ApplyFiducialCut = False
process.g4SimHits.HFShowerLibrary.ApplyFiducialCut = False

# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                process.out_step
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
