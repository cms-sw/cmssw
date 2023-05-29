import FWCore.ParameterSet.Config as cms

process = cms.Process("HFShowerLib")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Generator_cff')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
#process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13p6TeVEarly2022Collision_cfi')
process.load("Configuration.Geometry.GeometryExtended2021_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("SimG4CMS.ShowerLibraryProducer.hfShowerLibaryAnalysis_cfi")
from Configuration.AlCa.GlobalTag import GlobalTag 
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HFShower = dict()

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    PGunParameters = cms.PSet(
        PartID   = cms.vint32(13),
        MinTheta = cms.double(0.019997),
        MaxTheta = cms.double(0.019997),
        MinPhi   = cms.double(3.14500926),
        MaxPhi   = cms.double(3.14500926),
        MinE     = cms.double(100.0),
        MaxE     = cms.double(100.0)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hfShowerDump.root')
)

process.genstepfilter.triggerConditions=cms.vstring("generation_step")

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.analysis_step = cms.EndPath(process.hfShowerLibaryAnalysis)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
#                                process.genfiltersummary_step,
                                process.simulation_step,
                                process.analysis_step)

#process.hfShowerLibaryAnalysis.Verbosity = True
#process.hfShowerLibaryAnalysis.EventPerBin = 5000
#process.hfShowerLibaryAnalysis.FileName = "HFShowerLibrary_npmt_noatt_eta4_16en_v4.root"

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
