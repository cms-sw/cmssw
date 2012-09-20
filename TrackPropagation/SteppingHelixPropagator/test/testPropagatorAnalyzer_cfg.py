import FWCore.ParameterSet.Config as cms

process = cms.Process("PROPAGATORTEST")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }

)

process.propAna = cms.EDAnalyzer("SteppingHelixPropagatorAnalyzer",
    ntupleTkHits = cms.bool(False),
    startFromPrevHit = cms.bool(False),
    radX0CorrectionMode = cms.bool(False),
    trkIndOffset = cms.int32(0),
    NtFile = cms.string('PropagatorDump.root'),
    testPCAPropagation = cms.bool(False),
    debug = cms.bool(False),
    g4SimName = cms.string('g4SimHits'),
    simTracksTag = cms.InputTag('g4SimHits'),
    simVertexesTag = cms.InputTag('g4SimHits')
)

process.p = cms.Path(process.propAna)
process.PoolSource.fileNames = [
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValSingleMuPt10/GEN-SIM/v1/0000/6C269129-E1F2-E111-9FC6-001A92971BC8.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValSingleMuPt10/GEN-SIM/v1/0000/5404C41C-E1F2-E111-9731-0026189438AD.root',
    '/store/relval/CMSSW_6_0_0-START60_V4/RelValSingleMuPt10/GEN-SIM/v1/0000/36D70436-E1F2-E111-BB26-001A92810AA2.root',
]

