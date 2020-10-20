import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0009/880AA097-75B7-DE11-B848-001D09F23C73.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/CC9C9FAC-86B6-DE11-8A1B-001D09F24FEC.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/A44075CB-7FB6-DE11-8010-000423D98868.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/6E433585-84B6-DE11-A3E1-001D09F2437B.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/285A7AC1-82B6-DE11-BA29-001D09F2525D.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/12755919-88B6-DE11-8FF2-000423D996C8.root', 
        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/04B93222-89B6-DE11-93E9-001D09F29524.root')
)
process.pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
    pdgId = cms.vint32(11, -11),
    src = cms.InputTag("pfNoPileUp")
)


process.gensource = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring('drop *', 
        'keep pdgId = 211', 
        'keep pdgId = -211')
)


process.pfPileUp = cms.EDProducer("PFPileUp",
    PFCandidates = cms.InputTag("particleFlow"),
    verbose = cms.untracked.bool(False),
    Vertices = cms.InputTag("offlinePrimaryVerticesWithBS")
)


process.pfNoPileUp = cms.EDProducer("TPPileUpPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlow"),
    topCollection = cms.InputTag("pfPileUp"),
    name = cms.untracked.string('pileUpOnPFCandidates'),
)


process.pfElectronBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    maxDeltaPhi = cms.double(0.5),
    BenchmarkLabel = cms.string('PFlowElectrons'),
    OnlyTwoJets = cms.bool(False),
    maxEta = cms.double(2.5),
    minEta = cms.double(-1),
    recPt = cms.double(2.0),
    minDeltaPhi = cms.double(-0.5),
    PlotAgainstRecoQuantities = cms.bool(False),
    minDeltaEt = cms.double(-100.0),
    OutputFile = cms.untracked.string('benchmark.root'),
    StartFromGen = cms.bool(False),
    deltaRMax = cms.double(0.05),
    maxDeltaEt = cms.double(50.0),
    InputTruthLabel = cms.InputTag("gensource"),
    InputRecoLabel = cms.InputTag("pfAllElectrons"),
    doMetPlots = cms.bool(False)
)


process.pfNoPileUpSequence = cms.Sequence(process.pfPileUp+process.pfNoPileUp)


process.electronBenchmarkGeneric = cms.Sequence(process.pfNoPileUpSequence+process.pfAllElectrons+process.gensource+process.pfElectronBenchmarkGeneric)


process.p = cms.Path(process.electronBenchmarkGeneric)


process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr_stats = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        output = cms.untracked.string('cerr'),
        optionalPSet = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(100),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        optionalPSet = cms.untracked.bool(True)
    ),
    suppressWarning = cms.untracked.vstring(),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring(),
    infos = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary')
)


process.DQM = cms.Service("DQM",
    filter = cms.untracked.string(''),
    publishFrequency = cms.untracked.double(5.0),
    collectorHost = cms.untracked.string('localhost'),
    collectorPort = cms.untracked.int32(9090),
    debug = cms.untracked.bool(False)
)


process.DQMStore = cms.Service("DQMStore",
    verboseQT = cms.untracked.int32(0),
    verbose = cms.untracked.int32(0),
)


process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


