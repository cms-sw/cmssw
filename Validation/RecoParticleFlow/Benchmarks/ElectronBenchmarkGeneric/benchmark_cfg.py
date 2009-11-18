import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_3_0_pre3/RelValZEE/GEN-SIM-RECO/MC_31X_V8-v1/0015/602FA642-089F-DE11-A825-001D09F24EE3.root', 
        '/store/relval/CMSSW_3_3_0_pre3/RelValZEE/GEN-SIM-RECO/MC_31X_V8-v1/0015/5C215805-059F-DE11-8E4B-001D09F24DA8.root', 
        '/store/relval/CMSSW_3_3_0_pre3/RelValZEE/GEN-SIM-RECO/MC_31X_V8-v1/0015/344D8640-069F-DE11-836A-001D09F29114.root', 
        '/store/relval/CMSSW_3_3_0_pre3/RelValZEE/GEN-SIM-RECO/MC_31X_V8-v1/0014/86C594B6-029F-DE11-90F9-001D09F26C5C.root')
)
process.pfAllElectrons = cms.EDProducer("PdgIdPFCandidateSelector",
    pdgId = cms.vint32(11, -11),
    src = cms.InputTag("pfNoPileUp")
)


process.gensource = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring('drop *', 
        'keep+ pdgId = 23', 
        'drop pdgId !=11 && pdgId !=-11')
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
    verbose = cms.untracked.bool(False)
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
    deltaRMax = cms.double(0.2),
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
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        optionalPSet = cms.untracked.bool(True)
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        optionalPSet = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        )
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
    categories = cms.untracked.vstring('FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
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
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False)
)


process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


