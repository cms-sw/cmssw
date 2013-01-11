import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackingTools.MaterialEffects.Propagators_cff")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")


process.GlobalTag.globaltag = 'GR_P_V32::All'

process.options   = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
#       '/store/user/farrell3/HSCPEDMMerged_14Sep2012/Run2012C_200000_200532.root'
#        '/store/user/farrell3/HSCPEDMUpdateData2012_12Sep2012/MC_8TeV_DYToMuMu/HSCP_10_1_AeT.root'
#        '/store/user/farrell3/HSCPEDMMerged_14Sep2012/MC_8TeV_DYToMuMu.root'
XXX_INPUT_XXX
#         'file:HSCP.root'
)
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000),
)    

process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.HLTTriggerMu = process.hltHighLevel.clone()
process.HLTTriggerMu.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
process.HLTTriggerMu.andOr = cms.bool( True ) #OR
process.HLTTriggerMu.throw = cms.bool( True )
process.HLTTriggerMu.HLTPaths = ["HLT_Mu40_eta2p1*"]
#process.HLTTriggerMuFilter = cms.Path(process.HLTTriggerMu   )

# Match to MC truth
process.muMcMatch = cms.EDProducer("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("muonsSkim"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles"),
)

process.tagMuons = cms.EDFilter("TagSelector",
    muons  = cms.untracked.InputTag("muonsSkim"),
    source = cms.untracked.InputTag("HSCParticleProducer"),
    inputDedxCollection =  cms.InputTag("dedxNPHarm2"),
    isProbe = cms.untracked.bool(False),
)

#process.tagMuons = cms.EDFilter("MuonRefSelector",
#  src = cms.InputTag("TightMuons"),
#  cut = cms.string(""),
#)

# Tk Probe collection
#process.tkProbes = cms.EDFilter("MuonRefSelector",
#    src = cms.InputTag("muonsSkim"),
#    cut = cms.string("isTrackerMuon && abs(eta)<2.1 && pt>40"),
#)

process.tkProbes = cms.EDFilter("TagSelector",
    muons  = cms.untracked.InputTag("muonsSkim"),
    source = cms.untracked.InputTag("HSCParticleProducer"),
    inputDedxCollection =  cms.InputTag("dedxNPHarm2"),
    isProbe = cms.untracked.bool(True),
)

process.PV = cms.EDProducer("PV",
    probes = cms.InputTag("muonsSkim"),
    makePV = cms.bool(True),
    makeSAEta = cms.bool(False),
    makeSAPt = cms.bool(False),
    makeWeight = cms.bool(False),
    isData = cms.bool(True),
)

if("XXX_OUTPUT_XXX".find("MC")!=-1):
    process.PV.isData = cms.bool(False)

process.SAPt = process.PV.clone()
process.SAPt.makePV = cms.bool(False)
process.SAPt.makeSAPt = cms.bool(True)

process.SAEta = process.PV.clone()
process.SAEta.makePV = cms.bool(False)
process.SAEta.makeSAEta = cms.bool(True)

process.Weight = process.PV.clone()
process.Weight.makePV = cms.bool(False)
process.Weight.makeWeight = cms.bool(True)

# SA Probe collection
#process.SAProbes = cms.EDFilter("MuonRefSelector",
#    src = cms.InputTag("muonsSkim"),
#    cut = cms.string("isStandAloneMuon")# && abs(SAEta)<2.1 && SAPt>40"),
#)

from RecoMuon.TrackingTools.MuonSegmentMatcher_cff import *
process.PassSA = cms.EDProducer("HSCP_TagProbe",
    MuonSegmentMatcher,
    src = cms.InputTag("tkProbes"),
    SAtracks = cms.InputTag("refittedStandAloneMuons", ""),
    Tktracks = cms.InputTag("TrackRefitter", ""),
    inputDedxCollection =  cms.InputTag("dedxNPHarm2"),
    SACut = cms.bool(True),
    TkCut = cms.bool(False),
    DzCut = cms.double(9999999.),
    DxyCut = cms.double(9999999.),
    StationCut = cms.int32(0),
    TOFNdofCut = cms.int32(0),
    TOFErrCut = cms.double(9999999.),
    SegSepCut = cms.double(-1.),
    SAPtCut = cms.double(-1.),
    SAEtaCut = cms.double(99999.9),
    minTrackHits = cms.int32(0),
    minPixelHits = cms.int32(0),
    minDeDxMeas = cms.uint32(0),
    maxV3D = cms.double(999999.),
    minQualityMask = cms.int32(0),
    maxTrackIso = cms.double(99999999.),
    maxEoP = cms.double(999999.),
    minFraction = cms.double(0),
    maxChi2 = cms.double(99999.),
    maxPtErr = cms.double(99999.),
    maxPhi = cms.double(-99999.),
    minPhi = cms.double(-99999.),
    timeRange = cms.double(-99999.)
)

process.SAProbes = process.PassSA.clone()
process.SAProbes.SAPtCut = cms.double(40.)
process.SAProbes.SAEtaCut = cms.double(2.1)
process.SAProbes.minPhi = cms.double(2.1)

process.PassStation = process.PassSA.clone()
process.PassStation.StationCut = cms.int32(2)

process.PassDxy = process.PassStation.clone()
process.PassDxy.DxyCut = cms.double(20.)

process.PassSegSep = process.PassDxy.clone()
process.PassSegSep.SegSepCut = cms.double(0.1)

process.PassPhi = process.PassSegSep.clone()
process.PassPhi.minPhi = cms.double(1.2)
process.PassPhi.maxPhi = cms.double(1.9)

process.PassTime = process.PassSegSep.clone()
process.PassTime.timeRange = cms.double(5.)

process.PassDz = process.PassTime.clone()
process.PassDz.DzCut = cms.double(15.)

process.PassTOFNdof = process.PassDz.clone()
process.PassTOFNdof.TOFNdofCut = cms.int32(8)

process.PassMuOnly = process.PassTOFNdof.clone()
process.PassMuOnly.TOFErrCut = cms.double(0.07)

process.AllButStation = process.PassMuOnly.clone()
process.AllButStation.StationCut = cms.int32(2)

process.AllButDxy = process.PassMuOnly.clone()
process.AllButDxy.DxyCut = cms.double(15.)

process.AllButSegSep = process.PassMuOnly.clone()
process.AllButSegSep.SegSepCut = cms.double(-1)

process.PassPxHits = process.PassSA.clone()
process.PassPxHits.SACut = cms.bool(False)
process.PassPxHits.TkCut = cms.bool(True)
process.PassPxHits.minPixelHits = cms.int32(2)

process.PassTkHits = process.PassPxHits.clone()
process.PassTkHits.minTrackHits = cms.int32(11)

process.PassDeDxMeas = process.PassPxHits.clone()
process.PassDeDxMeas.minDeDxMeas = cms.uint32(6)

process.PassV3D = process.PassDeDxMeas.clone()
process.PassV3D.maxV3D = cms.double(0.5)

process.PassQual = process.PassV3D.clone()
process.PassQual.minQualityMask = cms.int32(2)

process.PassTkIso = process.PassDeDxMeas.clone()
process.PassTkIso.maxTrackIso = cms.double(50)

process.PassEoP = process.PassTkIso.clone()
process.PassEoP.maxEoP = cms.double(0.3)

process.PassFrac = process.PassEoP.clone()
process.PassFrac.minFraction = cms.double(0.8)

process.PassChi2 = process.PassFrac.clone()
process.PassChi2.maxChi2 = cms.double(5)

process.PassPtErr = process.PassChi2.clone()
process.PassPtErr.maxPtErr = cms.double(0.25)

process.PassTk = process.PassPtErr.clone()

process.PassTkTOF = process.PassTk.clone()
process.PassTkTOF.TOFNdofCut = cms.int32(8)
process.PassTkTOF.TOFErrCut = cms.double(0.07)

process.PassTkSA = process.PassTk.clone()
process.PassTkSA.src = cms.InputTag("SAProbes")

process.PassTkTOFSA = process.PassTkTOF.clone()
process.PassTkTOFSA.src = cms.InputTag("SAProbes")

# Tag and Probe pairs
process.tkTagProbePairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("tagMuons@+ tkProbes@-"), # charge conjugate states are implied
    cut   = cms.string("70 < mass < 110"),
)

# Tag and Probe pairs
process.SATagProbePairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("tagMuons@+ SAProbes@-"), # charge conjugate states are implied
    cut   = cms.string("70 < mass < 110"),
)

process.probeMultiplicity = cms.EDProducer("ProbeMulteplicityProducer",
    pairs = cms.InputTag("tkTagProbePairs"),
)

# Make the fit tree and save it in the "MuonID" directory
process.SAMuonID = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tkTagProbePairs"),
    arbitration   = cms.string("None"),
    variables = cms.PSet(
        pt  = cms.string("pt"),
        eta = cms.string("eta"),
        phi = cms.string("phi"),
        PV = cms.InputTag("PV"),
        Weight  = cms.InputTag("Weight"),
    ),
    flags = cms.PSet(
        PassSAReco = cms.string("isStandAloneMuon"),
        PassSA = cms.InputTag("PassSA"),
        PassStation = cms.InputTag("PassStation"),
        PassDxy = cms.InputTag("PassDxy"),
	PassDz = cms.InputTag("PassDz"),
        PassTOFNdof = cms.InputTag("PassTOFNdof"),
        PassSegSep = cms.InputTag("PassSegSep"),
        PassTk = cms.InputTag("PassTk"),
        PassTkTOF = cms.InputTag("PassTkTOF"),
        PassMuOnly = cms.InputTag("PassMuOnly"),
        PassTkHits = cms.InputTag("PassTkHits"),
        PassPxHits = cms.InputTag("PassPxHits"),
        PassDeDxMeas = cms.InputTag("PassDeDxMeas"),
        PassV3D = cms.InputTag("PassV3D"),
        PassQual = cms.InputTag("PassQual"),
        PassTkIso = cms.InputTag("PassTkIso"),
        PassEoP = cms.InputTag("PassEoP"),
        PassFrac = cms.InputTag("PassFrac"),
        PassChi2 = cms.InputTag("PassChi2"),
        PassPtErr = cms.InputTag("PassPtErr"),
        AllButStation = cms.InputTag("AllButStation"),
        AllButDxy = cms.InputTag("AllButDxy"),
        AllButSegSep = cms.InputTag("AllButSegSep"),
    ),
    pairVariables = cms.PSet(
        dz      = cms.string("daughter(0).vz - daughter(1).vz"),
        pt      = cms.string("pt"), 
        rapidity = cms.string("rapidity"),
        deltaR   = cms.string("deltaR(daughter(0).eta, daughter(0).phi, daughter(1).eta, daughter(1).phi)"), 
        probeMultiplicity = cms.InputTag("probeMultiplicity"),
    ),
    pairFlags = cms.PSet(),
    isMC = cms.bool(False),
    #tagMatches = cms.InputTag("muMcMatch"),
    #probeMatches  = cms.InputTag("muMcMatch"),
    motherPdgId = cms.int32(443),
    makeMCUnbiasTree = cms.bool(False),
    #checkMotherInUnbiasEff = cms.bool(True),
    #allProbes     = cms.InputTag("trkProbes"),
    addRunLumiInfo = cms.bool(False),
)

process.TkMuonID = process.SAMuonID.clone()
process.TkMuonID.tagProbePairs = cms.InputTag("SATagProbePairs")
process.TkMuonID.flags = cms.PSet(
        PassTkSA = cms.InputTag("PassTkSA"),
        PassTkTOFSA = cms.InputTag("PassTkTOFSA"),
)
process.TkMuonID.variables = cms.PSet(
      pt  = cms.string("pt"),
      eta = cms.string("eta"),
      phi = cms.string("phi"),
      SAPt  = cms.InputTag("SAPt"),
      SAEta = cms.InputTag("SAEta"),
      PV = cms.InputTag("PV"),
)

process.tagAndProbe = cms.Path( 
    process.HLTTriggerMu *
#    process.TightMuons *
    process.tagMuons *
    process.tkProbes *
    process.SAProbes *
    process.PV *
    process.SAEta *
    process.SAPt *
    process.Weight *
    process.PassSA *
    process.PassStation *
    process.PassDxy *
    process.PassSegSep *
    process.PassPhi *
    process.PassTime *
    process.PassDz *
    process.PassTOFNdof *
    process.PassTkHits *
    process.PassPxHits *
    process.PassDeDxMeas *
    process.PassV3D *
    process.PassQual *
    process.PassTkIso *
    process.PassEoP *
    process.PassFrac *
    process.PassChi2 *
    process.PassPtErr *
    process.PassTk *
    process.PassTkSA *
    process.PassTkTOF *
    process.PassTkTOFSA *
    process.PassMuOnly *
    process.AllButStation *
    process.AllButDxy *
    process.AllButSegSep *
    process.probeMultiplicity *
    process.SATagProbePairs *
    process.tkTagProbePairs *
    process.SAMuonID 
#    process.TkMuonID
)

#process.end = cms.EndPath(process.Out)

process.p = cms.Schedule(process.tagAndProbe)

process.TFileService = cms.Service("TFileService", fileName = cms.string("XXX_PATH_XXX/TagProbeProducerRoot/XXX_OUTPUT_XXX.root"))

