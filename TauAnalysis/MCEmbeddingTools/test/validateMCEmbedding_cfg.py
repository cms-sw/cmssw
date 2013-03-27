import FWCore.ParameterSet.Config as cms

process = cms.Process("validateMCEmbedding")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.load('Configuration/Geometry/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string('START53_V7A::All')

print """
NOTE: You need to checkout the full set of TauAnalysis packages in order to run this config file.
      Installation instructions for the full set of TauAnalysis packages can be found at:
        https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTauAnalysis
"""      

#--------------------------------------------------------------------------------
# define configuration parameter default values

type = "EmbeddedMC"
##type = "MC"
srcReplacedMuons = 'genMuonsFromZs'
##srcReplacedMuons = ''
channel = 'etau'
##channel = 'mutau'
srcWeights = []
srcGenFilterInfo = "generator:minVisPtFilter"
##srcGenFilterInfo = ""
muonRadCorrectionsApplied = True
##muonRadCorrectionsApplied = False
##addTauPolValidationPlots = True
addTauPolValidationPlots = False
applyEmbeddingKineReweight = True
applyTauSpinnerWeight = True
##applyZmumuEvtSelEffCorrWeight = True
applyZmumuEvtSelEffCorrWeight = False
produceEmbeddingKineReweightNtuple = True
applyPileUpWeight = True
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define "hooks" for replacing configuration parameters
# in case running jobs on the CERN batch system/grid
#
#__type = "$type"
#__srcReplacedMuons = '$srcReplacedMuons'
#__channel = "$channel"
#__srcWeights = $srcWeights
#__srcGenFilterInfo = '$srcGenFilterInfo'
#__muonRadCorrectionsApplied = $muonRadCorrectionsApplied
#__addTauPolValidationPlots = $addTauPolValidationPlots
#__applyEmbeddingKineReweight = $applyEmbeddingKineReweight
#__applyTauSpinnerWeight = $applyTauSpinnerWeight
#__applyZmumuEvtSelEffCorrWeight = $applyZmumuEvtSelEffCorrWeight
#__produceEmbeddingKineReweightNtuple = $produceEmbeddingKineReweightNtuple
#__applyPileUpWeight = $applyPileUpWeight
isMC = None
if type == "MC" or type == "EmbeddedMC":
    isMC = True
elif type == "Data" or type == "EmbeddedData":
    isMC = False
else:
    raise ValueError("Invalid Configuration parameter 'type' = %s !!" % type)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data1/veelken/CMSSW_5_3_x/skims/simDYmumu_embedded_mutau_2013Mar19_AOD.root'
        ##'file:/data1/veelken/CMSSW_5_3_x/skims/simDYmumu_embedded_mutau_2013Mar19_woCaloRecHitMixing_AOD.root'                        
    ),
    ##eventsToProcess = cms.untracked.VEventRange(
    ##    '1:154452:61731259'
    ##)
    skipEvents = cms.untracked.uint32(0)            
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce collections of generator level electrons, muons and taus

# visible Pt cuts (same values applied on gen. and rec. level)
#electronPtThreshold = 13.0 # CV: for new e+MET+tau trigger
electronPtThreshold = 20.0 # CV: for comparison with HCP samples
#muonPtThreshold = 9.0 # CV: for new mu+MET+tau trigger
muonPtThreshold = 17.0 # CV: for comparison with HCP samples
tauPtThreshold = 20.0

process.genParticlesFromZs = cms.EDProducer("GenParticlesFromZsSelectorForMCEmbedding",
    src = cms.InputTag("genParticles"),
    pdgIdsMothers = cms.vint32(23, 22),
    pdgIdsDaughters = cms.vint32(15, 13, 11),
    maxDaughters = cms.int32(2),
    minDaughters = cms.int32(2),
    before_or_afterFSR = cms.string("afterFSR")
)
process.genTausFromZs = process.genParticlesFromZs.clone(
    pdgIdsDaughters = cms.vint32(15)
)
process.genZdecayToTaus = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('charge = 0'),
    decay = cms.string("genTausFromZs@+ genTausFromZs@-")
)
if not hasattr(process, "tauGenJets"):
    process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
process.genTauJetsFromZs = cms.EDProducer("TauGenJetMatchSelector",
    srcGenTauLeptons = cms.InputTag("genTausFromZs"),
    srcGenParticles = cms.InputTag("genParticles"),
    srcTauGenJets = cms.InputTag("tauGenJets"),
    dRmatchGenParticle = cms.double(0.1),
    dRmatchTauGenJet = cms.double(0.3)
)
process.genElectronsFromZtautauDecays = cms.EDFilter("TauGenJetDecayModeSelector",
    src = cms.InputTag("genTauJetsFromZs"),
    select = cms.vstring(
        "electron"
    ),
    filter = cms.bool(False)                                       
)
process.genElectronsFromZtautauDecaysWithinAcceptance = cms.EDFilter("GenJetSelector",
    src = cms.InputTag("genElectronsFromZtautauDecays"),
    cut = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold)
)
process.genMuonsFromZtautauDecays = cms.EDFilter("TauGenJetDecayModeSelector",
    src = cms.InputTag("genTauJetsFromZs"),
    # list of individual tau decay modes to be used for 'select' configuation parameter
    # define in PhysicsTools/JetMCUtils/src/JetMCTag.cc
    select = cms.vstring(
        "muon"
    ),
    filter = cms.bool(False)                                       
)
process.genMuonsFromZtautauDecaysWithinAcceptance = cms.EDFilter("GenJetSelector",
    src = cms.InputTag("genMuonsFromZtautauDecays"),
    cut = cms.string('pt > %1.1f & abs(eta) < 2.1' % muonPtThreshold)
)
process.genHadronsFromZtautauDecays = cms.EDFilter("TauGenJetDecayModeSelector",
    src = cms.InputTag("genTauJetsFromZs"),
    select = cms.vstring(
        "oneProng0Pi0",
        "oneProng1Pi0",
        "oneProng2Pi0",
        "oneProngOther",
        "threeProng0Pi0",
        "threeProng1Pi0",
        "threeProngOther",
        "rare"
    ),
    filter = cms.bool(False)                                       
)
process.genHadronsFromZtautauDecaysWithinAcceptance = cms.EDFilter("GenJetSelector",
    src = cms.InputTag("genHadronsFromZtautauDecays"),
    cut = cms.string('pt > %1.1f & abs(eta) < 2.3' % tauPtThreshold)
)
process.genLeptonSelectionSequence = cms.Sequence(
    process.genParticlesFromZs
   + process.genTausFromZs
   + process.genZdecayToTaus
   + process.tauGenJets
   + process.genTauJetsFromZs
   + process.genElectronsFromZtautauDecays
   + process.genElectronsFromZtautauDecaysWithinAcceptance
   + process.genMuonsFromZtautauDecays
   + process.genMuonsFromZtautauDecaysWithinAcceptance
   + process.genHadronsFromZtautauDecays
   + process.genHadronsFromZtautauDecaysWithinAcceptance
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce collections of reconstructed electrons, muons and taus

# define electron id & isolation selection
process.load("PhysicsTools/PatAlgos/patSequences_cff")
from PhysicsTools.PatAlgos.tools.pfTools import *
usePFIso(process)
process.patElectrons.addGenMatch = cms.bool(False)
process.genMatchedPatElectrons = cms.EDFilter("PATElectronAntiOverlapSelector",
    src = cms.InputTag("patElectrons"),
    srcNotToBeFiltered = cms.VInputTag("genElectronsFromZtautauDecaysWithinAcceptance"),
    dRmin = cms.double(0.3),
    invert = cms.bool(True),
    filter = cms.bool(False)                                                          
)
process.selectedElectronsIdMVA = cms.EDFilter("PATElectronIdSelector",
    src = cms.InputTag("genMatchedPatElectrons"),
    srcRecHitsEB = cms.InputTag("reducedEcalRecHitsEB"),
    srcRecHitsEE = cms.InputTag("reducedEcalRecHitsEE"),   
    srcVertex = cms.InputTag("goodVertex"),
    cut = cms.string("tight"),                                       
    filter = cms.bool(False)
)
process.selectedElectronsConversionVeto = cms.EDFilter("NPATElectronConversionFinder",
    src = cms.InputTag("selectedElectronsIdMVA"),                                                         
    maxMissingInnerHits = cms.int32(0),
    minMissingInnerHits = cms.int32(0),
    minRxy = cms.double(2.0),
    minFitProb = cms.double(1.e-6),
    maxHitsBeforeVertex = cms.int32(0),
    invertConversionVeto = cms.bool(False)
)
process.goodElectrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("selectedElectronsConversionVeto"),
    cut = cms.string(
        'pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold
    ),
    filter = cms.bool(False)
)
process.goodElectronsPFIso = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("goodElectrons"),
    cut = cms.string(
        '(chargedHadronIso()' + \
        ' + max(0., neutralHadronIso() + photonIso()' + \
        '          - puChargedHadronIso())) < 0.10*pt'
    ),                                
    filter = cms.bool(False)
)

process.recElectronSelectionSequence = cms.Sequence(
    process.pfParticleSelectionSequence
   + process.eleIsoSequence
   + process.patElectrons
   + process.genMatchedPatElectrons
   + process.selectedElectronsIdMVA
   + process.selectedElectronsConversionVeto
   + process.goodElectrons
   + process.goodElectronsPFIso
)

# define muon id & isolation selection
process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandaloneSelection_cff")
process.genMatchedPatMuons = cms.EDFilter("PATMuonAntiOverlapSelector",
    src = cms.InputTag("patMuonsForZmumuSelection"),
    srcNotToBeFiltered = cms.VInputTag("genMuonsFromZtautauDecaysWithinAcceptance"),
    dRmin = cms.double(0.3),
    invert = cms.bool(True),
    filter = cms.bool(False)                                                          
)
process.goodMuons.src = cms.InputTag("genMatchedPatMuons")
process.goodMuons.cut = cms.string(
    'pt > %1.1f & abs(eta) < 2.1 & isGlobalMuon & isPFMuon ' 
    ' & track.hitPattern.trackerLayersWithMeasurement > 5 & innerTrack.hitPattern.numberOfValidPixelHits > 0'
    ' & abs(dB) < 0.2 & globalTrack.normalizedChi2 < 10'
    ' & globalTrack.hitPattern.numberOfValidMuonHits > 0 & numberOfMatchedStations > 1' % muonPtThreshold
)
process.goodMuonsPFIso = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("goodMuons"),
    cut = cms.string(
        '(userIsolation("pat::User1Iso")' + \
        ' + max(0., userIsolation("pat::PfNeutralHadronIso") + userIsolation("pat::PfGammaIso")' + \
        '          - 0.5*userIsolation("pat::User2Iso"))) < 0.10*pt'
    ),
    filter = cms.bool(False)
)
process.recMuonSelectionSequence = cms.Sequence(
    process.muIsoSequence
   + process.patMuonsForZmumuSelection
   + process.genMatchedPatMuons
   + process.goodMuons
   + process.goodMuonsPFIso  
)

# define hadronic tau id selection
process.load("RecoTauTag/Configuration/RecoPFTauTag_cff")
switchToPFTauHPS(process)
process.load("PhysicsTools/PatAlgos/producersLayer1/tauProducer_cfi")
process.patTaus.addGenMatch = cms.bool(False)
process.patTaus.addGenJetMatch = cms.bool(False)
process.patTaus.isoDeposits = cms.PSet()
process.patTaus.userIsolation = cms.PSet()
process.genMatchedPatTaus = cms.EDFilter("PATTauAntiOverlapSelector",
    src = cms.InputTag("patTaus"),
    srcNotToBeFiltered = cms.VInputTag("genHadronsFromZtautauDecaysWithinAcceptance"),
    dRmin = cms.double(0.3),
    invert = cms.bool(True),
    filter = cms.bool(False)                                                          
)
tauDiscrByIsolation = "tauID('byLooseIsolationMVA') > 0.5"
tauDiscrAgainstElectrons = None
tauDiscrAgainstMuons = None
if channel == "etau":
    tauDiscrAgainstElectrons = "tauID('againstElectronMVA') > 0.5 & tauID('againstElectronTightMVA2') > 0.5 & "
    tauDiscrAgainstMuons = "tauID('againstMuonLoose') > 0.5"
elif channel == "mutau":
    tauDiscrAgainstElectrons = "tauID('againstElectronLoose') > 0.5"
    tauDiscrAgainstMuons = "tauID('againstMuonTight') > 0.5"
else:
    raise ValueError("Invalid Configuration parameter 'channel' = %s !!" % channel)
#--------------------------------------------------------------------------------
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
##process.printGenParticleListSIM = cms.EDAnalyzer("ParticleListDrawer",
##    src = cms.InputTag("genParticles::SIM"),
##    maxEventsToPrint = cms.untracked.int32(100)
##)
##process.printGenZsSIM = cms.EDAnalyzer("DumpGenZs",
##    src = cms.InputTag("genParticles::SIM")
##)  
##process.printGenParticleListEmbeddedRECO = cms.EDAnalyzer("ParticleListDrawer",
##    src = cms.InputTag("genParticles::EmbeddedRECO"),
##    maxEventsToPrint = cms.untracked.int32(100)
##)
##process.printGenZsEmbeddedRECO = cms.EDAnalyzer("DumpGenZs",
##    src = cms.InputTag("genParticles::EmbeddedRECO")
##)
##process.dumpVertices = cms.EDAnalyzer("DumpVertices",
##    src = cms.InputTag("offlinePrimaryVertices")
##)
##from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
##process.dumpTaus = cms.EDAnalyzer("DumpPATTausForRaman",
##    src = cms.InputTag("genMatchedPatTaus"),
##    srcVertex = cms.InputTag('goodVertex'),                              
##    minPt = cms.double(0.),
##    signalQualityCuts = PFTauQualityCuts.signalQualityCuts,
##    isolationQualityCuts = PFTauQualityCuts.isolationQualityCuts                           
##)
##process.dumpTaus.signalQualityCuts.maxDeltaZ = cms.double(1.e+3)
##process.dumpTaus.signalQualityCuts.maxTransverseImpactParameter = cms.double(1.e+3)
##process.dumpTaus.isolationQualityCuts.maxDeltaZ = cms.double(1.e+3)
##process.dumpTaus.isolationQualityCuts.maxTransverseImpactParameter = cms.double(1.e+3)
#--------------------------------------------------------------------------------
process.selectedTaus = cms.EDFilter("PATTauSelector",
    src = cms.InputTag("genMatchedPatTaus"),
    cut = cms.string(
        "pt > 20.0 & abs(eta) < 2.3 & tauID('decayModeFinding') > 0.5 & tauID('byLooseIsolationMVA') > 0.5"                                
    )
)
process.recTauSelectionSequence = cms.Sequence(
    process.recoTauCommonSequence
   + process.recoTauClassicHPSSequence
   + process.patTaus
   + process.genMatchedPatTaus
   ##+ process.printGenParticleListSIM
   ##+ process.printGenZsSIM
   ##+ process.printGenParticleListEmbeddedRECO
   ##+ process.printGenZsEmbeddedRECO
   ##+ process.dumpTaus
   + process.selectedTaus
)

process.recLeptonSelectionSequence = cms.Sequence(
    process.goodVertex # CV: defined in TauAnalysis/MCEmbeddingTools/python/ZmumuStandaloneSelection_cff.py
   + process.recElectronSelectionSequence
   + process.recMuonSelectionSequence
   + process.recTauSelectionSequence
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce collections of jets

# configure pat::Jet production
# (enable L2L3Residual corrections in case running on Data)
jetCorrections = [ 'L1FastJet', 'L2Relative', 'L3Absolute' ]
if not isMC:
    jetCorrections.append('L2L3Residual')

from PhysicsTools.PatAlgos.tools.jetTools import *
switchJetCollection(
    process,
    cms.InputTag('ak5PFJets'),
    doJTA = True,
    doBTagging = True,
    jetCorrLabel = ( 'AK5PF', cms.vstring(jetCorrections) ),
    doType1MET = False,
    doJetID = True,
    jetIdLabel = "ak5",
    outputModules = []
)

process.patJetsNotOverlappingWithLeptons = cms.EDFilter("PATJetAntiOverlapSelector",
    src = cms.InputTag('patJets'),
    srcNotToBeFiltered = cms.VInputTag(
        'goodElectrons',
        'goodMuons',
        'selectedTaus'
    ),
    dRmin = cms.double(0.5),
    invert = cms.bool(False),
    filter = cms.bool(False)                                                          
)

process.recJetSequence = cms.Sequence(process.jetTracksAssociatorAtVertex + process.btaggingAOD + process.makePatJets + process.patJetsNotOverlappingWithLeptons)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce collections of PFMET and CaloMET
# with and without Type-1 corrections applied.
# In case of CaloMET, additionally produce collections with and without
# HF calorimeter included in MET reconstruction.

process.load("PhysicsTools/PatUtils/patPFMETCorrections_cff")
process.recPFMetSequence = cms.Sequence(
    process.patPFMet
   + process.pfCandsNotInJet
   + process.selectedPatJetsForMETtype1p2Corr
   + process.patPFJetMETtype1p2Corr
   + process.type0PFMEtCorrection
   + process.patPFMETtype0Corr
   + process.pfCandMETcorr 
   + process.patType1CorrectedPFMet
)

process.patCaloMet = process.patMETs.clone(
    metSource = cms.InputTag('corMetGlobalMuons'),
    addMuonCorrections = cms.bool(False),
    genMETSource = cms.InputTag('genMetCalo')
)
process.load("RecoMET/METProducers/MetMuonCorrections_cff")
process.corMetGlobalMuonsNoHF = process.corMetGlobalMuons.clone(
    uncorMETInputTag = cms.InputTag('metNoHF')
)
process.patCaloMetNoHF = process.patCaloMet.clone(
    metSource = cms.InputTag('corMetGlobalMuonsNoHF')
)
process.load("JetMETCorrections/Type1MET/caloMETCorrections_cff")
process.caloJetMETcorr.srcMET = cms.InputTag('met')
process.caloJetMETcorr.jetCorrLabel = cms.string("ak5CaloL2L3")
process.caloType1CorrectedMet.src = cms.InputTag('corMetGlobalMuons')
process.patType1CorrectedCaloMet = process.patMETs.clone(
    metSource = cms.InputTag('caloType1CorrectedMet'),
    addMuonCorrections = cms.bool(False),
    genMETSource = cms.InputTag('genMetCalo')
)
process.caloType1CorrectedMetNoHF = process.caloType1CorrectedMet.clone(
    src = cms.InputTag('corMetGlobalMuonsNoHF')
)
process.patType1CorrectedCaloMetNoHF = process.patType1CorrectedCaloMet.clone(
    metSource = cms.InputTag('caloType1CorrectedMetNoHF')
)
process.recCaloMetSequence = cms.Sequence(
    process.patCaloMet
   + process.corMetGlobalMuonsNoHF
   + process.patCaloMetNoHF
   + process.caloJetMETcorr
   + process.caloType1CorrectedMet
   + process.patType1CorrectedCaloMet
   + process.caloType1CorrectedMetNoHF
   + process.patType1CorrectedCaloMetNoHF
)

process.load("RecoMET.METProducers.TrackMET_cfi")
process.patTrackMet = process.patMETs.clone(
    metSource = cms.InputTag('trackMet'),
    genMETSource = cms.InputTag('genMetTrue')
)
process.recTrackMetSequence = cms.Sequence(
    process.trackMet
   + process.patTrackMet
)    

process.recMetSequence = cms.Sequence(
    process.recPFMetSequence
   + process.recCaloMetSequence
   + process.recTrackMetSequence
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# require event to contain generator level tau lepton pair,
# decaying in the sprecified channel

process.genDecayModeFilterSequence = cms.Sequence()
process.genDecayModeAndAcceptanceFilterSequence = cms.Sequence()

numElectrons = None
numMuons     = None
numTauJets   = None
if channel == 'mutau':
    numElectrons = 0
    numMuons     = 1
    numTauJets   = 1
elif channel == 'etau':
    numElectrons = 1
    numMuons     = 0
    numTauJets   = 1
elif channel == 'emu':
    numElectrons = 1
    numMuons     = 1
    numTauJets   = 0
elif channel == 'tautau':
    numElectrons = 0
    numMuons     = 0
    numTauJets   = 2  
else:
    raise ValueError("Invalid Configuration parameter 'channel' = %s !!" % channel)

if numElectrons > 0:
    process.electronFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag('genElectronsFromZtautauDecays'),
        minNumber = cms.uint32(numElectrons),
        maxNumber = cms.uint32(numElectrons)                      
    )
    process.genDecayModeFilterSequence += process.electronFilter
    process.electronFilterWithinAcceptance = process.electronFilter.clone(
        src = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
    ) 
    process.genDecayModeAndAcceptanceFilterSequence += process.electronFilterWithinAcceptance

if numMuons > 0:
    process.muonFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag('genMuonsFromZtautauDecays'),
        minNumber = cms.uint32(numMuons),
        maxNumber = cms.uint32(numMuons)                      
    )
    process.genDecayModeFilterSequence += process.muonFilter
    process.muonFilterWithinAcceptance = process.muonFilter.clone(
        src = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance')
    )        
    process.genDecayModeAndAcceptanceFilterSequence += process.muonFilterWithinAcceptance

if numTauJets > 0:
    process.tauFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag('genHadronsFromZtautauDecays'),
        minNumber = cms.uint32(numTauJets),
        maxNumber = cms.uint32(numTauJets)                      
    )
    process.genDecayModeFilterSequence += process.tauFilter
    process.tauFilterWithinAcceptance = process.tauFilter.clone(
        src = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
    )
    process.genDecayModeAndAcceptanceFilterSequence += process.tauFilterWithinAcceptance
#--------------------------------------------------------------------------------

srcGenLeg1 = None
srcRecLeg1 = None
srcGenLeg2 = None
srcRecLeg2 = None
if channel == 'mutau':
    srcGenLeg1 = 'genMuonsFromZtautauDecays'
    srcRecLeg1 = 'goodMuonsPFIso'
    srcGenLeg2 = 'genHadronsFromZtautauDecays'
    srcRecLeg2 = 'selectedTaus'
elif channel == 'etau':
    srcGenLeg1 = 'genElectronsFromZtautauDecays'
    srcRecLeg1 = 'goodElectronsPFIso'
    srcGenLeg2 = 'genHadronsFromZtautauDecays'
    srcRecLeg2 = 'selectedTaus'
elif channel == 'emu':
    srcGenLeg1 = 'genElectronsFromZtautauDecays'
    srcRecLeg1 = 'goodElectronsPFIso'
    srcGenLeg2 = 'genHadronsFromZtautauDecays'
    srcRecLeg2 = 'selectedTaus'
elif channel == 'tautau':
    srcGenLeg1 = 'genHadronsFromZtautauDecays'
    srcRecLeg1 = 'selectedTaus'
    srcGenLeg2 = 'genHadronsFromZtautauDecays'
    srcRecLeg2 = 'selectedTaus'

process.embeddingWeightProdSequence = cms.Sequence()

if applyEmbeddingKineReweight:
    process.load("TauAnalysis/MCEmbeddingTools/embeddingKineReweight_cff")
    if channel == 'mutau':
        process.embeddingKineReweightGENtoEmbedded.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoEmbedded_mutau.root")
    elif channel == 'etau':
        process.embeddingKineReweightGENtoEmbedded.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoEmbedded_etau.root")
    else:
        raise ValueError("No makeEmbeddingKineReweightLUTs_GENtoEmbedded file defined for channel = %s !!" % channel)
    process.embeddingWeightProdSequence += process.embeddingKineReweightSequence
    srcWeights.extend([
        'embeddingKineReweightGENtoEmbedded:genDiTauMassVsGenDiTauPt',
        'embeddingKineReweightGENtoEmbedded:genTau2PtVsGenTau1Pt'
    ])
if applyTauSpinnerWeight:
    process.load("TauSpinnerInterface/TauSpinnerInterface/TauSpinner_cfi")
    process.TauSpinnerReco.CMSEnergy = cms.double(8000.)
    process.embeddingWeightProdSequence += process.TauSpinnerReco
    srcWeights.extend([
        'TauSpinnerReco:TauSpinnerWT'
    ])
if applyZmumuEvtSelEffCorrWeight:
    #--------------------------------------------------------------------------------
    # CV: rerun ZmumuEvtSelEffCorrWeightProducer module
    #     in order to run on "old" Embedding samples produced by Armin
    process.load("TauAnalysis/MCEmbeddingTools/ZmumuEvtSelEffCorrWeightProducer_cfi")
    process.ZmumuEvtSelEffCorrWeightProducer.selectedMuons = cms.InputTag(srcReplacedMuons)
    process.embeddingWeightProdSequence += process.ZmumuEvtSelEffCorrWeightProducer
    #--------------------------------------------------------------------------------
    srcWeights.extend([
        'ZmumuEvtSelEffCorrWeightProducer:weight'
    ])
    if not hasattr(process, "embeddingKineReweightSequence"):
        process.load("TauAnalysis/MCEmbeddingTools/embeddingKineReweight_cff")
    process.embeddingWeightProdSequence += process.embeddingKineReweightSequence
    srcWeights.extend([
        'embeddingKineReweightGENtoREC:genDiTauPt',
        'embeddingKineReweightGENtoREC:genDiTauMass'
    ])
if applyPileUpWeight:
    process.load("TauAnalysis/MCEmbeddingTools/PileUpWeightProducer_cfi")
    process.embeddingWeightProdSequence += process.PileUpWeightProducerPUS10vs2012A
    srcWeights.extend(['PileUpWeightProducerPUS10vs2012A:weight'])
print "setting 'srcWeights' = %s" % srcWeights

process.validationAnalyzerDR00 = cms.EDAnalyzer("MCEmbeddingValidationAnalyzer",
    srcReplacedMuons = cms.InputTag(srcReplacedMuons),                                        
    replacedMuonPtThresholdHigh = cms.double(17.),
    replacedMuonPtThresholdLow = cms.double(8.),                                            
    srcRecMuons = cms.InputTag('muons'),
    srcRecTracks = cms.InputTag('generalTracks'),
    srcCaloTowers = cms.InputTag('towerMaker'),
    srcRecPFCandidates = cms.InputTag('particleFlow'),
    srcRecJets = cms.InputTag('patJetsNotOverlappingWithLeptons'),                                        
    srcTheRecVertex = cms.InputTag('goodVertex'),
    srcRecVertices = cms.InputTag('offlinePrimaryVertices'),
    srcRecVerticesWithBS = cms.InputTag('offlinePrimaryVerticesWithBS'),
    srcBeamSpot = cms.InputTag('offlineBeamSpot'),
    srcGenDiTaus = cms.InputTag('genZdecayToTaus'),
    dRminSeparation = cms.double(-1.), # CV: minimum separation in dR between replaced muons and embedded tau leptons
    ZPtMin = cms.double(-1.),
    ZPtMax = cms.double(-1.),
    srcGenLeg1 = cms.InputTag(srcGenLeg1),
    srcRecLeg1 = cms.InputTag(srcRecLeg1),                                        
    srcGenLeg2 = cms.InputTag(srcGenLeg2),
    srcRecLeg2 = cms.InputTag(srcRecLeg2),
    srcGenParticles = cms.InputTag('genParticles'),                                          
    srcL1ETM = cms.InputTag('l1extraParticles', 'MET'),
    srcGenMEt = cms.InputTag('genMetTrue'),
    srcRecPFMEt = cms.InputTag('patType1CorrectedPFMet'),                                        
    srcRecCaloMEt = cms.InputTag('patCaloMetNoHF'),
    srcMuonsBeforeRad = cms.InputTag('generator', 'muonsBeforeRad'),
    srcMuonsAfterRad = cms.InputTag('generator', 'muonsAfterRad'),                                        
    srcMuonRadCorrWeight = cms.InputTag('muonRadiationCorrWeightProducer', 'weight'),
    srcMuonRadCorrWeightUp = cms.InputTag('muonRadiationCorrWeightProducer', 'weightUp'),
    srcMuonRadCorrWeightDown = cms.InputTag('muonRadiationCorrWeightProducer', 'weightDown'),                                                
    srcOtherWeights = cms.VInputTag(srcWeights),
    srcGenFilterInfo = cms.InputTag(srcGenFilterInfo),                                        
    dqmDirectory = cms.string("validationAnalyzerDR00_%s" % channel),                                        
                                            
    # electron Pt, eta and phi distributions;
    # electron id & isolation and trigger efficiencies
    electronDistributions = cms.VPSet(
        #------------------------------------------------------------------------
        # CV: extra plots for checking electron Id MVA input variables
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            dqmDirectory = cms.string('genMatchedPatElectronDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & trackerDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),                                        
            dqmDirectory = cms.string('genMatchedPatPFElectronBarrelTrkSeedDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & trackerDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),                                        
            dqmDirectory = cms.string('genMatchedPatNonPFElectronBarrelTrkSeedDistributions')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & ecalDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),                                        
            dqmDirectory = cms.string('genMatchedPatPFElectronBarrelECALseedDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & ecalDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),                                        
            dqmDirectory = cms.string('genMatchedPatNonPFElectronBarrelECALseedDistributions')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & trackerDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),                                            
            dqmDirectory = cms.string('genMatchedPatPFElectronEndcapTrkSeedDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & trackerDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),                                            
            dqmDirectory = cms.string('genMatchedPatNonPFElectronEndcapTrkSeedDistributions')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & ecalDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),                                             
            dqmDirectory = cms.string('genMatchedPatPFElectronEndcapECALseedDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & ecalDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),                                             
            dqmDirectory = cms.string('genMatchedPatNonPFElectronEndcapECALseedDistributions')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedElectronsIdMVA'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold), 
            dqmDirectory = cms.string('selectedElectronIdMVAdistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedElectronsConversionVeto'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold),
            dqmDirectory = cms.string('selectedElectronConversionVetoDistributions')
        ),
        #------------------------------------------------------------------------
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectrons'),
            dqmDirectory = cms.string('goodElectronDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectronsPFIso'),
            dqmDirectory = cms.string('goodIsoElectronDistributions')
        )
    ),
    electronEfficiencies = cms.VPSet(
        #------------------------------------------------------------------------
        # CV: extra plots for checking electron Id MVA input variables
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            dqmDirectory = cms.string('genMatchedPatElectronEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & trackerDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatPFElectronBarrelTrkSeedEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & trackerDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatNonPFElectronBarrelTrkSeedEfficiencies')
        ),                                
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & ecalDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatPFElectronBarrelECALseedEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 1.479' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) < 1.444 & ecalDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatNonPFElectronBarrelECALseedEfficiencies')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & trackerDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatPFElectronEndcapTrkSeedEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & trackerDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatNonPFElectronEndcapTrkSeedEfficiencies')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & ecalDrivenSeed & superCluster.id.productIndex != pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatPFElectronEndcapECALseedEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) > 1.479 & abs(eta) < 2.1' % electronPtThreshold),
            cutRec = cms.string('abs(superCluster.eta) > 1.566 & ecalDrivenSeed & superCluster.id.productIndex = pflowSuperCluster.id.productIndex'),
            dqmDirectory = cms.string('genMatchedPatNonPFElectronEndcapECALseedEfficiencies')
        ),                                            
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatElectrons'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold), 
            dqmDirectory = cms.string('genMatchedPatElectronWithinAccEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedElectronsIdMVA'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold), 
            dqmDirectory = cms.string('selectedElectronIdMVAefficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedElectronsConversionVeto'),
            cutGen = cms.string('pt > %1.1f & abs(eta) < 2.1' % electronPtThreshold), 
            dqmDirectory = cms.string('selectedElectronConversionVetoEfficiencies')
        ),
        #------------------------------------------------------------------------                                        
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectrons'),
            dqmDirectory = cms.string('goodElectronEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectrons'),
            cutRec = cms.string("abs(superCluster.eta) < 1.444"),                                    
            dqmDirectory = cms.string('goodElectronEfficiencies/Barrel')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectrons'),
            cutRec = cms.string("abs(superCluster.eta) > 1.566"),                                    
            dqmDirectory = cms.string('goodElectronEfficiencies/Endcap')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectronsPFIso'),
            dqmDirectory = cms.string('goodIsoElectronEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectronsPFIso'),
            cutRec = cms.string("abs(superCluster.eta) < 1.444"),                                    
            dqmDirectory = cms.string('goodIsoElectronEfficiencies/Barrel')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectronsPFIso'),
            cutRec = cms.string("abs(superCluster.eta) > 1.566"),                                    
            dqmDirectory = cms.string('goodIsoElectronEfficiencies/Endcap')
        )
    ),		
    electronL1TriggerEfficiencies = cms.VPSet(					
        cms.PSet(
            srcRef = cms.InputTag('goodElectrons'),
            cutRef = cms.string("pt > %1.1f & abs(eta) < 2.1" % electronPtThreshold),                                        
            srcL1 = cms.InputTag('l1extraParticles', 'NonIsolated'),
            cutL1 = cms.string("pt > %1.1f & abs(eta) < 2.1" % (electronPtThreshold - 1.0)),
            dqmDirectory = cms.string('electronTriggerEfficiencyL1_Elec12wrtGoodElectrons')
        ),
        cms.PSet(
	    srcRef = cms.InputTag('goodElectronsPFIso'),
            cutRef = cms.string("pt > %1.1f & abs(eta) < 2.1" % electronPtThreshold),
            srcL1 = cms.InputTag('l1extraParticles', 'Isolated'),
            cutL1 = cms.string("pt > %1.1f & abs(eta) < 2.1" % (electronPtThreshold - 1.0)),	    
            dqmDirectory = cms.string('electronTriggerEfficiencyL1_IsoElec12wrtGoodIsoElectrons')
        )
    ),	

    # muon Pt, eta and phi distributions;
    # muon id & isolation and trigger efficiencies	
    muonDistributions = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodMuons'),
            dqmDirectory = cms.string('goodMuonDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodMuonsPFIso'),
            dqmDirectory = cms.string('goodIsoMuonDistributions')
        )
    ),
    muonEfficiencies = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodMuons'),
            dqmDirectory = cms.string('goodMuonEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodMuonsPFIso'),
            dqmDirectory = cms.string('goodIsoMuonEfficiencies')
        )
    ),	
    muonL1TriggerEfficiencies = cms.VPSet(					
        cms.PSet(
	    srcRef = cms.InputTag('goodMuons'),
            cutRef = cms.string("pt > %1.1f & abs(eta) < 2.1" % muonPtThreshold),
            srcL1 = cms.InputTag('l1extraParticles'),
            cutL1 = cms.string("pt > %1.1f & abs(eta) < 2.1" % (muonPtThreshold - 1.0)),	    
            dqmDirectory = cms.string('muonTriggerEfficiencyL1_Mu8wrtGoodMuons')
        ),
        cms.PSet(            
	    srcRef = cms.InputTag('goodMuonsPFIso'),
            cutRef = cms.string("pt > %1.1f  & abs(eta) < 2.1" % muonPtThreshold),
	    srcL1 = cms.InputTag('l1extraParticles'),
            cutL1 = cms.string("pt > %1.1f  & abs(eta) < 2.1" % (muonPtThreshold - 1.0)),
            dqmDirectory = cms.string('muonTriggerEfficiencyL1_Mu8wrtGoodIsoMuons')
        )
    ),		

    # tau Pt, eta and phi distributions;
    # tau id efficiency
    tauDistributions = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedTaus'),
            dqmDirectory = cms.string('selectedTauDistributions')
        )
    ),
    tauEfficiencies = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedTaus'),
            dqmDirectory = cms.string('selectedTauEfficiencies')
        )
    ),

    # Pt, eta and phi distribution of L1Extra objects
    # (electrons, muons, tau-jet, central and forward jets)                                     
    l1ElectronDistributions = cms.VPSet(
        cms.PSet(
	    src = cms.InputTag('l1extraParticles', 'NonIsolated'),
            cut = cms.string("pt > %1.1f & abs(eta) < 2.1" % (electronPtThreshold - 1.0)),
            dqmDirectory = cms.string('l1ElectronDistributions')
        ),                                        
        cms.PSet(
	    src = cms.InputTag('l1extraParticles', 'Isolated'),
            cut = cms.string("pt > %1.1f & abs(eta) < 2.1" % (electronPtThreshold - 1.0)),
            dqmDirectory = cms.string('l1IsoElectronDistributions')
        )
    ),
    l1MuonDistributions = cms.VPSet(					
        cms.PSet(
	    src = cms.InputTag('l1extraParticles'),
            cut = cms.string("pt > %1.1f & abs(eta) < 2.1" % (muonPtThreshold - 1.0)),
            dqmDirectory = cms.string('l1MuonDistributions')
        )
    ),
    l1TauDistributions = cms.VPSet(					
        cms.PSet(
	    src = cms.InputTag('l1extraParticles', 'Tau'),
            cut = cms.string("pt > %1.1f" % (tauPtThreshold - 5.0)),
            dqmDirectory = cms.string('l1TauDistributions')
        )
    ),
    l1CentralJetDistributions = cms.VPSet(					
        cms.PSet(
	    src = cms.InputTag('l1extraParticles', 'Central'),
            cut = cms.string("pt > 20."),
            dqmDirectory = cms.string('l1CentralJetDistributions')
        )
    ),
    l1ForwardJetDistributions = cms.VPSet(					
        cms.PSet(
	    src = cms.InputTag('l1extraParticles', 'Forward'),
            cut = cms.string("pt > 20."),
            dqmDirectory = cms.string('l1ForwardJetDistributions')
        )
    ),

    # MET Pt and phi distributions;
    # efficiency of L1 (Calo)MET trigger requirement
    metDistributions = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patCaloMet'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),	 
            dqmDirectory = cms.string('rawCaloMEtDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patType1CorrectedCaloMet'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('type1CorrCaloMEtDistributions')
        ),
	cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patCaloMetNoHF'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('rawCaloMEtNoHFdistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patType1CorrectedCaloMetNoHF'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('type1CorrCaloMEtNoHFdistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patTrackMet'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('trackMEtDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patPFMet'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('rawPFMEtDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genMetTrue'),
	    srcRec = cms.InputTag('patType1CorrectedPFMet'),
  	    srcGenZs = cms.InputTag('genZdecayToTaus'),
            dqmDirectory = cms.string('type1CorrPFMEtDistributions')
        )
    ),
    metL1TriggerEfficiencies = cms.VPSet(					
        cms.PSet(
	    srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(20.),
            cutL1Pt = cms.double(-1.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM20_et')
        ),
        cms.PSet(
	    srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(-1.),
            cutL1Pt = cms.double(20.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM20_pt')
        ),                                        
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(26.),
            cutL1Pt = cms.double(-1.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM26_et')
        ),
        cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(-1.),
            cutL1Pt = cms.double(26.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM26_pt')
        ),                                        
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(30.),
            cutL1Pt = cms.double(-1.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM30_et')
        ),
        cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(-1.),
            cutL1Pt = cms.double(30.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM30_pt')
        ),                                        
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(36.),
            cutL1Pt = cms.double(-1.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM36_et')
        ),
        cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(-1.),
            cutL1Pt = cms.double(36.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM36_pt')
        ),                                        
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(40.),
            cutL1Pt = cms.double(-1.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM40_et')
        ),
        cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Et = cms.double(-1.),
            cutL1Pt = cms.double(40.),                                        
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM40_pt')
        )                                        
    ),

    verbosity = cms.int32(0)
)

if muonRadCorrectionsApplied:
    process.validationAnalyzerDR00.srcMuonsBeforeRad = cms.InputTag('generator', 'muonsBeforeRad')
    process.validationAnalyzerDR00.srcMuonsAfterRad = cms.InputTag('generator', 'muonsAfterRad')
    process.validationAnalyzerDR00.srcMuonRadCorrWeight = cms.InputTag('muonRadiationCorrWeightProducer', 'weight')
    process.validationAnalyzerDR00.srcMuonRadCorrWeightUp = cms.InputTag('muonRadiationCorrWeightProducer', 'weightUp')
    process.validationAnalyzerDR00.srcMuonRadCorrWeightDown = cms.InputTag('muonRadiationCorrWeightProducer', 'weightDown')
else:
    process.validationAnalyzerDR00.srcMuonsBeforeRad = cms.InputTag('')
    process.validationAnalyzerDR00.srcMuonsAfterRad = cms.InputTag('')
    process.validationAnalyzerDR00.srcMuonRadCorrWeight = cms.InputTag('')
    process.validationAnalyzerDR00.srcMuonRadCorrWeightUp = cms.InputTag('')
    process.validationAnalyzerDR00.srcMuonRadCorrWeightDown = cms.InputTag('')

def excludeFromWeights(srcWeights, srcWeightsToExclude):
    retVal = []
    for srcWeight in srcWeights:
        isToBeExcluded = False
        for srcWeightToExclude in srcWeightsToExclude:
            if srcWeightToExclude == srcWeight:
                isToBeExcluded = True
        if not isToBeExcluded:
            retVal.append(srcWeight)
    return retVal

process.validationAnalyzerDR00_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPt = process.validationAnalyzerDR00.clone(
    srcWeights = cms.VInputTag(excludeFromWeights(srcWeights, [
        'embeddingKineReweightGENtoEmbedded:genDiTauMassVsGenDiTauPt'
    ])),
    dqmDirectory = cms.string("validationAnalyzer_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPt_%s" % channel),        
)
process.validationAnalyzerDR00_woWeightGENtoEmbedded_genTau2PtVsGenTau1Pt = process.validationAnalyzerDR00.clone(
    srcWeights = cms.VInputTag(excludeFromWeights(srcWeights, [
        'embeddingKineReweightGENtoEmbedded:genTau2PtVsGenTau1Pt'
    ])),
    dqmDirectory = cms.string("validationAnalyzer_woWeightGENtoEmbedded_genTau2PtVsGenTau1Pt_%s" % channel),        
)
process.validationAnalyzerDR00_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPtAndGenTau2PtVsGenTau1Pt = process.validationAnalyzerDR00.clone(
    srcWeights = cms.VInputTag(excludeFromWeights(srcWeights, [
        'embeddingKineReweightGENtoEmbedded:genDiTauMassVsGenDiTauPt',
        'embeddingKineReweightGENtoEmbedded:genTau2PtVsGenTau1Pt'
    ])),
    dqmDirectory = cms.string("validationAnalyzer_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPtAndGenTau2PtVsGenTau1Pt_%s" % channel),        
)

process.validationAnalyzerDR05 = process.validationAnalyzerDR00.clone(
    dRminSeparation = cms.double(0.5),
    dqmDirectory = cms.string("validationAnalyzerDR05_%s" % channel)
)    
process.validationAnalyzerDR07 = process.validationAnalyzerDR00.clone(
    dRminSeparation = cms.double(0.7),
    dqmDirectory = cms.string("validationAnalyzerDR07_%s" % channel)
)    
process.validationAnalyzerDR10 = process.validationAnalyzerDR00.clone(
    dRminSeparation = cms.double(1.0),
    dqmDirectory = cms.string("validationAnalyzerDR10_%s" % channel)
)
process.validationAnalyzerZPt0_10 = process.validationAnalyzerDR00.clone(
    ZPtMin = cms.double(0.0),
    ZPtMax = cms.double(10.0),
    dqmDirectory = cms.string("validationAnalyzerZPt0_10_%s" % channel)
)
process.validationAnalyzerZPt10_20 = process.validationAnalyzerDR00.clone(
    ZPtMin = cms.double(10.0),
    ZPtMax = cms.double(20.0),
    dqmDirectory = cms.string("validationAnalyzerZPt10_20_%s" % channel)
)
process.validationAnalyzerZPt20_40 = process.validationAnalyzerDR00.clone(
    ZPtMin = cms.double(20.0),
    ZPtMax = cms.double(40.0),
    dqmDirectory = cms.string("validationAnalyzerZPt20_40_%s" % channel)
)
process.validationAnalyzerZPt40_100 = process.validationAnalyzerDR00.clone(
    ZPtMin = cms.double(40.0),
    ZPtMax = cms.double(100.0),
    dqmDirectory = cms.string("validationAnalyzerZPt40_100_%s" % channel)
)
process.validationAnalyzerZPt100_inf = process.validationAnalyzerDR00.clone(
    ZPtMin = cms.double(100.0),
    ZPtMax = cms.double(-1),
    dqmDirectory = cms.string("validationAnalyzerZPt100_inf_%s" % channel)
)

process.validationAnalyzerSequence = cms.Sequence(
    process.validationAnalyzerDR00
   + process.validationAnalyzerDR00_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPt
   + process.validationAnalyzerDR00_woWeightGENtoEmbedded_genTau2PtVsGenTau1Pt
   + process.validationAnalyzerDR00_woWeightGENtoEmbedded_genDiTauMassVsGenDiTauPtAndGenTau2PtVsGenTau1Pt
   + process.validationAnalyzerDR05
   + process.validationAnalyzerDR07
   + process.validationAnalyzerDR10
   + process.validationAnalyzerZPt0_10
   + process.validationAnalyzerZPt10_20
   + process.validationAnalyzerZPt20_40
   + process.validationAnalyzerZPt40_100
   + process.validationAnalyzerZPt100_inf
)

#----------------------------------------------------------------------------------------------------
# CV: fill control plots for tau polarization and decay mode information,
#     as suggested by Ian Nugent

if addTauPolValidationPlots:
    process.tauValidationSequence = cms.Sequence()

    process.load("Validation.EventGenerator.TauValidation_cfi")
    process.tauValidationAnalyzerBeforeCuts = process.tauValidation.clone(
        UseWeightFromHepMC = cms.bool(True),
        dqmDirectory = cms.string("TauValidation/beforeCuts")
    )
    process.tauValidationSequence += process.tauValidationAnalyzerBeforeCuts
    process.genZptAnalyzerBeforeCutsEmbeddedRECO = cms.EDAnalyzer("GenZptAnalyzer",
        srcGenParticles = cms.InputTag('genParticles'),                                                                                          
        srcWeights = cms.VInputTag(),
        dqmDirectory = cms.string("genZptAnalyzer/beforeCuts/EmbeddedRECO")                         
    )
    process.tauValidationSequence += process.genZptAnalyzerBeforeCutsEmbeddedRECO
    process.genZptAnalyzerBeforeCutsSIM = process.genZptAnalyzerBeforeCutsEmbeddedRECO.clone(
        srcGenParticles = cms.InputTag('genParticles::SIM'),
        dqmDirectory = cms.string("genZptAnalyzer/beforeCuts/SIM")
    )
    process.tauValidationSequence += process.genZptAnalyzerBeforeCutsSIM   
            
    process.tauValidationSequence += process.genLeptonSelectionSequence
    process.tauValidationSequence += process.genDecayModeFilterSequence

    process.tauValidationAnalyzerAfterDecayModeCuts = process.tauValidationAnalyzerBeforeCuts.clone(
        dqmDirectory = cms.string("TauValidation/afterDecayModeCuts")
    )
    process.tauValidationSequence += process.tauValidationAnalyzerAfterDecayModeCuts
    process.genZptAnalyzerAfterDecayModeCutsEmbeddedRECO = process.genZptAnalyzerBeforeCutsEmbeddedRECO.clone(
        dqmDirectory = cms.string("genZptAnalyzer/afterDecayModeCuts/EmbeddedRECO")                         
    )
    process.tauValidationSequence += process.genZptAnalyzerAfterDecayModeCutsEmbeddedRECO
    process.genZptAnalyzerAfterDecayModeCutsSIM = process.genZptAnalyzerBeforeCutsSIM.clone(
        dqmDirectory = cms.string("genZptAnalyzer/afterDecayModeCuts/SIM")
    )
    process.tauValidationSequence += process.genZptAnalyzerAfterDecayModeCutsSIM
    
    process.tauValidationSequence += process.genDecayModeAndAcceptanceFilterSequence

    process.tauValidationAnalyzerAfterDecayModeAndVisPtCuts = process.tauValidationAnalyzerBeforeCuts.clone(
        dqmDirectory = cms.string("TauValidation/afterDecayModeAndVisPtCuts")
    )
    process.tauValidationSequence += process.tauValidationAnalyzerAfterDecayModeAndVisPtCuts
    process.genZptAnalyzerAfterDecayModeAndVisPtCutsEmbeddedRECO = process.genZptAnalyzerBeforeCutsEmbeddedRECO.clone(
        dqmDirectory = cms.string("genZptAnalyzer/afterDecayModeAndVisPtCuts/EmbeddedRECO")                         
    )
    process.tauValidationSequence += process.genZptAnalyzerAfterDecayModeAndVisPtCutsEmbeddedRECO
    process.genZptAnalyzerAfterDecayModeAndVisPtCutsSIM = process.genZptAnalyzerBeforeCutsSIM.clone(
        dqmDirectory = cms.string("genZptAnalyzer/afterDecayModeAndVisPtCuts/SIM")
    )
    process.tauValidationSequence += process.genZptAnalyzerAfterDecayModeAndVisPtCutsSIM
    
    process.tauValidationPath = cms.Path(process.tauValidationSequence)
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
# CV: produce Ntuple for computing embeddingKineReweight LUTs

process.ntupleProdSequence = cms.Sequence()

if produceEmbeddingKineReweightNtuple:
    embeddingKineReweightNtupleProducer_srcWeights = []
    if muonRadCorrectionsApplied:
        embeddingKineReweightNtupleProducer_srcWeights.append("muonRadiationCorrWeightProducer:weight")
    embeddingKineReweightNtupleProducer_srcWeights.extend(srcWeights)
    
    process.embeddingKineReweightNtupleProducer = cms.EDAnalyzer("EmbeddingKineReweightNtupleProducer",
        srcGenDiTaus = cms.InputTag('genZdecayToTaus'),
        srcGenParticles = cms.InputTag(''),
        srcSelectedMuons = cms.InputTag(''),
        srcWeights = cms.VInputTag(embeddingKineReweightNtupleProducer_srcWeights),
        srcGenFilterInfo = cms.InputTag(srcGenFilterInfo)
    )
    process.ntupleProdSequence += process.embeddingKineReweightNtupleProducer

    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("embeddingKineReweightNtuple.root")
    )
#----------------------------------------------------------------------------------------------------

process.DQMStore = cms.Service("DQMStore")

process.savePlots = cms.EDAnalyzer("DQMSimpleFileSaver",
    outputFileName = cms.string('validateMCEmbedding_plots.root')
)

process.p = cms.Path(
    process.genLeptonSelectionSequence
   + process.genDecayModeAndAcceptanceFilterSequence   
   + process.recLeptonSelectionSequence
   + process.recJetSequence
   + process.recMetSequence
   + process.embeddingWeightProdSequence
   + process.validationAnalyzerSequence
   + process.ntupleProdSequence
   + process.savePlots
)

# before starting to process 1st event, print event content
process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
process.filterFirstEvent = cms.EDFilter("EventCountFilter",
    numEvents = cms.int32(1)
)
process.printFirstEventContentPath = cms.Path(process.filterFirstEvent + process.printEventContent)

if addTauPolValidationPlots:
    process.schedule = cms.Schedule(process.printFirstEventContentPath, process.tauValidationPath, process.p)
else:
    process.schedule = cms.Schedule(process.printFirstEventContentPath, process.p)

##process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
##process.printGenParticleList = cms.EDAnalyzer("ParticleListDrawer",
##  #src = cms.InputTag("genParticles::SIM"),
##  src = cms.InputTag("genParticles"),
##  maxEventsToPrint = cms.untracked.int32(100)
##)
##process.printGenParticleListPath = cms.Path(process.printGenParticleList)
##
##process.schedule.extend([process.printGenParticleListPath])

##process.dumpSelMuonsEmbeddedRECO = cms.EDAnalyzer("DumpPATMuons",
##  src = cms.InputTag('goodMuons::EmbeddedRECO'),
##  minPt = cms.double(7.)
##)
##process.dumpSelMuons = process.dumpSelMuonsEmbeddedRECO.clone(
##  src = cms.InputTag('goodMuons')
##)    
##process.dumpSelMuonsPath = cms.Path(process.dumpSelMuonsEmbeddedRECO + process.dumpSelMuons)
##
##process.schedule.extend([process.dumpSelMuonsPath])

## print debug information whenever plugins get loaded dynamically from libraries
## (for debugging problems with plugin related dynamic library loading)
##process.add_(cms.Service("PrintLoadingPlugins"))

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

processDumpFile = open('validateMCEmbedding.dump', 'w')
print >> processDumpFile, process.dumpPython()
