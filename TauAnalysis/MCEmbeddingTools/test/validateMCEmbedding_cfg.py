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

isMC = True
#channel = 'etau'
channel = 'mutau'
srcWeights = []
srcGenFilterInfo = "generator:minVisPtFilter"
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define "hooks" for replacing configuration parameters
# in case running jobs on the CERN batch system/grid
#
#__isMC = $isMC
#__channel = "$channel"
#__srcWeights = $srcWeights
#__srcGenFilterInfo = "$srcGenFilterInfo"
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data1/veelken/CMSSW_5_3_x/skims/simDYmumu_embedded_mutau_2012Dec18_AOD.root'
        #'file:/data1/veelken/CMSSW_5_3_x/skims/simZplusJets_madgraph_AOD_1_1_txi.root'
        #'file:/tmp/veelken/rhembTauTau_data_Summer12_DYJetsToLL_DR53X_PU_S10_START53_V7A_v2_RECEmbed_2825_embed_AOD.root'                        
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
electronPtThreshold = 13.0
muonPtThreshold = 9.0
tauPtThreshold = 20.0

process.genParticlesFromZs = cms.EDProducer("GenParticlesFromZsSelectorForMCEmbedding",
    src = cms.InputTag("genParticles"),
    pdgIdsMothers = cms.vint32(23, 22),
    pdgIdsDaughters = cms.vint32(15, 13, 11),
    maxDaughters = cms.int32(2),
    minDaughters = cms.int32(2)
)
process.genTausFromZs = cms.EDProducer("GenParticlesFromZsSelectorForMCEmbedding",
    src = cms.InputTag("genParticles"),
    pdgIdsMothers = cms.vint32(23, 22),
    pdgIdsDaughters = cms.vint32(15),
    maxDaughters = cms.int32(2),
    minDaughters = cms.int32(2)
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
process.selectedElectronsPreId = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("genMatchedPatElectrons"),                                          
    cut = cms.string(
        '(abs(superCluster.eta) < 1.479 & electronID("eidRobustTight") > 0 & eSuperClusterOverP < 1.05 & eSuperClusterOverP > 0.95)'
        ' | (abs(superCluster.eta) > 1.479 & electronID("eidRobustTight") > 0 & eSuperClusterOverP < 1.12 & eSuperClusterOverP > 0.95)'
    ),
    filter = cms.bool(False)
)
process.selectedElectronsIdMVA = cms.EDFilter("PATElectronIdSelector",
    src = cms.InputTag("selectedElectronsPreId"),  
    srcVertex = cms.InputTag("goodVertex"),
    cut = cms.string("tight"),                                       
    filter = cms.bool(False)
)
process.selectedElectronsConversionVeto = cms.EDFilter("NPATElectronConversionFinder",
    src = cms.InputTag("selectedElectronsIdMVA"),                                                         
    maxMissingInnerHits = cms.int32(0),
    minMissingInnerHits = cms.int32(0),
    minRxy = cms.double(2.0),
    minFitProb = cms.double(1e-6),
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
   + process.selectedElectronsPreId
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
## process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
## process.printGenParticleListSIM = cms.EDAnalyzer("ParticleListDrawer",
##     src = cms.InputTag("genParticles::SIM"),
##     maxEventsToPrint = cms.untracked.int32(100)
## )
## process.printGenZsSIM = cms.EDAnalyzer("DumpGenZs",
##     src = cms.InputTag("genParticles::SIM")
## )  
## process.printGenParticleListEmbeddedRECO = cms.EDAnalyzer("ParticleListDrawer",
##     src = cms.InputTag("genParticles::EmbeddedRECO"),
##     maxEventsToPrint = cms.untracked.int32(100)
## )
## process.printGenZsEmbeddedRECO = cms.EDAnalyzer("DumpGenZs",
##     src = cms.InputTag("genParticles::EmbeddedRECO")
## )
## process.dumpVertices = cms.EDAnalyzer("DumpVertices",
##     src = cms.InputTag("offlinePrimaryVertices")
## )
## from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
## process.dumpTaus = cms.EDAnalyzer("DumpPATTausForRaman",
##     src = cms.InputTag("genMatchedPatTaus"),
##     srcVertex = cms.InputTag('goodVertex'),                              
##     minPt = cms.double(0.),
##     signalQualityCuts = PFTauQualityCuts.signalQualityCuts,
##     isolationQualityCuts = PFTauQualityCuts.isolationQualityCuts                           
## )
## process.dumpTaus.signalQualityCuts.maxDeltaZ = cms.double(1.e+3)
## process.dumpTaus.signalQualityCuts.maxTransverseImpactParameter = cms.double(1.e+3)
## process.dumpTaus.isolationQualityCuts.maxDeltaZ = cms.double(1.e+3)
## process.dumpTaus.isolationQualityCuts.maxTransverseImpactParameter = cms.double(1.e+3)
#--------------------------------------------------------------------------------
process.selectedTaus = cms.EDFilter("PATTauSelector",
    src = cms.InputTag("genMatchedPatTaus"),
    cut = cms.string(
        "pt > 20.0 & abs(eta) < 2.3 & tauID('decayModeFinding') > 0.5 & tauID('byLooseIsolationMVA') > 0.5"                                
    )                                
       # "pt > %1.1f & abs(eta) < 2.3 & tauID('decayModeFinding') > 0.5 & %s & %s & %s" % \
       #(tauPtThreshold, tauDiscrByIsolation, tauDiscrAgainstElectrons, tauDiscrAgainstMuons)
       #"pt > %1.1f & abs(eta) < 2.3 & tauID('decayModeFinding') > 0.5 & tauID('byLooseIsolationMVA') > 0.5"                                
)
process.recTauSelectionSequence = cms.Sequence(
    process.recoTauCommonSequence
   + process.recoTauClassicHPSSequence
   + process.patTaus
   + process.genMatchedPatTaus
   ###+ process.printGenParticleListSIM
   ##+ process.printGenZsSIM + process.printGenParticleListEmbeddedRECO + process.printGenZsEmbeddedRECO + process.dumpTaus
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

process.recJetSequence = cms.Sequence(process.jetTracksAssociatorAtVertex + process.btaggingAOD + process.makePatJets)
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

process.recMetSequence = cms.Sequence(
    process.recPFMetSequence
   + process.recCaloMetSequence
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# require event to contain generator level tau lepton pair,
# decaying in the sprecified channel
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
        src = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
        minNumber = cms.uint32(numElectrons),
        maxNumber = cms.uint32(numElectrons)                      
    ) 
    process.genLeptonSelectionSequence += process.electronFilter

if numMuons > 0:
    process.muonFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag('genMuonsFromZtautauDecaysWithinAcceptance'),
        minNumber = cms.uint32(numMuons),
        maxNumber = cms.uint32(numMuons)                      
    )
    process.genLeptonSelectionSequence += process.muonFilter

if numTauJets > 0:
    process.tauFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
        minNumber = cms.uint32(numTauJets),
        maxNumber = cms.uint32(numTauJets)                      
    )
    process.genLeptonSelectionSequence += process.tauFilter
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

process.validationAnalyzer = cms.EDAnalyzer("MCEmbeddingValidationAnalyzer",
    #----------------------------------------------------------------------------                                        
    # NOTE: configuration parameter 'srcReplacedMuons' needs to match value used when producing embedding sample;
    #          values of 'replacedMuonPtThresholdHigh' and 'replacedMuonPtThresholdLow' configuration parameters
    #          need to match muon Pt thresholds defined in TauAnalysis/MCEmbeddingTools/python/ZmumuStandaloneSelection_cff.py
    srcReplacedMuons = cms.InputTag('goldenZmumuCandidatesGe2IsoMuons'),
    replacedMuonPtThresholdHigh = cms.double(20.),
    replacedMuonPtThresholdLow = cms.double(10.),
    #----------------------------------------------------------------------------                                              
    srcRecMuons = cms.InputTag('muons'),
    srcRecTracks = cms.InputTag('generalTracks'),
    srcRecPFCandidates = cms.InputTag('particleFlow'),
    srcRecVertex = cms.InputTag('goodVertex'),                                        
    srcGenDiTaus = cms.InputTag('genZdecayToTaus'),
    srcGenLeg1 = cms.InputTag(srcGenLeg1),
    srcRecLeg1 = cms.InputTag(srcRecLeg1),                                        
    srcGenLeg2 = cms.InputTag(srcGenLeg2),
    srcRecLeg2 = cms.InputTag(srcRecLeg2),
    srcWeights = cms.VInputTag(srcWeights),
    srcGenFilterInfo = cms.InputTag(srcGenFilterInfo),                                        
    dqmDirectory = cms.string("validationAnalyzer_%s" % channel),                                        
                                            
    # electron Pt, eta and phi distributions;
    # electron id & isolation and trigger efficiencies
    electronDistributions = cms.VPSet(					
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
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectrons'),
            dqmDirectory = cms.string('goodElectronEfficiencies')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genElectronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('goodElectronsPFIso'),
            dqmDirectory = cms.string('goodIsoElectronEfficiencies')
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
        ),
        #------------------------------------------------------------------------                                        
        cms.PSet(
            srcGen = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('patTaus'),
            dqmDirectory = cms.string('patTauDistributions')
        ),
        cms.PSet(
            srcGen = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('genMatchedPatTaus'),
            dqmDirectory = cms.string('genMatchedTauDistributions')
        )
        #------------------------------------------------------------------------ 
    ),
    tauEfficiencies = cms.VPSet(					
        cms.PSet(
            srcGen = cms.InputTag('genHadronsFromZtautauDecaysWithinAcceptance'),
	    srcRec = cms.InputTag('selectedTaus'),
            dqmDirectory = cms.string('selectedTauEfficiencies')
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
            cutL1Pt = cms.double(20.),	    
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM20')
        ),
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Pt = cms.double(26.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM26')
        ),
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Pt = cms.double(30.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM30')
        ),
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Pt = cms.double(36.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM36')
        ),
	cms.PSet(
            srcRef = cms.InputTag('patCaloMetNoHF'),
            srcL1 = cms.InputTag('l1extraParticles', 'MET'),
            cutL1Pt = cms.double(40.),
            dqmDirectory = cms.string('metTriggerEfficiencyL1_ETM40')
        )
    )
)

process.DQMStore = cms.Service("DQMStore")

process.savePlots = cms.EDAnalyzer("DQMSimpleFileSaver",
    outputFileName = cms.string('validateMCEmbedding_plots.root')
)

process.p = cms.Path(
    process.genLeptonSelectionSequence
   + process.recLeptonSelectionSequence
   + process.recJetSequence
   + process.recMetSequence
   + process.validationAnalyzer
   + process.savePlots
)

# before starting to process 1st event, print event content
process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
process.filterFirstEvent = cms.EDFilter("EventCountFilter",
    numEvents = cms.int32(1)
)
process.printFirstEventContentPath = cms.Path(process.filterFirstEvent + process.printEventContent)

process.schedule = cms.Schedule(process.printFirstEventContentPath, process.p)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

processDumpFile = open('validateMCEmbedding.dump', 'w')
print >> processDumpFile, process.dumpPython()
