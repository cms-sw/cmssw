# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.coreTools import *

###############################
####### Parameters ############
###############################
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')

options.register ('useData',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Run this on real data")

options.register ('hltProcess',
                  'HLT',
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.string,
                  "HLT process name to use.")

options.register ('writeFat',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Output tracks and PF candidates (and GenParticles for MC)")

options.register ('writeSimpleInputs',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Write four-vector and ID of PF candidates")

options.register ('writeGenParticles',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Output GenParticles collection")

options.register ('forceCheckClosestZVertex',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Force the check of the closest z vertex")


options.register ('useSusyFilter',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Use the SUSY event filter")


options.register ('useExtraJetColls',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Write extra jet collections for substructure studies")

options.parseArguments()


if not options.useData :
	inputJetCorrLabel = ('AK5PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'])

	process.source.fileNames = [
		'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V5-v1/0000/FEBE99BB-3881-E111-B1F3-003048D42DC8.root'
		]    		

else :
	inputJetCorrLabel = ('AK5PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual'])
	process.source.fileNames = [
		'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V5-v1/0000/FEBE99BB-3881-E111-B1F3-003048D42DC8.root'
		]

	

#process.source.eventsToProcess = cms.untracked.VEventRange( ['1:86747'] )

#process.source.skipEvents = cms.untracked.uint32(17268) 

print options

print 'Running jet corrections: '
print inputJetCorrLabel

import sys


###############################
####### Global Setup ##########
###############################



# 4.2.x or 52x configuration
fileTag = "52x"
if options.useData :
	process.GlobalTag.globaltag = cms.string( 'GR_R_52_V9B::All' )
else :
	process.GlobalTag.globaltag = cms.string( 'START52_V11B::All' )


# require scraping filter
process.scrapingVeto = cms.EDFilter("FilterOutScraping",
                                    applyfilter = cms.untracked.bool(True),
                                    debugOn = cms.untracked.bool(False),
                                    numtrack = cms.untracked.uint32(10),
                                    thresh = cms.untracked.double(0.2)
                                    )
# HB + HE noise filtering
process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')
# Modify defaults setting to avoid an over-efficiency in the presence of OFT PU
process.HBHENoiseFilter.minIsolatedNoiseSumE = cms.double(999999.)
process.HBHENoiseFilter.minNumIsolatedNoiseChannels = cms.int32(999999)
process.HBHENoiseFilter.minIsolatedNoiseSumEt = cms.double(999999.)


# switch on PAT trigger
#from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
#switchOnTrigger( process, hltProcess=options.hltProcess )




###############################
####### DAF PV's     ##########
###############################

pvSrc = 'offlinePrimaryVertices'

process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag("goodOfflinePrimaryVertices"),
                                           minimumNDOF = cms.uint32(3) , # this is > 3
                                           maxAbsZ = cms.double(24), 
                                           maxd0 = cms.double(2) 
                                           )




from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

process.goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( maxZ = cms.double(24.0),
                                     minNdof = cms.double(4.0) # this is >= 4
                                     ),
    src=cms.InputTag(pvSrc)
    )


###############################
########## Gen Setup ##########
###############################

process.load("RecoJets.Configuration.GenJetParticles_cff")
from RecoJets.JetProducers.ca4GenJets_cfi import ca4GenJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
process.ca8GenJetsNoNu = ca4GenJets.clone( rParam = cms.double(0.8),
                                           src = cms.InputTag("genParticlesForJetsNoNu"))

process.ak8GenJetsNoNu = ak5GenJets.clone( rParam = cms.double(0.8),
                                           src = cms.InputTag("genParticlesForJetsNoNu"))


process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

# add the flavor history
process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryPaths_cfi")


# prune gen particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.prunedGenParticles = cms.EDProducer("GenParticlePruner",
                                            src = cms.InputTag("genParticles"),
                                            select = cms.vstring(
                                                "drop  *"
                                                ,"keep status = 3" #keeps  particles from the hard matrix element
                                                ,"keep (abs(pdgId) >= 11 & abs(pdgId) <= 16) & status = 1" #keeps e/mu and nus with status 1
                                                ,"keep (abs(pdgId)  = 15) & status = 3" #keeps taus
                                                )
                                            )


## process.prunedGenParticles = cms.EDProducer("GenParticlePruner",
##                                             src = cms.InputTag("genParticles"),
##                                             select = cms.vstring(
##                                                 "drop  *"
##                                                 ,"keep++ (abs(pdgId) =6) "
##                                                 )
##                                             )

###############################
#### Jet RECO includes ########
###############################

from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.CATopJetParameters_cfi import *
from RecoJets.JetProducers.GenJetParameters_cfi import *


###############################
########## PF Setup ###########
###############################

# Default PF2PAT with AK5 jets. Make sure to turn ON the L1fastjet stuff. 
from PhysicsTools.PatAlgos.tools.pfTools import *
postfix = "PFlow"
usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5', runOnMC=not options.useData, postfix=postfix,
	  jetCorrections=inputJetCorrLabel, pvCollection=cms.InputTag('goodOfflinePrimaryVertices'))
if not options.forceCheckClosestZVertex :
    process.pfPileUpPFlow.checkClosestZVertex = False


postfixLoose = "PFlowLoose"
usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5', runOnMC=not options.useData, postfix=postfixLoose,
	  jetCorrections=inputJetCorrLabel, pvCollection=cms.InputTag('goodOfflinePrimaryVertices'))
if not options.forceCheckClosestZVertex :
    process.pfPileUpPFlowLoose.checkClosestZVertex = False

# Turn on the delta-beta corrections
process.pfIsolatedElectronsPFlow.doDeltaBetaCorrection = True
process.pfIsolatedMuonsPFlow.doDeltaBetaCorrection = True
process.pfIsolatedElectronsPFlowLoose.doDeltaBetaCorrection = True
process.pfIsolatedMuonsPFlowLoose.doDeltaBetaCorrection = True

# Set up "loose" leptons. 
process.pfIsolatedMuonsPFlowLoose.isolationCut = cms.double(999.0) 
process.pfIsolatedElectronsPFlowLoose.isolationCut = cms.double(999.0)
process.patMuonsPFlowLoose.pfMuonSource = "pfMuonsPFlowLoose"
process.patElectronsPFlowLoose.pfElectronSource = "pfElectronsPFlowLoose"


# turn to false when running on data
if options.useData :
    removeMCMatching( process, ['All'] )


###############################
###### Electron ID ############
###############################

process.load('EGamma.EGammaAnalysisTools.electronIdMVAProducer_cfi') 
process.eidMVASequence = cms.Sequence(  process.mvaTrigV0 + process.mvaNonTrigV0 )
#Electron ID
process.patElectronsPFlow.electronIDSources.mvaTrigV0    = cms.InputTag("mvaTrigV0")
process.patElectronsPFlow.electronIDSources.mvaNonTrigV0 = cms.InputTag("mvaNonTrigV0") 
process.patPF2PATSequencePFlow.replace( process.patElectronsPFlow, process.eidMVASequence * process.patElectronsPFlow )

process.patElectronsPFlowLoose.electronIDSources.mvaTrigV0    = cms.InputTag("mvaTrigV0")
process.patElectronsPFlowLoose.electronIDSources.mvaNonTrigV0 = cms.InputTag("mvaNonTrigV0") 
process.patPF2PATSequencePFlowLoose.replace( process.patElectronsPFlowLoose, process.eidMVASequence * process.patElectronsPFlowLoose )

###############################
###### Bare KT 0.6 jets #######
###############################

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets

###############################
###### Bare CA 0.8 jets #######
###############################
from RecoJets.JetProducers.ca4PFJets_cfi import ca4PFJets
process.ca8PFJetsPFlow = ca4PFJets.clone(
    rParam = cms.double(0.8),
    src = cms.InputTag('pfNoElectron'+postfix),
    doAreaFastjet = cms.bool(True),
    doRhoFastjet = cms.bool(True),
    Rho_EtaMax = cms.double(6.0),
    Ghost_EtaMax = cms.double(7.0)
    )



###############################
###### AK 0.7 jets ############
###############################
process.ak7PFlow = process.pfJetsPFlow.clone(
	rParam = cms.double(0.7)
    )


###############################
###### AK 0.8 jets ############
###############################
process.ak8PFlow = process.pfJetsPFlow.clone(
	rParam = cms.double(0.8)
    )


###############################
###### AK 0.5 jets groomed ####
###############################

from RecoJets.JetProducers.ak5PFJetsTrimmed_cfi import ak5PFJetsTrimmed
process.ak5TrimmedPFlow = ak5PFJetsTrimmed.clone(
    src = process.pfJetsPFlow.src,
    doAreaFastjet = cms.bool(True)
    )

from RecoJets.JetProducers.ak5PFJetsFiltered_cfi import ak5PFJetsFiltered
process.ak5FilteredPFlow = ak5PFJetsFiltered.clone(
    src = process.pfJetsPFlow.src,
    doAreaFastjet = cms.bool(True)
    )

from RecoJets.JetProducers.ak5PFJetsPruned_cfi import ak5PFJetsPruned
process.ak5PrunedPFlow = ak5PFJetsPruned.clone(
    src = process.pfJetsPFlow.src,
    doAreaFastjet = cms.bool(True)
    )



###############################
###### AK 0.7 jets groomed ####
###############################

process.ak7TrimmedPFlow = process.ak5TrimmedPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.7)
    )

process.ak7FilteredPFlow = process.ak5FilteredPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.7)
	)

process.ak7PrunedPFlow = process.ak5PrunedPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.7)
    )


process.ak7TrimmedGenJetsNoNu = ak5GenJets.clone(
	rParam = cms.double(0.7),
	src = cms.InputTag("genParticlesForJetsNoNu"),
	useTrimming = cms.bool(True),
	rFilt = cms.double(0.2),
	trimPtFracMin = cms.double(0.03),
	)

process.ak7FilteredGenJetsNoNu = ak5GenJets.clone(
	rParam = cms.double(0.7),
	src = cms.InputTag("genParticlesForJetsNoNu"),
	useFiltering = cms.bool(True),
	nFilt = cms.int32(3),
	rFilt = cms.double(0.3),
	writeCompound = cms.bool(True),
	jetCollInstanceName=cms.string("SubJets")
	)



process.ak7PrunedGenJetsNoNu = ak5GenJets.clone(
	SubJetParameters,
	rParam = cms.double(0.7),
	src = cms.InputTag("genParticlesForJetsNoNu"),
	usePruning = cms.bool(True),
	writeCompound = cms.bool(True),
	jetCollInstanceName=cms.string("SubJets")
	)



###############################
###### AK 0.8 jets groomed ####
###############################

process.ak8TrimmedPFlow = process.ak5TrimmedPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.8)
    )

process.ak8FilteredPFlow = process.ak5FilteredPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.8)
	)

process.ak8PrunedPFlow = process.ak5PrunedPFlow.clone(
	src = process.pfJetsPFlow.src,
	rParam = cms.double(0.8)
    )

###############################
###### CA8 Pruning Setup ######
###############################


# Pruned PF Jets
process.caPrunedPFlow = process.ak5PrunedPFlow.clone(
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(0.8)
)


process.caPrunedGen = process.ca8GenJetsNoNu.clone(
	SubJetParameters,
	usePruning = cms.bool(True),
	useExplicitGhosts = cms.bool(True),
	writeCompound = cms.bool(True),
	jetCollInstanceName=cms.string("SubJets")
)

###############################
###### CA8 Filtered Setup #####
###############################


# Filtered PF Jets
process.caFilteredPFlow = ak5PFJetsFiltered.clone(
	src = cms.InputTag('pfNoElectron'+postfix),
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(1.2),
	writeCompound = cms.bool(True),
	doAreaFastjet = cms.bool(True),
	jetPtMin = cms.double(100.0)
)

from RecoJets.JetProducers.ak5PFJetsFiltered_cfi import ak5PFJetsMassDropFiltered
process.caMassDropFilteredPFlow = ak5PFJetsMassDropFiltered.clone(
	src = cms.InputTag('pfNoElectron'+postfix),
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(1.2),
	writeCompound = cms.bool(True),
	doAreaFastjet = cms.bool(True),
	jetPtMin = cms.double(100.0)
)


process.caFilteredGenJetsNoNu = process.ca8GenJetsNoNu.clone(
	nFilt = cms.int32(2),
	rFilt = cms.double(0.3),
	useFiltering = cms.bool(True),
	useExplicitGhosts = cms.bool(True),
	writeCompound = cms.bool(True),
	rParam       = cms.double(1.2),
	jetCollInstanceName=cms.string("SubJets"),
	jetPtMin = cms.double(100.0)
)

process.caMassDropFilteredGenJetsNoNu = process.caFilteredGenJetsNoNu.clone(
        src = cms.InputTag('genParticlesForJetsNoNu'),
	useMassDropTagger = cms.bool(True),
	muCut = cms.double(0.667),
	yCut = cms.double(0.08)
)



###############################
#### CATopTag Setup ###########
###############################

# CATopJet PF Jets
# with adjacency 
process.caTopTagPFlow = cms.EDProducer(
    "CATopJetProducer",
    PFJetParameters.clone( src = cms.InputTag('pfNoElectron'+postfix),
                           doAreaFastjet = cms.bool(True),
                           doRhoFastjet = cms.bool(False),
			   jetPtMin = cms.double(100.0)
                           ),
    AnomalousCellParameters,
    CATopJetParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = cms.double(0.8),
    writeCompound = cms.bool(True)
    )

process.CATopTagInfosPFlow = cms.EDProducer("CATopJetTagger",
                                    src = cms.InputTag("caTopTagPFlow"),
                                    TopMass = cms.double(171),
                                    TopMassMin = cms.double(0.),
                                    TopMassMax = cms.double(250.),
                                    WMass = cms.double(80.4),
                                    WMassMin = cms.double(0.0),
                                    WMassMax = cms.double(200.0),
                                    MinMassMin = cms.double(0.0),
                                    MinMassMax = cms.double(200.0),
                                    verbose = cms.bool(False)
                                    )



process.caTopTagGen = cms.EDProducer(
    "CATopJetProducer",
    GenJetParameters.clone(src = cms.InputTag("genParticlesForJetsNoNu"),
                           doAreaFastjet = cms.bool(False),
                           doRhoFastjet = cms.bool(False)),
    AnomalousCellParameters,
    CATopJetParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = cms.double(0.8),
    writeCompound = cms.bool(True)
    )

process.CATopTagInfosGen = cms.EDProducer("CATopJetTagger",
                                          src = cms.InputTag("caTopTagGen"),
                                          TopMass = cms.double(171),
                                          TopMassMin = cms.double(0.),
                                          TopMassMax = cms.double(250.),
                                          WMass = cms.double(80.4),
                                          WMassMin = cms.double(0.0),
                                          WMassMax = cms.double(200.0),
                                          MinMassMin = cms.double(0.0),
                                          MinMassMax = cms.double(200.0),
                                          verbose = cms.bool(False)
                                          )



# CATopJet PF Jets

for ipostfix in [postfix] :
    for module in (
        getattr(process,"ca8PFJets" + ipostfix),
        getattr(process,"CATopTagInfos" + ipostfix),
        getattr(process,"caTopTag" + ipostfix),
        getattr(process,"caPruned" + ipostfix)
        ) :
        getattr(process,"patPF2PATSequence"+ipostfix).replace( getattr(process,"pfNoElectron"+ipostfix), getattr(process,"pfNoElectron"+ipostfix)*module )


    if options.useExtraJetColls : 
	    for module in (
		getattr(process,"ak5Trimmed" + ipostfix),
		getattr(process,"ak5Filtered" + ipostfix),
		getattr(process,"ak5Pruned" + ipostfix),
		getattr(process,"ak7Trimmed" + ipostfix),
		getattr(process,"ak7Filtered" + ipostfix),
		getattr(process,"ak7Pruned" + ipostfix),
		getattr(process,"ak7" + ipostfix),
		getattr(process,"ak8Trimmed" + ipostfix),
		getattr(process,"ak8Filtered" + ipostfix),
		getattr(process,"ak8Pruned" + ipostfix),
		getattr(process,"ak8" + ipostfix),
		getattr(process,"caFiltered" + ipostfix),
		getattr(process,"caMassDropFiltered" + ipostfix)
		) :
		    getattr(process,"patPF2PATSequence"+ipostfix).replace( getattr(process,"pfNoElectron"+ipostfix), getattr(process,"pfNoElectron"+ipostfix)*module )



# Use the good primary vertices everywhere. 
for imod in [process.patMuonsPFlow,
             process.patMuonsPFlowLoose,
             process.patElectronsPFlow,
             process.patElectronsPFlowLoose,
             process.patMuons,
             process.patElectrons] :
    imod.pvSrc = "goodOfflinePrimaryVertices"
    imod.embedTrack = True
    

addJetCollection(process, 
                 cms.InputTag('ca8PFJetsPFlow'),
                 'CA8', 'PF',
                 doJTA=True,
                 doBTagging=True,
                 jetCorrLabel=inputJetCorrLabel,
                 doType1MET=False,
                 doL1Cleaning=False,
                 doL1Counters=False,
                 genJetCollection = cms.InputTag("ca8GenJetsNoNu"),
                 doJetID = False
                 )


addJetCollection(process, 
                 cms.InputTag('caPrunedPFlow'),
                 'CA8Pruned', 'PF',
                 doJTA=False,
                 doBTagging=False,
                 jetCorrLabel=inputJetCorrLabel,
                 doType1MET=False,
                 doL1Cleaning=False,
                 doL1Counters=False,
                 genJetCollection = cms.InputTag("ca8GenJetsNoNu"),
                 doJetID = False
                 )



addJetCollection(process, 
                 cms.InputTag('caTopTagPFlow'),
                 'CATopTag', 'PF',
                 doJTA=True,
                 doBTagging=True,
                 jetCorrLabel=inputJetCorrLabel,
                 doType1MET=False,
                 doL1Cleaning=False,
                 doL1Counters=False,
                 genJetCollection = cms.InputTag("ca8GenJetsNoNu"),
                 doJetID = False
                 )


if options.useExtraJetColls: 
	addJetCollection(process, 
			 cms.InputTag('caFilteredPFlow'),
			 'CA12Filtered', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ca8GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('caMassDropFilteredPFlow'),
			 'CA12MassDropFiltered', 'PF',
			 doJTA=True,
			 doBTagging=True,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ca8GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('caMassDropFilteredPFlow', 'SubJets'),         # Jet collection; must be already in the event when patLayer0 sequence is executed
			 'CA12MassDropFilteredSubjets', 'PF',
			 doJTA=True,            # Run Jet-Track association & JetCharge
			 doBTagging=True,       # Run b-tagging
			 jetCorrLabel=None,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak5GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak5PrunedPFlow'),
			 'AK5Pruned', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak5GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('ak5FilteredPFlow'),
			 'AK5Filtered', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak5GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak5TrimmedPFlow'),
			 'AK5Trimmed', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak5GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('ak7PFlow'),
			 'AK7', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak7GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak7PrunedPFlow'),
			 'AK7Pruned', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak7GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('ak7FilteredPFlow'),
			 'AK7Filtered', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak7GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak7TrimmedPFlow'),
			 'AK7Trimmed', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak7GenJetsNoNu"),
			 doJetID = False
			 )





	addJetCollection(process, 
			 cms.InputTag('ak8PFlow'),
			 'AK8', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak8GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak8PrunedPFlow'),
			 'AK8Pruned', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak8GenJetsNoNu"),
			 doJetID = False
			 )


	addJetCollection(process, 
			 cms.InputTag('ak8FilteredPFlow'),
			 'AK8Filtered', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak8GenJetsNoNu"),
			 doJetID = False
			 )

	addJetCollection(process, 
			 cms.InputTag('ak8TrimmedPFlow'),
			 'AK8Trimmed', 'PF',
			 doJTA=False,
			 doBTagging=False,
			 jetCorrLabel=inputJetCorrLabel,
			 doType1MET=False,
			 doL1Cleaning=False,
			 doL1Counters=False,
			 genJetCollection = cms.InputTag("ak8GenJetsNoNu"),
			 doJetID = False
			 )




switchJetCollection(process,cms.InputTag('ak5PFJets'),
		    doJTA        = False,
		    doBTagging   = False,
		    jetCorrLabel = inputJetCorrLabel,
		    doType1MET   = False,
		    genJetCollection=cms.InputTag("ak5GenJetsNoNu"),
		    doJetID      = False
		    )

for icorr in [process.patJetCorrFactors,
	      process.patJetCorrFactorsCATopTagPF,
              process.patJetCorrFactorsCA8PrunedPF,
              process.patJetCorrFactorsCA8PF ] :
    icorr.rho = cms.InputTag("kt6PFJets", "rho")


if options.useExtraJetColls: 
	for icorr in [process.patJetCorrFactorsAK5PrunedPF,
		      process.patJetCorrFactorsAK5FilteredPF,
		      process.patJetCorrFactorsAK5TrimmedPF,
		      process.patJetCorrFactorsAK7PF,
		      process.patJetCorrFactorsAK7PrunedPF,
		      process.patJetCorrFactorsAK7FilteredPF,
		      process.patJetCorrFactorsAK7TrimmedPF,
		      process.patJetCorrFactorsAK8PF,
		      process.patJetCorrFactorsAK8PrunedPF,
		      process.patJetCorrFactorsAK8FilteredPF,
		      process.patJetCorrFactorsAK8TrimmedPF] :
	    icorr.rho = cms.InputTag("kt6PFJets", "rho")



###############################
### TagInfo and Matching Setup#
###############################

# Do some configuration of the jet substructure things
for jetcoll in (process.patJetsPFlow,
		process.patJets,
                process.patJetsCA8PF,
                process.patJetsCA8PrunedPF,
                process.patJetsCATopTagPF
                ) :
    if options.useData == False :
        jetcoll.embedGenJetMatch = False
        jetcoll.getJetMCFlavour = True
        jetcoll.addGenPartonMatch = True
    # Add the calo towers and PFCandidates.
    # I'm being a little tricksy here, because I only
    # actually keep the products if the "writeFat" switch
    # is on. However, this allows for overlap checking
    # with the Refs so satisfies most use cases without
    # having to add to the object size
    jetcoll.addBTagInfo = False
    jetcoll.embedCaloTowers = True
    jetcoll.embedPFCandidates = True

# Add CATopTag and b-tag info... piggy-backing on b-tag functionality
process.patJetsPFlow.addBTagInfo = True
process.patJetsCATopTagPF.addBTagInfo = True



# Do some configuration of the jet substructure things
if options.useExtraJetColls: 
	for jetcoll in (process.patJetsAK5TrimmedPF,
			process.patJetsAK5PrunedPF,
			process.patJetsAK5FilteredPF,
			process.patJetsAK7PF,
			process.patJetsAK7TrimmedPF,
			process.patJetsAK7PrunedPF,
			process.patJetsAK7FilteredPF,
			process.patJetsAK8PF,
			process.patJetsAK8TrimmedPF,
			process.patJetsAK8PrunedPF,
			process.patJetsAK8FilteredPF,
			process.patJetsCA12FilteredPF,
			process.patJetsCA12MassDropFilteredPF
			) :
	    if options.useData == False :
		jetcoll.embedGenJetMatch = False
		jetcoll.getJetMCFlavour = True
		jetcoll.addGenPartonMatch = True
	    # Add the calo towers and PFCandidates.
	    # I'm being a little tricksy here, because I only
	    # actually keep the products if the "writeFat" switch
	    # is on. However, this allows for overlap checking
	    # with the Refs so satisfies most use cases without
	    # having to add to the object size
	    jetcoll.addBTagInfo = False
	    jetcoll.embedCaloTowers = True
	    jetcoll.embedPFCandidates = True

	# Add CATopTag and b-tag info... piggy-backing on b-tag functionality
	process.patJetsCA12MassDropFilteredPF.addBTagInfo = True


#################################################
#### Fix the PV collections for the future ######
#################################################
for module in [process.patJetCorrFactors,
               process.patJetCorrFactorsPFlow,
               process.patJetCorrFactorsCATopTagPF,
               process.patJetCorrFactorsCA8PrunedPF,
               process.patJetCorrFactorsCA8PF
               ]:
    module.primaryVertices = "goodOfflinePrimaryVertices"

    
if options.useExtraJetColls: 
	for module in [process.patJetCorrFactorsCA12FilteredPF,
		       process.patJetCorrFactorsCA12MassDropFilteredPF,
		       process.patJetCorrFactorsAK5TrimmedPF,
		       process.patJetCorrFactorsAK5PrunedPF,
		       process.patJetCorrFactorsAK5FilteredPF,
		       process.patJetCorrFactorsAK7PF,
		       process.patJetCorrFactorsAK7TrimmedPF,
		       process.patJetCorrFactorsAK7PrunedPF,
		       process.patJetCorrFactorsAK7FilteredPF,
		       process.patJetCorrFactorsAK8PF,
		       process.patJetCorrFactorsAK8TrimmedPF,
		       process.patJetCorrFactorsAK8PrunedPF,
		       process.patJetCorrFactorsAK8FilteredPF
		       ]:
	    module.primaryVertices = "goodOfflinePrimaryVertices"


###############################
#### Selections Setup #########
###############################

# AK5 Jets
process.selectedPatJetsPFlow.cut = cms.string("pt > 5")
process.patJetsPFlow.addTagInfos = True
process.patJetsPFlow.tagInfoSources = cms.VInputTag(
    cms.InputTag("secondaryVertexTagInfosAODPFlow")
    )
process.patJetsPFlow.userData.userFunctions = cms.vstring( "? hasTagInfo('secondaryVertex') && tagInfoSecondaryVertex('secondaryVertex').nVertices() > 0 ? "
                                                      "tagInfoSecondaryVertex('secondaryVertex').secondaryVertex(0).p4().mass() : 0")
process.patJetsPFlow.userData.userFunctionLabels = cms.vstring('secvtxMass')

# CA8 jets
process.selectedPatJetsCA8PF.cut = cms.string("pt > 20")

# CA8 Pruned jets
process.selectedPatJetsCA8PrunedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")


# CA8 TopJets
process.selectedPatJetsCATopTagPF.cut = cms.string("pt > 150 & abs(rapidity) < 2.5")
process.patJetsCATopTagPF.addTagInfos = True
process.patJetsCATopTagPF.tagInfoSources = cms.VInputTag(
    cms.InputTag('CATopTagInfosPFlow')
    )

if options.useExtraJetColls: 
	# CA12 Filtered jets
	process.selectedPatJetsCA12FilteredPF.cut = cms.string("pt > 150 & abs(rapidity) < 2.5")
	process.selectedPatJetsCA12MassDropFilteredPF.cut = cms.string("pt > 150 & abs(rapidity) < 2.5")

	# AK5 groomed jets
	process.selectedPatJetsAK5PrunedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK5TrimmedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK5FilteredPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")


	# AK7 groomed jets
	process.selectedPatJetsAK7PF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK7PrunedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK7TrimmedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK7FilteredPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")


	# AK8 groomed jets
	process.selectedPatJetsAK8PF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK8PrunedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK8TrimmedPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	process.selectedPatJetsAK8FilteredPF.cut = cms.string("pt > 20 & abs(rapidity) < 2.5")
	


# electrons
process.selectedPatElectrons.cut = cms.string('pt > 10.0 & abs(eta) < 2.5')
process.patElectrons.embedTrack = cms.bool(True)
process.selectedPatElectronsPFlow.cut = cms.string('pt > 10.0 & abs(eta) < 2.5')
process.patElectronsPFlow.embedTrack = cms.bool(True)
process.selectedPatElectronsPFlowLoose.cut = cms.string('pt > 10.0 & abs(eta) < 2.5')
process.patElectronsPFlowLoose.embedTrack = cms.bool(True)
# muons
process.selectedPatMuons.cut = cms.string('pt > 10.0 & abs(eta) < 2.5')
process.patMuons.embedTrack = cms.bool(True)
process.selectedPatMuonsPFlow.cut = cms.string("pt > 10.0 & abs(eta) < 2.5")
process.patMuonsPFlow.embedTrack = cms.bool(True)
process.selectedPatMuonsPFlowLoose.cut = cms.string("pt > 10.0 & abs(eta) < 2.5")
process.patMuonsPFlowLoose.embedTrack = cms.bool(True)
# taus
process.selectedPatTausPFlow.cut = cms.string("pt > 10.0 & abs(eta) < 3")
process.selectedPatTaus.cut = cms.string("pt > 10.0 & abs(eta) < 3")
process.patTausPFlow.isoDeposits = cms.PSet()
process.patTaus.isoDeposits = cms.PSet()
# photons
process.patPhotonsPFlow.isoDeposits = cms.PSet()
process.patPhotons.isoDeposits = cms.PSet()


# Apply jet ID to all of the jets upstream. We aren't going to screw around
# with this, most likely. So, we don't really to waste time with it
# at the analysis level. 
from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector
process.goodPatJetsPFlow = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                        filterParams = pfJetIDSelector.clone(),
                                        src = cms.InputTag("selectedPatJetsPFlow")
                                        )
process.goodPatJetsCA8PF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                        filterParams = pfJetIDSelector.clone(),
                                        src = cms.InputTag("selectedPatJetsCA8PF")
                                        )
process.goodPatJetsCA8PrunedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                              filterParams = pfJetIDSelector.clone(),
                                              src = cms.InputTag("selectedPatJetsCA8PrunedPF")
                                              )

process.goodPatJetsCATopTagPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                             filterParams = pfJetIDSelector.clone(),
                                             src = cms.InputTag("selectedPatJetsCATopTagPF")
                                             )


if options.useExtraJetColls:
	process.goodPatJetsCA12FilteredPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsCA12FilteredPF")
						      )

	process.goodPatJetsCA12MassDropFilteredPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsCA12MassDropFilteredPF")
						      )

	process.goodPatJetsAK5PrunedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK5PrunedPF")
						      )
	process.goodPatJetsAK5FilteredPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK5FilteredPF")
						      )
	process.goodPatJetsAK5TrimmedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK5TrimmedPF")
						      )

	process.goodPatJetsAK7PF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK7PF")
						      )
	process.goodPatJetsAK7PrunedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK7PrunedPF")
						      )
	process.goodPatJetsAK7FilteredPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK7FilteredPF")
						      )
	process.goodPatJetsAK7TrimmedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK7TrimmedPF")
						      )



	process.goodPatJetsAK8PF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK8PF")
						      )
	process.goodPatJetsAK8PrunedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK8PrunedPF")
						      )
	process.goodPatJetsAK8FilteredPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK8FilteredPF")
						      )
	process.goodPatJetsAK8TrimmedPF = cms.EDFilter("PFJetIDSelectionFunctorFilter",
						      filterParams = pfJetIDSelector.clone(),
						      src = cms.InputTag("selectedPatJetsAK8TrimmedPF")
						      )



if options.writeSimpleInputs :
	process.pfInputs = cms.EDProducer(
	    "CandViewNtpProducer", 
	    src = cms.InputTag('selectedPatJetsCA8PF', 'pfCandidates'),
	    lazyParser = cms.untracked.bool(True),
	    eventInfo = cms.untracked.bool(False),
	    variables = cms.VPSet(
		cms.PSet(
		    tag = cms.untracked.string("px"),
		    quantity = cms.untracked.string("px")
		    ),
		cms.PSet(
		    tag = cms.untracked.string("py"),
		    quantity = cms.untracked.string("py")
		    ),
		cms.PSet(
		    tag = cms.untracked.string("pz"),
		    quantity = cms.untracked.string("pz")
		    ),
		cms.PSet(
		    tag = cms.untracked.string("energy"),
		    quantity = cms.untracked.string("energy")
		    ),
		cms.PSet(
		    tag = cms.untracked.string("pdgId"),
		    quantity = cms.untracked.string("pdgId")
		    )
		)
	)


if options.useExtraJetColls:
	process.ak5Lite = cms.EDProducer(
	    "CandViewNtpProducer", 
	    src = cms.InputTag('goodPatJetsPFlow'),
	    lazyParser = cms.untracked.bool(True),
	    eventInfo = cms.untracked.bool(False),
	    variables = cms.VPSet(
			cms.PSet(
				tag = cms.untracked.string("px"),
				quantity = cms.untracked.string("px")
				),
			cms.PSet(
				tag = cms.untracked.string("py"),
				quantity = cms.untracked.string("py")
				),
			cms.PSet(
				tag = cms.untracked.string("pz"),
				quantity = cms.untracked.string("pz")
				),
			cms.PSet(
				tag = cms.untracked.string("energy"),
				quantity = cms.untracked.string("energy")
				),
			cms.PSet(
				tag = cms.untracked.string("jetArea"),
				quantity = cms.untracked.string("jetArea")
				),
			cms.PSet(
				tag = cms.untracked.string("jecFactor"),
				quantity = cms.untracked.string("jecFactor(0)")
				)
				)
	)


	process.ak5TrimmedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK5TrimmedPF')
		)

	process.ak5PrunedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK5PrunedPF')
		)

	process.ak5FilteredLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK5FilteredPF')
		)

	process.ak7Lite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK7PF')
		)

	process.ak7TrimmedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK7TrimmedPF')
		)

	process.ak7PrunedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK7PrunedPF')
		)

	process.ak7FilteredLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK7FilteredPF')
		)




	process.ak7TrimmedGenLite = cms.EDProducer(
	    "CandViewNtpProducer", 
	    src = cms.InputTag('ak7TrimmedGenJetsNoNu'),
	    lazyParser = cms.untracked.bool(True),
	    eventInfo = cms.untracked.bool(False),
	    variables = cms.VPSet(
			cms.PSet(
				tag = cms.untracked.string("px"),
				quantity = cms.untracked.string("px")
				),
			cms.PSet(
				tag = cms.untracked.string("py"),
				quantity = cms.untracked.string("py")
				),
			cms.PSet(
				tag = cms.untracked.string("pz"),
				quantity = cms.untracked.string("pz")
				),
			cms.PSet(
				tag = cms.untracked.string("energy"),
				quantity = cms.untracked.string("energy")
				)
				)
	)


	process.ak7PrunedGenLite = process.ak7TrimmedGenLite.clone(
		src = cms.InputTag('ak7PrunedGenJetsNoNu')
		)

	process.ak7FilteredGenLite = process.ak7TrimmedGenLite.clone(
		src = cms.InputTag('ak7FilteredGenJetsNoNu')
		)

        process.ca8PrunedGenLite = process.ak7TrimmedGenLite.clone(
                src = cms.InputTag('caPrunedGen')
                )

        process.ca12FilteredGenLite = process.ak7TrimmedGenLite.clone(
                src = cms.InputTag('caFilteredGenJetsNoNu')
                )

        process.ca12MassDropFilteredGenLite = process.ak7TrimmedGenLite.clone(
                src = cms.InputTag('caMassDropFilteredGenJetsNoNu')
                )



	process.ak8Lite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK8PF')
		)

	process.ak8TrimmedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK8TrimmedPF')
		)

	process.ak8PrunedLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK8PrunedPF')
		)

	process.ak8FilteredLite = process.ak5Lite.clone(
		src = cms.InputTag('goodPatJetsAK8FilteredPF')
		)


## IVF and BCandidate producer for Vbb cross check analysis
process.load('RecoVertex/AdaptiveVertexFinder/inclusiveVertexing_cff')


# let it run

process.patseq = cms.Sequence(
    process.scrapingVeto*
    process.HBHENoiseFilter*
    #process.offlinePrimaryVerticesDAF*    
    process.goodOfflinePrimaryVertices*
    process.primaryVertexFilter*
    process.softElectronCands*
    process.inclusiveVertexing*
    process.genParticlesForJetsNoNu*
    process.ca8GenJetsNoNu*
    process.ak8GenJetsNoNu*
    process.caFilteredGenJetsNoNu*
    process.caMassDropFilteredGenJetsNoNu*
    getattr(process,"patPF2PATSequence"+postfix)*
    process.patDefaultSequence*
    process.goodPatJetsPFlow*
    process.goodPatJetsCA8PF*
    process.goodPatJetsCA8PrunedPF*
    process.goodPatJetsCATopTagPF*
    process.flavorHistorySeq*
    process.prunedGenParticles*
    process.caPrunedGen*
    process.caTopTagGen*
    process.CATopTagInfosGen*
    getattr(process,"patPF2PATSequence"+postfixLoose)#*
#    process.miniPFLeptonSequence
    )

if options.useExtraJetColls:
	process.extraJetSeq = cms.Sequence(
		process.ak7TrimmedGenJetsNoNu*
		process.ak7FilteredGenJetsNoNu*
		process.ak7PrunedGenJetsNoNu*
		process.goodPatJetsCA12FilteredPF*
		process.goodPatJetsCA12MassDropFilteredPF*
		process.goodPatJetsAK5TrimmedPF*
		process.goodPatJetsAK5FilteredPF*
		process.goodPatJetsAK5PrunedPF*
		process.goodPatJetsAK7PF*
		process.goodPatJetsAK7TrimmedPF*
		process.goodPatJetsAK7FilteredPF*
		process.goodPatJetsAK7PrunedPF*
		process.goodPatJetsAK8PF*
		process.goodPatJetsAK8TrimmedPF*
		process.goodPatJetsAK8FilteredPF*
		process.goodPatJetsAK8PrunedPF*
		process.ak5Lite*
		process.ak5TrimmedLite*
		process.ak5FilteredLite*
		process.ak5PrunedLite*
		process.ak7Lite*
		process.ak7TrimmedLite*
		process.ak7FilteredLite*
		process.ak7PrunedLite*
		process.ak7TrimmedGenLite*
		process.ak7FilteredGenLite*
		process.ak7PrunedGenLite*
		process.ak8Lite*
		process.ak8TrimmedLite*
		process.ak8FilteredLite*
		process.ak8PrunedLite*
                process.ca8PrunedGenLite*
                process.ca12FilteredGenLite*
                process.ca12MassDropFilteredGenLite
	)
	process.patseq *= process.extraJetSeq


if options.useData == True :
    process.patseq.remove( process.genParticlesForJetsNoNu )
    process.patseq.remove( process.genJetParticles )
    process.patseq.remove( process.ak8GenJetsNoNu )
    process.patseq.remove( process.ca8GenJetsNoNu )
    process.patseq.remove( process.caFilteredGenJetsNoNu )
    process.patseq.remove( process.flavorHistorySeq )
    process.patseq.remove( process.caPrunedGen )
    process.patseq.remove( process.caTopTagGen )
    process.patseq.remove( process.CATopTagInfosGen )
    process.patseq.remove( process.prunedGenParticles )
    if options.useExtraJetColls:
	    process.patseq.remove( process.ak8GenJetsNoNu )
	    process.patseq.remove( process.caFilteredGenJetsNoNu )
	    process.patseq.remove( process.ak7TrimmedGenJetsNoNu )
	    process.patseq.remove( process.ak7FilteredGenJetsNoNu )
	    process.patseq.remove( process.ak7PrunedGenJetsNoNu )
	    process.patseq.remove( process.ak7TrimmedGenLite )
	    process.patseq.remove( process.ak7FilteredGenLite )
	    process.patseq.remove( process.ak7PrunedGenLite )
            process.patseq.remove( process.ca8PrunedGenLite )
            process.patseq.remove( process.ca12FilteredGenLite )
            process.patseq.remove( process.ca12MassDropFilteredGenLite )


if options.writeSimpleInputs :
	process.patseq *= cms.Sequence(process.pfInputs)

if options.useSusyFilter :
	process.patseq.remove( process.HBHENoiseFilter )
	process.load( 'PhysicsTools.HepMCCandAlgos.modelfilter_cfi' )
	process.modelSelector.parameterMins = [500.,    0.] # mstop, mLSP
	process.modelSelector.parameterMaxs  = [7000., 200.] # mstop, mLSP
	process.p0 = cms.Path(
		process.modelSelector *
		process.patseq
	)



else :
	process.p0 = cms.Path(
		process.patseq
	)





process.out.SelectEvents.SelectEvents = cms.vstring('p0')

# rename output file
if options.useData :
    if options.writeFat :
        process.out.fileName = cms.untracked.string('ttbsm_' + fileTag + '_data_fat.root')
    else :
        process.out.fileName = cms.untracked.string('ttbsm_' + fileTag + '_data.root')
else :
    if options.writeFat :
        process.out.fileName = cms.untracked.string('ttbsm_' + fileTag + '_mc_fat.root')
    else :
        process.out.fileName = cms.untracked.string('ttbsm_' + fileTag + '_mc.root')


# reduce verbosity
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)


# process all the events
process.maxEvents.input = 100
process.options.wantSummary = True
process.out.dropMetaData = cms.untracked.string("DROPPED")


process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")



process.out.outputCommands = [
    'drop *_cleanPat*_*_*',
    'keep *_selectedPat*_*_*',
    'keep *_goodPat*_*_*',
    'drop patJets_selectedPat*_*_*',
    'keep patJets_selectedPatJetsCA12MassDropFilteredSubjetsPF*_*_*',
    'drop *_selectedPatJets_*_*',    
    'keep *_patMETs*_*_*',
#    'keep *_offlinePrimaryVertices*_*_*',
#    'keep *_kt6PFJets*_*_*',
    'keep *_goodOfflinePrimaryVertices*_*_*',    
    'drop patPFParticles_*_*_*',
#    'drop patTaus_*_*_*',
    'keep recoPFJets_caPruned*_*_*',
    'keep recoPFJets_ca*Filtered*_*_*',
    'keep recoPFJets_caTopTag*_*_*',
    'keep patTriggerObjects_patTriggerPFlow_*_*',
    'keep patTriggerFilters_patTriggerPFlow_*_*',
    'keep patTriggerPaths_patTriggerPFlow_*_*',
    'keep patTriggerEvent_patTriggerEventPFlow_*_*',
    'keep *_cleanPatPhotonsTriggerMatch*_*_*',
    'keep *_cleanPatElectronsTriggerMatch*_*_*',
    'keep *_cleanPatMuonsTriggerMatch*_*_*',
    'keep *_cleanPatTausTriggerMatch*_*_*',
    'keep *_cleanPatJetsTriggerMatch*_*_*',
    'keep *_patMETsTriggerMatch*_*_*',
    'keep double_*_*_PAT',
    'keep *_TriggerResults_*_*',
    'keep *_hltTriggerSummaryAOD_*_*',
    'keep *_caTopTagPFlow_*_*',
    'keep *_caPrunedPFlow_*_*',
    'keep *_CATopTagInfosPFlow_*_*',
    'keep *_prunedGenParticles_*_*',
    'drop recoPFCandidates_selectedPatJets*_*_*',
    'keep recoPFCandidates_selectedPatJetsPFlow_*_*',
    'drop CaloTowers_selectedPatJets*_*_*',
    'drop recoBasicJets_*_*_*',
    'keep *_*Lite_*_*',
    'drop patJets_goodPatJetsAK5FilteredPF_*_*',
    'drop patJets_goodPatJetsAK5PrunedPF_*_*',
    'drop patJets_goodPatJetsAK5TrimmedPF_*_*',
    'drop patJets_goodPatJetsAK7PF_*_*',
    'drop patJets_goodPatJetsAK7FilteredPF_*_*',
    'drop patJets_goodPatJetsAK7PrunedPF_*_*',
    'drop patJets_goodPatJetsAK7TrimmedPF_*_*',
    'drop patJets_goodPatJetsAK8PF_*_*',
    'drop patJets_goodPatJetsAK8FilteredPF_*_*',
    'drop patJets_goodPatJetsAK8PrunedPF_*_*',
    'drop patJets_goodPatJetsAK8TrimmedPF_*_*',
    'drop recoGenJets_selectedPatJets*_*_*',
    'keep *_*_rho_*',
    'drop *_*PFlowLoose*_*_*',
    'keep patElectrons_selected*PFlowLoose*_*_*',
    'keep patMuons_selected*PFlowLoose*_*_*',
    #'keep patTaus_*PFlowLoose*_*_*',
    'keep *_offlineBeamSpot_*_*',
    'drop *_*atTaus_*_*',
    'keep *_pfType1CorrectedMet_*_*',
    'keep *_pfType1p2CorrectedMet_*_*'
    #'keep recoTracks_generalTracks_*_*'
    ]

if options.useData :
    process.out.outputCommands += ['drop *_MEtoEDMConverter_*_*',
                                   'keep LumiSummary_lumiProducer_*_*'
                                   ]
else :
    process.out.outputCommands += ['keep recoGenJets_ca8GenJetsNoNu_*_*',
				   'keep recoGenJets_ak5GenJetsNoNu_*_*',
				   'keep recoGenJets_ak7GenJetsNoNu_*_*',
				   'keep recoGenJets_ak8GenJetsNoNu_*_*',
				   'keep recoGenJets_caFilteredGenJetsNoNu_*_*',
				   'keep recoGenJets_caPrunedGen_*_*',
				   'keep *_caTopTagGen_*_*',
                                   'keep GenRunInfoProduct_generator_*_*',
                                   'keep GenEventInfoProduct_generator_*_*',
                                   'keep *_flavorHistoryFilter_*_*',
                                   'keep PileupSummaryInfos_*_*_*',
				   'keep recoGenJets_selectedPatJetsPFlow_*_*',
                                   ]

if options.writeFat :

    process.out.outputCommands += [
        'keep *_pfNoElectron*_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoPFCandidates_selectedPatJets*_*_*',
        'keep recoBaseTagInfosOwned_selectedPatJets*_*_*',
        'keep CaloTowers_selectedPatJets*_*_*'
        ]
if options.writeFat or options.writeGenParticles :
    if options.useData == False :
        process.out.outputCommands += [
            'keep *_genParticles_*_*'
            ]


if options.writeSimpleInputs :
	process.out.outputCommands += [
		'keep *_pfInputs_*_*'
		]


open('junk.py','w').write(process.dumpPython())
