from __future__ import print_function
# As of 1 Feb 2017:
# This configuration appears to be already broken. It
# fails to even run only under python.
# For this reason, it was not converted to use Tasks.
# If it is ever fixed, it will also need to be migrated
# to use Tasks to run unscheduled modules.

## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

###############################
####### Parameters ############
###############################
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')


options.register ('tlbsmTag',
                  'tlbsm_71x_v1',
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.string,
                  'TLBSM tag use in production')


options.register ('usePythia8',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Use status codes from Pythia8 rather than Pythia6")


options.register ('usePythia6andPythia8',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "Use status codes from Pythia8 and Pythia6")

options.parseArguments()

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoJets.Configuration.GenJetParticles_cff")
from TopQuarkAnalysis.TopPairBSM.filters_cff import applyFilters

################################################################################################
############################ Run filters #######################################################
################################################################################################

######### TO DO : TURN ON FILTERS ###########
#applyFilters(process)
print('CAVEAT : Filters are not yet implemented')

######### TO DO : TURN ON TRIGGERS! #########


###############################
####### DAF PV's     ##########
###############################

pvSrc = 'offlinePrimaryVertices'

## The good primary vertex filter ____________________________________________||
process.primaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake & ndof > 4 & abs(z) <= 24 & position.Rho <= 2"),
    filter = cms.bool(True)
    )


from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

process.goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( maxZ = cms.double(24.0),
                                     minNdof = cms.double(4.0) # this is >= 4
                                     ),
    src=cms.InputTag(pvSrc)
    )

################################################################################################
############################ Pruned GenParticles ###############################################
################################################################################################

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

if options.usePythia8 :
    process.prunedGenParticles.select = cms.vstring(
                                                "drop  *"
                                                ,"keep status = 21" #keeps  particles from the hard matrix element
                                                ,"keep status = 22" #keeps  particles from the hard matrix element
                                                ,"keep status = 23" #keeps  particles from the hard matrix element
                                                ,"keep (abs(pdgId) >= 11 & abs(pdgId) <= 16) & status = 1" #keeps e/mu and nus with status 1
                                                ,"keep (abs(pdgId)  = 15) & (status = 21 || status = 22 || status = 23) " #keeps taus
                                                )
if options.usePythia6andPythia8 :
    process.prunedGenParticles.select = cms.vstring(
                                                "drop  *"
                                                ,"keep status = 3" #keeps  particles from the hard matrix element
                                                ,"keep status = 21" #keeps  particles from the hard matrix element
                                                ,"keep status = 22" #keeps  particles from the hard matrix element
                                                ,"keep status = 23" #keeps  particles from the hard matrix element
                                                ,"keep (abs(pdgId) >= 11 & abs(pdgId) <= 16) & status = 1" #keeps e/mu and nus with status 1
                                                ,"keep (abs(pdgId)  = 15) & (status = 3 || status = 21 || status = 22 || status = 23)" #keeps taus
                                                )                                      


    


################################################################################################
############################ Configure leptons #################################################
################################################################################################


postfix = 'EI'

from PhysicsTools.PatAlgos.tools.pfTools import adaptPFMuons, adaptPFElectrons
from PhysicsTools.PatAlgos.tools.helpers import loadWithPostfix
from PhysicsTools.PatAlgos.tools.helpers import applyPostfix

#loadWithPostfix(process,'PhysicsTools.PatAlgos.patSequences_cff',postfix)


# Electrons

adaptPFElectrons(process,
                 process.patElectrons,
                 postfix)

# Muons

adaptPFMuons(process,
             process.patMuons,
             postfix,
             muonMatchModule=process.muonMatch
             )


# Taus
################ TO DO : We need a tau expert to do this. ###################
#process.patTaus.tauSource = cms.InputTag("pfTaus"+postfix)


################################################################################################
############################ Run extra MET reconstruction ######################################
################################################################################################


from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
addMETCollection(process, labelName='patMETPF', metSource='pfType1CorrectedMet')


################################################################################################
############################ Run extra jet reconstruction ######################################
################################################################################################
from RecoJets.Configuration.RecoPFJets_cff import *
process.ca15PFJetsCHS = ca8PFJetsCHS.clone(rParam=1.5,src='pfNoPileUpJME',doAreaFastjet=False)
process.hepTopTagPFJetsCHS = hepTopTagPFJetsCHS.clone(src='pfNoPileUpJME')

from RecoJets.JetProducers.caTopTaggers_cff import caTopTagInfos, hepTopTagInfos

process.caTopTagInfos = caTopTagInfos.clone()


################################################################################################
############################ Configure jets in PAT #############################################
################################################################################################


## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

## uncomment the following lines to add ak4PFJetsCHS to your PAT output
addJetCollection(
   process,
   labelName = 'AK4PFCHS',
   jetSource = cms.InputTag('ak4PFJetsCHS'),
   algo='ak',
   rParam=0.4,
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]   
   )

addJetCollection(
   process,
   labelName = 'CA8PFCHS',
   jetSource = cms.InputTag('ca8PFJetsCHS'),
   algo='ca',
   rParam=0.8,
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA8CMSTopTag',
   jetSource = cms.InputTag('cmsTopTagPFJetsCHS',''),
   getJetMCFlavour=False,
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
    btagInfos = [
        'caTopTagInfos'
         ]   
   )

addJetCollection(
   process,
   labelName = 'CA8CMSTopTagSubjets',
   jetSource = cms.InputTag('cmsTopTagPFJetsCHS','caTopSubJets'),
   algo='ca',
   rParam=0.8,
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]
   )

addJetCollection(
   process,
   labelName = 'CA8Pruned',
   jetSource = cms.InputTag('ca8PFJetsCHSPruned',''),
   getJetMCFlavour=False,
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA8PrunedSubjets',
   jetSource = cms.InputTag('ca8PFJetsCHSPruned','SubJets'),
   algo='ca',
   rParam=0.8,
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]   
   )


addJetCollection(
   process,
   labelName = 'CA15HEPTopTag',
   jetSource = cms.InputTag('hepTopTagPFJetsCHS',''),
   getJetMCFlavour=False,
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA15HEPTopTagSubjets',
   jetSource = cms.InputTag('hepTopTagPFJetsCHS','caTopSubJets'),
   algo='ca',
   rParam=1.5,
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]   
   )

addJetCollection(
   process,
   labelName = 'EI',
   jetSource = cms.InputTag('pfJetsEI'),
   algo='ak',
   rParam=0.4,
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]   
   )

switchJetCollection(
    process,
    jetSource = cms.InputTag('ak4PFJets'),
    algo='ak',
    rParam=0.4,
    jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
    btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
    btagInfos = [
        'secondaryVertexTagInfos'
         ]
   )


# Subjet flavor still requires some manual tweaking until it is better integrated into PAT
# Adjust the flavor of subjets of pruned CA8 jets
process.patJetFlavourAssociationCA8PrunedSubjets = process.patJetFlavourAssociationCA8PrunedSubjets.clone(
    jets = cms.InputTag('ca8PFJetsCHS'),
    groomedJets = cms.InputTag('ca8PFJetsCHSPruned'),
    subjets = cms.InputTag('ca8PFJetsCHSPruned','SubJets')
)
process.patJetsCA8PrunedSubjets.JetFlavourInfoSource = cms.InputTag('patJetFlavourAssociationCA8PrunedSubjets','SubJets')
# Adjust the flavor of subjets of CMSTopTag jets
process.patJetFlavourAssociationCA8CMSTopTagSubjets = process.patJetFlavourAssociationCA8CMSTopTagSubjets.clone(
    jets = cms.InputTag('ca8PFJetsCHS'),
    groomedJets = cms.InputTag('cmsTopTagPFJetsCHS'),
    subjets = cms.InputTag('cmsTopTagPFJetsCHS','caTopSubJets')
)
process.patJetsCA8CMSTopTagSubjets.JetFlavourInfoSource = cms.InputTag('patJetFlavourAssociationCA8CMSTopTagSubjets','SubJets')
# Adjust the flavor of subjets of HEPTopTag jets
process.patJetFlavourAssociationCA15HEPTopTagSubjets = process.patJetFlavourAssociationCA15HEPTopTagSubjets.clone(
    jets = cms.InputTag('ca15PFJetsCHS'),
    groomedJets = cms.InputTag('hepTopTagPFJetsCHS'),
    subjets = cms.InputTag('hepTopTagPFJetsCHS','caTopSubJets')
)
process.patJetsCA15HEPTopTagSubjets.JetFlavourInfoSource = cms.InputTag('patJetFlavourAssociationCA15HEPTopTagSubjets','SubJets')


# Add some user functions for the secondary vertex mass. 
for mod in [process.patJets,
            process.patJetsAK4PFCHS,
            process.patJetsEI,
            process.patJetsCA8PFCHS,
            process.patJetsCA8CMSTopTagSubjets,
            process.patJetsCA8PrunedSubjets,
            process.patJetsCA15HEPTopTagSubjets ] :
    mod.userData.userFunctions = cms.vstring( "? hasTagInfo('secondaryVertex') && tagInfoSecondaryVertex('secondaryVertex').nVertices() > 0 ? "
                                                      "tagInfoSecondaryVertex('secondaryVertex').secondaryVertex(0).p4().mass() : 0")
    mod.userData.userFunctionLabels = cms.vstring('secvtxMass')


# Add the top-tagging info which piggy-backs on the b-tagging tag info
process.patJetsCA8CMSTopTag.addTagInfos = True
process.patJetsCA8CMSTopTag.tagInfoSources = cms.VInputTag(
    cms.InputTag('caTopTagInfos')
    )

process.patJetsCA15HEPTopTag.addTagInfos = True
process.patJetsCA15HEPTopTag.tagInfoSources = cms.VInputTag(
    cms.InputTag('hepTopTagInfos')
    )



# Apply jet ID to all of the jets upstream. We aren't going to screw around
# with this, most likely. So, we don't really to waste time with it
# at the analysis level. 
from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector
for ilabel in ['PatJets',
               'PatJetsAK4PFCHS',
               'PatJetsEI',
               'PatJetsCA8PFCHS',
               'PatJetsCA8CMSTopTag',
               'PatJetsCA8Pruned',
               'PatJetsCA15HEPTopTag'] :
    ifilter = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                            filterParams = pfJetIDSelector.clone(),
                            src = cms.InputTag("selected" + ilabel)
                            )
    setattr( process, 'good' + ilabel, ifilter )


# Next, "pack" the pat::Jets that use substructure so we can run b-tagging and JEC's on the subjets. 

for ilabel in ['PatJetsCA8CMSTopTag',
               'PatJetsCA8Pruned',
               'PatJetsCA15HEPTopTag'] :
    imerger = cms.EDProducer("BoostedJetMerger",
                            jetSrc=cms.InputTag("good" + ilabel ),
                            subjetSrc=cms.InputTag("selected" + ilabel + "Subjets")
    )
    setattr( process, 'good' + ilabel + 'Packed', imerger )


#print process.out.outputCommands

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = ['/store/relval/CMSSW_7_2_0_pre1/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/0E9DF649-2BFE-E311-8F3F-0025905B85E8.root']
#                                         ##
process.maxEvents.input = 10
#                                         ##
process.out.outputCommands += [
    'keep GenRunInfoProduct_generator_*_*',
    'keep GenEventInfoProduct_generator_*_*',
    'keep PileupSummaryInfos_*_*_*',
    'keep *_goodOfflinePrimaryVertices*_*_*',    
    'keep *_ak4GenJetsNoNu_*_*',
    'keep *_ca8GenJetsNoNu_*_*',    
    'keep *_fixedGrid_*_*',
    'drop *_*_rho*_*',
    'drop *_*_sigma*_*',    
    'keep patJets_goodPatJets_*_*',
    'keep patJets_goodPatJetsAK4PFCHS_*_*',
    'keep patJets_goodPatJetsCA15HEPTopTagPacked_*_*',
    'keep patJets_goodPatJetsCA8CMSTopTagPacked_*_*',
    'keep patJets_goodPatJetsCA8PFCHS_*_*',
    'keep patJets_goodPatJetsCA8PrunedPacked_*_*',
    'keep patJets_goodPatJetsEI_*_*',
    'drop patJets_selected*_*_*',                        # Drop all of the "selected" ones as they are duplicates...
    'keep patJets_selected*Subjets_*_*',                 # ... except subjets
    'drop CaloTowers_*_*_*',
    'drop recoGenJets_*_genJets_*',
    'drop recoPFCandidates_*_pfCandidates_*',
    'keep *_particleFlow__*',
    'keep *_prunedGenParticles_*_*',
    'keep patTriggerObjects_patTrigger_*_*',
    'keep patTriggerFilters_patTrigger_*_*',
    'keep patTriggerPaths_patTrigger_*_*',
    'keep patTriggerEvent_patTriggerEvent_*_*',
    'keep *_cleanPatPhotonsTriggerMatch*_*_*',
    'keep *_cleanPatElectronsTriggerMatch*_*_*',
    'keep *_cleanPatMuonsTriggerMatch*_*_*',
    'keep *_cleanPatTausTriggerMatch*_*_*',
    'keep *_cleanPatJetsTriggerMatch*_*_*',
    'keep *_patMETsTriggerMatch*_*_*',
    'keep *_TriggerResults_*_*',
    'keep *_hltTriggerSummaryAOD_*_*',    
   ]
#                                         ##
process.out.fileName = 'patTuple_tlbsm_train_' + options.tlbsmTag + '.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
