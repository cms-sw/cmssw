## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

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
print 'CAVEAT : Filters are not yet implemented'


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
process.hepTopTagPFJetsCHS = hepTopTagPFJetsCHS.clone(src='pfNoPileUpJME')

from RecoJets.JetProducers.caTopTaggers_cff import CATopTagInfos, HEPTopTagInfos

process.CATopTagInfos = CATopTagInfos.clone()


################################################################################################
############################ Configure jets in PAT #############################################
################################################################################################


## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

## uncomment the following lines to add ak5PFJetsCHS to your PAT output
addJetCollection(
   process,
   labelName = 'AK5PFCHS',
   jetSource = cms.InputTag('ak5PFJetsCHS'),
   algo='ak5',
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
   algo='ca8',
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA8CMSTopTag',
   jetSource = cms.InputTag('cmsTopTagPFJetsCHS',''),
   algo='ca8',
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
    btagInfos = [
        'CATopTagInfos'
         ]   
   )

addJetCollection(
   process,
   labelName = 'CA8CMSTopTagSubjets',
   jetSource = cms.InputTag('cmsTopTagPFJetsCHS','caTopSubJets'),
   algo='ca8',
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
   algo='ca8',
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA8PrunedSubjets',
   jetSource = cms.InputTag('ca8PFJetsCHSPruned','SubJets'),
   algo='ca8',
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
   algo='ca8',
   jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
   )

addJetCollection(
   process,
   labelName = 'CA15HEPTopTagSubjets',
   jetSource = cms.InputTag('hepTopTagPFJetsCHS','caTopSubJets'),
   algo='ca8',
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
   algo='ak5',
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
    jetSource = cms.InputTag('ak5PFJets'),
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


# Add some user functions for the secondary vertex mass. 
for mod in [process.patJets,
            process.patJetsAK5PFCHS,
            process.patJetsEI,
            process.patJetsCA8CMSTopTagSubjets,
            process.patJetsCA8PrunedSubjets,
            process.patJetsCA15HEPTopTagSubjets ] :
    mod.userData.userFunctions = cms.vstring( "? hasTagInfo('secondaryVertex') && tagInfoSecondaryVertex('secondaryVertex').nVertices() > 0 ? "
                                                      "tagInfoSecondaryVertex('secondaryVertex').secondaryVertex(0).p4().mass() : 0")
    mod.userData.userFunctionLabels = cms.vstring('secvtxMass')


# Add the top-tagging info which piggy-backs on the b-tagging tag info
process.patJetsCA8CMSTopTag.addTagInfos = True
process.patJetsCA8CMSTopTag.tagInfoSources = cms.VInputTag(
    cms.InputTag('CATopTagInfos')
    )

process.patJetsCA15HEPTopTag.addTagInfos = True
process.patJetsCA15HEPTopTag.tagInfoSources = cms.VInputTag(
    cms.InputTag('HEPTopTagInfos')
    )



# Apply jet ID to all of the jets upstream. We aren't going to screw around
# with this, most likely. So, we don't really to waste time with it
# at the analysis level. 
#from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector
#process.goodPatJetsPFlow = cms.EDFilter("PFJetIDSelectionFunctorFilter",
#                                        filterParams = pfJetIDSelector.clone(),
#                                        src = cms.InputTag("selectedPatJetsPFlow")
#                                        )


# Next, "pack" the pat::Jets that use substructure so we can run b-tagging and JEC's on the subjets. 


process.selectedPatJetsCA8PrunedPFPacked = cms.EDProducer("BoostedJetMerger",
                                                      jetSrc=cms.InputTag("selectedPatJetsCA8Pruned"),
                                                      subjetSrc=cms.InputTag("selectedPatJetsCA8PrunedSubjets")
    )

process.selectedPatJetsCATopTagPFPacked = cms.EDProducer("BoostedJetMerger",
                                                      jetSrc=cms.InputTag("selectedPatJetsCA8CMSTopTag"),
                                                      subjetSrc=cms.InputTag("selectedPatJetsCA8CMSTopTagSubjets")
    )


process.selectedPatJetsCAHEPTopTagPFPacked = cms.EDProducer("BoostedJetMerger",
                                                      jetSrc=cms.InputTag("selectedPatJetsCA15HEPTopTag"),
                                                      subjetSrc=cms.InputTag("selectedPatJetsCA15HEPTopTagSubjets")
    )



#print process.out.outputCommands

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = ['dcap:///pnfs/cms/WAX/11/store/relval/CMSSW_7_0_0/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/POSTLS170_V3-v1/00000/56210C05-B596-E311-B433-002618943832.root']
#                                         ##
process.maxEvents.input = 10
#                                         ##
process.out.outputCommands += [
   'keep *_ak5GenJetsNoNu_*_*',
   'keep *_ca8GenJetsNoNu_*_*',
   'keep *_particleFlow_*_*'
   ]
#                                         ##
process.out.fileName = 'patTuple_tlbsm_train.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
