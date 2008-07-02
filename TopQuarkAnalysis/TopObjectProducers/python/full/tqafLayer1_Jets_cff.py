import FWCore.ParameterSet.Config as cms

#
# L1 input
#

## import module
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import allLayer1Jets

## configure for tqaf
allLayer1Jets.jetSource              = 'allLayer0Jets'
allLayer1Jets.embedCaloTowers        = True
allLayer1Jets.getJetMCFlavour        = True
allLayer1Jets.JetPartonMapSource     = 'jetFlavourAssociation'
allLayer1Jets.addGenPartonMatch      = True
allLayer1Jets.genPartonMatch         = 'jetPartonMatch'
allLayer1Jets.addGenJetMatch         = True
allLayer1Jets.genJetMatch            = 'jetGenJetMatch'
allLayer1Jets.addPartonJetMatch      = False
allLayer1Jets.partonJetSource        = 'nonsenseName'
allLayer1Jets.addResolutions         = True
allLayer1Jets.useNNResolutions       = False
allLayer1Jets.caliJetResoFile        = 'PhysicsTools/PatUtils/data/Resolutions_lJets_MCJetCorJetIcone5.root'
allLayer1Jets.caliBJetResoFile       = 'PhysicsTools/PatUtils/data/Resolutions_bJets_MCJetCorJetIcone5.root'
allLayer1Jets.addBTagInfo            = True
allLayer1Jets.addDiscriminators      = True
allLayer1Jets.discriminatorModule    = 'layer0BTags'
allLayer1Jets.discriminatorNames     = ['*']
allLayer1Jets.addTagInfoRefs         = True
allLayer1Jets.tagInfoModule          = 'layer0TagInfos'
allLayer1Jets.tagInfoNames           = ['secondaryVertexTagInfos',
                                        'softElectronTagInfos',
                                        'softMuonTagInfos',
                                        'impactParameterTagInfos']
allLayer1Jets.addAssociatedTracks    = True
allLayer1Jets.trackAssociationSource = 'layer0JetTracksAssociator'
allLayer1Jets.addJetCharge           = True
allLayer1Jets.jetChargeSource        = 'layer0JetCharge'

#
# L1 selection
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedLayer1Jets

## configure for tqaf
selectedLayer1Jets.src               = 'allLayer1Jets'
selectedLayer1Jets.cut               = 'et > 20. & abs(eta) < 3.0 & nConstituents > 0'

