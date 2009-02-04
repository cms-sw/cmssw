import FWCore.ParameterSet.Config as cms

#----------------------------------------------------------------------------------------
#
#
# adds more jet collections to the tqafLayer1 output. Make sure to call this macro before
# any jet replacement is done. Needs the following file(s) to be known before it can be
# executed:
#
#  * from TopQuarkAnalysis.TopObjectProducers.tqafLayer1_EventContent_cff   import *
#
#
#----------------------------------------------------------------------------------------

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

def tqafAddJetCollections(process):    
## add kt4 CaloJet collection
    addJetCollection(process,
                     'kt4CaloJets',
                     'CaloKt4',
                     runCleaner   = "CaloJet",
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = ('KT4', 'Calo'),
                     doType1MET   = True,
                     doL1Counters = False
                     )
## add kt5 CaloJet collection
    addJetCollection(process,
                     'kt6CaloJets',
                     'CaloKt6',
                     runCleaner   = "CaloJet",
                     doJTA        = True,
                     doBTagging   = False,
                     jetCorrLabel = ('KT6', 'Calo'),
                     doType1MET   = True,
                     doL1Counters = False
                     )
    
## add additional jets to EventContent
    tqafLayer1EventContent_jets = cms.PSet(
        outputCommands = cms.untracked.vstring(
        'keep *_selectedLayer1JetsCaloKt4_*_*',
        'keep *_selectedLayer1JetsCaloKt6_*_*'
        )
    )
    process.tqafEventContent.outputCommands.extend(tqafLayer1EventContent_jets.outputCommands)
    
    return ()
