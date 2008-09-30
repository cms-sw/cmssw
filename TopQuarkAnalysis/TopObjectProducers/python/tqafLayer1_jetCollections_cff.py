import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------------
#
#
# adds more jet collections to the patLayer1 output. Make sure to 
# call this macro before any jet replacelment is done.
#
#
#-----------------------------------------------------------------

from PhysicsTools.PatAlgos.tools.jetTools import *

def tqafLayer1JetCollections(process):    
## add kt4 CaloJet collection
    addJetCollection(process,
                     'kt4CaloJets',
                     'CaloKt4',
                     runCleaner   = "CaloJet",
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = 'FKt4',
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
                     jetCorrLabel = 'FKt6',
                     doType1MET   = True,
                     doL1Counters = False
                     )
## add kt4 PflowJet collection
    addJetCollection(process,
                     'sisCone5PFJets',
                     'PflowSisCone5',
                     runCleaner   = "PFJet",
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = 'FKt4',
                     doType1MET   = True,
                     doL1Counters = False
                     )
## add kt4 PflowJet collection
    addJetCollection(process,
                     'kt4PFJets',
                     'PflowKt4',
                     runCleaner   = "PFJet",
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = 'FKt4',
                     doType1MET   = True,
                     doL1Counters = False
                     )
## add kt6 PflowJet collection
    addJetCollection(process,
                     'kt6PFJets',
                     'PflowKt6',
                     runCleaner   = "PFJet",
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = 'FKt6',
                     doType1MET   = True,
                     doL1Counters = False
                     )
    
## add jets to EventContent
    tqafLayer1EventContent_jets = cms.PSet(
        outputCommands = cms.untracked.vstring(
        'keep *_selectedLayer1JetsCaloKt4_*_*',
        'keep *_selectedLayer1JetsCaloKt6_*_*',
        'keep *_selectedLayer1JetsPflowSisCone5_*_*',
        'keep *_selectedLayer1JetsPflowKt4_*_*',
        'keep *_selectedLayer1JetsPflowKt6_*_*'
        )
        )
    process.tqafEventContent.outputCommands.extend(tqafLayer1EventContent_jets.outputCommands)
    
    return ()
