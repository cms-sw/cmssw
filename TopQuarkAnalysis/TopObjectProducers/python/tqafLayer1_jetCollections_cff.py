import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.jetTools import *

def tqafLayer1JetCollections(process):    
## add kt4 CaloJet collection
    addJetCollection(process, 'kt4CaloJets', 'CaloKt4', runCleaner="CaloJet",
                     doJTA=True, doBTagging=True, jetCorrLabel='FKt4', doType1MET=True, doL1Counters=False)
## add kt5 CaloJet collection
    addJetCollection(process, 'kt6CaloJets', 'CaloKt6', runCleaner="CaloJet",
                     doJTA=True, doBTagging=False,jetCorrLabel='FKt6', doType1MET=True, doL1Counters=False)
## add kt4 PflowJet collection
    addJetCollection(process,'sisCone5PFJets', 'PflowSisCone5', runCleaner="PFJet",
                     doJTA=True, doBTagging=True, jetCorrLabel='FKt4', doType1MET=True, doL1Counters=False)
## add kt4 PflowJet collection
    addJetCollection(process,'kt4PFJets', 'PflowKt4', runCleaner="PFJet",
                     doJTA=True, doBTagging=True, jetCorrLabel='FKt4', doType1MET=True, doL1Counters=False)
## add kt6 PflowJet collection
    addJetCollection(process,'kt6PFJets', 'PflowKt6', runCleaner="PFJet",
                     doJTA=True, doBTagging=True, jetCorrLabel='FKt6', doType1MET=True, doL1Counters=False)

## switch the jet collection to sisCone5
    switchJetCollection(process, 
                        'sisCone5CaloJets',    # jet collection; must be already in the event when patLayer0 sequence is executed
                        layers=[0,1],          # if you're not running patLayer1, set 'layers=[0]' 
                        runCleaner="CaloJet",  # =None if not to clean
                        doJTA=True,            # run jet-track association & JetCharge
                        doBTagging=True,       # run b-tagging
                        jetCorrLabel='Scone5', # example jet correction name; set to None for no JEC
                        doType1MET=True        # recompute Type1 MET using these jets
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
