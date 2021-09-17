from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *
from Validation.RecoTau.RecoTauValidation_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoTauV = DQMEDAnalyzer(
    "DQMEventInfo",
    subSystemFolder = cms.untracked.string('RecoTauV')
    )


produceDenomsData = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData
    )

seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForALLQCDDataset'), hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*') ) ) )
TauValNumeratorAndDenominatorRealData.visit(seqModifier)

seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForALLEleDataset'), hltPaths = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v*') ) ) )
TauValNumeratorAndDenominatorRealElectronsData.visit(seqModifier)

seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForALLMuDataset'), hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*') ) ) )
TauValNumeratorAndDenominatorRealMuonsData.visit(seqModifier)

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData+
    dqmInfoTauV
    )

runTauEff = cms.Sequence(
    efficienciesRealData+
    efficienciesRealDataSummary+
    efficienciesRealElectronsData+
    efficienciesRealElectronsDataSummary+
    efficienciesRealMuonsData+
    efficienciesRealMuonsDataSummary+
    efficienciesTauValidationMiniAODRealData+
    efficienciesTauValidationMiniAODRealElectronsData+
    efficienciesTauValidationMiniAODRealMuonsData+
    normalizePlotsRealMuonsData
    )

#----------------------------------------------------------------------------------------------------------------------------------------
#                                                  Denominators according to dataset
#----------------------------------------------------------------------------------------------------------------------------------------
## produceDenomsMu = cms.Sequence(
##     produceDenominatorRealData+
##     produceDenominatorRealMuonsData
##     )
produceDenomsSingleMu = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealMuonsData
    )
produceDenomsJet = cms.Sequence(produceDenominatorRealData)
produceDenomsMultiJet = cms.Sequence(produceDenomsJet)

produceDenomsDoubleElectron = cms.Sequence(produceDenominatorRealElectronsData)
produceDenomsTauPlusX = cms.Sequence(produceDenomsDoubleElectron)

#----------------------------------------------------------------------------------------------------------------------------------------
#                                                  Main modules according to dataset
#----------------------------------------------------------------------------------------------------------------------------------------
proc.GeneralMuSequence = cms.Sequence( proc.TauValNumeratorAndDenominatorRealData * proc.TauValNumeratorAndDenominatorRealMuonsData )

#Mu Dataset
## procAttributes = dir(proc) #Takes a snapshot of what there in the process
## helpers.cloneProcessingSnippet( proc, proc.GeneralMuSequence, 'AtMu') #clones the sequence inside the process with AtMu postfix
## seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForMuDataset'), hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*') ) ) )
## proc.GeneralMuSequenceAtMu.visit(seqModifier)
## #checks what's new in the process (the cloned sequences and modules in them)
## newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtMu') != -1), dir(proc) )
## #spawns a local variable with the same name as the proc attribute, needed for future process.load
## for newAttr in newProcAttributes:
##     locals()[newAttr] = getattr(proc,newAttr)
 
## pfTauRunDQMValidationMu = cms.Sequence(
##     TauValNumeratorAndDenominatorRealDataAtMu+
##     TauValNumeratorAndDenominatorRealMuonsDataAtMu+
##     dqmInfoTauV
##     )

#SingleMu Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.GeneralMuSequence, 'AtSingleMu') #clones the sequence inside the process with AtSingleMu postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForSingleMuDataset'), hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*') ) ) )
proc.GeneralMuSequenceAtSingleMu.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('AtSingleMu') != -1)]
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationSingleMu = cms.Sequence(
    GeneralMuSequenceAtSingleMu+
    dqmInfoTauV
    )

#Jet Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorRealData, 'AtJet') #clones the sequence inside the process with AtJet postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForJetDataset'), hltPaths = cms.vstring('HLT_Jet30_L1FastJet_v*') ) ) )
proc.TauValNumeratorAndDenominatorRealDataAtJet.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('AtJet') != -1)]
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationJet = cms.Sequence(
    TauValNumeratorAndDenominatorRealDataAtJet+
    dqmInfoTauV
    )

#MultiJet Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorRealData, 'AtMultiJet') #clones the sequence inside the process with AtMultiJet postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForMultiJetDataset'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.TauValNumeratorAndDenominatorRealDataAtMultiJet.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('AtMultiJet') != -1)]
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationMultiJet = cms.Sequence(
    TauValNumeratorAndDenominatorRealDataAtMultiJet+
    dqmInfoTauV
    )

#DoubleElectron Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorRealElectronsData, 'AtDoubleElectron') #clones the sequence inside the process with AtDoubleElectron postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForDoubleElectronDataset'), hltPaths = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v*') ) ) ) 
proc.TauValNumeratorAndDenominatorRealElectronsDataAtDoubleElectron.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('AtDoubleElectron') != -1)]
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationDoubleElectron = cms.Sequence(
    TauValNumeratorAndDenominatorRealElectronsDataAtDoubleElectron+
    dqmInfoTauV
    )

#TauPlusX Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorRealElectronsData, 'TauPlusX') #clones the sequence inside the process with TauPlusX postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('TauTriggerForTauPlusXDataset'), hltPaths = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v*') ) ) )
proc.TauValNumeratorAndDenominatorRealElectronsDataTauPlusX.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('TauPlusX') != -1)]
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationTauPlusX = cms.Sequence(
    TauValNumeratorAndDenominatorRealElectronsDataTauPlusX+
    dqmInfoTauV
    )

#----------------------------------------------------------------------------------------------------------------------------------------
#                                                      Efficiencies production according to dataset
#----------------------------------------------------------------------------------------------------------------------------------------
## runTauEffMu = cms.Sequence(
##     efficienciesRealMuonsData+
##     efficienciesRealData+
##     normalizePlotsRealMuonsData
##     )
runTauEffSingleMu =  cms.Sequence(
    efficienciesRealMuonsData+
    efficienciesRealData+
    normalizePlotsRealMuonsData
    )       

runTauEffJet = cms.Sequence(TauEfficienciesRealData)
runTauEffMutiJet = cms.Sequence(runTauEffJet)

runTauEffDoubleElectron = cms.Sequence(produceDenominatorRealElectronsData)
runTauEffTauPlusX = cms.Sequence(runTauEffDoubleElectron)

##Full sequences, including normalizations
## TauEfficienciesRealData+
## TauEfficienciesRealElectronsData+
## TauEfficienciesRealMuonsData

