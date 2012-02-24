from Validation.RecoTau.dataTypes.ValidateTausOnRealData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealElectronsData_cff import *
from Validation.RecoTau.dataTypes.ValidateTausOnRealMuonsData_cff import *

dqmInfoTauV = cms.EDAnalyzer(
    "DQMEventInfo",
    subSystemFolder = cms.untracked.string('RecoTauV')
    )


produceDenoms = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealElectronsData+
    produceDenominatorRealMuonsData
    )

pfTauRunDQMValidation = cms.Sequence(
    TauValNumeratorAndDenominatorRealData+
    TauValNumeratorAndDenominatorRealElectronsData+
    TauValNumeratorAndDenominatorRealMuonsData+
    dqmInfoTauV
    )

runTauEff = cms.Sequence(
    efficienciesRealData+
    efficienciesRealElectronsData+
    efficienciesRealMuonsData+
    normalizePlotsRealMuonsData
    )

#----------------------------------------------------------------------------------------------------------------------------------------
#                                                  Denominators according to dataset
#----------------------------------------------------------------------------------------------------------------------------------------
produceDenomsMu = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealMuonsData
    )
produceDenomsSingleMu = cms.Sequence(
    produceDenominatorRealData+
    produceDenominatorRealMuonsData
    )
produceDenomsJet = produceDenominatorRealData
produceDenomsMultiJet = produceDenomsJet

produceDenomsDoubleElectron = produceDenominatorRealElectronsData
produceDenomsTauPlusX = produceDenomsDoubleElectron

#----------------------------------------------------------------------------------------------------------------------------------------
#                                                  Main modules according to dataset
#----------------------------------------------------------------------------------------------------------------------------------------
#Mu Dataset
proc.GeneralMuSequence = cms.Sequence( proc.TauValNumeratorAndDenominatorRealData * proc.TauValNumeratorAndDenominatorRealMuonsData )
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.GeneralMuSequence, 'AtMu') #clones the sequence inside the process with AtMu postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.GeneralMuSequenceAtMu.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtMu') != -1), dir(proc) )
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)
    
pfTauRunDQMValidationMu = cms.Sequence(
    TauValNumeratorAndDenominatorRealDataAtMu+
    TauValNumeratorAndDenominatorRealMuonsDataAtMu+
    dqmInfoTauV
    )

#SingleMu Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.GeneralMuSequence, 'AtSingleMu') #clones the sequence inside the process with AtSingleMu postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.GeneralMuSequenceAtSingleMu.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtSingleMu') != -1), dir(proc) )
#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

pfTauRunDQMValidationSingleMu = cms.Sequence(
    TauValNumeratorAndDenominatorRealDataAtSingleMu+
    TauValNumeratorAndDenominatorRealMuonsDataAtSingleMu+
    dqmInfoTauV
    )

#Jet Dataset
procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorRealData, 'AtJet') #clones the sequence inside the process with AtJet postfix
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.TauValNumeratorAndDenominatorRealDataAtJet.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtJet') != -1), dir(proc) )
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
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.TauValNumeratorAndDenominatorRealDataAtMultiJet.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtMultiJet') != -1), dir(proc) )
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
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.TauValNumeratorAndDenominatorRealElectronsDataAtDoubleElectron.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('AtDoubleElectron') != -1), dir(proc) )
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
seqModifier = ApplyFunctionToSequence( lambda module: setTrigger( module, cms.PSet( hltDBKey = cms.string('OUR_CUSTOM_STRING'), hltPaths = cms.vstring('OUR_HLT_FALLBACK_PATH') ) ) )
proc.TauValNumeratorAndDenominatorRealElectronsDataTauPlusX.visit(seqModifier)
#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('TauPlusX') != -1), dir(proc) )
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
runTauEffMu = cms.Sequence(
    efficienciesRealMuonsData+
    efficienciesRealData+
    normalizePlotsRealMuonsData
    )
runTauEffSingleMu = runTauEffMu

runTauEffJet = TauEfficienciesRealData
runTauEffMutiJet = runTauEffJet

runTauEffDoubleElectron = produceDenominatorRealElectronsData
runTauEffTauPlusX = runTauEffDoubleElectron

##Full sequences, including normalizations
## TauEfficienciesRealData+
## TauEfficienciesRealElectronsData+
## TauEfficienciesRealMuonsData

