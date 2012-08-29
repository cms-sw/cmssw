import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import ApplyFunctionToSequence, SetValidationExtention
import PhysicsTools.PatAlgos.tools.helpers as helpers

proc = cms.Process('helper')

proc.load('Validation.RecoTau.dataTypes.ValidateTausOnZEE_cff')# import *

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.runTauValidation, 'FastSim') #clones the sequence inside the process with ZTT postfix
#helpers.cloneProcessingSnippet( proc, proc.TauEfficienciesZTT, 'FastSim') #clones the sequence inside the process with ZTT postfix
#proc.produceDenominatorZTTFastSim = helpers.cloneProcessingSnippet( proc, proc.produceDenominatorZTT, 'FastSim')

#adds to TauValNumeratorAndDenominator modules in the sequence FastSim to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'FastSim')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZEEFastSim.visit(zttModifier)

#Sets the correct naming to efficiency histograms
proc.efficienciesZEEFastSim.streamTag = cms.InputTag("ZEEFastSim")

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('FastSim') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorZEEFastSim = proc.produceDenominatorFastSim
produceDenominator = produceDenominatorZEEFastSim

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZEEFastSim
      +TauValNumeratorAndDenominatorZEEFastSim
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesZEEFastSim
      )
