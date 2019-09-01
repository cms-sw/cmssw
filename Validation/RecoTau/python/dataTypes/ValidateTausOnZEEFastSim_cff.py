import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.ValidationUtils as Utils
from Validation.RecoTau.RecoTauValidation_cfi import ApplyFunctionToSequence, SetValidationExtention
import PhysicsTools.PatAlgos.tools.helpers as helpers

proc = cms.Process('helper')

proc.load('Validation.RecoTau.dataTypes.ValidateTausOnZEE_cff')# import *

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorZEE, 'FastSim') #clones the sequence inside the process with ZEE postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficienciesZEE, 'FastSim') #clones the sequence inside the process with ZEE postfix
proc.produceDenominatorZEEFastSim = helpers.cloneProcessingSnippet( proc, proc.produceDenominatorZEE, 'FastSim')

#adds to TauValNumeratorAndDenominator modules in the sequence FastSim to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'FastSim')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZEEFastSim.visit(zttModifier)

#Set discriminators
discs_to_retain = ['ByDecayModeFinding', 'ElectronRejection']
proc.RunHPSValidationZEEFastSim.discriminators = cms.VPSet([p for p in proc.RunHPSValidationZEEFastSim.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

#Sets the correct naming to efficiency histograms
proc.efficienciesZEEFastSim.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZEEFastSim)
proc.efficienciesZEEFastSimSummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/hpsPFTauProducerZEEFastSim_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/hpsPFTauProducerZEEFastSim_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/hpsPFTauProducerZEEFastSim_Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('FastSim') != -1)]

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominator = cms.Sequence(produceDenominatorZEEFastSim)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZEEFastSim
      +TauValNumeratorAndDenominatorZEEFastSim
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesZEEFastSim
      )
