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
proc.RunHPSValidationZEEFastSim.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6VLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6MediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6TightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6VTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False))
)

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
