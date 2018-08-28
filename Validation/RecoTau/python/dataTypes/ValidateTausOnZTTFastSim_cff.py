import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.ValidationUtils as Utils
from Validation.RecoTau.RecoTauValidation_cfi import ApplyFunctionToSequence, SetValidationExtention
import PhysicsTools.PatAlgos.tools.helpers as helpers

proc = cms.Process('helper')

proc.load('Validation.RecoTau.dataTypes.ValidateTausOnZTT_cff')# import *

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominatorZTT, 'FastSim') #clones the sequence inside the process with ZTT postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficienciesZTT, 'FastSim') #clones the sequence inside the process with ZTT postfix
proc.produceDenominatorZTTFastSim = helpers.cloneProcessingSnippet( proc, proc.produceDenominatorZTT, 'FastSim')

#adds to TauValNumeratorAndDenominator modules in the sequence FastSim to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'FastSim')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZTTFastSim.visit(zttModifier)

#Set discriminators
proc.RunHPSValidationZTTFastSim.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(False))
)

#Sets the correct naming to efficiency histograms
proc.efficienciesZTTFastSim.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZTTFastSim)
proc.efficienciesZTTFastSimSummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/hpsPFTauProducerZTTFastSim_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/hpsPFTauProducerZTTFastSim_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/hpsPFTauProducerZTTFastSim_Summary/#PAR#PlotNum'),
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

produceDenominator = cms.Sequence(produceDenominatorZTTFastSim)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZTTFastSim
      +TauValNumeratorAndDenominatorZTTFastSim
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesZTTFastSim
      )

