import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

selectElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 11",
    "keep++ pdgId = -11",
    )
)

selectStableElectrons = genParticlesForJets.clone(src = cms.InputTag("selectElectrons"))

#objectTypeSelectedTauValDenominatorModule = copy.deepcopy(iterativeCone5GenJets)
#objectTypeSelectedTauValDenominatorModule.src = cms.InputTag("selectElectronsForGenJets")

kinematicSelectedTauValDenominatorZEE = cms.EDFilter(
   "CandPtrSelector",
   src = cms.InputTag('selectStableElectrons'),
   cut = kinematicSelectedTauValDenominatorCut,#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZEE') #clones the sequence inside the process with ZEE postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'ZEE') #clones the sequence inside the process with ZEE postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorZEE, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorZEE') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence ZEE to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'ZEE')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZEE.visit(zttModifier)

#Set discriminators
proc.RunHPSValidationZEE.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6VLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6MediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6TightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA6VTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False))
)

#Sets the correct naming to efficiency histograms
proc.efficienciesZEE.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZEE)
proc.efficienciesZEESummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/hpsPFTauProducerZEE_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/hpsPFTauProducerZEE_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/hpsPFTauProducerZEE_Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('ZEE') != -1)]

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorZEE = cms.Sequence(
    selectElectrons*
    selectStableElectrons*
    kinematicSelectedTauValDenominatorZEE
    )

produceDenominator = cms.Sequence(produceDenominatorZEE)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZEE*
      TauValNumeratorAndDenominatorZEE
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode*
      TauEfficienciesZEE
      )

