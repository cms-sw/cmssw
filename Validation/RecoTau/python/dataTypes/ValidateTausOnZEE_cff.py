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
    src = cms.InputTag("prunedGenParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 11",
    "keep++ pdgId = -11",
    )
)

selectStableElectrons = genParticlesForJets.clone(src = "selectElectrons")

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
discs_to_retain = ['ByDecayModeFinding', 'ElectronRejection']
proc.RunHPSValidationZEE.discriminators = cms.VPSet([p for p in proc.RunHPSValidationZEE.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

#Sets the correct naming to efficiency histograms
proc.efficienciesZEE.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZEE)
proc.efficienciesZEESummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZEE_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZEE_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZEE_Summary/#PAR#PlotNum'),
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
    selectElectrons
    +cms.ignore(selectStableElectrons)
    +cms.ignore(kinematicSelectedTauValDenominatorZEE)
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
