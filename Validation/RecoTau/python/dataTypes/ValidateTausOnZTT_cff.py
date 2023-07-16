import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

tauGenJetsForVal = tauGenJets.clone(GenParticles = 'prunedGenParticles')

# require generated tau to decay hadronically
objectTypeSelectedTauValDenominatorModuleZTT = cms.EDFilter("TauGenJetDecayModeSelector",
     src = cms.InputTag("tauGenJetsForVal"),
     select = cms.vstring('oneProng0Pi0', 'oneProng1Pi0', 'oneProng2Pi0', 'oneProngOther',
                          'threeProng0Pi0', 'threeProng1Pi0', 'threeProngOther', 'rare'),
     filter = cms.bool(False)
)

# require generator level hadrons produced in tau-decay to have transverse momentum above threshold
kinematicSelectedTauValDenominatorZTT = cms.EDFilter(
   "GenJetSelector", #"GenJetSelector"
   src = cms.InputTag('objectTypeSelectedTauValDenominatorModuleZTT'),
   cut = kinematicSelectedTauValDenominatorCut,#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZTT') #clones the sequence inside the process with ZTT postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'ZTT') #clones the sequence inside the process with ZTT postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorZTT, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorZTT') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence ZTT to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'ZTT')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZTT.visit(zttModifier)

#Set discriminators
discs_to_retain = ['ByDecayModeFinding', 'CombinedIsolationDBSumPtCorr3Hits', 'IsolationMVArun2v1DBoldDMwLT', 'IsolationMVArun2v1DBnewDMwLT', 'MuonRejection', 'ElectronRejection']
proc.RunHPSValidationZTT.discriminators = cms.VPSet([p for p in proc.RunHPSValidationZTT.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

#Sets the correct naming to efficiency histograms
proc.efficienciesZTT.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZTT)
proc.efficienciesZTTSummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZTT_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZTT_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerZTT_Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('ZTT') != -1)]

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorZTT = cms.Sequence(
      tauGenJetsForVal
      +cms.ignore(objectTypeSelectedTauValDenominatorModuleZTT)
      +cms.ignore(kinematicSelectedTauValDenominatorZTT)
      )

produceDenominator = cms.Sequence(produceDenominatorZTT)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZTT
      +TauValNumeratorAndDenominatorZTT
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesZTT
      )
