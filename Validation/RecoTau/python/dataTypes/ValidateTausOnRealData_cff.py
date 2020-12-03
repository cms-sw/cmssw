import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

kinematicSelectedPFJets = cms.EDFilter(
    "TauValPFJetSelector",
    src = cms.InputTag('ak4PFJets'),
    cut = cms.string("pt > 15 & abs(eta) < 2.5"),
    filter = cms.bool(False)
	)

PFJetsId = cms.EDFilter(
    "TauValPFJetSelector",
    src = cms.InputTag('kinematicSelectedPFJets'),
    cut = cms.string("chargedHadronEnergyFraction > 0.0 & neutralHadronEnergyFraction < 0.99 & neutralHadronEnergyFraction < 0.99 & chargedEmEnergyFraction < 0.99 & chargedEmEnergyFraction < 0.99 & neutralEmEnergyFraction < 0.99 & chargedMultiplicity > 0 & nConstituents > 1"),
    filter = cms.bool(False)
	)
CleanedPFJets = cms.EDProducer("TauValJetViewCleaner",
    srcObject            = cms.InputTag( "kinematicSelectedPFJets" ),
    srcObjectsToRemove   = cms.VInputTag( cms.InputTag("muons"), cms.InputTag("gedGsfElectrons") ),
    deltaRMin            = cms.double(0.15)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealData') #clones the sequence inside the process with RealData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealData') #clones the sequence inside the process with RealData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealData, 'kinematicSelectedTauValDenominator', 'CleanedPFJets') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealData.visit(zttModifier)

binning = cms.PSet(
	pt = cms.PSet( nbins = cms.int32(25), min = cms.double(0.), max = cms.double(250.) ), #hinfo(75, 0., 150.)
	eta = cms.PSet( nbins = cms.int32(4), min = cms.double(-3.), max = cms.double(3.) ), #hinfo(60, -3.0, 3.0);
	phi = cms.PSet( nbins = cms.int32(4), min = cms.double(-180.), max = cms.double(180.) ), #hinfo(36, -180., 180.);
	pileup = cms.PSet( nbins = cms.int32(18), min = cms.double(0.), max = cms.double(72.) ),#hinfo(25, 0., 25.0);
	)
zttModifier = ApplyFunctionToSequence(lambda m: setBinning(m,binning))
proc.TauValNumeratorAndDenominatorRealData.visit(zttModifier)
#-----------------------------------------

#Set discriminators
discs_to_retain = ['ByDecayModeFinding', 'CombinedIsolationDBSumPtCorr3Hits', 'IsolationMVArun2v1DBoldDMwLT', 'IsolationMVArun2v1DBnewDMwLT']
proc.RunHPSValidationRealData.discriminators = cms.VPSet([p for p in proc.RunHPSValidationRealData.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

#Sets the correct naming to efficiency histograms
proc.efficienciesRealData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealData)
proc.efficienciesRealDataSummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealData_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealData_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealData_Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('RealData') != -1)]

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)


produceDenominatorRealData = cms.Sequence(
      cms.ignore(kinematicSelectedPFJets) *
      cms.ignore(PFJetsId) *
      CleanedPFJets
      )

produceDenominator = cms.Sequence(produceDenominatorRealData)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorRealData
      +TauValNumeratorAndDenominatorRealData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealData
      )
