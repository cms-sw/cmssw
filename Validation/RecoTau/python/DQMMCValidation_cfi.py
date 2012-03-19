import FWCore.ParameterSet.Config as cms
#import Validation.RecoTau.ValidateTausOnZTT_cff as zttVal
from Validation.RecoTau.ValidateTausOnZTT_cff import *
import copy
import Validation.RecoTau.ValidationUtils as Utils

#------------------------------------------------------------
#                     Producing Num e Denom
#------------------------------------------------------------

def PrintSeq(seq, tau=False):
    scanner = Utils.Scanner()
    seq.visit(scanner)
    for module in scanner.modules():
        print type(module)
        if type(module) is cms.EDAnalyzer and tau:# or type(module) is cms.EDFilter:
            print module.TauProducer.value() + module.ExtensionName.value()

def SetSignalPars(module):
    module.ExtensionName = (module.ExtensionName.value()+"_Signal")
    module.RefCollection = "zttKinemSelection"

def SetFakePars(module):
    module.ExtensionName = (module.ExtensionName.value()+"_Fakes")
    module.RefCollection = "qcdKinemSelection"

pfTauRunDQMValidation = cms.Sequence()

tauGenJets = copy.deepcopy(tauGenJets)
zttDenominator = objectTypeSelectedTauValDenominator.clone()
zttKinemSelection= kinematicSelectedTauValDenominator.clone(src = cms.InputTag("zttDenominator"))

zttModifier = ApplyFunctionToSequence(SetSignalPars)
TauValNumeratorAndDenominator.visit(zttModifier)        
pfTauRunDQMValidation += TauValNumeratorAndDenominator

from Validation.RecoTau.ValidateTausOnQCD_cff import *

genParticlesForJetsQCD= genParticlesForJets.clone()
qcdDenominator = objectTypeSelectedTauValDenominator.clone()
qcdKinemSelection= kinematicSelectedTauValDenominator.clone(src = cms.InputTag("qcdDenominator"))

qcdModifier = ApplyFunctionToSequence(SetFakePars)
TauValNumeratorAndDenominator2.visit(qcdModifier)        
pfTauRunDQMValidation += TauValNumeratorAndDenominator2

produceDenoms = cms.Sequence(
    tauGenJets
    *zttDenominator
    *zttKinemSelection
    +genParticlesForJets
    *qcdDenominator
    *qcdKinemSelection
    )

#-------------------------------------------------------------------------------------------------------
#                     Producing Efficiencies (postValidation)
#-------------------------------------------------------------------------------------------------------


plotPsetSignal = Utils.SetPlotSequence(TauValNumeratorAndDenominator)
plotPsetFake = Utils.SetPlotSequence(TauValNumeratorAndDenominator2)
TauEfficienciesFake = TauEfficiencies.clone(plots = plotPsetFake)
TauEfficiencies.plots = plotPsetSignal

runTauEff = cms.Sequence(TauEfficienciesFake + TauEfficiencies)

#--------------------------------------------------------------------------
#         Making histograms look nicer (not working yet)
#--------------------------------------------------------------------------

def SetSignalPlotSet(module):
    module.PrintToFile = False
    del module.drawJobs.TauIdEffStepByStep
    for subsetName in module.drawJobs.parameterNames_():
        subset = getattr(module.drawJobs,subsetName)
        if hasattr(subset,'plots'):
            mEs = []
            for monitorEl in subset.plots.dqmMonitorElements:
                correcectME = monitorEl[13:]
                lastUnderscore = correcectME.rfind('_',0,correcectME.rfind('/'))
                correcectME = correcectME[:lastUnderscore]+'_Signal'+correcectME[lastUnderscore:]
                mEs.append(correcectME)
            subset.plots.dqmMonitorElements = cms.vstring(mEs)


def SetFakePlotSet(module):
    module.PrintToFile = False
    del module.drawJobs.TauIdEffStepByStep
    for subsetName in module.drawJobs.parameterNames_():
        subset = getattr(module.drawJobs,subsetName)
        if hasattr(subset,'plots'):
#            subset.drawOptionSet = 'fakeRate'
            subset.yAxis = 'fakeRate'
#            subset.legend = 'fakeRate'
            mEs = []
            for monitorEl in subset.plots.dqmMonitorElements:
                correcectME = monitorEl[13:]
                lastUnderscore = correcectME.rfind('_',0,correcectME.rfind('/'))
                correcectME = correcectME[:lastUnderscore]+'_Fakes'+correcectME[lastUnderscore:]
                mEs.append(correcectME)
            subset.plots.dqmMonitorElements = cms.vstring(mEs)

zttModifier = ApplyFunctionToSequence(SetSignalPlotSet)
plotTauValidation.visit(zttModifier)

qcdModifier = ApplyFunctionToSequence(SetFakePlotSet)
plotTauValidation2.visit(qcdModifier)        

makeBetterPlots = cms.Sequence(plotTauValidation+plotTauValidation2)
