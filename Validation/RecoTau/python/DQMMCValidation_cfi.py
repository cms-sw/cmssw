import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.ValidateTausOnZTT_cff import *
from Validation.RecoTau.ValidateTausOnQCD_cff import *
from Validation.RecoTau.ValidateTausOnZEE_cff import *
from Validation.RecoTau.ValidateTausOnZMM_cff import *

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


produceDenoms = cms.Sequence()
produceDenoms += produceDenominatorZTT
produceDenoms += produceDenominatorQCD
produceDenoms += produceDenominatorZMM
produceDenoms += produceDenominatorZEE

pfTauRunDQMValidation = cms.Sequence()
pfTauRunDQMValidation += runTauValidationBatchModeZTT
pfTauRunDQMValidation += runTauValidationBatchModeQCD
pfTauRunDQMValidation += runTauValidationBatchModeZMM
pfTauRunDQMValidation += runTauValidationBatchModeZEE

#-------------------------------------------------------------------------------------------------------
#                     Producing Efficiencies (postValidation)
#-------------------------------------------------------------------------------------------------------

runTauEff = cms.Sequence()
runTauEff += TauEfficienciesZTT
runTauEff += TauEfficienciesQCD
runTauEff += TauEfficienciesZMM
runTauEff += TauEfficienciesZEE

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
