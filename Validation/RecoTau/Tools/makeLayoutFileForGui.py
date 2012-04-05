#!/usr/bin/env python
#Is there a better way? Can we make it in the DQM, probably, but the documentation is not complete and when you try to make a loop or more then a function the GUI gets crazy. So...
from string import Template

stripDiscName = lambda x: (x.split('/')[-1]).split('Eff')[0]
defaultPlots = [
    ['RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff%s',
    'RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEff%s',
    'RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEff%s'],
    ['RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEff%s',
    'RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEff%s',
    'RecoTauV/hpsPFTauProducer%s_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEff%s'],
    ]

muonrejplots = ['RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEff%s', 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEff%s', 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEff%s']

elerejplots = [
    ['RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEff%s',
     'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEff%s',],
    ['RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEff%s',
     'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEff%s']
     ]

variables = {'a':'pt','b':'pileup','c':'eta','d':'phi'}

datasetNames = {'RealData' : 'QCD Jets', 'RealMuonsData' : 'muons from Z', 'RealElectronsData' : 'electrons from Z'}

outputFileString = 'def shiftpftaulayout(i, p, *rows): i["00 Shift/Tau/" + p] = DQMItem(layout=rows)\n\n%s'
stdFill=Template('shiftpftaulayout(\n\tdqmitems,\n\t"$locpath",\n\t$rows\n\t)')

def defaults(locpath,dataType,var):
    rows = [[{ 'path': plot % (dataType,var), 'description': '%s fake rate from %s' % (stripDiscName(plot), datasetNames[dataType])} for plot in row] for row in defaultPlots]
    strow = ',\n\t'.join([row.__repr__() for row in rows])
    return stdFill.substitute(locpath=locpath,rows=strow)

def muRej(locpath,var):
    rows = [{ 'path': plot % var, 'description': '%s fake rate' % (stripDiscName(plot))} for plot in muonrejplots]
    return stdFill.substitute(locpath=locpath,rows=rows.__repr__())

def eleRej(locpath,var):
    rows = [[{ 'path': plot % var, 'description': '%s fake rate' % (stripDiscName(plot))} for plot in row] for row in elerejplots]
    return stdFill.substitute(locpath=locpath,rows=rows.__repr__())

#
#SingleMu
#
toAdd = [ defaults('SingleMu/00%s - Fake rate from muons vs %s' % item,'RealMuonsData',item[1]) for item in variables.items() ]
toAdd.extend( [ muRej('SingleMu/01%s - Muon rejection fake rate vs %s' % item,item[1]) for item in variables.items() ] )
toAdd.extend( [ defaults('SingleMu/02%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )

#
#Jet
#
toAdd.extend( [ defaults('Jet/00%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )

#
#DoubleE / TauPlusX
#
toAdd.extend( [ defaults('DoubleElectron_OR_TauPlusX/00%s - Fake rate from electrons vs %s' % item,'RealElectronsData',item[1]) for item in variables.items() ] )
toAdd.extend( [ eleRej('DoubleElectron_OR_TauPlusX/01%s - Electron rejection fake rate vs %s' % item,item[1]) for item in variables.items() ] )

layout = open('shift_pftau_T0_layout.py','w')
layout.write(outputFileString % '\n'.join(toAdd) )
