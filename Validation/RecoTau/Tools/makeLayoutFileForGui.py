#!/usr/bin/env python3
######################################################################################################################################################
#                                SHIFT LAYOUTS
######################################################################################################################################################
#Is there a better way? Can we make it in the DQM, probably, but the documentation is not complete and when you try to make a loop or more then a function the GUI gets crazy. So...
from string import Template

stripDiscName = lambda x: (x.split('/')[-1]).split('Eff')[0]

defaultPlots = [
    ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEff%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEff%s'],
    ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEff%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEff%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEff%s'],
    ]

muonrejplots = ['RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEff%s', 'RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEff%s', 'RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEff%s']

elerejplots = [
    ['RecoTauV/standardValidation/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEff%s',
     'RecoTauV/standardValidation/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEff%s',],
    ['RecoTauV/standardValidation/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEff%s',
     'RecoTauV/standardValidation/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEff%s']
     ]

variables = {'a':'pt','b':'pileup','c':'eta','d':'phi'}

datasetNames = {'RealData' : 'QCD Jets', 'RealMuonsData' : 'muons from Z', 'RealElectronsData' : 'electrons from Z'}

outputFileString = 'def shiftpftaulayout(i, p, *rows): i["00 Shift/Tau/" + p] = DQMItem(layout=rows)\n\n%s'
stdFill=Template('shiftpftaulayout(\n\tdqmitems,\n\t"$locpath",\n\t$rows\n\t)')

def defaults(locpath,dataType,var):
    rows = [[{ 'path': plot % (dataType,var), 'description': '%s fake rate from %s' % (stripDiscName(plot), datasetNames[dataType]),'draw': {'drawopts': "e"}} for plot in row] for row in defaultPlots]
    strow = ',\n\t'.join([row.__repr__() for row in rows])
    return stdFill.substitute(locpath=locpath,rows=strow)

def muRej(locpath,var):
    rows = [{ 'path': plot % var, 'description': '%s fake rate' % (stripDiscName(plot)),'draw': {'drawopts': "e"}} for plot in muonrejplots]
    return stdFill.substitute(locpath=locpath,rows=rows.__repr__())

def eleRej(locpath,var):
    rows = [[{ 'path': plot % var, 'description': '%s fake rate' % (stripDiscName(plot)),'draw': {'drawopts': "e"}} for plot in row] for row in elerejplots]
    strow = ',\n\t'.join([row.__repr__() for row in rows])
    return stdFill.substitute(locpath=locpath,rows=strow)

#
#SingleMu
#
toAdd = [ defaults('SingleMu/00%s - Fake rate from muons vs %s' % item,'RealMuonsData',item[1]) for item in variables.items() ]
toAdd.extend( [ muRej('SingleMu/01%s - Muon rejection fake rate vs %s' % item,item[1]) for item in variables.items() ] )
toAdd.extend( [ defaults('SingleMu/02%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )

#
#Jet
#
toAdd.extend([ defaults('Jet/00%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )

#
#DoubleE / TauPlusX
#
toAdd.extend([ defaults('DoubleElectron_OR_TauPlusX/00%s - Fake rate from electrons vs %s' % item,'RealElectronsData',item[1]) for item in variables.items() ] )
toAdd.extend( [ eleRej('DoubleElectron_OR_TauPlusX/01%s - Electron rejection fake rate vs %s' % item,item[1]) for item in variables.items() ] )

layout = open('shift_pftau_T0_layout.py','w')
layout.write(outputFileString % '\n'.join(toAdd) )
layout.close()

######################################################################################################################################################
#                                 LAYOUTS
######################################################################################################################################################
layoutString = 'def pftaulayout(i, p, *rows): i["RecoTauV/standardValidation/Layouts/" + p] = DQMItem(layout=rows)\n\n%s'
stdFill=Template('pftaulayout(\n\tdqmitems,\n\t"$locpath",\n\t$rows\n\t)')

layoutDefaultPlots = [
    [
        ['RecoTauV/standardValidation/hpsPFTauProducer%s_Matched/PFJetMatchingEff%s',
        'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff%s'],
        ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEff%s',
         'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEff%s'],
         ],
    [
         ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEff%s',
         'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEff%s'],
         ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEff%s',
         'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEff%s'],
         ]
    ]

layoutscone = [
    ['RecoTauV/standardValidation/shrinkingConePFTauProducerLeadingPion%s_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationUsingLeadingPionEff%s',
     'RecoTauV/standardValidation/shrinkingConePFTauProducerLeadingPion%s_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff%s'],
    ['RecoTauV/standardValidation/shrinkingConePFTauProducerLeadingPion%s_shrinkingConePFTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff%s',
     'RecoTauV/standardValidation/shrinkingConePFTauProducerLeadingPion%s_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationUsingLeadingPionEff%s']
    ]

layoutSizeSumpt = [
    ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_%s'],
    ['RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_%s',
    'RecoTauV/standardValidation/hpsPFTauProducer%s_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_%s'],
    ]

import string

def layDefaults(locpath,dataType,var):
    framerows = [[[{ 'path': plot % (dataType,var), 'description': '%s fake rate from %s' % (stripDiscName(plot), datasetNames[dataType]),'draw': {'drawopts': "e"}} for plot in row] for row in frame] for frame in layoutDefaultPlots]
    strows = [',\n\t'.join([row.__repr__() for row in rows]) for rows in framerows]
    ret = [stdFill.substitute(locpath=locpath % postfix, rows=strow) for postfix,strow in zip(string.letters,strows)]
    return '\n'.join(ret)

def sCone(locpath,dataType,var):
    rows = [[{ 'path': plot % (dataType,var), 'description': '%s fake rate from %s' % (stripDiscName(plot), datasetNames[dataType]),'draw': {'drawopts': "e"}} for plot in row] for row in layoutscone]
    strow = ',\n\t'.join([row.__repr__() for row in rows])
    return stdFill.substitute(locpath=locpath,rows=strow)

def sizeSumpt(locpath,dataType,var):
    typ = lambda x: x.split('_')[-2]
    if typ == 'SumPt':
        typ += ' distribution'
    rows = [[{ 'path': plot % (dataType,var), 'description': '%s faking taus %s of %s' % (datasetNames[dataType],typ(plot), var),'draw': {'drawopts': "e"}} for plot in row] for row in layoutSizeSumpt]
    strow = ',\n\t'.join([row.__repr__() for row in rows])
    return stdFill.substitute(locpath=locpath,rows=strow)

variables = {'a%s':'pt','b%s':'pileup','c%s':'eta','d%s':'phi'}
sumVar = dict( list(zip(string.letters,[elem for elem in ['signalCands','isolationChargedHadrCands','isolationGammaCands','isolationNeutrHadrCands'] ])) )


#
#SingleMu
#
toAdd = [ layDefaults('SingleMu/00%s - Fake rate from muons vs %s' % item,'RealMuonsData',item[1]) for item in variables.items() ]
toAdd.extend( [ muRej('SingleMu/01%s - Muon rejection fake rate vs %s' % (item[0]%'',item[1]),item[1]) for item in variables.items() ] )
toAdd.extend( [ layDefaults('SingleMu/02%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )
toAdd.extend( [ sizeSumpt('SingleMu/03%s - Distributions of size and sumPt for %s, %s faking taus' % (item[0], item[1],datasetNames['RealMuonsData']),'RealMuonsData',item[1]) for item in sumVar.items() ] )
toAdd.extend( [ sizeSumpt('SingleMu/04%s - Distributions of size and sumPt for %s, %s faking taus' % (item[0], item[1],datasetNames['RealData']),'RealData',item[1]) for item in sumVar.items() ] )

#
#Jet
#
toAdd.extend([ layDefaults('Jet/00%s - Fake rate from jets vs %s' % item,'RealData',item[1]) for item in variables.items() ] )
toAdd.extend( [ sizeSumpt('Jet/01%s - Distributions of size and sumPt for %s, %s faking taus' % (item[0], item[1],datasetNames['RealData']),'RealData',item[1]) for item in sumVar.items() ] )

#
#DoubleE / TauPlusX
#
toAdd.extend([ layDefaults('DoubleElectron_OR_TauPlusX/00%s - Fake rate from electrons vs %s' % item,'RealElectronsData',item[1]) for item in variables.items() ] )
toAdd.extend( [ eleRej('DoubleElectron_OR_TauPlusX/01%s - Electron rejection fake rate vs %s' % (item[0].replace('%s',''),item[1]),item[1]) for item in variables.items() ] )
toAdd.extend( [ sizeSumpt('DoubleElectron_OR_TauPlusX/04%s - Distributions of size and sumPt for %s, %s faking taus' % (item[0], item[1],datasetNames['RealElectronsData']),'RealElectronsData',item[1]) for item in sumVar.items() ] )

layout = open('pftatau_T0_layouts.py','w')
layout.write(layoutString % '\n'.join(toAdd) )
layout.close()
