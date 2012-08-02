import FWCore.ParameterSet.Config as cms
import copy

def CreatePlotEntry(analyzer, discriminatorLabel=None, step=True):
    """CreatePlotEntry(analyzer, discriminatorLabel)\n
    Creates a PSet with the informations used by TauDQMHistEffProducer\n
    where to find the numerator and denominator\n
    where to put the new plot and how to name it\n
    which variables control"""

    producer = analyzer.TauProducer.pythonValue()[1:-1]
    ext = analyzer.ExtensionName.pythonValue()[1:-1]
    if discriminatorLabel == None:
        num = 'RecoTauV/%s%s_Matched/%sMatched_vs_#PAR#TauVisible'%(producer,ext,producer)
        #out = 'RecoTauV/%s%s_Matched/PFJetMatchingEff#PAR#'%(producer,ext)
        if producer.find('caloReco') != -1:
            out = 'RecoTauV/%s%s_Matched/CaloJetMatchingEff#PAR#'%(producer,ext)
        else:
            out = 'RecoTauV/%s%s_Matched/PFJetMatchingEff#PAR#'%(producer,ext)
    else:
        num = 'RecoTauV/%s%s_%s/%s_vs_#PAR#TauVisible'%(producer,ext,discriminatorLabel,discriminatorLabel)
        if discriminatorLabel.find('DiscriminationBy') != -1:
            hname = discriminatorLabel[(discriminatorLabel.find('DiscriminationBy')+len('DiscriminationBy')):]
        else:
            hname = discriminatorLabel[(discriminatorLabel.find('Discrimination')+len('Discrimination')):]
        out = 'RecoTauV/%s%s_%s/%sEff#PAR#'%(producer,ext,discriminatorLabel,hname)

    den = 'RecoTauV/%s%s_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'%(producer,ext)
    ret = cms.PSet(
        numerator = cms.string(num),
        denominator = cms.string(den),
        efficiency = cms.string(out),
        parameter = cms.vstring('pt', 'eta', 'phi', 'pileup'),
        stepByStep = cms.bool(step)
        )
    return ret

def NameVariable(analyzer, discriminatorLabel=None):
    """NameVariable(analyzer, discriminatorLabel)\n
    returns a string with the name of the pset created by CreatePlotEntry"""
    #This part is messy, there is no way to directly link the producer and the name of the variable. There are more exception than rules! IF THE DISCRIMINATOR NAME CHANGES YOU HAVE TO CHANHE IT HERE TOO!
    if analyzer.TauProducer.pythonValue()[1:-1] == 'shrinkingConePFTauProducer':
        if analyzer.ExtensionName.pythonValue()[1:-1] == 'Tanc':
            first='ShrinkingConeTanc'
        elif analyzer.ExtensionName.pythonValue()[1:-1] == 'LeadingPion':
            first='PFTauHighEfficiencyLeadingPion'
        elif analyzer.ExtensionName.pythonValue()[1:-1] == "":
            first='PFTauHighEfficiency'
        else:
            #print 'Case not found check the available cases in Validation/RecoTau/python/ValidationUtils.py -- NameVariable'
            first=analyzer.TauProducer.pythonValue()[1:-1]+analyzer.ExtensionName.pythonValue()[1:-1]
    elif analyzer.TauProducer.pythonValue()[1:-1] == 'hpsPFTauProducer':
        first='HPS'
    elif analyzer.TauProducer.pythonValue()[1:-1] == 'hpsTancTaus':
        first='HPSTanc'+analyzer.ExtensionName.value()
    elif analyzer.TauProducer.pythonValue()[1:-1] == 'caloRecoTauProducer':
        first='CaloTau'
    else:
        #print 'Case not found check the available cases in Validation/RecoTau/python/ValidationUtils.py -- NameVariable'
        first=analyzer.TauProducer.pythonValue()[1:-1]+analyzer.ExtensionName.pythonValue()[1:-1]
    
    if discriminatorLabel == None:
        last = 'Matching'
    else:
        if discriminatorLabel.find('DiscriminationBy') != -1:
            last = discriminatorLabel[(discriminatorLabel.find('DiscriminationBy')+len('DiscriminationBy')):]
            if last.find('TaNCfr') != -1:
                last = last[len('TaNCfr'):]
        else:
            last = discriminatorLabel[(discriminatorLabel.find('DiscriminationAgainst')+len('DiscriminationAgainst')):]+"Rejection"

    return first+"ID"+last+"Efficiencies"

def PlotAnalyzer(pset, analyzer):
    """PlotAnalyzer(pset, analyzer)\n
    fills a PSet that contains all the performance plots for a anlyzer\n
    pset is the PSet to fill/add"""

    setattr(pset,NameVariable(analyzer),CreatePlotEntry(analyzer))

    for currentDiscriminator in analyzer.discriminators:
        label = currentDiscriminator.discriminator.pythonValue()[1:-1]
        step = currentDiscriminator.plotStep.value()
        setattr(pset,NameVariable(analyzer,label),CreatePlotEntry(analyzer,label,step))

class Scanner(object):
    """Class to scan a sequence and give a list of analyzer used and a list of their names"""
    def __init__(self):
        self._analyzerRef = []
    def enter(self,visitee):
        self._analyzerRef.append(visitee)
    def modules(self):
        return self._analyzerRef
    def leave(self, visitee):
        pass

def DisableQCuts(sequence):
   scanner = Scanner()
   sequence.visit(scanner)
   disabled = cms.PSet(
    isolationQualityCuts = cms.PSet(
        minTrackHits = cms.uint32(0),
        minTrackVertexWeight = cms.double(-1),
        minTrackPt = cms.double(0),
        maxTrackChi2 = cms.double(9999),
        minTrackPixelHits = cms.uint32(0),
        minGammaEt = cms.double(0),
        maxDeltaZ = cms.double(0.2),
        maxTransverseImpactParameter = cms.double(9999)
        ),
    pvFindingAlgo = cms.string('highestWeightForLeadTrack'),
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    signalQualityCuts = cms.PSet(
        minTrackHits = cms.uint32(0),
        minTrackVertexWeight = cms.double(-1),
        minTrackPt = cms.double(0),
        maxTrackChi2 = cms.double(9999),
        minTrackPixelHits = cms.uint32(0),
        minGammaEt = cms.double(0),
        maxDeltaZ = cms.double(0.2),
        maxTransverseImpactParameter = cms.double(9999)
        )
    )
   for module in scanner.modules():
      if hasattr(module,'qualityCuts'):
         setattr(module,'qualityCuts',disabled)


def SetPlotSequence(sequence):
    """SetSequence(seqence)\n
    This Function return a PSet of the sequence given to be used by TauDQMHistEffProducer"""
    pset = cms.PSet()
    scanner = Scanner()
    sequence.visit(scanner)
    for analyzer in scanner.modules():#The first one is the sequence itself
        if type(analyzer) is cms.EDAnalyzer:
            PlotAnalyzer(pset, analyzer)
    return pset

def SpawnPSet(lArgument, subPset):
    """SpawnPSet(lArgument, subPset) --> cms.PSet\n
    lArgument is a list containing a list of three strings/values:\n
           1-name to give to the spawned pset\n
           2-variable(s) to be changed\n
           3-value(s) of the variable(s): SAME LENGTH OF 2-!\n
           Supported types: int string float(converted to double)"""
    ret = cms.PSet()
    for spawn in lArgument:
        if len(spawn) != 3:
            print "ERROR! SpawnPSet uses argument of three data\n"
            print self.__doc__
            return None
        if len(spawn[1]) != len(spawn[2]):
            print "ERROR! Lists of arguments to replace must have the same length"
            print self.__doc__
            return None
        spawnArg = copy.deepcopy(subPset)
        for par, val in zip(spawn[1],spawn[2]):
            if type(val) is str :
                setattr(spawnArg,par,cms.string(val))
            elif type(val) is int :
                setattr(spawnArg,par,cms.int32(val))
            elif type(val) is float :
                setattr(spawnArg,par,cms.double(val))
        setattr(ret,spawn[0],spawnArg)
    return ret

def SetpByStep(analyzer, plotPset, useOnly):
    """SetpByStep(analyzer, plotPset) --> PSet\n
     This function produces the parameter set stepBystep for the EDAnalyzer TauDQMHistPlotter starting from the PSet produced for TauDQMHistEffProducer and the analyzer to plot"""
    standardEfficiencyOverlay = cms.PSet(
        parameter = cms.vstring('pt', 'eta', 'phi', 'pileup'),
        title = cms.string('TauId step by step efficiencies'),
        xAxis = cms.string('#PAR#'),
        yAxis = cms.string('efficiency'),
        legend = cms.string('efficiency_overlay'),
        labels = cms.vstring('pt', 'eta')
        )
    ret = cms.PSet(
        standardEfficiencyOverlay,
        plots = cms.VPSet()
        )
    producer = analyzer.TauProducer.pythonValue()[1:-1]
    ext = analyzer.ExtensionName.pythonValue()[1:-1]
    keyword = producer + ext + "_"
    counter = 0
    tancDisc = ['Matching','DecayModeSelection','LeadingPionPtCut','LeadingTrackFinding','LeadingTrackPtCut','Tanc','TancVLoose','TancLoose','TancMedium','TancRaw','TancTight','AgainstElectron','AgainstMuon']
    hpsDisc = ['Matching','DecayModeSelection','LooseIsolation','MediumIsolation','TightIsolation']
    for parName in plotPset.parameterNames_():
        isToBePlotted = getattr(plotPset,parName).stepByStep.value()
        if isToBePlotted:
            effplot = getattr(plotPset,parName).efficiency.pythonValue()[1:-1]
            discriminator = parName[parName.find('ID')+len('ID'):-len('Efficiencies')]
            if useOnly == 'tanc':
                useThis = discriminator in tancDisc
            elif useOnly == 'hps':
                useThis = discriminator in hpsDisc
            else :
                useThis = True
            if (effplot.find(keyword) != -1) and useThis:
                monEl = '#PROCESSDIR#/'+effplot
                counter = counter + 1
                drawOpt = 'eff_overlay0%s'%(counter)
                psetName = effplot[effplot.rfind('/')+1:-8]
                ret.plots.append(cms.PSet(
                    dqmMonitorElements = cms.vstring(monEl),
                    process = cms.string('test'),
                    drawOptionEntry = cms.string(drawOpt),
                    legendEntry = cms.string(psetName)
                    ))
    return ret

def SpawnDrawJobs(analyzer, plotPset, useOnly=None):
    """SpwnDrawJobs(analyzer, plotPset) --> cms.PSet\n
    This function produces the parameter set drawJobs for the EDAnalyzer TauDQMHistPlotter starting from the PSet produced for TauDQMHistEffProducer and the analyzer to plot"""
    standardEfficiencyParameters = cms.PSet(
        parameter = cms.vstring('pt', 'eta', 'phi', 'pileup'),
        xAxis = cms.string('#PAR#'),
        yAxis = cms.string('efficiency'),
        legend = cms.string('efficiency'),
        labels = cms.vstring('pt', 'eta'),
        drawOptionSet = cms.string('efficiency')
        )
    ret = cms.PSet()
    tancDisc = ['Matching','DecayModeSelection','LeadingPionPtCut','LeadingTrackFinding','LeadingTrackPtCut','Tanc','TancVLoose','TancLoose','TancMedium','TancRaw','TancTight','AgainstElectron','AgainstMuon']
    hpsDisc = ['Matching','DecayModeSelection','LooseIsolation','MediumIsolation','TightIsolation']
    producer = analyzer.TauProducer.pythonValue()[1:-1]
    ext = analyzer.ExtensionName.pythonValue()[1:-1]
    keyword = producer + ext + "_"
    for parName in plotPset.parameterNames_():
        effplot = getattr(plotPset,parName).efficiency.pythonValue()[1:-1]
        discriminator = parName[parName.find('ID')+len('ID'):-len('Efficiencies')]
        if useOnly == 'tanc':
            useThis = discriminator in tancDisc
        elif useOnly == 'hps':
            useThis = discriminator in hpsDisc
        else :
            useThis = True
        if (effplot.find(keyword) != -1) and useThis:
            monEl = '#PROCESSDIR#/'+effplot
            psetName = effplot[effplot.rfind('/')+1:-5]
            psetVal = cms.PSet(
                standardEfficiencyParameters,
                plots = cms.PSet(
                    dqmMonitorElements = cms.vstring(monEl),
                    processes = cms.vstring('test', 'reference')
                    )
            )
            setattr(ret,psetName,psetVal)
    setattr(ret,'TauIdEffStepByStep',SetpByStep(analyzer, plotPset,useOnly))
    return ret #control if it's ok
