import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.ValidationUtils as Utils
import copy
import re
import os

from Validation.RecoTau.ValidationOptions_cfi import *


"""

   RecoTauValidation_cfi.py

   Contains the standard tau validation parameters.  It is organized into
   the following sections.

   DENOMINATOR 
     
     Set common kinematic cuts (pt > 5 and eta < 2.5) on the denominator source.
     Note that the denominator depends on the type of test (signal/background/e etc)

     The denominator kinematic cutter requires that 

   HISTOGRAMS

     Produce numerator and denominator histgorams used to produce
     tau efficiency plots

        Provides sequence: 
          TauValNumeratorAndDenominator 
        Requires:
          tauValSelectedDenominator (filtered GenJet collection)
        
   EFFICIENCY
   
     Using numerator and denominators, calculate and store
     the efficiency curves

        Provides sequence:
          TauEfficiencies
        Requires:
          TauValNumeratorAndDenominator

   PLOTTING

     Plot curves calculated in efficiency, in both an overlay mode
     showing overall performance for a release, and the indvidual 
     discriminator efficiency compared to a given release

        Provides sequence:
          loadTau
          plotTauValidation
          loadAndPlotTauValidation

        Requires:
          TauEfficiencies, external root file to compare to

     Plotting must be executed in a separate cmsRun job!

   UTILITIES
     
     Various scripts to automate things...


"""

"""

DENOMINATOR

"""

# require generator level hadrons produced in tau-decay to have transverse momentum above threshold
kinematicSelectedTauValDenominator = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("objectTypeSelectedTauValDenominator"),
     cut = cms.string('pt > 5. && abs(eta) < 2.5'),
     filter = cms.bool(False)
)

denominator = cms.InputTag("kinematicSelectedTauValDenominator")

"""

HISTOGRAMS

        Plot the pt/eta/energy/phi spectrum of PFTaus that pass 
        a series of PFTauDiscriminator cuts.

        These will be used as the numerator/denominators of the
        efficiency calculations
"""

StandardMatchingParameters = cms.PSet(
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False), #TRUE FOR TEST ONLY
   #RefCollection                = cms.InputTag("TauGenJetProducer","selectedGenTauDecaysToHadronsPt5Cumulative"),
   RefCollection                = cms.InputTag("kinematicSelectedTauValDenominator"),
)

PFTausHighEfficiencyLeadingPionBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string("LeadingPion"),
   TauProducer                  = cms.InputTag('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5),plotStep = cms.bool(True))
 )
)

PFTausHighEfficiencyBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.InputTag('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5),plotStep = cms.bool(True))
 )
)

RunTancValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunTancValidation.ExtensionName = "Tanc"
RunTancValidation.discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5),plotStep = cms.bool(True))
)

RunHPSValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunHPSValidation.ExtensionName = ""
RunHPSValidation.TauProducer   = cms.InputTag('hpsPFTauProducer')
RunHPSValidation.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False))
)

RunHPSTanc_HPSValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunHPSTanc_HPSValidation.ExtensionName = "_HPS"
RunHPSTanc_HPSValidation.TauProducer   = cms.InputTag('hpsTancTaus')
RunHPSTanc_HPSValidation.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByDecayModeSelection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByVLooseIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLooseIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByMediumIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTightIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByMediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLooseMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTightMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False))
)

RunHPSTanc_TANCValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunHPSTanc_TANCValidation.ExtensionName = "_TANC"
RunHPSTanc_TANCValidation.TauProducer   = cms.InputTag('hpsTancTaus')
RunHPSTanc_TANCValidation.discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByDecayModeSelection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTanc"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTancVLoose"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTancLoose"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTancMedium"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTancRaw"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
    cms.PSet( discriminator = cms.string("hpsTancTausDiscriminationByTancTight"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
)

CaloTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.InputTag('caloRecoTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), #not plotted
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5),plotStep = cms.bool(True))
 )
)

TauValNumeratorAndDenominator = cms.Sequence(
#      PFTausBothProngs+ OLD
      RunHPSTanc_HPSValidation*
      RunHPSTanc_TANCValidation*
      CaloTausBothProngs *
      PFTausHighEfficiencyBothProngs*
      PFTausHighEfficiencyLeadingPionBothProngs*
      RunTancValidation*
      RunHPSValidation
   )

"""

EFFICIENCY

        Tau efficiency calculations

        Define the Efficiency curves to produce.  Each
        efficiency producer takes the numberator and denominator
        histograms and the dependent variables.
"""

plotPset = Utils.SetPlotSequence(TauValNumeratorAndDenominator)
TauEfficiencies = cms.EDAnalyzer("DQMHistEffProducer",
    plots = plotPset    
)

 
"""

PLOTTING

        loadTau:  load two separate TauVal root files into the DQM
                  so the plotter can access them

"""

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    #inputFileNames = cms.vstring('/afs/cern.ch/user/f/friis/scratch0/MyValidationArea/310pre6NewTags/src/Validation/RecoTau/test/CMSSW_3_1_0_pre6_ZTT_0505Fixes.root'),
    inputFileNames = cms.vstring('/opt/sbg/cms/ui4_data1/dbodin/CMSSW_3_5_1/src/TauID/QCD_recoFiles/TauVal_CMSSW_3_6_0_QCD.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/opt/sbg/cms/ui4_data1/dbodin/CMSSW_3_5_1/src/TauID/QCD_recoFiles/TauVal_CMSSW_3_6_0_QCD.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

# Lots of junk to define the plot style

# standard drawing stuff
xAxisStuff = cms.PSet(
   xAxisTitle = cms.string('P_{T} / GeV'),
   xAxisTitleOffset = cms.double(0.9),
   xAxisTitleSize = cms.double(0.05)
)
xModifiers = [['pt',['xAxisTitle'],['P_{T} / GeV']],['eta',['xAxisTitle'],['#eta']],['phi',['xAxisTitle'],['#phi']],['energy',['xAxisTitle'],['E / GeV']]]

yAxisStuff =cms.PSet(
   yScale = cms.string('linear'), # linear/log
   minY_linear = cms.double(0.),
   maxY_linear = cms.double(1.6),
   minY_log = cms.double(0.001),
   maxY_log = cms.double(1.8),
   yAxisTitle = cms.string('#varepsilon'), 
   yAxisTitleOffset = cms.double(1.1),
   yAxisTitleSize = cms.double(0.05)
)
yModifiers = [['efficiency',['yScale'],['linear']],['fakeRate',['yScale'],['log']]]

legStuff = cms.PSet(
   posX = cms.double(0.50),
   posY = cms.double(0.72),
   sizeX = cms.double(0.39),
   sizeY = cms.double(0.17),
   header = cms.string(''),
   option = cms.string('brNDC'),
   borderSize = cms.int32(0),
   fillColor = cms.int32(0)
)
legModifiers = [['efficiency',['posY','sizeY'],[0.72,0.17]],['efficiency_overlay',['posY','sizeY'],[0.66,0.23]]]

drawOptStuff = cms.PSet(
   markerColor = cms.int32(1),
   markerSize = cms.double(1.),
   markerStyle = cms.int32(20),
   lineColor = cms.int32(1),
   lineStyle = cms.int32(1),
   lineWidth = cms.int32(2),
   drawOption = cms.string('ex0'),
   drawOptionLegend = cms.string('p')
)
drawOptModifiers = [['eff_overlay01',['markerColor','lineColor'],[1,1]],['eff_overlay02',['markerColor','lineColor'],[2,2]],['eff_overlay03',['markerColor','lineColor'],[3,3]],['eff_overlay04',['markerColor','lineColor'],[4,4]],['eff_overlay05',['markerColor','lineColor'],[6,6]],['eff_overlay06',['markerColor','lineColor'],[5,5]],['eff_overlay07',['markerColor','lineColor'],[7,7]],['eff_overlay08',['markerColor','lineColor'],[28,28]],['eff_overlay09',['markerColor','lineColor','markerStyle'],[2,2,29]],['eff_overlay010',['markerColor','lineColor','markerStyle'],[4,4,29]],['eff_overlay011',['markerColor','lineColor','markerStyle'],[6,6,29]]]

standardDrawingStuff = cms.PSet(
  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         
  indOutputFileName = cms.string('#PLOT#.png'),
  xAxes = Utils.SpawnPSet(xModifiers,xAxisStuff),
  yAxes = Utils.SpawnPSet(yModifiers,yAxisStuff),
  legends =  Utils.SpawnPSet(legModifiers,legStuff),
  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV') #vstring not supported by SpawnPSet
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),
  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
        test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
  drawOptionEntries =  Utils.SpawnPSet(drawOptModifiers,drawOptStuff)
)

standardCompareTestAndReference = cms.PSet(
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = cms.string('no test label'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = cms.string('no ref label'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
)
        
##################################################
#
#   The plotting of all the HPS TaNC efficiencies
#
##################################################
plotPFTauEfficiencies_hps = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(RunHPSTanc_HPSValidation, plotPset),
  outputFilePath = cms.string('./HPSTancTaus_HPS/'),
)

plotPFTauEfficiencies_tanc = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(RunHPSTanc_TANCValidation, plotPset),
  outputFilePath = cms.string('./HPSTancTaus_TANC/'),
)

##################################################
#
#   The plotting of HPS Efficiencies
#
##################################################


plotHPSEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(RunHPSValidation, plotPset),
  outputFilePath = cms.string('./hpsPFTauProducer/')
)      

##################################################
#
#   The plotting of all the PFTauHighEfficiencies ID efficiencies
#
##################################################
plotPFTauHighEfficiencyEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(PFTausHighEfficiencyBothProngs, plotPset),
  outputFilePath = cms.string('./shrinkingConePFTauProducer/'),
)      

##################################################
#
#   The plotting of all the CaloTau ID efficiencies
#
##################################################

plotCaloTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(CaloTausBothProngs, plotPset),
  outputFilePath = cms.string('./caloRecoTauProducer/'),
)

##################################################
#
#   The plotting of all the TaNC efficiencies
#
##################################################

plotTancValidation = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(RunTancValidation, plotPset),
  outputFilePath = cms.string('./shrinkingConePFTauProducerTanc/'),
)

##################################################
#
#   The plotting of all the Shrinking cone leading pion efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = Utils.SpawnDrawJobs(PFTausHighEfficiencyLeadingPionBothProngs, plotPset),
  outputFilePath = cms.string('./shrinkingConePFTauProducerLeadingPion/'),
)

plotTauValidation = cms.Sequence(
      plotPFTauEfficiencies_hps
      +plotPFTauEfficiencies_tanc
      +plotPFTauHighEfficiencyEfficiencies
      +plotCaloTauEfficiencies
      +plotTancValidation
      +plotPFTauHighEfficiencyEfficienciesLeadingPion
      +plotHPSEfficiencies
      )

loadAndPlotTauValidation = cms.Sequence(
      loadTau
      +plotTauValidation
      )

"""

UTILITIES

"""

class ApplyFunctionToSequence:
   """ Helper class that applies a given function to all modules
       in a sequence """
   def __init__(self,function):
      self.functor = function
   def enter(self, module):
      self.functor(module)
   def leave(self, module):
      pass

def TranslateToLegacyProdNames(input):
   input = re.sub('fixedConePFTauProducer', 'pfRecoTauProducer', input)
   #fixedDiscriminationRegex = re.compile('fixedConePFTauDiscrimination( \w* )')
   fixedDiscriminationRegex = re.compile('fixedConePFTauDiscrimination(\w*)')
   input = fixedDiscriminationRegex.sub(r'pfRecoTauDiscrimination\1', input)
   input = re.sub('shrinkingConePFTauProducer', 'pfRecoTauProducerHighEfficiency', input)
   shrinkingDiscriminationRegex = re.compile('shrinkingConePFTauDiscrimination(\w*)')
   input = shrinkingDiscriminationRegex.sub(r'pfRecoTauDiscrimination\1HighEfficiency', input)
   return input


def ConvertDrawJobToLegacyCompare(input):
   """ Converts a draw job defined to compare 31X named PFTau validtion efficiencies
       to comapre a 31X to a 22X named validation """
   # get the list of drawjobs { name : copyOfPSet }
   if not hasattr(input, "drawJobs"):
      return
   myDrawJobs = input.drawJobs.parameters_()
   for drawJobName, drawJobData in myDrawJobs.iteritems():
      print drawJobData
      if not drawJobData.plots.pythonTypeName() == "cms.PSet":
         continue
      pSetToInsert = cms.PSet(
            standardEfficiencyParameters,
            plots = cms.VPSet(
               # test plot w/ modern names
               cms.PSet(
                  dqmMonitorElements = drawJobData.plots.dqmMonitorElements,
                  process = cms.string('test'),
                  drawOptionEntry = cms.string('eff_overlay01'),
                  legendEntry = cms.string(input.processes.test.legendEntry.value())
                  ),
               # ref plot w/ vintage name
               cms.PSet(
                  # translate the name
                  dqmMonitorElements = cms.vstring(TranslateToLegacyProdNames(drawJobData.plots.dqmMonitorElements.value()[0])),
                  process = cms.string('reference'),
                  drawOptionEntry = cms.string('eff_overlay02'),
                  legendEntry = cms.string(input.processes.reference.legendEntry.value())
                  )
               )
            )
      input.drawJobs.__setattr__(drawJobName, pSetToInsert)

def MakeLabeler(TestLabel, ReferenceLabel):
   def labeler(module):
      if hasattr(module, 'processes'):
         if module.processes.hasParameter(['test', 'legendEntry']) and module.processes.hasParameter([ 'reference', 'legendEntry']):
            module.processes.test.legendEntry = TestLabel
            module.processes.reference.legendEntry = ReferenceLabel
            print "Set test label to %s and reference label to %s for plot producer %s" % (TestLabel, ReferenceLabel, module.label())
         else:
            print "ERROR in RecoTauValidation_cfi::MakeLabeler - trying to set test/reference label but %s does not have processes.(test/reference).legendEntry parameters!" % module.label()
   return labeler

def SetYmodulesToLog(matchingNames = []):
   ''' set all modules whose name contains one of the matching names to log y scale'''
   def yLogger(module):
      ''' set a module to use log scaling in the yAxis'''
      if hasattr(module, 'drawJobs'):
         print "EK DEBUG"
         drawJobParamGetter = lambda subName : getattr(module.drawJobs, subName)
         #for subModule in [getattr(module.drawJobs, subModuleName) for subModuleName in dir(module.drawJobs)]:
         attrNames = dir(module.drawJobs)
         for subModuleName, subModule in zip(attrNames, map(drawJobParamGetter, attrNames)):
            matchedNames = [name for name in matchingNames if subModuleName.find( name) > -1] # matching sub strings
            if len(matchingNames) == 0:
               matchedNames = ['take','everything','and','dont','bother']
            if hasattr(subModule, "yAxis") and len(matchedNames):
               print "Setting drawJob: ", subModuleName, " to log scale."
               subModule.yAxis = cms.string('fakeRate') #'fakeRate' configuration specifies the log scaling
         if len(matchingNames) == 0: 
            module.yAxes.efficiency.maxY_log = 40
            module.yAxes.fakeRate.maxY_log = 40
   return yLogger


def SetBaseDirectory(Directory):
   def BaseDirectorizer(module):
      newPath = Directory
      #if module.hasParameter("outputFilePath"):
      if hasattr(module, "outputFilePath"):
         oldPath = module.outputFilePath.value()
         newPath = os.path.join(newPath, oldPath)
         if not os.path.exists(newPath):
            os.makedirs(newPath)
         print newPath
         module.outputFilePath = cms.string("%s" % newPath)
   return BaseDirectorizer

def RemoveComparisonPlotCommands(module):
   if hasattr(module, 'drawJobs'):
      #get draw job parameter names
      drawJobs = module.drawJobs.parameterNames_()
      for drawJob in drawJobs:
         if drawJob != "TauIdEffStepByStep":
            module.drawJobs.__delattr__(drawJob)
            print "Removing comparison plot", drawJob

def SetPlotDirectory(myPlottingSequence, directory):
   myFunctor = ApplyFunctionToSequence(SetBaseDirectory(directory))
   myPlottingSequence.visit(myFunctor)

def SetTestAndReferenceLabels(myPlottingSequence, TestLabel, ReferenceLabel):
   myFunctor = ApplyFunctionToSequence(MakeLabeler(TestLabel, ReferenceLabel))
   myPlottingSequence.visit(myFunctor)

def SetCompareToLegacyProductNames(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(ConvertDrawJobToLegacyCompare)
   myPlottingSequence.visit(myFunctor)

def SetTestFileToPlot(myProcess, FileLoc):
   myProcess.loadTau.test.inputFileNames = cms.vstring(FileLoc)

def SetReferenceFileToPlot(myProcess, FileLoc):
   if FileLoc == None:
      del myProcess.loadTau.reference
   else:
      myProcess.loadTau.reference.inputFileNames = cms.vstring(FileLoc)

def SetLogScale(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(SetYmodulesToLog())
   myPlottingSequence.visit(myFunctor)

def SetSmartLogScale(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(SetYmodulesToLog(['Electron', 'Muon', 'Isolation', 'TaNC']))
   myPlottingSequence.visit(myFunctor)

def SetPlotOnlyStepByStep(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(RemoveComparisonPlotCommands)
   myPlottingSequence.visit(myFunctor)
