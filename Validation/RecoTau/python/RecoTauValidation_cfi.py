import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.ValidationUtils as Utils
import copy
import re
import os

#from Validation.RecoTau.ValidationOptions_cff import *


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

kinematicSelectedTauValDenominatorCut = cms.string('pt > 5. && abs(eta) < 2.5')
denominator = cms.InputTag("kinematicSelectedTauValDenominator")

"""

HISTOGRAMS

        Plot the pt/eta/energy/phi spectrum of PFTaus that pass 
        a series of PFTauDiscriminator cuts.

        These will be used as the numerator/denominators of the
        efficiency calculations
"""

#Helper process to make future cloning easier
proc = cms.Process('helper')

StandardMatchingParameters = cms.PSet(
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False), #TRUE FOR TEST ONLY
   #RefCollection                = cms.InputTag("TauGenJetProducer","selectedGenTauDecaysToHadronsPt5Cumulative"),
   RefCollection                = denominator,
   TauPtCut                     = cms.double(0.), #almost deprecated, since recoCuts provides more flexibility
   recoCuts                     = cms.string(''), #filter reconstructed candidates. leave this empty to select all. or use sth like: pt > 20 & abs(eta) < 2.3
   genCuts                      = cms.string(''), #filter generated candidates. leave this empty to select all. or use sth like: pt > 20 & abs(eta) < 2.3
   chainCuts                    = cms.bool(False) #Decide whether to chain discriminators or not
)

GenericTriggerSelectionParameters = cms.PSet(
   andOr          = cms.bool( False ),#specifies the logical combination of the single filters' (L1, HLT and DCS) decisions at top level (True=OR)
   dbLabel        = cms.string("PFTauDQMTrigger"),#specifies the label under which the DB payload is available from the ESSource or Global Tag
   andOrHlt       = cms.bool(True),#specifies the logical combination of the single HLT paths' decisions (True=OR)
   hltInputTag    = cms.InputTag("TriggerResults::HLT"),
   #hltDBKey       = cms.string('jetmet_highptjet'),#Tag of the record in the database, where IOV-based HLT paths are found. This record overwrites the configuration parameter hltPaths
   hltPaths       = cms.vstring('HLT_IsoMu*_eta*_LooseIsoPFTau*_v*','HLT_DoubleIsoPFTau*_Trk*_eta*_v*'),#Lists logical expressions of HLT paths, which should have accepted the event (fallback in case DB unaccessible)
   errorReplyHlt  = cms.bool(False),#specifies the desired return value of the HLT filter and the single HLT path filter in case of certain errors
   verbosityLevel = cms.uint32(0) #0: complete silence (default), needed for T0 processing;
)

proc.templateAnalyzer = cms.EDAnalyzer(
   "TauTagValidation",
   StandardMatchingParameters,
   GenericTriggerSelection = GenericTriggerSelectionParameters,
   ExtensionName           = cms.string(""),
   TauProducer             = cms.InputTag(''),
   discriminators          = cms.VPSet(
   )
)

proc.RunHPSValidation = proc.templateAnalyzer.clone()
proc.RunHPSValidation.ExtensionName = ""
#RunHPSValidation.TauPtCut = cms.double(15.)
proc.RunHPSValidation.TauProducer   = cms.InputTag('hpsPFTauProducer')
proc.RunHPSValidation.discriminators = cms.VPSet(
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFindingOldDMs"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseChargedIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseChargedIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightChargedIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolation"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA5VLooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA5LooseElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA5MediumElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA5TightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVA5VTightElectronRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(True)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseMuonRejection2"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumMuonRejection2"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightMuonRejection2"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseMuonRejection3"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightMuonRejection3"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVALooseMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVAMediumMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)),
   cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMVATightMuonRejection"),selectionCut = cms.double(0.5),plotStep = cms.bool(False)), 
)

proc.TauValNumeratorAndDenominator = cms.Sequence(
      proc.RunHPSValidation
   )

"""

EFFICIENCY

        Tau efficiency calculations

        Define the Efficiency curves to produce.  Each
        efficiency producer takes the numberator and denominator
        histograms and the dependent variables.
"""

plotPset = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominator)
proc.efficiencies = cms.EDAnalyzer(
   "TauDQMHistEffProducer",
   plots = plotPset    
   )


################################################
#
#         Normalizes All the histograms
#
################################################

proc.normalizePlots = cms.EDAnalyzer(
   "DQMHistNormalizer",
   plotNamesToNormalize = cms.vstring('*_pTRatio_*','*_Size_*','*_SumPt_*','*_dRTauRefJet*'),
   reference = cms.string('*_pTRatio_allHadronic')
   )

proc.TauEfficiencies = cms.Sequence(
   proc.efficiencies*
   proc.normalizePlots
   )

"""

PLOTTING

        loadTau:  load two separate TauVal root files into the DQM
                  so the plotter can access them

"""

loadTau = cms.EDAnalyzer("TauDQMFileLoader",
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
xModifiers = [['pt',['xAxisTitle'],['P_{T} / GeV']],['eta',['xAxisTitle'],['#eta']],['phi',['xAxisTitle'],['#phi']],['pileup',['xAxisTitle'],['# of Reco Vertices']]]

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
yModifiers = [['efficiency',['yScale','yAxisTitle'],['linear','#varepsilon']],['fakeRate',['yScale','yAxisTitle'],['log','Fake rate']]]

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
#   The plotting of HPS Efficiencies
#
##################################################
## plotHPSEfficiencies = cms.EDAnalyzer("TauDQMHistPlotter",
##                                      standardDrawingStuff,
##                                      standardCompareTestAndReference,
##                                      drawJobs = Utils.SpawnDrawJobs(RunHPSValidation, plotPset),
##                                      outputFilePath = cms.string('./hpsPFTauProducer/'),
##                                      PrintToFile = cms.bool(True)
##                                      )
## #clone for DQM
## plotHPSEfficiencies2 = plotHPSEfficiencies.clone()


##################################################
#
#   The plotting of all the Shrinking cone leading pion efficiencies
#
##################################################
## plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("TauDQMHistPlotter",
##                                                                 standardDrawingStuff,
##                                                                 standardCompareTestAndReference,
##                                                                 drawJobs = Utils.SpawnDrawJobs(PFTausHighEfficiencyLeadingPionBothProngs, plotPset),
##                                                                 outputFilePath = cms.string('./shrinkingConePFTauProducerLeadingPion/'),
##                                                                 PrintToFile = cms.bool(True)
##                                                                 )
## #clone for DQM
## plotPFTauHighEfficiencyEfficienciesLeadingPion2 = plotPFTauHighEfficiencyEfficienciesLeadingPion.clone()


## plotTauValidation = cms.Sequence(
##       plotPFTauHighEfficiencyEfficienciesLeadingPion
##       +plotHPSEfficiencies
##       )

## plotTauValidation2 = cms.Sequence(
##       plotPFTauHighEfficiencyEfficienciesLeadingPion2
##       +plotHPSEfficiencies2
##       )


## loadAndPlotTauValidation = cms.Sequence(
##       loadTau
##       +plotTauValidation
##       )

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

def SetValidationExtention(module, extension):
    module.ExtensionName = module.ExtensionName.value()+extension

def setBinning(module,pset):
    if module._TypedParameterizable__type == 'TauTagValidation':
        module.histoSettings = pset

def setTrigger(module,pset):
   if hasattr(module,'_TypedParameterizable__type') and module._TypedParameterizable__type == 'TauTagValidation':
      setattr(module,'turnOnTrigger',cms.bool(True)) #Turns on trigger (in case is off)
      for item in pset.parameters_().items():
         setattr(module.GenericTriggerSelection,item[0],item[1])
