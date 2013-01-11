
print 'Running for XXX_OUTPUT_XXX'

Dir = "/uscms_data/d2/farrell3/WorkArea/14Aug2012/CMSSW_5_3_3/src/PhysicsTools/TagAndProbe/test/"

TkCut = False
TkTOFCut = False
MuOnlyCut = False
NdofCut = False
SACut = False
StationCut = False
DxyCut = False
SegSepCut = False
DzCut = False

Eta = False
Pt = False
PV = False

OutputDir = "Tk"

if("XXX_OUTPUT_XXX".find("TkCut")!=-1):
    TkCut = True
    OutputDir = "Tk"
if("XXX_OUTPUT_XXX".find("TkTOFCut")!=-1):
    TkTOFCut = True
    OutputDir = "TkTOF"
if("XXX_OUTPUT_XXX".find("MuOnlyCut")!=-1):
    MuOnlyCut = True
    OutputDir = "MuOnly"
if("XXX_OUTPUT_XXX".find("NdofCut")!=-1):
    NdofCut = True
    OutputDir = "Ndof"
if("XXX_OUTPUT_XXX".find("SACut")!=-1):
    SACut = True
    OutputDir = "SA"
if("XXX_OUTPUT_XXX".find("StationCut")!=-1):
    StationCut = True
    OutputDir = "Station"
if("XXX_OUTPUT_XXX".find("DxyCut")!=-1):
    DxyCut = True
    OutputDir = "Dxy"
if("XXX_OUTPUT_XXX".find("SegSepCut")!=-1):
    SegSepCut = True
    OutputDir = "SegSep"
if("XXX_OUTPUT_XXX".find("DzCut")!=-1):
    DzCut = True
    OutputDir = "Dz"


if("XXX_OUTPUT_XXX".find("Eta")!=-1):
    Eta = True
if("XXX_OUTPUT_XXX".find("Pt")!=-1):
    Pt = True
if("XXX_OUTPUT_XXX".find("PV")!=-1):
    PV = True

if("XXX_OUTPUT_XXX".find("MC")!=-1):
    OutputDir = OutputDir + "MC"    

InputDir = "SAMuonID"
if("XXX_OUTPUT_XXX".find("SAProbe")!=-1):
    InputDir = "TkMuonID"
    OutputDir = OutputDir + "_SAProbe"

import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )    

process.Analyzer = cms.EDAnalyzer("TagProbeFitTreeAnalyzer",
    # IO parameters:
    InputFileNames = cms.vstring(XXX_INPUT_XXX),
    #InputFileNames = cms.vstring('erase_Skim.root'),
    InputDirectoryName = cms.string(InputDir),
    InputTreeName = cms.string("fitter_tree"),
    #OutputFileName = cms.string("temp.root"),
    OutputFileName = cms.string(Dir + "TagProbeAnalyzerRoot/HSCPTagProbe_XXX_OUTPUT_XXX.root"),
    OutputDirectoryName = cms.string(OutputDir),
    #number of CPUs to use for fitting
    NumCPU = cms.uint32(1),
    # specifies wether to save the RooWorkspace containing the data for each bin and
    # the pdf object with the initial and final state snapshots
    SaveWorkspace = cms.bool(True),

    WeightVariable = cms.string("Weight"),

    # defines all the real variables of the probes available in the input tree and intended for use in the efficiencies
    Variables = cms.PSet(
        mass = cms.vstring("Tag-Probe Mass", "70", "110", "GeV/c^{2}"),
        pt = cms.vstring("Probe p_{T}", "20", "200", "GeV/c"),
        eta = cms.vstring("Probe #eta", "-2.1", "2.1", ""),
        PV = cms.vstring("Primary Vertices", "0", "50", ""),
        Weight = cms.vstring("Weight", "0", "50", ""),
        #SAPt = cms.vstring("Probe p_{T}", "20", "200", "GeV/c"),
    ),

    # defines all the discrete variables of the probes available in the input tree and intended for use in the efficiency calculations
    Categories = cms.PSet(
        PassTk = cms.vstring("PassTk", "dummy[true=1,false=0]"),
        PassTkTOF = cms.vstring("PassTkTOF", "dummy[true=1,false=0]"),
        PassMuOnly = cms.vstring("PassMuOnly", "dummy[true=1,false=0]"),
        PassTkSA = cms.vstring("PassTkSA", "dummy[true=1,false=0]"),
        PassTkTOFSA = cms.vstring("PassTkTOFSA", "dummy[true=1,false=0]"),
        PassTOFNdof = cms.vstring("PassTOFNdof", "dummy[true=1,false=0]"),
        PassSA = cms.vstring("PassSA", "dummy[true=1,false=0]"),
        PassStation = cms.vstring("PassStation", "dummy[true=1,false=0]"),
        PassDxy = cms.vstring("PassDxy", "dummy[true=1,false=0]"),
        PassSegSep = cms.vstring("PassSegSep", "dummy[true=1,false=0]"),
        PassDz = cms.vstring("PassDz", "dummy[true=1,false=0]"),
    ),

    # defines all the PDFs that will be available for the efficiency calculations; uses RooFit's "factory" syntax;
    # each pdf needs to define "signal", "backgroundPass", "backgroundFail" pdfs, "efficiency[0.9,0,1]" and "signalFractionInPassing[0.9]" are used for initial values  
    PDFs = cms.PSet(
        vpvPlusExpo = cms.vstring(
            "Voigtian::signal1(mass, mean1[90,80,100], width[2.495], sigma1[2,1,3])",
            "Voigtian::signal2(mass, mean2[90,80,100], width,        sigma2[4,2,10])",
            "SUM::signal(vFrac[0.8,0,1]*signal1, signal2)",
            "Exponential::backgroundPass(mass, lp[-0.1,-1,0.1])",
            "Exponential::backgroundFail(mass, lf[-0.1,-1,0.1])",
            "efficiency[0.9,0,1]",
            "signalFractionInPassing[0.9]"
        ),
    ),

    # defines a set of efficiency calculations, what PDF to use for fitting and how to bin the data;
    # there will be a separate output directory for each calculation that includes a simultaneous fit, side band subtraction and counting. 
    Efficiencies = cms.PSet()
)

if Eta:
    process.Analyzer.Efficiencies = cms.PSet(
        #the name of the parameter set becomes the name of the directory
          eta = cms.PSet(
            #specifies the efficiency of which category and state to measure
            EfficiencyCategoryAndState = cms.vstring("PassTk","true"),
            #specifies what unbinned variables to include in the dataset, the mass is needed for the fit
            UnbinnedVariables = cms.vstring("mass", "Weight"),
            #specifies the binning of parameters
            BinnedVariables = cms.PSet(
                eta = cms.vdouble(-2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2,1.5, 1.8, 2.1)
            ),
            #first string is the default followed by binRegExp - PDFname pairs
            BinToPDFmap = cms.vstring("vpvPlusExpo")
          )
    )

if Pt:
  process.Analyzer.Efficiencies = cms.PSet(
        #the name of the parameter set becomes the name of the directory
      pt = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("PassTk","true"),
	        UnbinnedVariables = cms.vstring("mass", "Weight"),
            BinnedVariables = cms.PSet(
                pt = cms.vdouble(40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 56.0, 60.0, 70.0, 80.0, 90.0, 100.0, 140.0, 240.0),
		    ),
            BinToPDFmap = cms.vstring("vpvPlusExpo")
        )
    )
if PV:
    process.Analyzer.Efficiencies = cms.PSet(
        #the name of the parameter set becomes the name of the directory
        PV = cms.PSet(
            EfficiencyCategoryAndState = cms.vstring("PassTk","true"),
            UnbinnedVariables = cms.vstring("mass", "Weight"),
            BinnedVariables = cms.PSet(
                PV = cms.vdouble(5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,31.5,32.5,33.5,34.5\
,35.5),
            ),
            BinToPDFmap = cms.vstring("vpvPlusExpo")
        )
     )


if TkTOFCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassTkTOF","true")
if TkTOFCut and Pt:
  process.Analyzer.Efficiencies.pt.EfficiencyCategoryAndState = cms.vstring("PassTkTOF","true")
if TkTOFCut and PV:
   process.Analyzer.Efficiencies.PV.EfficiencyCategoryAndState = cms.vstring("PassTkTOF","true")

if MuOnlyCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassMuOnly","true")
if MuOnlyCut and Pt:
  process.Analyzer.Efficiencies.pt.EfficiencyCategoryAndState = cms.vstring("PassMuOnly","true")
if MuOnlyCut and PV:
   process.Analyzer.Efficiencies.PV.EfficiencyCategoryAndState = cms.vstring("PassMuOnly","true")

if NdofCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassTOFNdof","true")
if NdofCut and Pt:
  process.Analyzer.Efficiencies.pt.EfficiencyCategoryAndState = cms.vstring("PassTOFNdof","true")
if NdofCut and PV:
   process.Analyzer.Efficiencies.PV.EfficiencyCategoryAndState = cms.vstring("PassTOFNdof","true")

if SACut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassSA","true")

if StationCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassStation","true")

if DxyCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassDxy","true")

if SegSepCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassSegSep","true")

if DzCut and Eta:
   process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassDz","true")

if("XXX_OUTPUT_XXX".find("SAProbe")!=-1):
   if TkCut and Eta:
      process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassTkSA","true")
   if TkCut and Pt:
       process.Analyzer.Efficiencies.pt.EfficiencyCategoryAndState = cms.vstring("PassTkSA","true")
   if TkCut and PV:
       process.Analyzer.Efficiencies.PV.EfficiencyCategoryAndState = cms.vstring("PassTkSA","true")

   if TkTOFCut and Eta:
      process.Analyzer.Efficiencies.eta.EfficiencyCategoryAndState = cms.vstring("PassTkTOFSA","true")
   if TkTOFCut and Pt:
       process.Analyzer.Efficiencies.pt.EfficiencyCategoryAndState = cms.vstring("PassTkTOFSA","true")
   if TkTOFCut and PV:
       process.Analyzer.Efficiencies.PV.EfficiencyCategoryAndState = cms.vstring("PassTkTOFSA","true")

process.fitness = cms.Path(
    process.Analyzer
)

