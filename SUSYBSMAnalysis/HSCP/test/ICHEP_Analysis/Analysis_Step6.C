#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TCutG.h" 
#include "TPaveText.h"
#include "tdrstyle.C"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_Samples.h"
//#include "CL95.h"
#include "roostats_cl95.C"
#include "nSigma.C"

using namespace std;



struct stAllInfo{
   double Mass;
   double MassMean;
   double MassSigma;
   double MassCut;
   double XSec_Th;
   double XSec_Err;
   double XSec_Exp;
   double XSec_ExpUp;
   double XSec_ExpDown;
   double XSec_Exp2Up;
   double XSec_Exp2Down;
   double XSec_Obs;
   double Eff;
   double Eff_SYSTP;
   double Eff_SYSTI;
   double Eff_SYSTM;
   double Eff_SYSTT;
   double Eff_SYSTPU;
   double Significance;
   double Index;
   double WP_Pt;
   double WP_I;
   double WP_TOF;
   float  NData;
   float  NPred;
   float  NPredErr;
   float  NSign;

   stAllInfo(string path=""){
     Mass=-1; XSec_Th=-1; XSec_Err=-1; XSec_Exp=-1; XSec_ExpUp=-1;XSec_ExpDown=-1;XSec_Exp2Up=-1;XSec_Exp2Down=-1; XSec_Obs=-1; Eff=-1; Eff_SYSTP=-1; Eff_SYSTI=-1;  Eff_SYSTM=-1; Eff_SYSTT=-1; Eff_SYSTPU=-1;
      if(path=="")return;
      FILE* pFile = fopen(path.c_str(),"r");
      if(!pFile){printf("Can't open %s\n",path.c_str()); return;}
      fscanf(pFile,"Mass         : %lf\n",&Mass);
      fscanf(pFile,"MassMean     : %lf\n",&MassMean);
      fscanf(pFile,"MassSigma    : %lf\n",&MassSigma);
      fscanf(pFile,"MassCut      : %lf\n",&MassCut);
      fscanf(pFile,"Index        : %lf\n",&Index);
      fscanf(pFile,"WP_Pt        : %lf\n",&WP_Pt);
      fscanf(pFile,"WP_I         : %lf\n",&WP_I);
      fscanf(pFile,"WP_TOF       : %lf\n",&WP_TOF);
      fscanf(pFile,"Eff          : %lf\n",&Eff);
      fscanf(pFile,"Eff_SystP    : %lf\n",&Eff_SYSTP);
      fscanf(pFile,"Eff_SystI    : %lf\n",&Eff_SYSTI);
      fscanf(pFile,"Eff_SystM    : %lf\n",&Eff_SYSTM);
      fscanf(pFile,"Eff_SystT    : %lf\n",&Eff_SYSTT);
      fscanf(pFile,"Eff_SystPU   : %lf\n",&Eff_SYSTPU);
      fscanf(pFile,"Signif       : %lf\n",&Significance);
      fscanf(pFile,"XSec_Th      : %lf\n",&XSec_Th);
      fscanf(pFile,"XSec_Exp     : %lf\n",&XSec_Exp);
      fscanf(pFile,"XSec_ExpUp   : %lf\n",&XSec_ExpUp);
      fscanf(pFile,"XSec_ExpDown : %lf\n",&XSec_ExpDown);
      fscanf(pFile,"XSec_Exp2Up  : %lf\n",&XSec_Exp2Up);
      fscanf(pFile,"XSec_Exp2Down: %lf\n",&XSec_Exp2Down);
      fscanf(pFile,"XSec_Obs     : %lf\n",&XSec_Obs);
      fscanf(pFile,"NData        : %E\n" ,&NData);
      fscanf(pFile,"NPred        : %E\n" ,&NPred);
      fscanf(pFile,"NPredErr     : %E\n" ,&NPredErr);
      fscanf(pFile,"NSign        : %E\n" ,&NSign);
      fclose(pFile);
   }

};


struct stGraph{
   TGraph* Stop;
   TGraph* StopN;
   TGraph* GluinoF0;
   TGraph* GluinoF1;
   TGraph* GluinoF5;
   TGraph* GluinoNF0;
   TGraph* GluinoNF1;
   TGraph* GluinoNF5;
   TGraph* GMStau;
   TGraph* PPStau;
   TGraph* DCRho08HyperK;
   TGraph* DCRho12HyperK;
   TGraph* DCRho16HyperK;
   TGraph* GluinoTh;
   TGraph* StopTh;
   TGraph* GMStauTh;
   TCutG*  GluinoThErr;
   TCutG*  StopThErr;
};

double PlotMinScale = 0.0005;
double PlotMaxScale = 3;

TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string syst, string ModelName, int XSectionType=2, string Mass0="", string Mass1="", string Mass2="", string Mass3="", string Mass4="", string Mass5="", string Mass6="", string Mass7="", string Mass8="", string Mass9="",string Mass10="", string Mass11="", string Mass12="", string Mass13="");


stAllInfo Exclusion(string pattern, string modelName, string signal, double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1, string syst="");
int      JobIdToIndex(string JobId);

void GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange);
double FindIntersection(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty=0, bool debug=false);
int ReadXSection(string InputFile, double* Mass, double* XSec, double* Low, double* High,  double* ErrLow, double* ErrHigh);
TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High, double MinLow=PlotMinScale, double MaxHigh=PlotMaxScale);
void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern);
void DrawModelLimitWithBand(string InputPattern, string inputmodel);
std::vector<string> GetModels(string inputmodel);
string GetModelName(string inputmodel);
void DrawRatioBands(string InputPattern, string inputmodel);

double MinRange = 0;
double MaxRange = 1999;

char Buffer[2048];

int    CurrentSampleIndex;
string InputPath;
string OutputPath;

TH1D* MassSign      = NULL;
TH1D* MassMCTr      = NULL;
TH1D* MassData      = NULL;
TH1D* MassPred      = NULL;
TH1D* MassSignPDF   = NULL;
TH1D* MassPredPDF   = NULL;
double FitParam[10];
TF1* Stau_MMC_Fit   = NULL;
TF1* Stop_MMC_Fit   = NULL;
TF1* MGStop_MMC_Fit = NULL;
TF1* Gluino_MMC_Fit = NULL;
TF1* Stau_SMC_Fit   = NULL;
TF1* Stop_SMC_Fit   = NULL;
TF1* MGStop_SMC_Fit = NULL;
TF1* Gluino_SMC_Fit = NULL;

std::vector<stSignal> signals;
std::vector<double> signalsMeanHSCPPerEvent;
std::vector<double> signalsMeanHSCPPerEvent_SYSTP;
std::vector<double> signalsMeanHSCPPerEvent_SYSTT;
std::vector<double> signalsMeanHSCPPerEvent_SYSTM;
std::vector<double> signalsMeanHSCPPerEvent_SYSTI;

double RescaleFactor;
double RescaleError;
int Mode=0;
void Analysis_Step6(string MODE="COMPILE", string InputPattern="", string modelName="", string signal="", double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1, string syst=""){
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.12);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505,"X");
   gStyle->SetNdivisions(550,"Y");

   if(MODE=="COMPILE")return;

   if(MODE=="ANALYSE"){
      stAllInfo result = Exclusion(InputPattern, modelName, signal, Ratio_0C, Ratio_1C, Ratio_2C, syst);
      return;
   }
   
   string MuPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type2/";
   string TkPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type0/";

   string outpath = string("Results/EXCLUSION/");
   MakeDirectories(outpath);

   std::vector<string> ModelNames;
//    ModelNames.push_back("Hyperk");
   ModelNames.push_back("All");

   for(int i=0;i<ModelNames.size();i++){
      DrawRatioBands(TkPattern,ModelNames[i]);      
      DrawRatioBands(MuPattern,ModelNames[i] );

   }


   std::vector<string> Models;
   Models.push_back("Gluinof1");
   Models.push_back("Gluinof5");
   Models.push_back("GluinoN");
   Models.push_back("Stop");
   Models.push_back("StopN");
   Models.push_back("GMStau");
   Models.push_back("PPStau");
   Models.push_back("DCRho08");
   Models.push_back("DCRho12");
   Models.push_back("DCRho16");

   for(int i=0;i<Models.size();i++){
//      DrawModelLimitWithBand(MuPattern, Models[i]);
//      DrawModelLimitWithBand(TkPattern, Models[i]);
   }


   TCanvas* c1;

   FILE* pFile = fopen((string("Analysis_Step6_Result") + syst + ".txt").c_str(),"w");

   FILE* talkFile = fopen((outpath + "TalkPlots" + syst + ".txt").c_str(),"w");

   fprintf(pFile, "\\documentclass{article}\n");
   fprintf(pFile, "\\begin{document}\n\n");
   fprintf(pFile, "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");

   fprintf(talkFile, "\\documentclass{article}\n");
   fprintf(talkFile, "\\usepackage{rotating}\n");
   fprintf(talkFile, "\\begin{document}\n\n");
   fprintf(talkFile, "\\begin{tiny}\n\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & TOF & Mass Cut (GeV) & N pred & N observed & Eff & Signif \\\\\n");
   fprintf(talkFile, "\\hline\n");

//   TGraph* Tk_Obs_Gluino2C  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino  (2C)     ", 2, "Gluino300_2C" , "Gluino400_2C" , "Gluino500_2C" , "Gluino600_2C" , "Gluino700_2C", "Gluino800_2C", "Gluino900_2C", "Gluino1000_2C" );
//   TGraph* Tk_Obs_GluinoF0  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino  (f=00\\%)", 2, "Gluino300_f0" , "Gluino400_f0" , "Gluino500_f0" , "Gluino600_f0" , "Gluino700_f0", "Gluino800_f0", "Gluino900_f0", "Gluino1000_f0" );
   TGraph* Tk_Obs_GluinoF1  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino (f=10\\%)", 2, "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino700_f1", "Gluino800_f1", "Gluino900_f1", "Gluino1000_f1", "Gluino1100_f1", "Gluino1200_f1" );
   TGraph* Tk_Obs_GluinoZF1 = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino Z2 (f=10\\%)", 2, "Gluino600Z_f1" , "Gluino700Z_f1", "Gluino800Z_f1");
   TGraph* Tk_Obs_GluinoF5  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino (f=50\\%)", 2, "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino700_f5", "Gluino800_f5", "Gluino900_f5", "Gluino1000_f5", "Gluino1100_f5", "Gluino1200_f5" );
//   TGraph* Tk_Obs_GluinoNF0 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=00\\%)", 2, "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino700N_f0", "Gluino800N_f0", "Gluino900N_f0", "Gluino1000N_f0" );
   TGraph* Tk_Obs_GluinoNF1 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=10\\%)", 2, "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino700N_f1", "Gluino800N_f1", "Gluino900N_f1", "Gluino1000N_f1", "Gluino1100N_f1", "Gluino1200N_f1" );
//   TGraph* Tk_Obs_GluinoNF5 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=50\\%)", 2, "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino700N_f5", "Gluino800N_f5" , "Gluino900N_f5" , "Gluino1000N_f5" );
//   TGraph* Tk_Obs_Stop2C    = MakePlot(pFile,talkFile,TkPattern,syst,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop400_2C"   , "Stop500_2C"   , "Stop600_2C" , "Stop700_2C" , "Stop800_2C"                    );
   TGraph* Tk_Obs_Stop      = MakePlot(pFile,talkFile,TkPattern,syst,"Stop"               , 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop400"      , "Stop500"      , "Stop600"    , "Stop700"    , "Stop800"                       );
   //TGraph* Tk_Obs_StopZ     = MakePlot(pFile,talkFile,TkPattern,syst,"Stop Z2"            , 2, "Stop300Z"     , "Stop400Z"     , "Stop500Z");
   TGraph* Tk_Obs_StopN     = MakePlot(pFile,talkFile,TkPattern,syst,"StopN"              , 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop400N"     , "Stop500N"     , "Stop600N"   , "Stop700N"   , "Stop800N"                      );
   TGraph* Tk_Obs_GMStau    = MakePlot(pFile,talkFile,TkPattern,syst,"GMSB Stau"          , 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* Tk_Obs_PPStau    = MakePlot(pFile,talkFile,TkPattern,syst,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247","PPStau308");
   TGraph* Tk_Obs_DCRho08HyperK    = MakePlot(pFile,talkFile,TkPattern,syst,"DiChamp    Rho08  ", 2, "DCRho08HyperK100"    , "DCRho08HyperK121"    , "DCRho08HyperK182"    , "DCRho08HyperK242"    , "DCRho08HyperK302",    "DCRho08HyperK350"    ,    "DCRho08HyperK370"    , "DCRho08HyperK390"    ,  "DCRho08HyperK395"    ,  "DCRho08HyperK400"    ,  "DCRho08HyperK410"    ,    "DCRho08HyperK420"    ,  "DCRho08HyperK500");  
   TGraph* Tk_Obs_DCRho12HyperK    = MakePlot(pFile,talkFile,TkPattern,syst,"DiChamp    Rho12  ", 2,"DCRho12HyperK100"    , "DCRho12HyperK182"    , "DCRho12HyperK302"    , "DCRho12HyperK500" , "DCRho12HyperK530"  , "DCRho12HyperK570"     ,"DCRho12HyperK590"     , "DCRho12HyperK595" , "DCRho12HyperK600"   , "DCRho12HyperK610"    ,"DCRho12HyperK620"    ,"DCRho12HyperK700");
   TGraph* Tk_Obs_DCRho16HyperK    = MakePlot(pFile,talkFile,TkPattern,syst,"DiChamp    Rho16  ", 2,"DCRho16HyperK100"    , "DCRho16HyperK182"    , "DCRho16HyperK302"    , "DCRho16HyperK500"    , "DCRho16HyperK700" , "DCRho16HyperK730" , "DCRho16HyperK770"   , "DCRho16HyperK790"     , "DCRho16HyperK795" , "DCRho16HyperK800"     ,"DCRho16HyperK820"    ,"DCRho16HyperK900");

   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile, "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");




   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");

//   TGraph* Mu_Obs_Gluino2C  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino  (2C)     ", 2, "Gluino300_2C" , "Gluino400_2C" , "Gluino500_2C" , "Gluino600_2C" , "Gluino700_2C", "Gluino800_2C", "Gluino900_2C", "Gluino1000_2C" );
//   TGraph* Mu_Obs_GluinoF0  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino  (f=00\\%)", 2, "Gluino300_f0" , "Gluino400_f0" , "Gluino500_f0" , "Gluino600_f0" , "Gluino700_f0", "Gluino800_f0", "Gluino900_f0", "Gluino1000_f0" );
   TGraph* Mu_Obs_GluinoF1  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino (f=10\\%)", 2, "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino700_f1", "Gluino800_f1", "Gluino900_f1", "Gluino1000_f1", "Gluino1100_f1", "Gluino1200_f1" );
   //TGraph* Mu_Obs_GluinoZF1 = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino Z2 (f=10\\%)", 2, "Gluino600Z_f1" , "Gluino700Z_f1", "Gluino800Z_f1");
   TGraph* Mu_Obs_GluinoF5  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino (f=50\\%)", 2, "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino700_f5", "Gluino800_f5", "Gluino900_f5", "Gluino1000_f5", "Gluino1100_f5", "Gluino1200_f5" );
//   TGraph* Mu_Obs_GluinoNF0 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=00\\%)", 2, "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino700N_f0", "Gluino800N_f0", "Gluino900N_f0", "Gluino1000N_f0" );
   TGraph* Mu_Obs_GluinoNF1 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=10\\%)", 2, "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino700N_f1", "Gluino800N_f1", "Gluino900N_f1", "Gluino1000N_f1", "Gluino1100N_f1", "Gluino1200N_f1" );
//   TGraph* Mu_Obs_GluinoNF5 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=50\\%)", 2, "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino700N_f5", "Gluino800N_f5" , "Gluino900N_f5" , "Gluino1000N_f5" );
//   TGraph* Mu_Obs_Stop2C    = MakePlot(pFile,talkFile,MuPattern,syst,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop400_2C"   , "Stop500_2C"   , "Stop600_2C" , "Stop700_2C" , "Stop800_2C"                    );
   TGraph* Mu_Obs_Stop      = MakePlot(pFile,talkFile,MuPattern,syst,"Stop"               , 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop400"      , "Stop500"      , "Stop600"    , "Stop700"    , "Stop800"                       );
   //TGraph* Mu_Obs_StopZ     = MakePlot(pFile,talkFile,MuPattern,syst,"Stop Z2"            , 2, "Stop300Z"     , "Stop400Z"     , "Stop500Z");
   TGraph* Mu_Obs_StopN     = MakePlot(pFile,talkFile,MuPattern,syst,"StopN"              , 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop400N"     , "Stop500N"     , "Stop600N"   , "Stop700N"   , "Stop800N"                      );
   TGraph* Mu_Obs_GMStau    = MakePlot(pFile,talkFile,MuPattern,syst,"GMSB Stau"          , 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* Mu_Obs_PPStau    = MakePlot(pFile,talkFile,MuPattern,syst,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247", "PPStau308" );
   TGraph* Mu_Obs_DCRho08HyperK    = MakePlot(pFile,talkFile,MuPattern,syst,"DiChamp    Rho08  ", 2, "DCRho08HyperK100"    , "DCRho08HyperK121"    , "DCRho08HyperK182"    , "DCRho08HyperK242"    , "DCRho08HyperK302",    "DCRho08HyperK350"    ,    "DCRho08HyperK370"    , "DCRho08HyperK390"    ,  "DCRho08HyperK395"    ,  "DCRho08HyperK400"    ,  "DCRho08HyperK410"    ,    "DCRho08HyperK420"    ,  "DCRho08HyperK500");
   TGraph* Mu_Obs_DCRho12HyperK    = MakePlot(pFile,talkFile,MuPattern,syst,"DiChamp    Rho12  ", 2, "DCRho12HyperK100"    , "DCRho12HyperK182"    , "DCRho12HyperK302"    , "DCRho12HyperK500" , "DCRho12HyperK530"  , "DCRho12HyperK570"     , "DCRho12HyperK590"     ,"DCRho12HyperK595" , "DCRho12HyperK600"   , "DCRho12HyperK610"    ,"DCRho12HyperK620"    ,"DCRho12HyperK700");

   TGraph* Mu_Obs_DCRho16HyperK    = MakePlot(pFile,talkFile,MuPattern,syst,"DiChamp    Rho16  ", 2, "DCRho16HyperK100"    , "DCRho16HyperK182"    , "DCRho16HyperK302"    , "DCRho16HyperK500"    , "DCRho16HyperK700" , "DCRho16HyperK730" , "DCRho16HyperK770"   , "DCRho16HyperK790"     , "DCRho16HyperK795" , "DCRho16HyperK800"     ,"DCRho16HyperK820"    ,"DCRho16HyperK900");

   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile,"\\end{document}\n\n");

   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile,"\\end{document}\n\n");

   CheckSignalUncertainty(pFile,talkFile,TkPattern);
   CheckSignalUncertainty(pFile,talkFile,MuPattern);


   TGraph* GMStauXSec = MakePlot(NULL,NULL,TkPattern,"","GMSB Stau        ", 0, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* PPStauXSec = MakePlot(NULL,NULL,TkPattern,"","Pair Prod. Stau  ", 0, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247"    , "PPStau308");

   TGraph* DCRho08HyperKXSec = MakePlot(NULL,NULL,TkPattern,"","DiChamp    Rho08  ", 0,  "DCRho08HyperK100"    , "DCRho08HyperK121"    , "DCRho08HyperK182"    , "DCRho08HyperK242"    , "DCRho08HyperK302",    "DCRho08HyperK350"    ,    "DCRho08HyperK370"    , "DCRho08HyperK390"    ,  "DCRho08HyperK395"    ,  "DCRho08HyperK400"    ,  "DCRho08HyperK410"    ,    "DCRho08HyperK420"    ,  "DCRho08HyperK500");  

   TGraph* DCRho12HyperKXSec = MakePlot(NULL,NULL,TkPattern,"","DiChamp    Rho12  ", 0, "DCRho12HyperK100"    , "DCRho12HyperK182"    , "DCRho12HyperK302"    , "DCRho12HyperK500" , "DCRho12HyperK530"  , "DCRho12HyperK570"     ,"DCRho12HyperK590"     , "DCRho12HyperK595" , "DCRho12HyperK600"   , "DCRho12HyperK610"    ,"DCRho12HyperK620"    ,"DCRho12HyperK700");

   TGraph* DCRho16HyperKXSec = MakePlot(NULL,NULL,TkPattern,"","DiChamp    Rho16  ", 0, "DCRho16HyperK100"    , "DCRho16HyperK182"    , "DCRho16HyperK302"    , "DCRho16HyperK500"    , "DCRho16HyperK700" , "DCRho16HyperK730" , "DCRho16HyperK770"   , "DCRho16HyperK790"     , "DCRho16HyperK795" , "DCRho16HyperK800"     , "DCRho16HyperK810"     ,"DCRho16HyperK820"    ,"DCRho16HyperK900");

   double ThGluinoMass [100]; double ThGluinoXSec [100];  double ThGluinoLow  [100]; double ThGluinoHigh [100]; double ThGluinoErrLow  [100];  double ThGluinoErrHigh [100];
   int ThGluinoN = ReadXSection("gluino_XSec.txt", ThGluinoMass,ThGluinoXSec,ThGluinoLow,ThGluinoHigh, ThGluinoErrLow, ThGluinoErrHigh);
   TGraph* GluinoXSec    = new TGraph(ThGluinoN,ThGluinoMass,ThGluinoXSec);
   TGraph* GluinoXSecLow = new TGraph(ThGluinoN,ThGluinoMass,ThGluinoLow);
   GluinoXSec->SetTitle("");
   GluinoXSec->GetYaxis()->SetTitleOffset(1.70);
   TCutG* GluinoXSecErr = GetErrorBand("gluinoErr",ThGluinoN,ThGluinoMass,ThGluinoLow,ThGluinoHigh);

   double ThStopMass [100]; double ThStopXSec [100];  double ThStopLow  [100];  double ThStopHigh [100];  double ThStopErrLow  [100];  double ThStopErrHigh [100];
   int ThStopN = ReadXSection("stop_XSec.txt", ThStopMass,ThStopXSec,ThStopLow,ThStopHigh, ThStopErrLow, ThStopErrHigh);
   TGraph* StopXSec    = new TGraph(ThStopN,ThStopMass,ThStopXSec);
   TGraph* StopXSecLow = new TGraph(ThStopN,ThStopMass,ThStopLow);
   StopXSec->SetTitle("");
   StopXSec->GetYaxis()->SetTitleOffset(1.70);
   TCutG* StopXSecErr = GetErrorBand("StopErr", ThStopN,ThStopMass,ThStopLow,ThStopHigh);


   int ThStauN = 9 ; double ThStauMass [100]; double ThStauXSec [100];  double ThStauLow  [100];  double ThStauHigh [100];
   ThStauMass[0] = 100; ThStauXSec[0] = 1.3398;  ThStauLow[0] = 1.18163;  ThStauHigh[0] = 1.48684;
   ThStauMass[1] = 126; ThStauXSec[1] = 0.274591;  ThStauLow[1] = 0.242982;  ThStauHigh[1] = 0.304386;
   ThStauMass[2] = 156; ThStauXSec[2] = 0.0645953;  ThStauLow[2] = 0.0581651;  ThStauHigh[2] = 0.0709262;
   ThStauMass[3] = 200; ThStauXSec[3] = 0.0118093;  ThStauLow[3] = 0.0109992;  ThStauHigh[3] = 0.012632;
   ThStauMass[4] = 247; ThStauXSec[4] = 0.00342512;  ThStauLow[4] = 0.00324853;  ThStauHigh[4] = 0.00358232;
   ThStauMass[5] = 308; ThStauXSec[5] = 0.00098447;  ThStauLow[5] = 0.00093519;  ThStauHigh[5] = 0.00102099;
   ThStauMass[6] = 370; ThStauXSec[6] = 0.000353388; ThStauLow[6] = 0.000335826;  ThStauHigh[6] = 0.000366819;
   ThStauMass[7] = 432; ThStauXSec[7] = 0.000141817; ThStauLow[7] = 0.000134024;  ThStauHigh[7] = 0.000147665;
   ThStauMass[8] = 494; ThStauXSec[8] = 6.17749e-05; ThStauLow[8] =5.83501e-05 ;  ThStauHigh[8] = 6.45963e-05;
   TCutG* StauXSecErr = GetErrorBand("StauErr", ThStauN,ThStauMass,ThStauLow,ThStauHigh);

   int ThPPStauN = 6 ; double ThPPStauMass [100]; double ThPPStauXSec [100];  double ThPPStauLow  [100];  double ThPPStauHigh [100];
   ThPPStauMass[0] = 100; ThPPStauXSec[0] = 0.038200;  ThPPStauLow[0] = 0.037076;  ThPPStauHigh[0] = 0.0391443;
   ThPPStauMass[1] = 126; ThPPStauXSec[1] = 0.0161;  ThPPStauLow[1] = 0.0155927;  ThPPStauHigh[1] = 0.016527;
   ThPPStauMass[2] = 156; ThPPStauXSec[2] = 0.007040;  ThPPStauLow[2] = 0.0067891;  ThPPStauHigh[2] = 0.00723151;
   ThPPStauMass[3] = 200; ThPPStauXSec[3] = 0.002470;  ThPPStauLow[3] = 0.00237277;  ThPPStauHigh[3] = 0.00253477;
   ThPPStauMass[4] = 247; ThPPStauXSec[4] = 0.001010;  ThPPStauLow[4] = 0.00096927;  ThPPStauHigh[4] = 0.00103844;
   ThPPStauMass[5] = 308; ThPPStauXSec[5] = 0.000353;  ThPPStauLow[5] = 0.000335308;  ThPPStauHigh[5] = 0.000363699;
   TCutG* PPStauXSecErr = GetErrorBand("PPStauErr", ThPPStauN,ThPPStauMass,ThPPStauLow,ThPPStauHigh);

   int ThDCRho08HyperKN = 13; double ThDCRho08HyperKMass [100]; double ThDCRho08HyperKXSec [100];  double ThDCRho08HyperKLow  [100];  double ThDCRho08HyperKHigh [100];
   ThDCRho08HyperKMass[0] = 100; ThDCRho08HyperKXSec[0] = 1.405000;  ThDCRho08HyperKLow[0] = ThDCRho08HyperKXSec[0]*0.85;  ThDCRho08HyperKHigh[0] = ThDCRho08HyperKXSec[0]*1.15;
   ThDCRho08HyperKMass[1] = 121; ThDCRho08HyperKXSec[1] = 0.979000;  ThDCRho08HyperKLow[1] = ThDCRho08HyperKXSec[1]*0.85;  ThDCRho08HyperKHigh[1] = ThDCRho08HyperKXSec[1]*1.15;
   ThDCRho08HyperKMass[2] = 182; ThDCRho08HyperKXSec[2] = 0.560000;  ThDCRho08HyperKLow[2] = ThDCRho08HyperKXSec[2]*0.85;  ThDCRho08HyperKHigh[2] = ThDCRho08HyperKXSec[2]*1.15;
   ThDCRho08HyperKMass[3] = 242; ThDCRho08HyperKXSec[3] = 0.489000;  ThDCRho08HyperKLow[3] = ThDCRho08HyperKXSec[3]*0.85;  ThDCRho08HyperKHigh[3] = ThDCRho08HyperKXSec[3]*1.15;
   ThDCRho08HyperKMass[4] = 302; ThDCRho08HyperKXSec[4] = 0.463000;  ThDCRho08HyperKLow[4] = ThDCRho08HyperKXSec[4]*0.85;  ThDCRho08HyperKHigh[4] = ThDCRho08HyperKXSec[4]*1.15;
   ThDCRho08HyperKMass[5] = 350; ThDCRho08HyperKXSec[5] = 0.473000;  ThDCRho08HyperKLow[5] = ThDCRho08HyperKXSec[5]*0.85;  ThDCRho08HyperKHigh[5] = ThDCRho08HyperKXSec[5]*1.15;
   ThDCRho08HyperKMass[6] = 370; ThDCRho08HyperKXSec[6] = 0.48288105;  ThDCRho08HyperKLow[6] = ThDCRho08HyperKXSec[6]*0.85;  ThDCRho08HyperKHigh[6] = ThDCRho08HyperKXSec[6]*1.15;
   ThDCRho08HyperKMass[7] = 390; ThDCRho08HyperKXSec[7] = 0.47132496;  ThDCRho08HyperKLow[7] = ThDCRho08HyperKXSec[7]*0.85;  ThDCRho08HyperKHigh[7] = ThDCRho08HyperKXSec[7]*1.15;
   ThDCRho08HyperKMass[8] = 395; ThDCRho08HyperKXSec[8] = 0.420000;  ThDCRho08HyperKLow[8] = ThDCRho08HyperKXSec[8]*0.85;  ThDCRho08HyperKHigh[8] = ThDCRho08HyperKXSec[8]*1.15;
   ThDCRho08HyperKMass[9] = 400; ThDCRho08HyperKXSec[9] = 0.473000;  ThDCRho08HyperKLow[9] = ThDCRho08HyperKXSec[9]*0.85;  ThDCRho08HyperKHigh[9] = ThDCRho08HyperKXSec[9]*1.15;
   ThDCRho08HyperKMass[10] = 410; ThDCRho08HyperKXSec[10] = 0.0060812129;  ThDCRho08HyperKLow[10] = ThDCRho08HyperKXSec[10]*0.85;  ThDCRho08HyperKHigh[10] = ThDCRho08HyperKXSec[10]*1.15;
   ThDCRho08HyperKMass[11] = 420; ThDCRho08HyperKXSec[11] = 0.0035;  ThDCRho08HyperKLow[11] = ThDCRho08HyperKXSec[11]*0.85;  ThDCRho08HyperKHigh[11] = ThDCRho08HyperKXSec[11]*1.15;
   ThDCRho08HyperKMass[12] = 500; ThDCRho08HyperKXSec[12] = 0.0002849;  ThDCRho08HyperKLow[12] = ThDCRho08HyperKXSec[12]*0.85;  ThDCRho08HyperKHigh[12] = ThDCRho08HyperKXSec[12]*1.15;
   TCutG* DCRho08HyperKXSecErr = GetErrorBand("DCRho08HyperKErr", ThDCRho08HyperKN,ThDCRho08HyperKMass,ThDCRho08HyperKLow,ThDCRho08HyperKHigh);

   int ThDCRho12HyperKN = 12; double ThDCRho12HyperKMass [100]; double ThDCRho12HyperKXSec [100];  double ThDCRho12HyperKLow  [100];  double ThDCRho12HyperKHigh [100];
   ThDCRho12HyperKMass[0] = 100; ThDCRho12HyperKXSec[0] = 0.8339415992;  ThDCRho12HyperKLow[0] = ThDCRho12HyperKXSec[0]*0.85;  ThDCRho12HyperKHigh[0] = ThDCRho12HyperKXSec[0]*1.15;
   ThDCRho12HyperKMass[1] = 182; ThDCRho12HyperKXSec[1] = 0.168096952140;  ThDCRho12HyperKLow[1] = ThDCRho12HyperKXSec[1]*0.85;  ThDCRho12HyperKHigh[1] = ThDCRho12HyperKXSec[1]*1.15;
   ThDCRho12HyperKMass[2] = 302; ThDCRho12HyperKXSec[2] = 0.079554948387;  ThDCRho12HyperKLow[2] = ThDCRho12HyperKXSec[2]*0.85;  ThDCRho12HyperKHigh[2] = ThDCRho12HyperKXSec[2]*1.15;
   ThDCRho12HyperKMass[3] = 500; ThDCRho12HyperKXSec[3] = 0.063996737;  ThDCRho12HyperKLow[3] = ThDCRho12HyperKXSec[3]*0.85;  ThDCRho12HyperKHigh[3] = ThDCRho12HyperKXSec[3]*1.15;
   ThDCRho12HyperKMass[4] = 530; ThDCRho12HyperKXSec[4] = 0.064943882;  ThDCRho12HyperKLow[4] = ThDCRho12HyperKXSec[4]*0.85;  ThDCRho12HyperKHigh[4] = ThDCRho12HyperKXSec[4]*1.15;
   ThDCRho12HyperKMass[5] = 570; ThDCRho12HyperKXSec[5] = 0.0662920530;  ThDCRho12HyperKLow[5] = ThDCRho12HyperKXSec[5]*0.85;  ThDCRho12HyperKHigh[5] = ThDCRho12HyperKXSec[5]*1.15;
   ThDCRho12HyperKMass[6] = 590; ThDCRho12HyperKXSec[6] = 0.060748383;  ThDCRho12HyperKLow[6] = ThDCRho12HyperKXSec[6]*0.85;  ThDCRho12HyperKHigh[6] = ThDCRho12HyperKXSec[6]*1.15;
   ThDCRho12HyperKMass[7] = 595; ThDCRho12HyperKXSec[7] = 0.04968409;  ThDCRho12HyperKLow[7] = ThDCRho12HyperKXSec[7]*0.85;  ThDCRho12HyperKHigh[7] = ThDCRho12HyperKXSec[7]*1.15;
   ThDCRho12HyperKMass[8] = 600; ThDCRho12HyperKXSec[8] = 0.0026232721237;  ThDCRho12HyperKLow[8] = ThDCRho12HyperKXSec[8]*0.85;  ThDCRho12HyperKHigh[8] = ThDCRho12HyperKXSec[8]*1.15;
   ThDCRho12HyperKMass[9] = 610; ThDCRho12HyperKXSec[9] = 0.00127431;  ThDCRho12HyperKLow[9] = ThDCRho12HyperKXSec[9]*0.85;  ThDCRho12HyperKHigh[9] = ThDCRho12HyperKXSec[9]*1.15;
   ThDCRho12HyperKMass[10] = 620; ThDCRho12HyperKXSec[10] = 0.00056965104319;  ThDCRho12HyperKLow[10] = ThDCRho12HyperKXSec[10]*0.85;  ThDCRho12HyperKHigh[10] = ThDCRho12HyperKXSec[10]*1.15;
   ThDCRho12HyperKMass[11] = 700; ThDCRho12HyperKXSec[11] = 0.00006122886211;  ThDCRho12HyperKLow[11] = ThDCRho12HyperKXSec[11]*0.85;  ThDCRho12HyperKHigh[11] = ThDCRho12HyperKXSec[11]*1.15;
   TCutG* DCRho12HyperKXSecErr = GetErrorBand("DCRho12HyperKErr", ThDCRho12HyperKN,ThDCRho12HyperKMass,ThDCRho12HyperKLow,ThDCRho12HyperKHigh);

   int ThDCRho16HyperKN = 12; double ThDCRho16HyperKMass [100]; double ThDCRho16HyperKXSec [100];  double ThDCRho16HyperKLow  [100];  double ThDCRho16HyperKHigh [100];
   ThDCRho16HyperKMass[0] = 100; ThDCRho16HyperKXSec[0] = 0.711518686800;  ThDCRho16HyperKLow[0] = ThDCRho16HyperKXSec[0]*0.85;  ThDCRho16HyperKHigh[0] = ThDCRho16HyperKXSec[0]*1.15;
   ThDCRho16HyperKMass[1] = 182; ThDCRho16HyperKXSec[1] = 0.089726059780;  ThDCRho16HyperKLow[1] = ThDCRho16HyperKXSec[1]*0.85;  ThDCRho16HyperKHigh[1] = ThDCRho16HyperKXSec[1]*1.15;
   ThDCRho16HyperKMass[2] = 302; ThDCRho16HyperKXSec[2] = 0.019769637301;  ThDCRho16HyperKLow[2] = ThDCRho16HyperKXSec[2]*0.85;  ThDCRho16HyperKHigh[2] = ThDCRho16HyperKXSec[2]*1.15;
   ThDCRho16HyperKMass[3] = 500; ThDCRho16HyperKXSec[3] = 0.0063302286576;  ThDCRho16HyperKLow[3] = ThDCRho16HyperKXSec[3]*0.85;  ThDCRho16HyperKHigh[3] = ThDCRho16HyperKXSec[3]*1.15;
   ThDCRho16HyperKMass[4] = 700; ThDCRho16HyperKXSec[4] = 0.002536779850;  ThDCRho16HyperKLow[4] = ThDCRho16HyperKXSec[4]*0.85;  ThDCRho16HyperKHigh[4] = ThDCRho16HyperKXSec[4]*1.15;
   ThDCRho16HyperKMass[5] = 730; ThDCRho16HyperKXSec[5] = 0.00213454921;  ThDCRho16HyperKLow[5] = ThDCRho16HyperKXSec[5]*0.85;  ThDCRho16HyperKHigh[5] = ThDCRho16HyperKXSec[5]*1.15;
   ThDCRho16HyperKMass[6] = 770; ThDCRho16HyperKXSec[6] = 0.001737551;  ThDCRho16HyperKLow[6] = ThDCRho16HyperKXSec[6]*0.85;  ThDCRho16HyperKHigh[6] = ThDCRho16HyperKXSec[6]*1.15;
   ThDCRho16HyperKMass[7] = 790; ThDCRho16HyperKXSec[7] =0.00161578593 ;  ThDCRho16HyperKLow[7] = ThDCRho16HyperKXSec[7]*0.85;  ThDCRho16HyperKHigh[7] = ThDCRho16HyperKXSec[7]*1.15;
   ThDCRho16HyperKMass[8] = 795; ThDCRho16HyperKXSec[8] = 0.00153513713;  ThDCRho16HyperKLow[8] = ThDCRho16HyperKXSec[8]*0.85;  ThDCRho16HyperKHigh[8] = ThDCRho16HyperKXSec[8]*1.15;
   ThDCRho16HyperKMass[9] = 800; ThDCRho16HyperKXSec[9] = 0.000256086965;  ThDCRho16HyperKLow[9] = ThDCRho16HyperKXSec[9]*0.85;  ThDCRho16HyperKHigh[9] = ThDCRho16HyperKXSec[9]*1.15;
   ThDCRho16HyperKMass[10] = 820; ThDCRho16HyperKXSec[10] = 0.000097929923655;  ThDCRho16HyperKLow[10] = ThDCRho16HyperKXSec[10]*0.85;  ThDCRho16HyperKHigh[10] = ThDCRho16HyperKXSec[10]*1.15;
   ThDCRho16HyperKMass[11] = 900; ThDCRho16HyperKXSec[11] = 0.000013146066;  ThDCRho16HyperKLow[11] = ThDCRho16HyperKXSec[11]*0.85;  ThDCRho16HyperKHigh[11] = ThDCRho16HyperKXSec[11]*1.15;
   TCutG* DCRho16HyperKXSecErr = GetErrorBand("DCRho16HyperKErr", ThDCRho16HyperKN,ThDCRho16HyperKMass,ThDCRho16HyperKLow,ThDCRho16HyperKHigh);



/*
   fprintf(pFile,"-----------------------\n0%% TK ONLY       \n-------------------------\n");
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Tk_Obs_Gluino2C,  GluinoXSecLow, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Tk_Obs_GluinoF0,  GluinoXSecLow, 300, 1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Tk_Obs_GluinoF1,  GluinoXSecLow, 300, 1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Tk_Obs_GluinoZF1, GluinoXSecLow, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Tk_Obs_GluinoF5,  GluinoXSecLow, 300, 1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Tk_Obs_GluinoNF0, GluinoXSecLow, 300, 1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Tk_Obs_GluinoNF1, GluinoXSecLow, 300, 1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Tk_Obs_GluinoNF5, GluinoXSecLow, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Tk_Obs_Stop2C   , StopXSecLow  , 130,  800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Tk_Obs_Stop     , StopXSecLow  , 130,  800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Tk_Obs_StopZ    , StopXSecLow  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Tk_Obs_StopN    , StopXSecLow  , 130,  800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Tk_Obs_GMStau   , GMStauXSec   , 100,  494, 1, 0.15));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Tk_Obs_PPStau   , PPStauXSec   , 100,  308, 1, 0.15));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Tk_Obs_DCStau   , DCStauXSec   , 121,  302, 1, 0.15));

   fprintf(pFile,"-----------------------\n0%% TK TOF        \n-------------------------\n");
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Mu_Obs_Gluino2C,  GluinoXSecLow, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Mu_Obs_GluinoF0,  GluinoXSecLow, 300, 1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Mu_Obs_GluinoF1,  GluinoXSecLow, 300, 1100, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Mu_Obs_GluinoZF1, GluinoXSecLow, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Mu_Obs_GluinoF5,  GluinoXSecLow, 300, 1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Mu_Obs_GluinoNF0, GluinoXSecLow, 300, 1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Mu_Obs_GluinoNF1, GluinoXSecLow, 300, 1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Mu_Obs_GluinoNF5, GluinoXSecLow, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Mu_Obs_Stop2C   , StopXSecLow  , 130,  800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Mu_Obs_Stop     , StopXSecLow  , 130,  800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Mu_Obs_StopZ    , StopXSecLow  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Mu_Obs_StopN    , StopXSecLow  , 130,  800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Mu_Obs_GMStau   , GMStauXSec   , 100,  494, 1, 0.15));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Mu_Obs_PPStau   , PPStauXSec   , 100,  308, 1, 0.15));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Mu_Obs_DCStau   , DCStauXSec   , 121,  302, 1, 0.15));

*/
   fprintf(pFile,"-----------------------\nNO TH UNCERTAINTY ACCOUNTED FOR   \n-------------------------\n");

   fprintf(pFile,"-----------------------\n0%% TK ONLY       \n-------------------------\n");
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Tk_Obs_Gluino2C,  GluinoXSec, 300, 900, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Tk_Obs_GluinoF0,  GluinoXSec, 300, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Tk_Obs_GluinoF1,  GluinoXSec, 300, 1200, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Tk_Obs_GluinoZF1, GluinoXSec, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Tk_Obs_GluinoF5,  GluinoXSec, 300, 1200, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Tk_Obs_GluinoNF0, GluinoXSec, 300, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Tk_Obs_GluinoNF1, GluinoXSec, 300, 1200, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Tk_Obs_GluinoNF5, GluinoXSec, 300, 900, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Tk_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Tk_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Tk_Obs_StopZ    , StopXSec  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Tk_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Tk_Obs_GMStau   , GMStauXSec, 100, 494, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Tk_Obs_PPStau   , PPStauXSec, 100, 308, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho08HyperK   \n", FindIntersection(Tk_Obs_DCRho08HyperK   , DCRho08HyperKXSec, 100, 500, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho12HyperK   \n", FindIntersection(Tk_Obs_DCRho12HyperK   , DCRho12HyperKXSec, 100, 700, 1, 0.0));
    fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho16HyperK   \n", FindIntersection(Tk_Obs_DCRho16HyperK   , DCRho16HyperKXSec, 100, 900, 1, 0.0));



   
   fprintf(pFile,"-----------------------\n0%% TK TOF        \n-------------------------\n");
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Mu_Obs_Gluino2C,  GluinoXSec, 300,1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Mu_Obs_GluinoF0,  GluinoXSec, 300,1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Mu_Obs_GluinoF1,  GluinoXSec, 300,1200, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Mu_Obs_GluinoZF1, GluinoXSec, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Mu_Obs_GluinoF5,  GluinoXSec, 300,1200, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Mu_Obs_GluinoNF0, GluinoXSec, 300,1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Mu_Obs_GluinoNF1, GluinoXSec, 300,1200, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Mu_Obs_GluinoNF5, GluinoXSec, 300,1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Mu_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Mu_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Mu_Obs_StopZ    , StopXSec  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Mu_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Mu_Obs_GMStau   , GMStauXSec, 100, 494, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Mu_Obs_PPStau   , PPStauXSec, 100, 308, 1, 0.0));

   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho08HyperK   \n", FindIntersection(Mu_Obs_DCRho08HyperK   , DCRho08HyperKXSec, 100, 500, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho12HyperK   \n", FindIntersection(Mu_Obs_DCRho12HyperK   , DCRho12HyperKXSec, 100, 700, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCRho16HyperK   \n", FindIntersection(Mu_Obs_DCRho16HyperK   , DCRho16HyperKXSec, 100, 900, 1, 0.0));

   fclose(pFile);
   if(syst!="")return;


   GluinoXSec      ->SetLineColor(4);  GluinoXSec      ->SetMarkerColor(4);   GluinoXSec      ->SetLineWidth(5);   GluinoXSec      ->SetLineStyle(3);  GluinoXSec      ->SetMarkerStyle(1);
   Mu_Obs_GluinoF1 ->SetLineColor(4);  Mu_Obs_GluinoF1 ->SetMarkerColor(4);   Mu_Obs_GluinoF1 ->SetLineWidth(2);   Mu_Obs_GluinoF1 ->SetLineStyle(1);  Mu_Obs_GluinoF1 ->SetMarkerStyle(22);
   Mu_Obs_GluinoF5 ->SetLineColor(4);  Mu_Obs_GluinoF5 ->SetMarkerColor(4);   Mu_Obs_GluinoF5 ->SetLineWidth(2);   Mu_Obs_GluinoF5 ->SetLineStyle(1);  Mu_Obs_GluinoF5 ->SetMarkerStyle(23);
   Mu_Obs_GluinoNF1->SetLineColor(4);  Mu_Obs_GluinoNF1->SetMarkerColor(4);   Mu_Obs_GluinoNF1->SetLineWidth(2);   Mu_Obs_GluinoNF1->SetLineStyle(1);  Mu_Obs_GluinoNF1->SetMarkerStyle(26);
   Tk_Obs_GluinoF1 ->SetLineColor(4);  Tk_Obs_GluinoF1 ->SetMarkerColor(4);   Tk_Obs_GluinoF1 ->SetLineWidth(2);   Tk_Obs_GluinoF1 ->SetLineStyle(1);  Tk_Obs_GluinoF1 ->SetMarkerStyle(22);
   Tk_Obs_GluinoF5 ->SetLineColor(4);  Tk_Obs_GluinoF5 ->SetMarkerColor(4);   Tk_Obs_GluinoF5 ->SetLineWidth(2);   Tk_Obs_GluinoF5 ->SetLineStyle(1);  Tk_Obs_GluinoF5 ->SetMarkerStyle(23);
   Tk_Obs_GluinoNF1->SetLineColor(4);  Tk_Obs_GluinoNF1->SetMarkerColor(4);   Tk_Obs_GluinoNF1->SetLineWidth(2);   Tk_Obs_GluinoNF1->SetLineStyle(1);  Tk_Obs_GluinoNF1->SetMarkerStyle(26);
   StopXSec        ->SetLineColor(2);  StopXSec        ->SetMarkerColor(2);   StopXSec        ->SetLineWidth(5);   StopXSec        ->SetLineStyle(2);  StopXSec        ->SetMarkerStyle(1);
   Mu_Obs_Stop     ->SetLineColor(2);  Mu_Obs_Stop     ->SetMarkerColor(2);   Mu_Obs_Stop     ->SetLineWidth(2);   Mu_Obs_Stop     ->SetLineStyle(1);  Mu_Obs_Stop     ->SetMarkerStyle(21);
   Mu_Obs_StopN    ->SetLineColor(2);  Mu_Obs_StopN    ->SetMarkerColor(2);   Mu_Obs_StopN    ->SetLineWidth(2);   Mu_Obs_StopN    ->SetLineStyle(1);  Mu_Obs_StopN    ->SetMarkerStyle(25);
   Tk_Obs_Stop     ->SetLineColor(2);  Tk_Obs_Stop     ->SetMarkerColor(2);   Tk_Obs_Stop     ->SetLineWidth(2);   Tk_Obs_Stop     ->SetLineStyle(1);  Tk_Obs_Stop     ->SetMarkerStyle(21);
   Tk_Obs_StopN    ->SetLineColor(2);  Tk_Obs_StopN    ->SetMarkerColor(2);   Tk_Obs_StopN    ->SetLineWidth(2);   Tk_Obs_StopN    ->SetLineStyle(1);  Tk_Obs_StopN    ->SetMarkerStyle(25);
   GMStauXSec      ->SetLineColor(1);  GMStauXSec      ->SetMarkerColor(1);   GMStauXSec      ->SetLineWidth(5);   GMStauXSec      ->SetLineStyle(1);  GMStauXSec      ->SetMarkerStyle(1);
   PPStauXSec      ->SetLineColor(6);  PPStauXSec      ->SetMarkerColor(6);   PPStauXSec      ->SetLineWidth(5);   PPStauXSec      ->SetLineStyle(4);  PPStauXSec      ->SetMarkerStyle(1);
   DCRho08HyperKXSec      ->SetLineColor(4);  DCRho08HyperKXSec      ->SetMarkerColor(4);   DCRho08HyperKXSec      ->SetLineWidth(5);   DCRho08HyperKXSec      ->SetLineStyle(3);  DCRho08HyperKXSec      ->SetMarkerStyle(1);
   DCRho12HyperKXSec      ->SetLineColor(2);  DCRho12HyperKXSec      ->SetMarkerColor(2);   DCRho12HyperKXSec      ->SetLineWidth(5);   DCRho12HyperKXSec      ->SetLineStyle(2);  DCRho12HyperKXSec      ->SetMarkerStyle(1);
   DCRho16HyperKXSec      ->SetLineColor(1);  DCRho16HyperKXSec      ->SetMarkerColor(1);   DCRho16HyperKXSec      ->SetLineWidth(5);   DCRho16HyperKXSec      ->SetLineStyle(1);  DCRho16HyperKXSec      ->SetMarkerStyle(1);


   Mu_Obs_GMStau   ->SetLineColor(1);  Mu_Obs_GMStau   ->SetMarkerColor(1);   Mu_Obs_GMStau   ->SetLineWidth(2);   Mu_Obs_GMStau   ->SetLineStyle(1);  Mu_Obs_GMStau   ->SetMarkerStyle(23);
   Mu_Obs_PPStau   ->SetLineColor(6);  Mu_Obs_PPStau   ->SetMarkerColor(6);   Mu_Obs_PPStau   ->SetLineWidth(2);   Mu_Obs_PPStau   ->SetLineStyle(1);  Mu_Obs_PPStau   ->SetMarkerStyle(23);
   Mu_Obs_DCRho08HyperK   ->SetLineColor(4);  Mu_Obs_DCRho08HyperK   ->SetMarkerColor(4);   Mu_Obs_DCRho08HyperK   ->SetLineWidth(2);   Mu_Obs_DCRho08HyperK   ->SetLineStyle(1);  Mu_Obs_DCRho08HyperK   ->SetMarkerStyle(22);
   Mu_Obs_DCRho12HyperK   ->SetLineColor(2);  Mu_Obs_DCRho12HyperK   ->SetMarkerColor(2);   Mu_Obs_DCRho12HyperK   ->SetLineWidth(2);   Mu_Obs_DCRho12HyperK   ->SetLineStyle(1);  Mu_Obs_DCRho12HyperK   ->SetMarkerStyle(23);
   Mu_Obs_DCRho16HyperK   ->SetLineColor(1);  Mu_Obs_DCRho16HyperK   ->SetMarkerColor(1);   Mu_Obs_DCRho16HyperK   ->SetLineWidth(2);   Mu_Obs_DCRho16HyperK   ->SetLineStyle(1);  Mu_Obs_DCRho16HyperK   ->SetMarkerStyle(26);


   Tk_Obs_GMStau   ->SetLineColor(1);  Tk_Obs_GMStau   ->SetMarkerColor(1);   Tk_Obs_GMStau   ->SetLineWidth(2);   Tk_Obs_GMStau   ->SetLineStyle(1);  Tk_Obs_GMStau   ->SetMarkerStyle(20);
   Tk_Obs_PPStau   ->SetLineColor(6);  Tk_Obs_PPStau   ->SetMarkerColor(6);   Tk_Obs_PPStau   ->SetLineWidth(2);   Tk_Obs_PPStau   ->SetLineStyle(1);  Tk_Obs_PPStau   ->SetMarkerStyle(20);
   Tk_Obs_DCRho08HyperK   ->SetLineColor(4);  Tk_Obs_DCRho08HyperK   ->SetMarkerColor(4);   Tk_Obs_DCRho08HyperK   ->SetLineWidth(2);   Tk_Obs_DCRho08HyperK   ->SetLineStyle(1);  Tk_Obs_DCRho08HyperK   ->SetMarkerStyle(22);
   Tk_Obs_DCRho12HyperK   ->SetLineColor(2);  Tk_Obs_DCRho12HyperK   ->SetMarkerColor(2);   Tk_Obs_DCRho12HyperK   ->SetLineWidth(2);   Tk_Obs_DCRho12HyperK   ->SetLineStyle(1);  Tk_Obs_DCRho12HyperK   ->SetMarkerStyle(23);
   Tk_Obs_DCRho16HyperK   ->SetLineColor(1);  Tk_Obs_DCRho16HyperK   ->SetMarkerColor(1);   Tk_Obs_DCRho16HyperK   ->SetLineWidth(2);   Tk_Obs_DCRho16HyperK   ->SetLineStyle(1);  Tk_Obs_DCRho16HyperK   ->SetMarkerStyle(26);



   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGMu = new TMultiGraph();
   MGMu->Add(GluinoXSec      ,"L");
   MGMu->Add(StopXSec        ,"L");
   MGMu->Add(GMStauXSec      ,"L");
   MGMu->Add(PPStauXSec      ,"L");
   MGMu->Add(Mu_Obs_GluinoF1      ,"LP");
   MGMu->Add(Mu_Obs_GluinoF5      ,"LP");
//   MGMu->Add(Mu_Obs_GluinoNF1     ,"LP");
   MGMu->Add(Mu_Obs_Stop          ,"LP");
//   MGMu->Add(Mu_Obs_StopN         ,"LP");
   MGMu->Add(Mu_Obs_GMStau        ,"LP");
   MGMu->Add(Mu_Obs_PPStau        ,"LP");
   MGMu->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr  ->Draw("f");
   StauXSecErr  ->Draw("f");
   PPStauXSecErr  ->Draw("f");
   MGMu->Draw("same");
   MGMu->SetTitle("");
   MGMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGMu->GetYaxis()->SetTitleOffset(1.70);
   MGMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   TLegend* LEGMu = new TLegend(0.45,0.65,0.65,0.90);   
//   LEGMu->SetHeader("95% C.L. Limits");
   LEGMu->SetHeader("Tracker + TOF");
   LEGMu->SetFillColor(0); 
   LEGMu->SetBorderSize(0);
   LEGMu->AddEntry(Mu_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(Mu_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
//   LEGMu->AddEntry(Mu_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGMu->AddEntry(Mu_Obs_Stop     , "stop"            ,"LP");
//   LEGMu->AddEntry(Mu_Obs_StopN    , "stop; ch. suppr.","LP");
   LEGMu->AddEntry(Mu_Obs_PPStau   , "Pair Prod. stau"       ,"LP");
   LEGMu->AddEntry(Mu_Obs_GMStau   , "GMSB stau"       ,"LP");
   //LEGMu->Draw();

   TLegend* LEGTh = new TLegend(0.15,0.7,0.48,0.9);
   LEGTh->SetHeader("Theoretical Prediction");
   LEGTh->SetFillColor(0);
   LEGTh->SetBorderSize(0);
   TGraph* GlThLeg = (TGraph*) GluinoXSec->Clone("GluinoThLeg");
   GlThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGTh->AddEntry(GlThLeg, "gluino (NLO+NLL)" ,"LF");
   TGraph* StThLeg = (TGraph*) StopXSec->Clone("StopThLeg");
   StThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGTh->AddEntry(StThLeg   ,"stop   (NLO+NLL)" ,"LF");

   TGraph* PPStauThLeg = (TGraph*) PPStauXSec->Clone("PPStauThLeg");
   PPStauThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGTh->AddEntry(PPStauThLeg   ,"Pair Prod. stau   (NLO)" ,"LF");
   TGraph* StauThLeg = (TGraph*) GMStauXSec->Clone("StauThLeg");
   StauThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGTh->AddEntry(StauThLeg   ,"GMSB stau   (NLO)" ,"LF");

   LEGTh->Draw();
   LEGMu->Draw();

//   c1->SetGridx(true);
//   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("MuExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuExclusionLog"));
   delete c1;


   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGTk = new TMultiGraph();
   MGTk->Add(GluinoXSec      ,"L");
   MGTk->Add(StopXSec        ,"L");
   MGTk->Add(GMStauXSec      ,"L");
   MGTk->Add(PPStauXSec      ,"L");
   MGTk->Add(Tk_Obs_GluinoF1      ,"LP");
   MGTk->Add(Tk_Obs_GluinoF5      ,"LP");
   MGTk->Add(Tk_Obs_GluinoNF1     ,"LP");
   MGTk->Add(Tk_Obs_Stop          ,"LP");
   MGTk->Add(Tk_Obs_StopN         ,"LP");
   MGTk->Add(Tk_Obs_GMStau        ,"LP");
   MGTk->Add(Tk_Obs_PPStau        ,"LP");
   MGTk->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr  ->Draw("f");
   StauXSecErr  ->Draw("f");
   PPStauXSecErr  ->Draw("f");
   MGTk->Draw("same");
   MGTk->SetTitle("");
   MGTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGTk->GetYaxis()->SetTitleOffset(1.70);
   MGTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGTk = new TLegend(0.45,0.58,0.795,0.9);
//   LEGTk->SetHeader("95% C.L. Limits");
   LEGTk->SetHeader("Tracker - Only");
   LEGTk->SetFillColor(0); 
   LEGTk->SetBorderSize(0);
   LEGTk->AddEntry(Tk_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_Stop     , "stop"            ,"LP");
   LEGTk->AddEntry(Tk_Obs_StopN    , "stop; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_PPStau   , "Pair Prod. stau"       ,"LP");
   LEGTk->AddEntry(Tk_Obs_GMStau   , "GMSB stau"       ,"LP");
   //LEGTk->Draw();

   LEGTh->Draw();
   LEGTk->Draw();

//   c1->SetGridx(true);
//   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("TkExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkExclusionLog"));
   delete c1;

    c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCMu = new TMultiGraph();
   MGDCMu->Add(DCRho08HyperKXSec      ,"L");
   MGDCMu->Add(Mu_Obs_DCRho08HyperK        ,"LP");
   MGDCMu->Add(DCRho12HyperKXSec      ,"L");
   MGDCMu->Add(Mu_Obs_DCRho12HyperK        ,"LP");
   MGDCMu->Add(DCRho16HyperKXSec      ,"L");
   MGDCMu->Add(Mu_Obs_DCRho16HyperK        ,"LP");
   MGDCMu->Draw("A");
//   DCRho08HyperKXSecErr  ->Draw("f");
//   DCRho12HyperKXSecErr  ->Draw("f");
//   DCRho16HyperKXSecErr  ->Draw("f");
   MGDCMu->Draw("same");
   MGDCMu->SetTitle("");
   MGDCMu->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCMu->GetYaxis()->SetTitleOffset(1.70);
   MGDCMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGDCMu = new TLegend(0.50,0.65,0.80,0.9);
   LEGDCMu->SetHeader("Tracker + TOF");
   LEGDCMu->SetFillColor(0); 
   LEGDCMu->SetBorderSize(0);
   LEGDCMu->AddEntry(Mu_Obs_DCRho08HyperK   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
   LEGDCMu->AddEntry(Mu_Obs_DCRho12HyperK   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCMu->AddEntry(Mu_Obs_DCRho16HyperK   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");
   //LEGDCMu->Draw();

   TLegend* LEGDCTh = new TLegend(0.15,0.7,0.49,0.9);
   LEGDCTh->SetHeader("Theoretical Prediction");
   LEGDCTh->SetFillColor(0);
   LEGDCTh->SetBorderSize(0);
   TGraph* DCRho08HyperKThLeg = (TGraph*) DCRho08HyperKXSec->Clone("DCRho08HyperKThLeg");
   DCRho08HyperKThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGDCTh->AddEntry(DCRho08HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 0.8 TeV   (LO)" ,"L");
   TGraph* DCRho12HyperKThLeg = (TGraph*) DCRho12HyperKXSec->Clone("DCRho12HyperKThLeg");
   DCRho12HyperKThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGDCTh->AddEntry(DCRho12HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.2 TeV   (LO)" ,"L");
   TGraph* DCRho16HyperKThLeg = (TGraph*) DCRho16HyperKXSec->Clone("DCRho16HyperKThLeg");
   DCRho16HyperKThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGDCTh->AddEntry(DCRho16HyperKThLeg   ,"Hyper-K, #tilde{#rho} = 1.6 TeV   (LO)" ,"L");
   LEGDCTh->Draw();
   LEGDCMu->Draw();
   SaveCanvas(c1, outpath, string("MuDCExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuDCExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCTk = new TMultiGraph();
   MGDCTk->Add(DCRho08HyperKXSec      ,"L");
   MGDCTk->Add(Tk_Obs_DCRho08HyperK        ,"LP");
   MGDCTk->Add(DCRho12HyperKXSec      ,"L");
   MGDCTk->Add(Tk_Obs_DCRho12HyperK        ,"LP");
   MGDCTk->Add(DCRho16HyperKXSec      ,"L");
   MGDCTk->Add(Tk_Obs_DCRho16HyperK        ,"LP");
   MGDCTk->Draw("A");
//   DCRho08HyperKXSecErr  ->Draw("f");
//   DCRho12HyperKXSecErr  ->Draw("f");
//   DCRho16HyperKXSecErr  ->Draw("f");
   MGDCTk->Draw("same");
   MGDCTk->SetTitle("");
   MGDCTk->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MGDCTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCTk->GetYaxis()->SetTitleOffset(1.70);
   MGDCTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);

   TLegend* LEGDCTk = new TLegend(0.50,0.65,0.80,0.90);
//   LEGDCTk->SetHeader("95% C.L. Limits");
   LEGDCTk->SetHeader("Tracker - Only");
   LEGDCTk->SetFillColor(0); 
   LEGDCTk->SetBorderSize(0);
   LEGDCTk->AddEntry(Tk_Obs_DCRho08HyperK   , "Hyper-K, #tilde{#rho} = 0.8 TeV"       ,"LP");
   LEGDCTk->AddEntry(Tk_Obs_DCRho12HyperK   , "Hyper-K, #tilde{#rho} = 1.2 TeV"       ,"LP");
   LEGDCTk->AddEntry(Tk_Obs_DCRho16HyperK   , "Hyper-K, #tilde{#rho} = 1.6 TeV"       ,"LP");
   LEGDCTk->Draw();

   LEGDCTh->Draw();

   SaveCanvas(c1, outpath, string("TkDCExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkDCExclusionLog"));
   delete c1;



   return; 
}


void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern){

  bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);

   std::vector<string> Models;
   Models.push_back("Gluino300_f1");
   Models.push_back("Gluino400_f1");
   Models.push_back("Gluino500_f1");
   Models.push_back("Gluino600_f1");
   Models.push_back("Gluino700_f1");
   Models.push_back("Gluino800_f1");
   Models.push_back("Gluino900_f1");
   Models.push_back("Gluino1000_f1");
   Models.push_back("Gluino1100_f1");
   Models.push_back("Gluino1200_f1");
   Models.push_back("Gluino300_f5");
   Models.push_back("Gluino400_f5");
   Models.push_back("Gluino500_f5");
   Models.push_back("Gluino600_f5");
   Models.push_back("Gluino700_f5");
   Models.push_back("Gluino800_f5");
   Models.push_back("Gluino900_f5");
   Models.push_back("Gluino1000_f5");
   Models.push_back("Gluino1100_f5");
   Models.push_back("Gluino1200_f5");
   Models.push_back("Gluino300N_f1");
   Models.push_back("Gluino400N_f1");
   Models.push_back("Gluino500N_f1");
   Models.push_back("Gluino600N_f1");
   Models.push_back("Gluino700N_f1");
   Models.push_back("Gluino800N_f1");
   Models.push_back("Gluino900N_f1");
   Models.push_back("Gluino1000N_f1");
   Models.push_back("Gluino1100N_f1");
   Models.push_back("Gluino1200N_f1");
   Models.push_back("Stop130");
   Models.push_back("Stop200");
   Models.push_back("Stop300");
   Models.push_back("Stop400");
   Models.push_back("Stop500");
   Models.push_back("Stop600");
   Models.push_back("Stop700");
   Models.push_back("Stop800");
   Models.push_back("Stop130N");
   Models.push_back("Stop200N");
   Models.push_back("Stop300N");
   Models.push_back("Stop400N");
   Models.push_back("Stop500N");
   Models.push_back("Stop600N");
   Models.push_back("Stop700N");
   Models.push_back("Stop800N");
   Models.push_back("GMStau100");
   Models.push_back("GMStau126");
   Models.push_back("GMStau156");
   Models.push_back("GMStau200");
   Models.push_back("GMStau247");
   Models.push_back("GMStau308");
   Models.push_back("GMStau370"); 
   Models.push_back("GMStau432"); 
   Models.push_back("GMStau494");
   Models.push_back("PPStau100");
   Models.push_back("PPStau126"); 
   Models.push_back("PPStau156"); 
   Models.push_back("PPStau200"); 
   Models.push_back("PPStau247");
   Models.push_back("PPStau308");
   Models.push_back("DCRho08HyperK100");
   Models.push_back("DCRho08HyperK121"); 
   Models.push_back("DCRho08HyperK182"); 
   Models.push_back("DCRho08HyperK242"); 
   Models.push_back("DCRho08HyperK302");  
   Models.push_back("DCRho08HyperK350");
   Models.push_back("DCRho08HyperK370");
   Models.push_back("DCRho08HyperK390");  
   Models.push_back("DCRho08HyperK395"); 
   Models.push_back("DCRho08HyperK400");
   Models.push_back("DCRho08HyperK410");
   Models.push_back("DCRho08HyperK420");
   Models.push_back("DCRho08HyperK500");
   Models.push_back("DCRho12HyperK100"); 
   Models.push_back("DCRho12HyperK182");
   Models.push_back("DCRho12HyperK302");
   Models.push_back("DCRho12HyperK500"); 
   Models.push_back("DCRho12HyperK530"); 
   Models.push_back("DCRho12HyperK570");
   Models.push_back("DCRho12HyperK590"); 
   Models.push_back("DCRho12HyperK595");
   Models.push_back("DCRho12HyperK600");
   Models.push_back("DCRho12HyperK610");
   Models.push_back("DCRho12HyperK620");
   Models.push_back("DCRho12HyperK700");
   Models.push_back("DCRho16HyperK100");
   Models.push_back("DCRho16HyperK182"); 
   Models.push_back("DCRho16HyperK302");
   Models.push_back("DCRho16HyperK500");
   Models.push_back("DCRho16HyperK700"); 
   Models.push_back("DCRho16HyperK730"); 
   Models.push_back("DCRho16HyperK770");
   Models.push_back("DCRho16HyperK790");
   Models.push_back("DCRho16HyperK795");
   Models.push_back("DCRho16HyperK800");
   Models.push_back("DCRho16HyperK820");
   Models.push_back("DCRho16HyperK900");
   

   if(IsTkOnly){
     fprintf(pFile, "%20s   Eff   --> PScale |  DeDxScale | PUScale || TotalUncertainty\n","Model");
     fprintf(talkFile, "\\hline\n%20s &  Eff   & PScale &  DeDxScale & PUScale & TotalUncertainty \\\\\n","Model");
   }
   else {
     fprintf(pFile, "%20s   Eff   --> PScale |  DeDxScale | PUScale | TOFScale || TotalUncertainty\n","Model");
     fprintf(talkFile, "\\hline\n%20s &  Eff   & PScale &  DeDxScale & PUScale & TOFScale & TotalUncertainty \\\\\n","Model");
   }

   for(unsigned int s=0;s<Models.size();s++){
        stAllInfo tmp(InputPattern+"/EXCLUSION" + "/"+Models[s]+".txt");
        double P = tmp.Eff - tmp.Eff_SYSTP;
        double I = tmp.Eff - tmp.Eff_SYSTI;
        double PU = tmp.Eff - tmp.Eff_SYSTPU;
        double T = tmp.Eff - tmp.Eff_SYSTT;
        bool IsStau = (Models[s].find("Stau",0)<std::string::npos);
        bool IsNeutral = (Models[s].find("N",0)<std::string::npos);

	double Ptemp=max(P, 0.0);
        double Itemp=max(I, 0.0);
        double PUtemp=max(PU, 0.0);
        double Ttemp=max(T, 0.0);

	if(IsTkOnly) fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, I/tmp.Eff, PU/tmp.Eff, sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);        

	else if(!IsNeutral) fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, I/tmp.Eff, PU/tmp.Eff, T/tmp.Eff, sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);

	if(IsTkOnly && (IsStau || (int)tmp.Mass%200==0)) {
	  fprintf(talkFile, "\\hline\n%20s &  %7.1f\\% & %7.1f\\%  &  %7.1f\\%  & %7.1f\\%  & %7.1f\\% \\\\\n",+Models[s].c_str(), 100.*tmp.Eff, 100.*P/tmp.Eff, 100.*I/tmp.Eff, 100.*PU/tmp.Eff, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);
	}
        if(!IsTkOnly && !IsNeutral) fprintf(talkFile, "\\hline\n%20s &  %7.1f\\% & %7.1f\\%  &  %7.1f\\%  & %7.1f\\%  & %7.1f\\% & %7.1f\\% \\\\\n",+Models[s].c_str(), 100.*tmp.Eff, 100.*P/tmp.Eff, 100.*I/tmp.Eff, 100.*PU/tmp.Eff, 100.*T/tmp.Eff, 100.*sqrt(Ptemp*Ptemp + Itemp*Itemp + PUtemp*PUtemp + Ttemp*Ttemp)/tmp.Eff);

   }
}



TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string syst, string ModelName, int XSectionType, string Mass0, string Mass1, string Mass2, string Mass3, string Mass4, string Mass5, string Mass6, string Mass7, string Mass8, string Mass9,string Mass10, string Mass11, string Mass12, string Mass13){
   unsigned int N=0;
   stAllInfo Infos[14];

   if(Mass0!=""){Infos[0] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass0+".txt"); N=1;}
   if(Mass1!=""){Infos[1] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass1+".txt"); N=2;}
   if(Mass2!=""){Infos[2] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass2+".txt"); N=3;}
   if(Mass3!=""){Infos[3] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass3+".txt"); N=4;}
   if(Mass4!=""){Infos[4] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass4+".txt"); N=5;}
   if(Mass5!=""){Infos[5] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass5+".txt"); N=6;}
   if(Mass6!=""){Infos[6] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass6+".txt"); N=7;}
   if(Mass7!=""){Infos[7] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass7+".txt"); N=8;}
   if(Mass8!=""){Infos[8] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass8+".txt"); N=9;}
   if(Mass9!=""){Infos[9] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass9+".txt"); N=10;}
   if(Mass10!=""){Infos[10] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass10+".txt"); N=11;}
   if(Mass11!=""){Infos[11] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass11+".txt"); N=12;}
   if(Mass12!=""){Infos[12] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass12+".txt"); N=13;}
   if(Mass13!=""){Infos[13] = stAllInfo(InputPattern+"/EXCLUSION" + syst + "/"+Mass13+".txt"); N=14;}

   double Mass   [14];for(unsigned int i=0;i<14;i++){Mass   [i]=Infos[i].Mass;    }
   double XSecTh [14];for(unsigned int i=0;i<14;i++){XSecTh [i]=Infos[i].XSec_Th; }
   double XSecObs[14];for(unsigned int i=0;i<14;i++){XSecObs[i]=Infos[i].XSec_Obs;}
   double XSecExp[14];for(unsigned int i=0;i<14;i++){XSecExp[i]=Infos[i].XSec_Exp;}



/*
   if(pFile){
      fprintf(pFile,"%40s",(ModelName + " mass (GeV/$c^2$)").c_str());for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.0f ",Infos[i].Mass);}     for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\\hline\n");
      fprintf(pFile,"%40s","Total acceptance (\\%)");                 for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",100.*Infos[i].Eff);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Expected 95\\% C.L. limit (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Exp);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Observed 95\\% C.L. limit (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Obs);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Theoretical cross section (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Th );} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\\hline\\hline\n");
   }
*/

   if(XSectionType>0 && syst=="")for(unsigned int i=0;i<N;i++)printf("%-18s %5.0f --> Pt>%+6.1f & I>%+5.3f & TOF>%+4.3f & M>%3.0f--> NData=%2.0f  NPred=%6.1E+-%6.1E  NSign=%6.1E (Eff=%3.2f) Local Significance %3.2f\n",ModelName.c_str(),Infos[i].Mass,Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NData, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NSign, Infos[i].Eff, Infos[i].Significance);

   if(XSectionType>0){
   for(unsigned int i=0;i<N;i++){
     if(Infos[i].WP_TOF==-1) fprintf(pFile,"%-20s & %4.0f & %6.0f & %5.3f & / & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %6.1E & %6.1E & %6.1E & %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].XSec_Th,Infos[i].XSec_Exp, Infos[i].XSec_Obs, Infos[i].Significance);
     else fprintf(pFile,"%-20s & %4.0f & %6.0f & %5.3f & %4.3f & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %6.1E & %6.1E & %6.1E & %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].XSec_Th,Infos[i].XSec_Exp, Infos[i].XSec_Obs, Infos[i].Significance);
     bool IsNeutral = (ModelName.find("N",0)<std::string::npos);
     if(Infos[i].WP_TOF==-1 && (ModelName=="GMSB Stau" || (int)Infos[i].Mass%200==0)) {
       fprintf(talkFile,"%-20s & %4.0f & %6.0f & %5.3f & / & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f & %3.2 \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].Significance);
       fprintf(talkFile, "\\hline\n");
     }
     if (Infos[i].WP_TOF!=-1 && !IsNeutral) {
       fprintf(talkFile,"%-20s & %4.0f & %6.0f & %5.3f & %4.3f & %4.0f & %6.3f $\\pm$ %6.3f & %2.0f & %4.3f %3.2f \\\\\n", ModelName.c_str(), Infos[i].Mass,  Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF,Infos[i].MassCut, Infos[i].NPred, Infos[i].NPredErr, Infos[i].NData, Infos[i].Eff, Infos[i].Significance);
       fprintf(talkFile, "\\hline\n");
     }
   }}
   
   TGraph* graph = NULL;
   if(XSectionType==0)graph = new TGraph(N,Mass,XSecTh);
   if(XSectionType==1)graph = new TGraph(N,Mass,XSecExp);
   if(XSectionType==2)graph = new TGraph(N,Mass,XSecObs);
   graph->SetTitle("");
   graph->GetYaxis()->SetTitle("CrossSection ( pb )");
   graph->GetYaxis()->SetTitleOffset(1.70);
   return graph;
}

stAllInfo Exclusion(string pattern, string modelName, string signal, double Ratio_0C, double Ratio_1C, double Ratio_2C, string syst){
   GetSignalDefinition(signals);
   CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return stAllInfo();  } 


   stAllInfo toReturn;
   toReturn.Mass      = signals[JobIdToIndex(signal)].Mass;
   toReturn.MassMean  = 0;
   toReturn.MassSigma = 0;
   toReturn.MassCut   = 0;
   toReturn.Index     = 0;
   toReturn.WP_Pt     = 0;
   toReturn.WP_I      = 0;
   toReturn.WP_TOF    = 0;
   toReturn.XSec_Th   = signals[JobIdToIndex(signal)].XSec;
   toReturn.XSec_Err  = signals[JobIdToIndex(signal)].XSec * 0.15;
   toReturn.XSec_Exp  = 1E50;
   toReturn.XSec_ExpUp    = 1E50;
   toReturn.XSec_ExpDown  = 1E50;
   toReturn.XSec_Exp2Up   = 1E50;
   toReturn.XSec_Exp2Down = 1E50;
   toReturn.XSec_Obs  = 1E50;
   toReturn.Eff       = 0;
   toReturn.Eff_SYSTP = 0;
   toReturn.Eff_SYSTI = 0;
   toReturn.Eff_SYSTM = 0;
   toReturn.Eff_SYSTT = 0;
   toReturn.NData     = 0;
   toReturn.NPred     = 0;
   toReturn.NPredErr  = 0;
   toReturn.NSign     = 0;



   double RescaleFactor = 1.0;
   double RescaleError  = 0.1;

   double RatioValue[] = {Ratio_0C, Ratio_1C, Ratio_2C};

   double MaxSOverB=-1; 
   int MaxSOverBIndex=-1;

   string outpath = pattern + "/EXCLUSION/";
   if(syst!=""){outpath = pattern + "/EXCLUSION" + syst + "/";}
   MakeDirectories(outpath);

   FILE* pFile = fopen((outpath+"/"+modelName+".info").c_str(),"w");
   if(!pFile)printf("Can't open file : %s\n",(outpath+"/"+modelName+".info").c_str());

   string InputPath     = pattern + "Histos_Data.root";
   TFile* InputFile     = new TFile(InputPath.c_str());

   TH1D*  HCuts_Pt      = (TH1D*)GetObjectFromPath(InputFile, "HCuts_Pt");
   TH1D*  HCuts_I       = (TH1D*)GetObjectFromPath(InputFile, "HCuts_I");
   TH1D*  HCuts_TOF     = (TH1D*)GetObjectFromPath(InputFile, "HCuts_TOF");
   TH1D*  H_A           = (TH1D*)GetObjectFromPath(InputFile, "H_A");
   TH1D*  H_B           = (TH1D*)GetObjectFromPath(InputFile, "H_B");
   TH1D*  H_C           = (TH1D*)GetObjectFromPath(InputFile, "H_C");
   TH1D*  H_D           = (TH1D*)GetObjectFromPath(InputFile, "H_D");
   TH1D*  H_E           = (TH1D*)GetObjectFromPath(InputFile, "H_E");
   TH1D*  H_F           = (TH1D*)GetObjectFromPath(InputFile, "H_F");
   TH1D*  H_G           = (TH1D*)GetObjectFromPath(InputFile, "H_G");
   TH1D*  H_H           = (TH1D*)GetObjectFromPath(InputFile, "H_H");
   TH2D*  MassData      = (TH2D*)GetObjectFromPath(InputFile, "Data/Mass");
   TH2D*  MassPred      = (TH2D*)GetObjectFromPath(InputFile, "Pred_Mass");
   TH2D*  MassSign[4];
   TH2D*  MassSignP[4];
   TH2D*  MassSignI[4];
   TH2D*  MassSignM[4];
   TH2D*  MassSignT[4];
   TH2D*  MassSignPU[4];

   string InputPathSign     = pattern + "Histos.root";
   TFile* InputFileSign     = new TFile(InputPathSign.c_str());

   TH1D* TotalE          = (TH1D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/TotalE" + syst);
   double norm=signals[CurrentSampleIndex].XSec*IntegratedLuminosity/TotalE->Integral();

   MassSign[0]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass" + syst);
   MassSign[1]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass" + syst);
   MassSign[2]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass" + syst);
   MassSign[3]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass" + syst);

   MassSignP[0]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass_SystP");
   MassSignP[1]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass_SystP");
   MassSignP[2]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass_SystP");
   MassSignP[3]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass_SystP");

   MassSignI[0]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass_SystI");
   MassSignI[1]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass_SystI");
   MassSignI[2]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass_SystI");
   MassSignI[3]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass_SystI");

   MassSignM[0]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass_SystM");
   MassSignM[1]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass_SystM");
   MassSignM[2]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass_SystM");
   MassSignM[3]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass_SystM");

   MassSignT[0]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass_SystT");
   MassSignT[1]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass_SystT");
   MassSignT[2]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass_SystT");
   MassSignT[3]         = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass_SystT");

   TH1D* TotalEPU          = (TH1D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/TotalEPU" + syst);
   double normPU=signals[CurrentSampleIndex].XSec*IntegratedLuminosity/TotalEPU->Integral();

   MassSignPU[0]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "/Mass_SystPU" + syst);
   MassSignPU[1]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC0/Mass_SystPU" + syst);
   MassSignPU[2]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC1/Mass_SystPU" + syst);
   MassSignPU[3]          = (TH2D*)GetObjectFromPath(InputFileSign, signals[CurrentSampleIndex].Name + "_NC2/Mass_SystPU" + syst);

   TH1D* MassSignProj[4];
   TH1D* MassSignPProj[4];
   TH1D* MassSignIProj[4];
   TH1D* MassSignMProj[4];
   TH1D* MassSignTProj[4];
   TH1D* MassSignPUProj[4];
   ///##############################################################################"
   MassSignProj[0] = MassSign[0]->ProjectionY("MassSignProj0",1,1);
   double Mean  = MassSignProj[0]->GetMean();
   double Width = MassSignProj[0]->GetRMS();
   MinRange = std::max(0.0, Mean-2*Width);
   MinRange = MassSignProj[0]->GetXaxis()->GetBinLowEdge(MassSignProj[0]->GetXaxis()->FindBin(MinRange)); //Round to a bin value to avoid counting prpoblem due to the binning. 
   delete MassSignProj[0];
   ///##############################################################################"

   //Going to first loop and find the cut with the min S over sqrt(B) because this is quick and normally gives a cut with a reach near the minimum
   stAllInfo CutInfo[MassData->GetNbinsX()];
   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++) CutInfo[CutIndex]=toReturn;

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      if(HCuts_Pt ->GetBinContent(CutIndex+1) < 45 ) continue;  // Be sure the pT cut is high enough to get some statistic for both ABCD and mass shape
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<25 || H_F->GetBinContent(CutIndex+1)<25 || H_G->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (AFG/EE) is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<25 || H_B->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (CB/A) is not reliable
      GetSignalMeanHSCPPerEvent(pattern,CutIndex, MinRange, MaxRange);
      TH1D* MassDataProj = MassData->ProjectionY("MassDataProj",CutIndex+1,CutIndex+1);
      TH1D* MassPredProj = MassPred->ProjectionY("MassPredProj",CutIndex+1,CutIndex+1);
      MassSignProj[0]    = MassSign [0]->ProjectionY("MassSignProj0",CutIndex+1,CutIndex+1); MassSignProj[0]->Scale(norm);
      MassSignProj[1]    = MassSign [1]->ProjectionY("MassSignProj1",CutIndex+1,CutIndex+1); MassSignProj[1]->Scale(norm);
      MassSignProj[2]    = MassSign [2]->ProjectionY("MassSignProj2",CutIndex+1,CutIndex+1); MassSignProj[2]->Scale(norm);
      MassSignProj[3]    = MassSign [3]->ProjectionY("MassSignProj3",CutIndex+1,CutIndex+1); MassSignProj[3]->Scale(norm);

      MassSignPProj[0]   = MassSignP[0]->ProjectionY("MassSignProP0",CutIndex+1,CutIndex+1); MassSignPProj[0]->Scale(norm);
      MassSignPProj[1]   = MassSignP[1]->ProjectionY("MassSignProP1",CutIndex+1,CutIndex+1); MassSignPProj[1]->Scale(norm);
      MassSignPProj[2]   = MassSignP[2]->ProjectionY("MassSignProP2",CutIndex+1,CutIndex+1); MassSignPProj[2]->Scale(norm);
      MassSignPProj[3]   = MassSignP[3]->ProjectionY("MassSignProP3",CutIndex+1,CutIndex+1); MassSignPProj[3]->Scale(norm);
      MassSignIProj[0]   = MassSignI[0]->ProjectionY("MassSignProI0",CutIndex+1,CutIndex+1); MassSignIProj[0]->Scale(norm);
      MassSignIProj[1]   = MassSignI[1]->ProjectionY("MassSignProI1",CutIndex+1,CutIndex+1); MassSignIProj[1]->Scale(norm);
      MassSignIProj[2]   = MassSignI[2]->ProjectionY("MassSignProI2",CutIndex+1,CutIndex+1); MassSignIProj[2]->Scale(norm);
      MassSignIProj[3]   = MassSignI[3]->ProjectionY("MassSignProI3",CutIndex+1,CutIndex+1); MassSignIProj[3]->Scale(norm);
      MassSignMProj[0]   = MassSignM[0]->ProjectionY("MassSignProM0",CutIndex+1,CutIndex+1); MassSignMProj[0]->Scale(norm);
      MassSignMProj[1]   = MassSignM[1]->ProjectionY("MassSignProM1",CutIndex+1,CutIndex+1); MassSignMProj[1]->Scale(norm);
      MassSignMProj[2]   = MassSignM[2]->ProjectionY("MassSignProM2",CutIndex+1,CutIndex+1); MassSignMProj[2]->Scale(norm);
      MassSignMProj[3]   = MassSignM[3]->ProjectionY("MassSignProM3",CutIndex+1,CutIndex+1); MassSignMProj[3]->Scale(norm);
      MassSignTProj[0]   = MassSignT[0]->ProjectionY("MassSignProT0",CutIndex+1,CutIndex+1); MassSignTProj[0]->Scale(norm);
      MassSignTProj[1]   = MassSignT[1]->ProjectionY("MassSignProT1",CutIndex+1,CutIndex+1); MassSignTProj[1]->Scale(norm);
      MassSignTProj[2]   = MassSignT[2]->ProjectionY("MassSignProT2",CutIndex+1,CutIndex+1); MassSignTProj[2]->Scale(norm);
      MassSignTProj[3]   = MassSignT[3]->ProjectionY("MassSignProT3",CutIndex+1,CutIndex+1); MassSignTProj[3]->Scale(norm);
      MassSignPUProj[0]   = MassSignPU[0]->ProjectionY("MassSignProPU0",CutIndex+1,CutIndex+1); MassSignPUProj[0]->Scale(normPU);
      MassSignPUProj[1]   = MassSignPU[1]->ProjectionY("MassSignProPU1",CutIndex+1,CutIndex+1); MassSignPUProj[1]->Scale(normPU);
      MassSignPUProj[2]   = MassSignPU[2]->ProjectionY("MassSignProPU2",CutIndex+1,CutIndex+1); MassSignPUProj[2]->Scale(normPU);
      MassSignPUProj[3]   = MassSignPU[3]->ProjectionY("MassSignProPU3",CutIndex+1,CutIndex+1); MassSignPUProj[3]->Scale(normPU);

      double NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
      double NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));
      double NPredErr    = pow(NPred*RescaleError,2);
      for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=pow(MassPredProj->GetBinError(i),2);}NPredErr=sqrt(NPredErr);

      if(isnan((float)NPred))continue;
      if(NPred<=0){continue;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
//    if(NPred<1E-4){continue;} //This will never be the selection which gives the best expected limit (cutting too much on signal) --> Slowdown computation for nothing...
      if(NPred>1000){continue;}  //When NPred is too big, expected limits just take an infinite time! 

      double Eff       = 0;
      double EffP      = 0;
      double EffI      = 0;
      double EffM      = 0;
      double EffT      = 0;
      double EffPU      = 0;
      if(RatioValue[0]<0 && RatioValue[1]<0 && RatioValue[2]<0){
            CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  } 
            double INTERN_ESign       = MassSignProj[0]->Integral(MassSignProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0]; 
            double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            Eff                       = INTERN_Eff;
            //fprintf(pFile  ,"%10s: INTERN_ESign=%6.2E   INTERN_Eff=%6.E   XSec=%6.2E   Lumi=%6.2E\n",signal.c_str(),INTERN_ESign,INTERN_Eff,signals[CurrentSampleIndex].XSec, IntegratedLuminosity);fflush(stdout);

            double INTERN_ESignP      = MassSignPProj[0]->Integral(MassSignPProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignPProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0];
            double INTERN_EffP        = INTERN_ESignP      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffP                      = INTERN_EffP;

            double INTERN_ESignI      = MassSignIProj[0]->Integral(MassSignIProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignIProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0];
            double INTERN_EffI        = INTERN_ESignI      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffI                      = INTERN_EffI;

            double INTERN_ESignM      = MassSignMProj[0]->Integral(MassSignMProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignMProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0];
            double INTERN_EffM        = INTERN_ESignM      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffM                      = INTERN_EffM;

            double INTERN_ESignT      = MassSignTProj[0]->Integral(MassSignTProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignTProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0];
            double INTERN_EffT        = INTERN_ESignT      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffT                      = INTERN_EffT;

            double INTERN_ESignPU      = MassSignPUProj[0]->Integral(MassSignPUProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignPUProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [0];
            double INTERN_EffPU        = INTERN_ESignPU      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffPU                      = INTERN_EffPU;
      }else{
         for(unsigned int i=0;i<3;i++){
            CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  }
            double INTERN_ESign       = MassSignProj[i+1]->Integral(MassSignProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [1+i]; 
            double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            Eff                      += INTERN_Eff   * RatioValue[i];

            double INTERN_ESignP      = MassSignPProj[i+1]->Integral(MassSignPProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignPProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [1+i];
            double INTERN_EffP        = INTERN_ESignP      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffP                     += INTERN_EffP  * RatioValue[i];

            double INTERN_ESignI      = MassSignIProj[i+1]->Integral(MassSignIProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignIProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [1+i];
            double INTERN_EffI        = INTERN_ESignI      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffI                     += INTERN_EffI  * RatioValue[i];

            double INTERN_ESignM      = MassSignMProj[i+1]->Integral(MassSignMProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignMProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [1+i];
            double INTERN_EffM        = INTERN_ESignM      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffM                     += INTERN_EffM  * RatioValue[i];

            double INTERN_ESignT      = MassSignTProj[i+1]->Integral(MassSignTProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignTProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [1+i];
            double INTERN_EffT        = INTERN_ESignT      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffT                     += INTERN_EffT  * RatioValue[i];

            double INTERN_ESignPU      = MassSignPUProj[i+1]->Integral(MassSignPUProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignPUProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [i+1];
            double INTERN_EffPU        = INTERN_ESignPU      / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            EffPU                     += INTERN_EffPU  * RatioValue[i];
         }
      }
      if(Eff==0)continue;
      NPred*=RescaleFactor;

     

     //fprintf(pFile ,"CutIndex=%4i ManHSCPPerEvents = %6.2f %6.2f %6.2f %6.2f   NTracks = %6.3f %6.3f %6.3f %6.3f\n",CutIndex,signalsMeanHSCPPerEvent[0], signalsMeanHSCPPerEvent[1],signalsMeanHSCPPerEvent[2],signalsMeanHSCPPerEvent[3], MassSignProj[0]->Integral(), MassSignProj[1]->Integral(), MassSignProj[2]->Integral(), MassSignProj[3]->Integral());

      fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f\n",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);
      fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f\n",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);

      if(Eff/sqrt(max(0.1, NPred))>MaxSOverB) {MaxSOverB=Eff/sqrt(max(0.1, NPred)); MaxSOverBIndex=CutIndex;}

     toReturn.MassMean  = Mean;
     toReturn.MassSigma = Width;
     toReturn.MassCut   = MinRange;
     toReturn.Index     = CutIndex;
     toReturn.WP_Pt     = HCuts_Pt ->GetBinContent(CutIndex+1);
     toReturn.WP_I      = HCuts_I  ->GetBinContent(CutIndex+1);
     toReturn.WP_TOF    = HCuts_TOF->GetBinContent(CutIndex+1);
     toReturn.XSec_Th   = signals[JobIdToIndex(signal)].XSec;
     toReturn.XSec_Err  = signals[JobIdToIndex(signal)].XSec * 0.15;
     toReturn.Eff       = Eff;
     toReturn.Eff_SYSTP = EffP;
     toReturn.Eff_SYSTI = EffI;
     toReturn.Eff_SYSTM = EffM;
     toReturn.Eff_SYSTT = EffT;
     toReturn.Eff_SYSTPU= EffPU;
     toReturn.NData     = NData;
     toReturn.NPred     = NPred;
     toReturn.NPredErr  = NPredErr;
     toReturn.NSign     = Eff*(signals[CurrentSampleIndex].XSec*IntegratedLuminosity);

     CutInfo[CutIndex]=toReturn;
   }

   fclose(pFile);   

   //Find reach for point with best S Over sqrt(B) first.
   double NPredSB=CutInfo[MaxSOverBIndex].NPred;
   double NPredErrSB=CutInfo[MaxSOverBIndex].NPredErr;
   double EffSB=CutInfo[MaxSOverBIndex].Eff;

   double FiveSigma=1E50;
   for (int n_obs=5; n_obs<1000; n_obs++) {
     if(nSigma(NPredSB, n_obs, NPredErrSB/NPredSB)>=5) {
       FiveSigma=n_obs;
       break;
     }
   }

   double MinReach=(FiveSigma-NPredSB)/(EffSB*IntegratedLuminosity);
   toReturn=CutInfo[MaxSOverBIndex]; // In case this point does give the best reach avoids rounding errors

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
     double NPred=CutInfo[CutIndex].NPred;
     double NPredErr=CutInfo[CutIndex].NPredErr;
     double Eff=CutInfo[CutIndex].Eff;
     if(Eff==0) continue;  //Eliminate points where prediction could not be made
     double FiveSigma=1E50;
     for (int n_obs=5; n_obs<1000; n_obs++) {
       if(n_obs<(NPred+3*sqrt(NPred))) continue;    //5 sigma implies more than 5 times sqrt(B) excess so can cut these points, put it at 3 to be safe
       double thisReach=(n_obs-NPred)/(Eff*IntegratedLuminosity);
       if(thisReach>=MinReach) break;    // This selection point will not give the optimum reach so move on
       if(nSigma(NPred, n_obs, NPredErr/NPred)>=5) {
	 FiveSigma=n_obs;
	 break;
       }
     }

     double Reach=(FiveSigma-NPred)/(Eff*IntegratedLuminosity);
     if(Reach>MinReach) continue;
     MinReach=Reach;
     toReturn=CutInfo[CutIndex];
   }

   LimitResult CLMResults;
   double signalUncertainty=0.07;
   double NPred=toReturn.NPred;
   double NPredErr=toReturn.NPredErr;
   double Eff=toReturn.Eff;
   double NData=toReturn.NData;

   CLMResults =  roostats_limit(IntegratedLuminosity, IntegratedLuminosity*0.022, Eff, Eff*signalUncertainty,NPred, NPredErr, NData, false, 1, "cls", "", 12345);

   double ExpLimit=CLMResults.GetExpectedLimit();
   double ExpLimitup    = CLMResults.GetOneSigmaHighRange();
   double ExpLimitdown  = CLMResults.GetOneSigmaLowRange();
   double ExpLimit2up   = CLMResults.GetTwoSigmaHighRange();
   double ExpLimit2down = CLMResults.GetTwoSigmaLowRange();
   double ObsLimit = CLMResults.GetObservedLimit();

   toReturn.XSec_Exp  = CLMResults.GetExpectedLimit();
   toReturn.XSec_ExpUp    = CLMResults.GetOneSigmaHighRange();
   toReturn.XSec_ExpDown  = CLMResults.GetOneSigmaLowRange();
   toReturn.XSec_Exp2Up   = CLMResults.GetTwoSigmaHighRange();
   toReturn.XSec_Exp2Down = CLMResults.GetTwoSigmaLowRange();
   toReturn.XSec_Obs  = CLMResults.GetObservedLimit();
   toReturn.Significance = nSigma(NPred, NData, NPredErr/NPred);

     FILE* pFile2 = fopen((outpath+"/"+modelName+".txt").c_str(),"w");
     if(!pFile2)printf("Can't open file : %s\n",(outpath+"/"+modelName+".txt").c_str());
     fprintf(pFile2,"Mass         : %f\n",signals[JobIdToIndex(signal)].Mass);
     fprintf(pFile2,"MassMean     : %f\n",toReturn.MassMean);
     fprintf(pFile2,"MassSigma    : %f\n",toReturn.MassSigma);
     fprintf(pFile2,"MassCut      : %f\n",toReturn.MassCut);
     fprintf(pFile2,"Index        : %f\n",toReturn.Index);
     fprintf(pFile2,"WP_Pt        : %f\n",toReturn.WP_Pt);
     fprintf(pFile2,"WP_I         : %f\n",toReturn.WP_I);
     fprintf(pFile2,"WP_TOF       : %f\n",toReturn.WP_TOF);
     fprintf(pFile2,"Eff          : %f\n",toReturn.Eff);
     fprintf(pFile2,"Eff_SystP    : %f\n",toReturn.Eff_SYSTP);
     fprintf(pFile2,"Eff_SystI    : %f\n",toReturn.Eff_SYSTI);
     fprintf(pFile2,"Eff_SystM    : %f\n",toReturn.Eff_SYSTM);
     fprintf(pFile2,"Eff_SystT    : %f\n",toReturn.Eff_SYSTT);
     fprintf(pFile2,"Eff_SystPU   : %f\n",toReturn.Eff_SYSTPU);
     fprintf(pFile2,"Signif       : %f\n",toReturn.Significance);
     fprintf(pFile2,"XSec_Th      : %f\n",toReturn.XSec_Th);
     fprintf(pFile2,"XSec_Exp     : %f\n",toReturn.XSec_Exp);
     fprintf(pFile2,"XSec_ExpUp   : %f\n",toReturn.XSec_ExpUp);
     fprintf(pFile2,"XSec_ExpDown : %f\n",toReturn.XSec_ExpDown);
     fprintf(pFile2,"XSec_Exp2Up  : %f\n",toReturn.XSec_Exp2Up);
     fprintf(pFile2,"XSec_Exp2Down: %f\n",toReturn.XSec_Exp2Down);
     fprintf(pFile2,"XSec_Obs     : %f\n",toReturn.XSec_Obs);     
     fprintf(pFile2,"NData        : %+6.2E\n",toReturn.NData);
     fprintf(pFile2,"NPred        : %+6.2E\n",toReturn.NPred);
     fprintf(pFile2,"NPredErr     : %+6.2E\n",toReturn.NPredErr);
     fprintf(pFile2,"NSign        : %+6.2E\n",toReturn.NSign);

     fclose(pFile2);
 
   return toReturn;
}






int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}


void GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange)
{

   string InputPath     = InputPattern + "Histos.root";
   TFile* InputFile     = new TFile(InputPath.c_str());

   signalsMeanHSCPPerEvent.clear();
   signalsMeanHSCPPerEvent_SYSTP.clear();
   signalsMeanHSCPPerEvent_SYSTI.clear();
   signalsMeanHSCPPerEvent_SYSTM.clear();
   signalsMeanHSCPPerEvent_SYSTT.clear();

   for(unsigned int n=0;n<4;n++){
      signalsMeanHSCPPerEvent.push_back(2.0);
      signalsMeanHSCPPerEvent_SYSTP.push_back(2.0);
      signalsMeanHSCPPerEvent_SYSTI.push_back(2.0); 
      signalsMeanHSCPPerEvent_SYSTM.push_back(2.0);
      signalsMeanHSCPPerEvent_SYSTT.push_back(2.0);
   }

   TH2D*  Mass     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "/Mass");
   TH2D*  MaxEventMass     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "/MaxEventMass");
   TH1D*  NTracksPassingSelection     = Mass->ProjectionY("NTracksPassingSelection",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection     = MaxEventMass->ProjectionY("NEventsPassingSelection",CutIndex+1,CutIndex+1);

   TH2D*  Mass_NC0     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC0/Mass");
   TH2D*  MaxEventMass_NC0     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC0/MaxEventMass");
   TH1D*  NTracksPassingSelection_NC0     = Mass_NC0->ProjectionY("NTracksPassingSelection_NC0",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection_NC0     = MaxEventMass_NC0->ProjectionY("NEventsPassingSelection_NC0",CutIndex+1,CutIndex+1);

   TH2D*  Mass_NC1     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC1/Mass");
   TH2D*  MaxEventMass_NC1     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC1/MaxEventMass");
   TH1D*  NTracksPassingSelection_NC1     = Mass_NC1->ProjectionY("NTracksPassingSelection_NC1",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection_NC1     = MaxEventMass_NC1->ProjectionY("NEventsPassingSelection_NC1",CutIndex+1,CutIndex+1);

   TH2D*  Mass_NC2     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC2/Mass");
   TH2D*  MaxEventMass_NC2     = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "_NC2/MaxEventMass");
   TH1D*  NTracksPassingSelection_NC2     = Mass_NC2->ProjectionY("NTracksPassingSelection_NC2",CutIndex+1,CutIndex+1);
   TH1D*  NEventsPassingSelection_NC2     = MaxEventMass_NC2->ProjectionY("NEventsPassingSelection_NC2",CutIndex+1,CutIndex+1);

   double NTracks       = NTracksPassingSelection->Integral(NTracksPassingSelection->GetXaxis()->FindBin(MinRange), NTracksPassingSelection->GetXaxis()->FindBin(MaxRange));
   double NEvents       = NEventsPassingSelection->Integral(NEventsPassingSelection->GetXaxis()->FindBin(MinRange), NEventsPassingSelection->GetXaxis()->FindBin(MaxRange));
   double NTracks_NC0   = NTracksPassingSelection_NC0->Integral(NTracksPassingSelection_NC0->GetXaxis()->FindBin(MinRange), NTracksPassingSelection_NC0->GetXaxis()->FindBin(MaxRange));
   double NEvents_NC0   = NEventsPassingSelection_NC0->Integral(NEventsPassingSelection_NC0->GetXaxis()->FindBin(MinRange), NEventsPassingSelection_NC0->GetXaxis()->FindBin(MaxRange));

   double NTracks_NC1   = NTracksPassingSelection_NC1->Integral(NTracksPassingSelection_NC1->GetXaxis()->FindBin(MinRange), NTracksPassingSelection_NC1->GetXaxis()->FindBin(MaxRange));
   double NEvents_NC1   = NEventsPassingSelection_NC1->Integral(NEventsPassingSelection_NC1->GetXaxis()->FindBin(MinRange), NEventsPassingSelection_NC1->GetXaxis()->FindBin(MaxRange));

   double NTracks_NC2   = NTracksPassingSelection_NC2->Integral(NTracksPassingSelection_NC2->GetXaxis()->FindBin(MinRange), NTracksPassingSelection_NC2->GetXaxis()->FindBin(MaxRange));
   double NEvents_NC2   = NEventsPassingSelection_NC2->Integral(NEventsPassingSelection_NC2->GetXaxis()->FindBin(MinRange), NEventsPassingSelection_NC2->GetXaxis()->FindBin(MaxRange));

   signalsMeanHSCPPerEvent[0] = (float)std::max(1.0,NTracks/ NEvents);
   signalsMeanHSCPPerEvent[1] = (float)std::max(1.0,NTracks_NC0/ NEvents_NC0);
   signalsMeanHSCPPerEvent[2] = (float)std::max(1.0,NTracks_NC1/ NEvents_NC1);
   signalsMeanHSCPPerEvent[3] = (float)std::max(1.0,NTracks_NC2/ NEvents_NC2);

   delete Mass;
   delete MaxEventMass;
   delete Mass_NC0;
   delete MaxEventMass_NC0;
   delete Mass_NC1;
   delete MaxEventMass_NC1;
   delete Mass_NC2;
   delete MaxEventMass_NC2;
   delete NTracksPassingSelection;
   delete NEventsPassingSelection;
   delete NTracksPassingSelection_NC0;
   delete NEventsPassingSelection_NC0;
   delete NTracksPassingSelection_NC1;
   delete NEventsPassingSelection_NC1;
   delete NTracksPassingSelection_NC2;
   delete NEventsPassingSelection_NC2;

   delete InputFile;
   return;
}

double FindIntersection(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty, bool debug){

   double Intersection = -1;

   double ThShift = 1.0-ThUncertainty;
   double PreviousX = Min;
   double PreviousV = obs->Eval(PreviousX, 0, "") - (ThShift * th->Eval(PreviousX, 0, "")) ;
   if(PreviousV>0)return -1;
   for(double x=Min+=Step;x<Max;x+=Step){                 
      double V = obs->Eval(x, 0, "") - (ThShift * th->Eval(x, 0, "") );
      if(debug){
         printf("%7.2f --> Obs=%6.2E  Th=%6.2E",x,obs->Eval(x, 0, ""),ThShift * th->Eval(x, 0, ""));
         if(V>=0)printf("   X\n");
         else printf("\n");
      }
      if(V<0){
         PreviousX = x;
         PreviousV = V;
      }else{
         Intersection = PreviousX;
      }
   }
   return Intersection;
}



int ReadXSection(string InputFile, double* Mass, double* XSec, double* Low, double* High, double* ErrLow, double* ErrHigh)
{
   FILE* pFile = fopen(InputFile.c_str(),"r");
   if(!pFile){ 
      printf("Not Found: %s\n",InputFile.c_str());
      return -1;
   }

   float tmpM, tmpX, tmpL, tmpH;
   
   int NPoints = 0;
   while ( ! feof (pFile) ){
     fscanf(pFile,"%f %E %E %E\n",&tmpM,&tmpX,&tmpH,&tmpL);
     Mass   [NPoints] = tmpM;
     XSec   [NPoints] = tmpX;
     Low    [NPoints] = tmpL;
     High   [NPoints] = tmpH;
     ErrLow [NPoints] = tmpX-tmpL;
     ErrHigh[NPoints] = tmpH-tmpX;
     NPoints++;

     //printf("%fGeV --> Error = %f\n", tmpM, 0.5*(tmpH-tmpL)/tmpX);
   }

   fclose(pFile);

   return NPoints;
}


TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High, double MinLow, double MaxHigh)
{
   TCutG* cutg = new TCutG(name.c_str(),2*N);
   cutg->SetFillColor(kGreen-7);
   for(int i=0;i<N;i++){
      double Min = std::max(Low[i],MinLow);
      cutg->SetPoint( i,Mass[i], Min);
   }
   for(int i=0;i<N;i++){
      double Max = std::min(High[N-1-i],MaxHigh);
      cutg->SetPoint(N+i,Mass[N-1-i], Max);
   }
   return cutg;
}

void DrawModelLimitWithBand(string InputPattern, string inputmodel)
{
   std::vector<string> Models;
   string modelname;
   if(inputmodel == "Gluinof1"){
      Models.push_back("Gluino300_f1");
      Models.push_back("Gluino400_f1");
      Models.push_back("Gluino500_f1");
      Models.push_back("Gluino600_f1");
      Models.push_back("Gluino700_f1");
      Models.push_back("Gluino800_f1");
      Models.push_back("Gluino900_f1");
      Models.push_back("Gluino1000_f1");
      modelname="gluino; 10% #tilde{g}g (NLO+NLL)";
   }
   else if(inputmodel == "Gluinof5"){
      Models.push_back("Gluino300_f5");
      Models.push_back("Gluino400_f5");
      Models.push_back("Gluino500_f5");
      Models.push_back("Gluino600_f5");
      Models.push_back("Gluino700_f5");
      Models.push_back("Gluino800_f5");
      Models.push_back("Gluino900_f5");
      Models.push_back("Gluino1000_f5");
      modelname="gluino; 50% #tilde{g}g (NLO+NLL)";
   }
   else if(inputmodel == "GluinoN"){
      Models.push_back("Gluino300N_f1");
      Models.push_back("Gluino400N_f1");
      Models.push_back("Gluino500N_f1");
      Models.push_back("Gluino600N_f1");
      Models.push_back("Gluino700N_f1");
      Models.push_back("Gluino800N_f1");
      Models.push_back("Gluino900N_f1");
      Models.push_back("Gluino1000N_f1");
      modelname="gluino; 10% #tilde{g}g; ch. suppr.(NLO+NLL)";

   }
   else if(inputmodel == "Stop"){
      Models.push_back("Stop130");
      Models.push_back("Stop200");
      Models.push_back("Stop300");
      Models.push_back("Stop400");
      Models.push_back("Stop500");
      Models.push_back("Stop600");
      Models.push_back("Stop700");
      Models.push_back("Stop800");
      modelname="stop (NLO+NLL)";
   }
   else if(inputmodel == "StopN"){
      Models.push_back("Stop130N");
      Models.push_back("Stop200N");
      Models.push_back("Stop300N");
      Models.push_back("Stop400N");
      Models.push_back("Stop500N");
      Models.push_back("Stop600N");
      Models.push_back("Stop700N");
      Models.push_back("Stop800N");
      modelname="stop;ch. suppr. (NLO+NLL)";
  }
   else if(inputmodel == "GMStau"){
      Models.push_back("GMStau100");
      Models.push_back("GMStau126");
      Models.push_back("GMStau156");
      Models.push_back("GMStau200");
      Models.push_back("GMStau247");
      Models.push_back("GMStau308");
      Models.push_back("GMStau370"); 
      Models.push_back("GMStau432"); 
      Models.push_back("GMStau494");
      modelname="GMSB stau (NLO)";
  }
   else if(inputmodel == "PPStau"){
      Models.push_back("PPStau100");
      Models.push_back("PPStau126"); 
      Models.push_back("PPStau156"); 
      Models.push_back("PPStau200"); 
      Models.push_back("PPStau247");
      Models.push_back("PPStau308");
      modelname="Pair Prod. stau (NLO)";
   }
   else if(inputmodel == "DCRho08"){
      Models.push_back("DCRho08HyperK100");
      Models.push_back("DCRho08HyperK121"); 
      Models.push_back("DCRho08HyperK182"); 
      Models.push_back("DCRho08HyperK242"); 
      Models.push_back("DCRho08HyperK302");  
      Models.push_back("DCRho08HyperK350");
      Models.push_back("DCRho08HyperK370");
      Models.push_back("DCRho08HyperK390");  
      Models.push_back("DCRho08HyperK395"); 
      Models.push_back("DCRho08HyperK400");
      Models.push_back("DCRho08HyperK410");
      Models.push_back("DCRho08HyperK420");
      Models.push_back("DCRho08HyperK500");
      modelname="Hyper-K, #tilde{#rho} = 0.8 TeV (LO)";
   }
   else if(inputmodel == "DCRho12"){
      Models.push_back("DCRho12HyperK100"); 
      Models.push_back("DCRho12HyperK182");
      Models.push_back("DCRho12HyperK302");
      Models.push_back("DCRho12HyperK500"); 
      Models.push_back("DCRho12HyperK530"); 
      Models.push_back("DCRho12HyperK570");
      Models.push_back("DCRho12HyperK590"); 
      Models.push_back("DCRho12HyperK595");
      Models.push_back("DCRho12HyperK600");
      Models.push_back("DCRho12HyperK610");
      Models.push_back("DCRho12HyperK620");
      Models.push_back("DCRho12HyperK700");
      modelname="Hyper-K, #tilde{#rho} = 1.2 TeV (LO)";
   }
   else if(inputmodel == "DCRho16"){
      Models.push_back("DCRho16HyperK100");
      Models.push_back("DCRho16HyperK182"); 
      Models.push_back("DCRho16HyperK302");
      Models.push_back("DCRho16HyperK500");
      Models.push_back("DCRho16HyperK700"); 
      Models.push_back("DCRho16HyperK730"); 
      Models.push_back("DCRho16HyperK770");
      Models.push_back("DCRho16HyperK790");
      Models.push_back("DCRho16HyperK795");
      Models.push_back("DCRho16HyperK800");
      Models.push_back("DCRho16HyperK820");
      Models.push_back("DCRho16HyperK900");
      modelname="Hyper-K, #tilde{#rho} = 1.6 TeV (LO)";
   }
   else{cout<<"no model specified"<<endl;}

   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   string prefix = "Mu"; 
   if(IsTkOnly) prefix ="Tk";


   unsigned int N = Models.size();
   stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
   for(int i=0;i<N;i++){
      Infos = stAllInfo(InputPattern+"EXCLUSION/" + Models[i] +".txt");
      Mass[i]=Infos.Mass;
      XSecTh [i]=Infos.XSec_Th;
      XSecObs[i]=Infos.XSec_Obs;
      XSecExp[i]=Infos.XSec_Exp;
      XSecExpUp[i]=Infos.XSec_ExpUp;
      XSecExpDown[i]=Infos.XSec_ExpDown;
      XSecExp2Up[i]=Infos.XSec_Exp2Up;
      XSecExp2Down[i]=Infos.XSec_Exp2Down;
   }

   TGraph* graphtheory = new TGraph(N,Mass,XSecTh);
   TGraph* graphobs = new TGraph(N,Mass,XSecObs);
   TGraph* graphexp = new TGraph(N,Mass,XSecExp);
   TCutG*  ExpErr = GetErrorBand("ExpErr",N,Mass,XSecExpDown,XSecExpUp);
   TCutG*  Exp2SigmaErr = GetErrorBand("Exp2SigmaErr",N,Mass,XSecExp2Down,XSecExp2Up);

   graphtheory->SetLineStyle(3);
   graphtheory->SetFillColor(kBlue);
   graphexp->SetLineStyle(4); 
   graphexp->SetLineColor(kRed);
   graphexp->SetMarkerStyle(); 
   graphexp->SetMarkerSize(0.); 
   Exp2SigmaErr->SetFillColor(kYellow);
   Exp2SigmaErr->SetLineColor(kWhite);
   ExpErr->SetFillColor(kGreen);
   ExpErr->SetLineColor(kWhite);
   graphobs->SetLineColor(kBlack);
   graphobs->SetLineWidth(2);
   graphobs->SetMarkerColor(kBlack);
   graphobs->SetMarkerStyle(23);

   TCanvas* c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MG = new TMultiGraph();

   MG->Add(graphexp      ,"LP");
   MG->Add(graphobs      ,"LP");
   MG->Add(graphtheory      ,"L");
   MG->Draw("A");
   Exp2SigmaErr->Draw("f");
   ExpErr  ->Draw("f");
   MG->Draw("same");
   MG->SetTitle("");
   MG->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
   MG->GetYaxis()->SetTitle("#sigma (pb)");
   MG->GetYaxis()->SetTitleOffset(1.70);
   MG->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEG = new TLegend(0.40,0.65,0.8,0.90);
   string headerstr;
   headerstr = "95% CL Limits (Tracker + TOF)";
   if(IsTkOnly) headerstr = "95% CL Limits (Tracker - Only)";
   LEG->SetHeader(headerstr.c_str());
   LEG->SetFillColor(0); 
   LEG->SetBorderSize(0);
   LEG->AddEntry(graphtheory,  modelname.c_str() ,"L");
   LEG->AddEntry(graphexp, "Expected"       ,"L");
   LEG->AddEntry(ExpErr, "Expected #pm 1#sigma","F");
   LEG->AddEntry(Exp2SigmaErr, "Expected #pm 2#sigma "       ,"F");
   LEG->AddEntry(graphobs, "Observed"       ,"LP");
   LEG->Draw();

   c1->SetLogy(true);


   if(IsTkOnly)   SaveCanvas(c1,"Results/EXCLUSION/", string("Tk"+ inputmodel + "ExclusionLog"));
   else    SaveCanvas(c1,"Results/EXCLUSION/", string("Mu"+ inputmodel + "ExclusionLog"));
   delete c1;


}
std::vector<string> GetModels(string inputmodel)
{
   std::vector<string> Models;
   string modelname;
   if(inputmodel == "Gluinof1"){
      Models.push_back("Gluino300_f1");
      Models.push_back("Gluino400_f1");
      Models.push_back("Gluino500_f1");
      Models.push_back("Gluino600_f1");
      Models.push_back("Gluino700_f1");
      Models.push_back("Gluino800_f1");
      Models.push_back("Gluino900_f1");
      Models.push_back("Gluino1000_f1");
      Models.push_back("Gluino1100_f1");
      Models.push_back("Gluino1200_f1");
      modelname="gluino; 10% #tilde{g}g (NLO+NLL)";
   }
   else if(inputmodel == "Gluinof5"){
      Models.push_back("Gluino300_f5");
      Models.push_back("Gluino400_f5");
      Models.push_back("Gluino500_f5");
      Models.push_back("Gluino600_f5");
      Models.push_back("Gluino700_f5");
      Models.push_back("Gluino800_f5");
      Models.push_back("Gluino900_f5");
      Models.push_back("Gluino1000_f5");
      Models.push_back("Gluino1100_f5");
      Models.push_back("Gluino1200_f5");
     modelname="gluino; 50% #tilde{g}g (NLO+NLL)";
   }
   else if(inputmodel == "GluinoN"){
      Models.push_back("Gluino300N_f1");
      Models.push_back("Gluino400N_f1");
      Models.push_back("Gluino500N_f1");
      Models.push_back("Gluino600N_f1");
      Models.push_back("Gluino700N_f1");
      Models.push_back("Gluino800N_f1");
      Models.push_back("Gluino900N_f1");
      Models.push_back("Gluino1000N_f1");
      Models.push_back("Gluino1100N_f1");
      Models.push_back("Gluino1200N_f1");
      modelname="gluino; 10% #tilde{g}g; ch. suppr.(NLO+NLL)";
   }
   else if(inputmodel == "Stop"){
      Models.push_back("Stop130");
      Models.push_back("Stop200");
      Models.push_back("Stop300");
      Models.push_back("Stop400");
      Models.push_back("Stop500");
      Models.push_back("Stop600");
      Models.push_back("Stop700");
      Models.push_back("Stop800");
      modelname="stop (NLO+NLL)";
   }
   else if(inputmodel == "StopN"){
      Models.push_back("Stop130N");
      Models.push_back("Stop200N");
      Models.push_back("Stop300N");
      Models.push_back("Stop400N");
      Models.push_back("Stop500N");
      Models.push_back("Stop600N");
      Models.push_back("Stop700N");
      Models.push_back("Stop800N");
      modelname="stop;ch. suppr. (NLO+NLL)";
  }
   else if(inputmodel == "GMStau"){
      Models.push_back("GMStau100");
      Models.push_back("GMStau126");
      Models.push_back("GMStau156");
      Models.push_back("GMStau200");
      Models.push_back("GMStau247");
      Models.push_back("GMStau308");
      Models.push_back("GMStau370"); 
      Models.push_back("GMStau432"); 
      Models.push_back("GMStau494");
      modelname="GMSB stau (NLO)";
  }
   else if(inputmodel == "PPStau"){
      Models.push_back("PPStau100");
      Models.push_back("PPStau126"); 
      Models.push_back("PPStau156"); 
      Models.push_back("PPStau200"); 
      Models.push_back("PPStau247");
      Models.push_back("PPStau308");
      modelname="Pair Prod. stau (NLO)";
   }
   else if(inputmodel == "DCRho08"){
      Models.push_back("DCRho08HyperK100");
      Models.push_back("DCRho08HyperK121"); 
      Models.push_back("DCRho08HyperK182"); 
      Models.push_back("DCRho08HyperK242"); 
      Models.push_back("DCRho08HyperK302");  
      Models.push_back("DCRho08HyperK350");
      Models.push_back("DCRho08HyperK370");
      Models.push_back("DCRho08HyperK390");  
      Models.push_back("DCRho08HyperK395"); 
      Models.push_back("DCRho08HyperK400");
      Models.push_back("DCRho08HyperK410");
      Models.push_back("DCRho08HyperK420");
      Models.push_back("DCRho08HyperK500");
      modelname="Hyper-K, #tilde{#rho} = 0.8 TeV (LO)";
   }
   else if(inputmodel == "DCRho12"){
      Models.push_back("DCRho12HyperK100"); 
      Models.push_back("DCRho12HyperK182");
      Models.push_back("DCRho12HyperK302");
      Models.push_back("DCRho12HyperK500"); 
      Models.push_back("DCRho12HyperK530"); 
      Models.push_back("DCRho12HyperK570");
      Models.push_back("DCRho12HyperK590"); 
      Models.push_back("DCRho12HyperK595");
      Models.push_back("DCRho12HyperK600");
      Models.push_back("DCRho12HyperK610");
      Models.push_back("DCRho12HyperK620");
      Models.push_back("DCRho12HyperK700");
      modelname="Hyper-K, #tilde{#rho} = 1.2 TeV (LO)";
   }
   else if(inputmodel == "DCRho16"){
      Models.push_back("DCRho16HyperK100");
      Models.push_back("DCRho16HyperK182"); 
      Models.push_back("DCRho16HyperK302");
      Models.push_back("DCRho16HyperK500");
      Models.push_back("DCRho16HyperK700"); 
      Models.push_back("DCRho16HyperK730"); 
      Models.push_back("DCRho16HyperK770");
      Models.push_back("DCRho16HyperK790");
      Models.push_back("DCRho16HyperK795");
      Models.push_back("DCRho16HyperK800");
      Models.push_back("DCRho16HyperK820");
      Models.push_back("DCRho16HyperK900");
      modelname="Hyper-K, #tilde{#rho} = 1.6 TeV (LO)";
   }
   else{cout<<"no model specified"<<endl;}
   return Models;

}
string GetModelName(string inputmodel)
{
   string modelname;
   if(inputmodel == "Gluinof1"){
      modelname="gluino; 10% #tilde{g}g";
   }
   else if(inputmodel == "Gluinof5"){
     modelname="gluino; 50% #tilde{g}g";
   }
   else if(inputmodel == "GluinoN"){
      modelname="gluino; 10% #tilde{g}g; ch. suppr.";
   }
   else if(inputmodel == "Stop"){
      modelname="stop";
   }
   else if(inputmodel == "StopN"){
      modelname="stop;ch. suppr.";
  }
   else if(inputmodel == "GMStau"){
      modelname="GMSB stau";
  }
   else if(inputmodel == "PPStau"){
      modelname="Pair Prod. stau";
   }
   else if(inputmodel == "DCRho08"){
      modelname="Hyper-K, #tilde{#rho} = 0.8 TeV";
   }
   else if(inputmodel == "DCRho12"){
      modelname="Hyper-K, #tilde{#rho} = 1.2 TeV";
   }
   else if(inputmodel == "DCRho16"){
      modelname="Hyper-K, #tilde{#rho} = 1.6 TeV";
   }
   else{cout<<"no model specified"<<endl;}
   return modelname;


}


void DrawRatioBands(string InputPattern, string inputmodel)
{
   bool IsTkOnly = (InputPattern.find("Type0",0)<std::string::npos);
   string prefix = "Mu"; 
   if(IsTkOnly) prefix ="Tk";

   std::vector<string> TModels;
   std::vector<bool> isNeutral;
   if(inputmodel == "Gluino"){
     TModels.push_back("Gluinof1");
     isNeutral.push_back(false);
     TModels.push_back("Gluinof5");
     isNeutral.push_back(false);
     TModels.push_back("GluinoN");
     isNeutral.push_back(true);
   }

   else if(inputmodel == "Stop"){
     TModels.push_back("Stop");
     isNeutral.push_back(false);
     TModels.push_back("StopN");
     isNeutral.push_back(true);
   }

   else if(inputmodel == "Stau"){
      TModels.push_back("GMStau");
      isNeutral.push_back(false);
      TModels.push_back("PPStau");
      isNeutral.push_back(false);
   }
   else if(inputmodel == "Hyperk"){
      TModels.push_back("DCRho08");
      isNeutral.push_back(false);
      TModels.push_back("DCRho12");
      isNeutral.push_back(false);
      TModels.push_back("DCRho16");
      isNeutral.push_back(false);
   }
   else if(inputmodel == "All"){
      TModels.push_back("Gluinof1");
      isNeutral.push_back(false);
      TModels.push_back("Gluinof5");
      isNeutral.push_back(false);
      TModels.push_back("GluinoN");
      isNeutral.push_back(true);
      TModels.push_back("Stop");
      isNeutral.push_back(false);
      TModels.push_back("StopN");
      isNeutral.push_back(true);
      TModels.push_back("GMStau");
      isNeutral.push_back(false);
      TModels.push_back("PPStau");
      isNeutral.push_back(false);
      TModels.push_back("DCRho08");
      isNeutral.push_back(false);
      TModels.push_back("DCRho12");
      isNeutral.push_back(false);
      TModels.push_back("DCRho16");
      isNeutral.push_back(false);
   }


   else {
      cout<<"no model specified"<<endl;
      return;
   }




   TCanvas* c1 = new TCanvas("c1", "c1",600,800);

   TGraph** graphAtheory = new TGraph*[TModels.size()];
   TGraph** graphAobs =  new TGraph*[TModels.size()];
   TGraph** graphAexp =  new TGraph*[TModels.size()];
   TCutG**  ExpAErr = new TCutG*[TModels.size()];
   TCutG**  Exp2SigmaAErr= new TCutG*[TModels.size()];
   TPad** padA= new  TPad*[TModels.size()];
   string  ModelNames[TModels.size()];
   double step, top;

   top= 1.0/(TModels.size()+2);
   step=(1.0-2.*top)/(TModels.size());

   for(int k=0;k<TModels.size();k++){
     if(!IsTkOnly && isNeutral[k]) continue;
     TPad* pad;
     //TPad* pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,1-(k+2)*step,0.9,1-step*(k+1));//lower left x, y, topright x, y
     if(k<(TModels.size()-1)) {
       pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,1-top-(k+1)*step,0.9,1-top-step*k);//lower left x, y, topright x, y
       pad->SetBottomMargin(0.);
     }
     else {
       pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,0.0,0.9,1-top-step*(k));//lower left x, y, topright x, y
       //pad = new TPad(Form("pad%i",k),Form("ExpErr%i",k),0.1,1-2*top-(k+1)*step+0.00001,0.9,1-top-step*(k));//lower left x, y, topright x, y
       pad->SetBottomMargin(top/(step+top));
     }
      pad->SetLeftMargin(0.1);
      pad->SetRightMargin(0.);
      pad->SetTopMargin(0.);
      padA[k] = pad;  
      padA[k]->Draw();
   }

   for(int k=0;k<TModels.size();k++){
     if(!IsTkOnly && isNeutral[k]) continue;
      std::vector<string> Models = GetModels(TModels[k]);
      ModelNames[k]=GetModelName(TModels[k]);

      TMultiGraph* MG = new TMultiGraph();
      unsigned int N = Models.size();
      stAllInfo Infos;double Mass[N], XSecTh[N], XSecExp[N],XSecObs[N], XSecExpUp[N],XSecExpDown[N],XSecExp2Up[N],XSecExp2Down[N];
      for(int i=0;i<N;i++){
         Infos = stAllInfo(InputPattern+"EXCLUSION/" + Models[i] +".txt");
         Mass[i]=Infos.Mass;
         XSecTh [i]=Infos.XSec_Th;
         XSecObs[i]=Infos.XSec_Obs/Infos.XSec_Exp;
         XSecExp[i]=1.;
         XSecExpUp[i]=Infos.XSec_ExpUp/Infos.XSec_Exp;
         XSecExpDown[i]=Infos.XSec_ExpDown/Infos.XSec_Exp;
         XSecExp2Up[i]=Infos.XSec_Exp2Up/Infos.XSec_Exp;
         XSecExp2Down[i]=Infos.XSec_Exp2Down/Infos.XSec_Exp;
      }

      TGraph* graphtheory= new TGraph(N,Mass,XSecTh);
      TGraph* graphobs = new TGraph(N,Mass,XSecObs);
      TGraph* graphexp = new TGraph(N,Mass,XSecExp);
      TCutG*  ExpErr = GetErrorBand(Form("ExpErr%i",k),N,Mass,XSecExpDown,XSecExpUp,0.0, 3.0);
      TCutG*  Exp2SigmaErr = GetErrorBand(Form("Exp2SigmaErr%i",k),N,Mass,XSecExp2Down,XSecExp2Up, 0.0, 3.0);

      graphAtheory[k] = graphtheory;      
      graphAobs[k] =graphobs;
      graphAexp[k] =graphexp;
      ExpAErr[k] = ExpErr;

      Exp2SigmaAErr[k] = Exp2SigmaErr;
      graphAtheory[k]->SetLineStyle(3);
      graphAexp[k]->SetLineStyle(4); 
      graphAexp[k]->SetLineColor(kRed);
      graphAexp[k]->SetMarkerStyle(); 
      graphAexp[k]->SetMarkerSize(0.); 
      Exp2SigmaAErr[k]->SetFillColor(kYellow);
      Exp2SigmaAErr[k]->SetLineColor(kWhite);
      ExpAErr[k]->SetFillColor(kGreen);
      ExpAErr[k]->SetLineColor(kWhite);
      graphAobs[k]->SetLineColor(kBlack);
      graphAobs[k]->SetLineWidth(2);
      graphAobs[k]->SetMarkerColor(kBlack);
      graphAobs[k]->SetMarkerStyle(23);


      padA[k]->cd();

      int masst[2] = {0,1250};
      int xsect[2] = {2, 1};
      TGraph* graph = new TGraph(2,masst,xsect); //fake graph to set xaxis right
      graph->SetMarkerSize(0.);
      MG->Add(graph      ,"P");
      MG->Add(graphAobs[k]      ,"LP");
      MG->Draw("A");
      if(k==0){
	TLegend* LEG;
	LEG = new TLegend(0.13,0.01,0.32,0.99);
         string headerstr;
         headerstr = "Tracker + TOF";
         if(IsTkOnly) headerstr = "Tracker - Only";
         LEG->SetHeader(headerstr.c_str());
         LEG->SetFillColor(0); 
         LEG->SetBorderSize(0);
         LEG->AddEntry(ExpAErr[0], "Expected #pm 1#sigma","F");
         //LEG->AddEntry(Exp2SigmaAErr[0], "Expected #pm 2#sigma","F");
         //LEG->AddEntry(graphAobs[0],"Observed" ,"LP");
         LEG->SetMargin(0.1);
         LEG->Draw();
      }  

      if(k==1){
        TLegend* LEG;
        LEG = new TLegend(0.13,0.01,0.32,0.99);
	string headerstr;
	//headerstr = "Tracker + TOF";
	//if(IsTkOnly) headerstr = "Tracker - Only";
	//LEG->SetHeader(headerstr.c_str());
	LEG->SetFillColor(0);
	LEG->SetBorderSize(0);
	//LEG->AddEntry(ExpAErr[0], "Expected #pm 1#sigma","F");
	LEG->AddEntry(Exp2SigmaAErr[0], "Expected #pm 2#sigma","F");
	LEG->AddEntry(graphAobs[0],"Observed" ,"LP");
	LEG->SetMargin(0.1);
	LEG->Draw();
      }

      Exp2SigmaAErr[k]->Draw("f");
      ExpAErr[k]  ->Draw("f");
      MG->Draw("same");
      MG->SetTitle("");
      if(k==TModels.size()-1) {
         MG->GetXaxis()->SetTitle("Mass (GeV/#font[12]{c}^{2})");
         MG->GetXaxis()->SetTitleSize(0.2);
         MG->GetXaxis()->SetLabelSize(0.2);
      }

      TPaveText *pt;
      if(IsTkOnly) {
      if(k!=TModels.size()-1) pt = new TPaveText(0.45, 0.6, 0.95, 0.87,"LBNDC");
      else pt = new TPaveText(0.45, 0.82, 0.95, 0.935,"LBNDC");
      }
      else {
	if(k!=TModels.size()-1) pt = new TPaveText(0.55, 0.6, 0.95, 0.87,"LBNDC");
	else pt = new TPaveText(0.55, 0.82, 0.95, 0.935,"LBNDC");
      }

      pt->SetBorderSize(0);
      pt->SetLineWidth(0);
      pt->SetFillColor(kWhite);
      TText *text = pt->AddText(ModelNames[k].c_str()); 
      text ->SetTextAlign(12);
      text ->SetTextSize(0.3);
      if(k==TModels.size()-1) text ->SetTextSize(0.5*text ->GetTextSize());
      pt->Draw();
      
      MG->GetXaxis()->SetRangeUser(0,1250);    
      MG->GetXaxis()->SetNdivisions(506,"Z");

      MG->GetYaxis()->SetRangeUser(0.001,2.99);
      MG->GetYaxis()->SetNdivisions(303, "Z");
      MG->GetYaxis()->SetLabelSize(0.3);
      if(k==(TModels.size()-1)) {
	MG->GetYaxis()->SetLabelSize(0.15);

      }

   }
   c1->cd();
   DrawPreliminary(IntegratedLuminosity);

   TPaveText *pt = new TPaveText(0.1, 0., 0.15, 0.7,"NDC");
   string tmp = "95% CL Limits (Relative to Expected Limit)";
   TText *text = pt->AddText(tmp.c_str()); 
   text ->SetTextAlign(12);
   text ->SetTextAngle(90);
   text ->SetTextSize(0.04);
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->Draw();



   if(IsTkOnly)   SaveCanvas(c1,"Results/EXCLUSION/", string("Tk"+ inputmodel + "LimitsRatio"));
   else    SaveCanvas(c1,"Results/EXCLUSION/", string("Mu"+ inputmodel + "LimitsRatio"));

   delete c1;


}



