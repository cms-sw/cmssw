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

using namespace std;



struct stAllInfo{
   double Mass;
   double MassMean;
   double MassSigma;
   double MassCut;
   double XSec_Th;
   double XSec_Err;
   double XSec_Exp;
   double XSec_Obs;
   double Eff;
   double Eff_SYSTP;
   double Eff_SYSTI;
   double Eff_SYSTM;
   double Eff_SYSTT;
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
      Mass=-1; XSec_Th=-1; XSec_Err=-1; XSec_Exp=-1; XSec_Obs=-1; Eff=-1; Eff_SYSTP=-1; Eff_SYSTI=-1;  Eff_SYSTM=-1; Eff_SYSTT=-1;
      if(path=="")return;
      FILE* pFile = fopen(path.c_str(),"r");
      if(!pFile){printf("Can't open %s\n",path.c_str()); return;}
      fscanf(pFile,"Mass      : %lf\n",&Mass);
      fscanf(pFile,"MassMean  : %lf\n",&MassMean);
      fscanf(pFile,"MassSigma : %lf\n",&MassSigma);
      fscanf(pFile,"MassCut   : %lf\n",&MassCut);
      fscanf(pFile,"Index     : %lf\n",&Index);
      fscanf(pFile,"WP_Pt     : %lf\n",&WP_Pt);
      fscanf(pFile,"WP_I      : %lf\n",&WP_I);
      fscanf(pFile,"WP_TOF    : %lf\n",&WP_TOF);
      fscanf(pFile,"Eff       : %lf\n",&Eff);
      fscanf(pFile,"Eff_SystP : %lf\n",&Eff_SYSTP);
      fscanf(pFile,"Eff_SystI : %lf\n",&Eff_SYSTI);
      fscanf(pFile,"Eff_SystM : %lf\n",&Eff_SYSTM);
      fscanf(pFile,"Eff_SystT : %lf\n",&Eff_SYSTT);
      fscanf(pFile,"Signif    : %lf\n",&Significance);
      fscanf(pFile,"XSec_Th   : %lf\n",&XSec_Th);
      fscanf(pFile,"XSec_Exp  : %lf\n",&XSec_Exp);
      fscanf(pFile,"XSec_Obs  : %lf\n",&XSec_Obs);
      fscanf(pFile,"NData     : %E\n" ,&NData);
      fscanf(pFile,"NPred     : %E\n" ,&NPred);
      fscanf(pFile,"NPredErr  : %E\n" ,&NPredErr);
      fscanf(pFile,"NSign     : %E\n" ,&NSign);
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
   TGraph* DCStau;
   TGraph* GluinoTh;
   TGraph* StopTh;
   TGraph* GMStauTh;
   TCutG*  GluinoThErr;
   TCutG*  StopThErr;
};

TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string syst, string ModelName, int XSectionType=2, string Mass0="", string Mass1="", string Mass2="", string Mass3="", string Mass4="", string Mass5="", string Mass6="", string Mass7="", string Mass8="", string Mass9="");


stAllInfo Exclusion(string pattern, string modelName, string signal, double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1, string syst="");
int      JobIdToIndex(string JobId);

void GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex, double MinRange, double MaxRange);
double FindIntersection(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty=0, bool debug=false);
int ReadXSection(string InputFile, double* Mass, double* XSec, double* Low, double* High,  double* ErrLow, double* ErrHigh);
TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High);
void CheckSignalUncertainty(FILE* pFile, FILE* talkFile, string InputPattern);
double getSignificance(double NData, double NPred, double signalUncertainty, double backgroundError, double luminosityError, string outpath);

//double PlotMinScale = 0.1;
//double PlotMaxScale = 50000;

//double PlotMinScale = 2;
//double PlotMaxScale = 800;

double PlotMinScale = 0.001;
double PlotMaxScale = 60;



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


   TCanvas* c1;

   string MuPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type2/";
   string TkPattern  = "Results/dedxASmi/combined/Eta15/PtMin45/Type0/";


   string outpath = string("Results/EXCLUSION/");
   MakeDirectories(outpath);



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
   TGraph* Tk_Obs_GluinoF1  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino (f=10\\%)", 2, "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino700_f1", "Gluino800_f1", "Gluino900_f1", "Gluino1000_f1", "Gluino1100_f1" );
   TGraph* Tk_Obs_GluinoZF1 = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino Z2 (f=10\\%)", 2, "Gluino600Z_f1" , "Gluino700Z_f1", "Gluino800Z_f1");
   TGraph* Tk_Obs_GluinoF5  = MakePlot(pFile,talkFile,TkPattern,syst,"Gluino (f=50\\%)", 2, "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino700_f5", "Gluino800_f5", "Gluino900_f5", "Gluino1000_f5", "Gluino1100_f5" );
//   TGraph* Tk_Obs_GluinoNF0 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=00\\%)", 2, "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino700N_f0", "Gluino800N_f0", "Gluino900N_f0", "Gluino1000N_f0" );
   TGraph* Tk_Obs_GluinoNF1 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=10\\%)", 2, "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino700N_f1", "Gluino800N_f1", "Gluino900N_f1", "Gluino1000N_f1", "Gluino1100N_f1" );
//   TGraph* Tk_Obs_GluinoNF5 = MakePlot(pFile,talkFile,TkPattern,syst,"GluinoN (f=50\\%)", 2, "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino700N_f5", "Gluino800N_f5" , "Gluino900N_f5" , "Gluino1000N_f5" );
//   TGraph* Tk_Obs_Stop2C    = MakePlot(pFile,talkFile,TkPattern,syst,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop400_2C"   , "Stop500_2C"   , "Stop600_2C" , "Stop700_2C" , "Stop800_2C"                    );
   TGraph* Tk_Obs_Stop      = MakePlot(pFile,talkFile,TkPattern,syst,"Stop"               , 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop400"      , "Stop500"      , "Stop600"    , "Stop700"    , "Stop800"                       );
   //TGraph* Tk_Obs_StopZ     = MakePlot(pFile,talkFile,TkPattern,syst,"Stop Z2"            , 2, "Stop300Z"     , "Stop400Z"     , "Stop500Z");
   TGraph* Tk_Obs_StopN     = MakePlot(pFile,talkFile,TkPattern,syst,"StopN"              , 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop400N"     , "Stop500N"     , "Stop600N"   , "Stop700N"   , "Stop800N"                      );
   TGraph* Tk_Obs_GMStau    = MakePlot(pFile,talkFile,TkPattern,syst,"GMSB Stau"          , 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* Tk_Obs_PPStau    = MakePlot(pFile,talkFile,TkPattern,syst,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247","PPStau308");
   TGraph* Tk_Obs_DCStau    = MakePlot(pFile,talkFile,TkPattern,syst,"DiChamp    Stau  ", 2, "DCStau100"    , "DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302",    "DCStau350"    ,     "DCStau395"    ,      "DCStau420"    ,  "DCStau500");  
   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile, "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");




   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile, "\\begin{sidewaystable}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   fprintf(talkFile,"Sample & Mass(GeV) & Pt(GeV) & $I_{as}$ & $#beta^{-1]$ & Mass Cut (GeV) & N pred & N observed & Eff \\\\\n");
   fprintf(talkFile, "\\hline\n");

//   TGraph* Mu_Obs_Gluino2C  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino  (2C)     ", 2, "Gluino300_2C" , "Gluino400_2C" , "Gluino500_2C" , "Gluino600_2C" , "Gluino700_2C", "Gluino800_2C", "Gluino900_2C", "Gluino1000_2C" );
//   TGraph* Mu_Obs_GluinoF0  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino  (f=00\\%)", 2, "Gluino300_f0" , "Gluino400_f0" , "Gluino500_f0" , "Gluino600_f0" , "Gluino700_f0", "Gluino800_f0", "Gluino900_f0", "Gluino1000_f0" );
   TGraph* Mu_Obs_GluinoF1  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino (f=10\\%)", 2, "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino700_f1", "Gluino800_f1", "Gluino900_f1", "Gluino1000_f1", "Gluino1100_f1" );
   //TGraph* Mu_Obs_GluinoZF1 = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino Z2 (f=10\\%)", 2, "Gluino600Z_f1" , "Gluino700Z_f1", "Gluino800Z_f1");
   TGraph* Mu_Obs_GluinoF5  = MakePlot(pFile,talkFile,MuPattern,syst,"Gluino (f=50\\%)", 2, "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino700_f5", "Gluino800_f5", "Gluino900_f5", "Gluino1000_f5", "Gluino1100_f5" );
//   TGraph* Mu_Obs_GluinoNF0 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=00\\%)", 2, "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino700N_f0", "Gluino800N_f0", "Gluino900N_f0", "Gluino1000N_f0" );
   TGraph* Mu_Obs_GluinoNF1 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=10\\%)", 2, "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino700N_f1", "Gluino800N_f1", "Gluino900N_f1", "Gluino1000N_f1", "Gluino1100N_f1" );
//   TGraph* Mu_Obs_GluinoNF5 = MakePlot(pFile,talkFile,MuPattern,syst,"GluinoN (f=50\\%)", 2, "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino700N_f5", "Gluino800N_f5" , "Gluino900N_f5" , "Gluino1000N_f5" );
//   TGraph* Mu_Obs_Stop2C    = MakePlot(pFile,talkFile,MuPattern,syst,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop400_2C"   , "Stop500_2C"   , "Stop600_2C" , "Stop700_2C" , "Stop800_2C"                    );
   TGraph* Mu_Obs_Stop      = MakePlot(pFile,talkFile,MuPattern,syst,"Stop"               , 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop400"      , "Stop500"      , "Stop600"    , "Stop700"    , "Stop800"                       );
   //TGraph* Mu_Obs_StopZ     = MakePlot(pFile,talkFile,MuPattern,syst,"Stop Z2"            , 2, "Stop300Z"     , "Stop400Z"     , "Stop500Z");
   TGraph* Mu_Obs_StopN     = MakePlot(pFile,talkFile,MuPattern,syst,"StopN"              , 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop400N"     , "Stop500N"     , "Stop600N"   , "Stop700N"   , "Stop800N"                      );
   TGraph* Mu_Obs_GMStau    = MakePlot(pFile,talkFile,MuPattern,syst,"GMSB Stau"          , 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* Mu_Obs_PPStau    = MakePlot(pFile,talkFile,MuPattern,syst,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247", "PPStau308" );
   TGraph* Mu_Obs_DCStau    = MakePlot(pFile,talkFile,MuPattern,syst,"DiChamp    Stau  ", 2,"DCStau121"    , "DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302"    ,"DCStau350"    ,"DCStau395"    ,"DCStau420"    ,"DCStau500");
   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile,"\\end{document}\n\n");

   fprintf(talkFile,"      \\end{tabular}\n\\end{sidewaystable}\n\n");
   fprintf(talkFile,"\\end{document}\n\n");

   CheckSignalUncertainty(pFile,talkFile,TkPattern);
   CheckSignalUncertainty(pFile,talkFile,MuPattern);


   TGraph* GMStauXSec = MakePlot(NULL,NULL,TkPattern,"","GMSB Stau        ", 0, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308", "GMStau370", "GMStau432", "GMStau494"    );
   TGraph* PPStauXSec = MakePlot(NULL,NULL,TkPattern,"","Pair Prod. Stau  ", 0, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247"    , "PPStau308");
   TGraph* DCStauXSec = MakePlot(NULL,NULL,TkPattern,"","DiChamp    Stau  ", 0, "DCStau100"    ,"DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302", "DCStau350"     , "DCStau395"     , "DCStau420"     , "DCStau500");

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


   int ThDCStauN = 9; double ThDCStauMass [100]; double ThDCStauXSec [100];  double ThDCStauLow  [100];  double ThDCStauHigh [100];
   ThDCStauMass[0] = 100; ThDCStauXSec[0] = 1.405000;  ThDCStauLow[0] = ThDCStauXSec[0]*0.85;  ThDCStauHigh[0] = ThDCStauXSec[0]*1.15;
   ThDCStauMass[1] = 121; ThDCStauXSec[1] = 0.979000;  ThDCStauLow[1] = ThDCStauXSec[1]*0.85;  ThDCStauHigh[1] = ThDCStauXSec[1]*1.15;
   ThDCStauMass[2] = 182; ThDCStauXSec[2] = 0.560000;  ThDCStauLow[2] = ThDCStauXSec[2]*0.85;  ThDCStauHigh[2] = ThDCStauXSec[2]*1.15;
   ThDCStauMass[3] = 242; ThDCStauXSec[3] = 0.489000;  ThDCStauLow[3] = ThDCStauXSec[3]*0.85;  ThDCStauHigh[3] = ThDCStauXSec[3]*1.15;
   ThDCStauMass[4] = 302; ThDCStauXSec[4] = 0.463000;  ThDCStauLow[4] = ThDCStauXSec[4]*0.85;  ThDCStauHigh[4] = ThDCStauXSec[4]*1.15;
   ThDCStauMass[5] = 350; ThDCStauXSec[5] = 0.473000;  ThDCStauLow[5] = ThDCStauXSec[5]*0.85;  ThDCStauHigh[5] = ThDCStauXSec[5]*1.15;
   ThDCStauMass[6] = 395; ThDCStauXSec[6] = 0.420000;  ThDCStauLow[6] = ThDCStauXSec[6]*0.85;  ThDCStauHigh[6] = ThDCStauXSec[6]*1.15;
   ThDCStauMass[7] = 420; ThDCStauXSec[7] = 0.0035;  ThDCStauLow[7] = ThDCStauXSec[7]*0.85;  ThDCStauHigh[7] = ThDCStauXSec[7]*1.15;
   ThDCStauMass[8] = 500; ThDCStauXSec[8] = 0.0002849;  ThDCStauLow[8] = ThDCStauXSec[8]*0.85;  ThDCStauHigh[8] = ThDCStauXSec[8]*1.15;
   TCutG* DCStauXSecErr = GetErrorBand("DCStauErr", ThDCStauN,ThDCStauMass,ThDCStauLow,ThDCStauHigh);

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
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Tk_Obs_GluinoF1,  GluinoXSec, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Tk_Obs_GluinoZF1, GluinoXSec, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Tk_Obs_GluinoF5,  GluinoXSec, 300, 1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Tk_Obs_GluinoNF0, GluinoXSec, 300, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Tk_Obs_GluinoNF1, GluinoXSec, 300, 900, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Tk_Obs_GluinoNF5, GluinoXSec, 300, 900, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Tk_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Tk_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Tk_Obs_StopZ    , StopXSec  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Tk_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Tk_Obs_GMStau   , GMStauXSec, 100, 494, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Tk_Obs_PPStau   , PPStauXSec, 100, 308, 1, 0.0));

   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Tk_Obs_DCStau   , DCStauXSec, 100, 500, 1, 0.0));
   
   fprintf(pFile,"-----------------------\n0%% TK TOF        \n-------------------------\n");
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Mu_Obs_Gluino2C,  GluinoXSec, 300,1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Mu_Obs_GluinoF0,  GluinoXSec, 300,1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Mu_Obs_GluinoF1,  GluinoXSec, 300,1100, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 Z2\n",FindIntersection(Mu_Obs_GluinoZF1, GluinoXSec, 600,800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Mu_Obs_GluinoF5,  GluinoXSec, 300,1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Mu_Obs_GluinoNF0, GluinoXSec, 300,1000, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Mu_Obs_GluinoNF1, GluinoXSec, 300,1100, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Mu_Obs_GluinoNF5, GluinoXSec, 300,1000, 1, 0.00));
//   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Mu_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Mu_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   //fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop Z2  \n", FindIntersection(Mu_Obs_StopZ    , StopXSec  , 300, 500, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Mu_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Mu_Obs_GMStau   , GMStauXSec, 100, 494, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Mu_Obs_PPStau   , PPStauXSec, 100, 308, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Mu_Obs_DCStau   , DCStauXSec, 100, 500, 1, 0.0));

   fclose(pFile);
   if(syst!="")return;


   GluinoXSec      ->SetLineColor(4);  GluinoXSec      ->SetMarkerColor(4);   GluinoXSec      ->SetLineWidth(1);   GluinoXSec      ->SetLineStyle(3);  GluinoXSec      ->SetMarkerStyle(1);
   Mu_Obs_GluinoF1 ->SetLineColor(4);  Mu_Obs_GluinoF1 ->SetMarkerColor(4);   Mu_Obs_GluinoF1 ->SetLineWidth(2);   Mu_Obs_GluinoF1 ->SetLineStyle(1);  Mu_Obs_GluinoF1 ->SetMarkerStyle(22);
   Mu_Obs_GluinoF5 ->SetLineColor(4);  Mu_Obs_GluinoF5 ->SetMarkerColor(4);   Mu_Obs_GluinoF5 ->SetLineWidth(2);   Mu_Obs_GluinoF5 ->SetLineStyle(1);  Mu_Obs_GluinoF5 ->SetMarkerStyle(23);
   Mu_Obs_GluinoNF1->SetLineColor(4);  Mu_Obs_GluinoNF1->SetMarkerColor(4);   Mu_Obs_GluinoNF1->SetLineWidth(2);   Mu_Obs_GluinoNF1->SetLineStyle(1);  Mu_Obs_GluinoNF1->SetMarkerStyle(26);
   Tk_Obs_GluinoF1 ->SetLineColor(4);  Tk_Obs_GluinoF1 ->SetMarkerColor(4);   Tk_Obs_GluinoF1 ->SetLineWidth(2);   Tk_Obs_GluinoF1 ->SetLineStyle(1);  Tk_Obs_GluinoF1 ->SetMarkerStyle(22);
   Tk_Obs_GluinoF5 ->SetLineColor(4);  Tk_Obs_GluinoF5 ->SetMarkerColor(4);   Tk_Obs_GluinoF5 ->SetLineWidth(2);   Tk_Obs_GluinoF5 ->SetLineStyle(1);  Tk_Obs_GluinoF5 ->SetMarkerStyle(23);
   Tk_Obs_GluinoNF1->SetLineColor(4);  Tk_Obs_GluinoNF1->SetMarkerColor(4);   Tk_Obs_GluinoNF1->SetLineWidth(2);   Tk_Obs_GluinoNF1->SetLineStyle(1);  Tk_Obs_GluinoNF1->SetMarkerStyle(26);
   StopXSec        ->SetLineColor(2);  StopXSec        ->SetMarkerColor(2);   StopXSec        ->SetLineWidth(1);   StopXSec        ->SetLineStyle(2);  StopXSec        ->SetMarkerStyle(1);
   Mu_Obs_Stop     ->SetLineColor(2);  Mu_Obs_Stop     ->SetMarkerColor(2);   Mu_Obs_Stop     ->SetLineWidth(2);   Mu_Obs_Stop     ->SetLineStyle(1);  Mu_Obs_Stop     ->SetMarkerStyle(21);
   Mu_Obs_StopN    ->SetLineColor(2);  Mu_Obs_StopN    ->SetMarkerColor(2);   Mu_Obs_StopN    ->SetLineWidth(2);   Mu_Obs_StopN    ->SetLineStyle(1);  Mu_Obs_StopN    ->SetMarkerStyle(25);
   Tk_Obs_Stop     ->SetLineColor(2);  Tk_Obs_Stop     ->SetMarkerColor(2);   Tk_Obs_Stop     ->SetLineWidth(2);   Tk_Obs_Stop     ->SetLineStyle(1);  Tk_Obs_Stop     ->SetMarkerStyle(21);
   Tk_Obs_StopN    ->SetLineColor(2);  Tk_Obs_StopN    ->SetMarkerColor(2);   Tk_Obs_StopN    ->SetLineWidth(2);   Tk_Obs_StopN    ->SetLineStyle(1);  Tk_Obs_StopN    ->SetMarkerStyle(25);
   GMStauXSec      ->SetLineColor(1);  GMStauXSec      ->SetMarkerColor(1);   GMStauXSec      ->SetLineWidth(1);   GMStauXSec      ->SetLineStyle(1);  GMStauXSec      ->SetMarkerStyle(1);
   PPStauXSec      ->SetLineColor(6);  PPStauXSec      ->SetMarkerColor(6);   PPStauXSec      ->SetLineWidth(1);   PPStauXSec      ->SetLineStyle(4);  PPStauXSec      ->SetMarkerStyle(1);
   DCStauXSec      ->SetLineColor(9);  PPStauXSec      ->SetMarkerColor(9);   PPStauXSec      ->SetLineWidth(1);   PPStauXSec      ->SetLineStyle(2);  PPStauXSec      ->SetMarkerStyle(1);
   Mu_Obs_GMStau   ->SetLineColor(1);  Mu_Obs_GMStau   ->SetMarkerColor(1);   Mu_Obs_GMStau   ->SetLineWidth(2);   Mu_Obs_GMStau   ->SetLineStyle(1);  Mu_Obs_GMStau   ->SetMarkerStyle(23);
   Mu_Obs_PPStau   ->SetLineColor(6);  Mu_Obs_PPStau   ->SetMarkerColor(6);   Mu_Obs_PPStau   ->SetLineWidth(2);   Mu_Obs_PPStau   ->SetLineStyle(1);  Mu_Obs_PPStau   ->SetMarkerStyle(23);
   Mu_Obs_DCStau   ->SetLineColor(9);  Mu_Obs_DCStau   ->SetMarkerColor(9);   Mu_Obs_DCStau   ->SetLineWidth(2);   Mu_Obs_DCStau   ->SetLineStyle(1);  Mu_Obs_DCStau   ->SetMarkerStyle(20);

   Tk_Obs_GMStau   ->SetLineColor(1);  Tk_Obs_GMStau   ->SetMarkerColor(1);   Tk_Obs_GMStau   ->SetLineWidth(2);   Tk_Obs_GMStau   ->SetLineStyle(1);  Tk_Obs_GMStau   ->SetMarkerStyle(20);
   Tk_Obs_PPStau   ->SetLineColor(6);  Tk_Obs_PPStau   ->SetMarkerColor(6);   Tk_Obs_PPStau   ->SetLineWidth(2);   Tk_Obs_PPStau   ->SetLineStyle(1);  Tk_Obs_PPStau   ->SetMarkerStyle(20);
   Tk_Obs_DCStau   ->SetLineColor(9);  Tk_Obs_DCStau   ->SetMarkerColor(9);   Tk_Obs_DCStau   ->SetLineWidth(2);   Tk_Obs_DCStau   ->SetLineStyle(1);  Tk_Obs_DCStau   ->SetMarkerStyle(20);


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
   MGMu->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   MGMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGMu->GetYaxis()->SetTitleOffset(1.70);
   MGMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGMu = new TLegend(0.50,0.65,0.80,0.90);
//   LEGMu->SetHeader("95% C.L. Limits");
   LEGMu->SetHeader("Tk + TOF");
   LEGMu->SetFillColor(0); 
   LEGMu->SetBorderSize(0);
   LEGMu->AddEntry(Mu_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(Mu_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
//   LEGMu->AddEntry(Mu_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGMu->AddEntry(Mu_Obs_Stop     , "stop"            ,"LP");
//   LEGMu->AddEntry(Mu_Obs_StopN    , "stop; ch. suppr.","LP");
   LEGMu->AddEntry(Mu_Obs_PPStau   , "Pair Prod. stau"       ,"LP");
   LEGMu->AddEntry(Mu_Obs_GMStau   , "GMSB stau"       ,"LP");
   LEGMu->Draw();

   TLegend* LEGTh = new TLegend(0.15,0.70,0.50,0.90);
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
   MGTk->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   MGTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGTk->GetYaxis()->SetTitleOffset(1.70);
   MGTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGTk = new TLegend(0.50,0.65,0.80,0.90);
//   LEGTk->SetHeader("95% C.L. Limits");
   LEGTk->SetHeader("Tk + only");
   LEGTk->SetFillColor(0); 
   LEGTk->SetBorderSize(0);
   LEGTk->AddEntry(Tk_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_Stop     , "stop"            ,"LP");
   LEGTk->AddEntry(Tk_Obs_StopN    , "stop; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_PPStau   , "Pair Prod. stau"       ,"LP");
   LEGTk->AddEntry(Tk_Obs_GMStau   , "GMSB stau"       ,"LP");
   LEGTk->Draw();

   LEGTh->Draw();

//   c1->SetGridx(true);
//   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("TkExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkExclusionLog"));
   delete c1;

    c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCMu = new TMultiGraph();
   MGDCMu->Add(DCStauXSec      ,"L");
   MGDCMu->Add(Mu_Obs_DCStau        ,"LP");
   MGDCMu->Draw("A");
   DCStauXSecErr  ->Draw("f");
   MGDCMu->Draw("same");
   MGDCMu->SetTitle("");
   MGDCMu->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   MGDCMu->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCMu->GetYaxis()->SetTitleOffset(1.70);
   MGDCMu->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGDCMu = new TLegend(0.50,0.65,0.80,0.90);
   LEGDCMu->SetHeader("Tk + TOF");
   LEGDCMu->SetFillColor(0); 
   LEGDCMu->SetBorderSize(0);
   LEGDCMu->AddEntry(Mu_Obs_DCStau   , "DiChamp stau"       ,"LP");
   LEGDCMu->Draw();

   TLegend* LEGDCTh = new TLegend(0.15,0.70,0.50,0.90);
   LEGDCTh->SetHeader("Theoretical Prediction");
   LEGDCTh->SetFillColor(0);
   LEGDCTh->SetBorderSize(0);
   TGraph* DCStauThLeg = (TGraph*) DCStauXSec->Clone("DCStauThLeg");
   DCStauThLeg->SetFillColor(GluinoXSecErr->GetFillColor());
   LEGDCTh->AddEntry(DCStauThLeg   ,"Dichamp stau   (LO)" ,"LF");
   LEGDCTh->Draw();
   SaveCanvas(c1, outpath, string("MuDCExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("MuDCExclusionLog"));
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGDCTk = new TMultiGraph();
   MGDCTk->Add(DCStauXSec      ,"L");
   MGDCTk->Add(Tk_Obs_DCStau        ,"LP");
   MGDCTk->Draw("A");
   DCStauXSecErr  ->Draw("f");
   MGDCTk->Draw("same");
   MGDCTk->SetTitle("");
   MGDCTk->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   MGDCTk->GetYaxis()->SetTitle("#sigma (pb)");
   MGDCTk->GetYaxis()->SetTitleOffset(1.70);
   MGDCTk->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* LEGDCTk = new TLegend(0.50,0.65,0.80,0.90);
//   LEGDCTk->SetHeader("95% C.L. Limits");
   LEGDCTk->SetHeader("Tk + only");
   LEGDCTk->SetFillColor(0); 
   LEGDCTk->SetBorderSize(0);
   LEGDCTk->AddEntry(Tk_Obs_DCStau   , "DiChamp stau"       ,"LP");
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
   Models.push_back("Gluino300_f5");
   Models.push_back("Gluino400_f5");
   Models.push_back("Gluino500_f5");
   Models.push_back("Gluino600_f5");
   Models.push_back("Gluino700_f5");
   Models.push_back("Gluino800_f5");
   Models.push_back("Gluino900_f5");
   Models.push_back("Gluino1000_f5");
   Models.push_back("Gluino300N_f1");
   Models.push_back("Gluino400N_f1");
   Models.push_back("Gluino500N_f1");
   Models.push_back("Gluino600N_f1");
   Models.push_back("Gluino700N_f1");
   Models.push_back("Gluino800N_f1");
   Models.push_back("Gluino900N_f1");
   Models.push_back("Gluino1000N_f1");
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

   if(IsTkOnly){
     fprintf(pFile, "%20s   Eff   --> PScale |  EstimScale | DiscrimScale || TotalUncertainty\n","Model");
     fprintf(talkFile, "%20s   Eff   --> PScale |  EstimScale | DiscrimScale || TotalUncertainty\n","Model");
   }
   else {
     fprintf(pFile, "%20s   Eff   --> PScale |  EstimScale | DiscrimScale | TOFScale || TotalUncertainty\n","Model");
     fprintf(talkFile, "%20s   Eff   --> PScale |  EstimScale | DiscrimScale | TOFScale || TotalUncertainty\n","Model");
   }

   for(unsigned int s=0;s<Models.size();s++){
        stAllInfo tmp(InputPattern+"/EXCLUSION" + "/"+Models[s]+".txt");
        double P = tmp.Eff - tmp.Eff_SYSTP;
        double I = tmp.Eff - tmp.Eff_SYSTI;
        double M = tmp.Eff - tmp.Eff_SYSTM;
        double T = tmp.Eff - tmp.Eff_SYSTT;
        bool IsStau = (Models[s].find("Stau",0)<std::string::npos);
        bool IsNeutral = (Models[s].find("N",0)<std::string::npos);

	if(IsTkOnly) fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, M/tmp.Eff, I/tmp.Eff, sqrt(P*P + I*I + M*M + T*T)/tmp.Eff);        

	else if(!IsNeutral) fprintf(pFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, M/tmp.Eff, I/tmp.Eff, T/tmp.Eff, sqrt(P*P + I*I + M*M + T*T)/tmp.Eff);

	if(IsTkOnly && (IsStau || (int)tmp.Mass%200==0)) {
	  fprintf(talkFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, M/tmp.Eff, I/tmp.Eff, T/tmp.Eff, sqrt(P*P + I*I + M*M + T*T)/tmp.Eff);
	}
        if(!IsTkOnly && !IsNeutral) fprintf(talkFile, "%20s   %7.3f --> %7.3f  |  %7.3f  | %7.3f  | %7.3f || %7.3f\n",+Models[s].c_str(), tmp.Eff, P/tmp.Eff, M/tmp.Eff, I/tmp.Eff, T/tmp.Eff, sqrt(P*P + I*I + M*M + T*T)/tmp.Eff);

   }
}



TGraph* MakePlot(FILE* pFile, FILE* talkFile, string InputPattern, string syst, string ModelName, int XSectionType, string Mass0, string Mass1, string Mass2, string Mass3, string Mass4, string Mass5, string Mass6, string Mass7, string Mass8, string Mass9){
   unsigned int N=0;
   stAllInfo Infos[10];

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


   double Mass   [10];for(unsigned int i=0;i<10;i++){Mass   [i]=Infos[i].Mass;    }
   double XSecTh [10];for(unsigned int i=0;i<10;i++){XSecTh [i]=Infos[i].XSec_Th; }
   double XSecObs[10];for(unsigned int i=0;i<10;i++){XSecObs[i]=Infos[i].XSec_Obs;}
   double XSecExp[10];for(unsigned int i=0;i<10;i++){XSecExp[i]=Infos[i].XSec_Exp;}

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

   string InputPathSign     = pattern + "Histos.root";
   TFile* InputFileSign     = new TFile(InputPathSign.c_str());

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


   TH1D* MassSignProj[4];
   TH1D* MassSignPProj[4];
   TH1D* MassSignIProj[4];
   TH1D* MassSignMProj[4];
   TH1D* MassSignTProj[4];
   ///##############################################################################"
   MassSignProj[0] = MassSign[0]->ProjectionY("MassSignProj0",1,1);
   double Mean  = MassSignProj[0]->GetMean();
   double Width = MassSignProj[0]->GetRMS();
   MinRange = std::max(0.0, Mean-2*Width);
   MinRange = MassSignProj[0]->GetXaxis()->GetBinLowEdge(MassSignProj[0]->GetXaxis()->FindBin(MinRange)); //Round to a bin value to avoid counting prpoblem due to the binning. 
   delete MassSignProj[0];
   ///##############################################################################"

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      if(HCuts_Pt ->GetBinContent(CutIndex+1) < 45 ) continue;  // Be sure the pT cut is high enough to get some statistic for both ABCD and mass shape
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<25 || H_F->GetBinContent(CutIndex+1)<25 || H_G->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (AFG/EE) is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<25 || H_B->GetBinContent(CutIndex+1)<25))continue;  //Skip events where Prediction (CB/A) is not reliable
      GetSignalMeanHSCPPerEvent(pattern,CutIndex, MinRange, MaxRange);
      TH1D* MassDataProj = MassData->ProjectionY("MassDataProj",CutIndex+1,CutIndex+1);
      TH1D* MassPredProj = MassPred->ProjectionY("MassPredProj",CutIndex+1,CutIndex+1);
      MassSignProj[0]    = MassSign [0]->ProjectionY("MassSignProj0",CutIndex+1,CutIndex+1);
      MassSignProj[1]    = MassSign [1]->ProjectionY("MassSignProj1",CutIndex+1,CutIndex+1);
      MassSignProj[2]    = MassSign [2]->ProjectionY("MassSignProj2",CutIndex+1,CutIndex+1);
      MassSignProj[3]    = MassSign [3]->ProjectionY("MassSignProj3",CutIndex+1,CutIndex+1);

      MassSignPProj[0]   = MassSignP[0]->ProjectionY("MassSignProP0",CutIndex+1,CutIndex+1);
      MassSignPProj[1]   = MassSignP[1]->ProjectionY("MassSignProP1",CutIndex+1,CutIndex+1);
      MassSignPProj[2]   = MassSignP[2]->ProjectionY("MassSignProP2",CutIndex+1,CutIndex+1);
      MassSignPProj[3]   = MassSignP[3]->ProjectionY("MassSignProP3",CutIndex+1,CutIndex+1);
      MassSignIProj[0]   = MassSignI[0]->ProjectionY("MassSignProI0",CutIndex+1,CutIndex+1);
      MassSignIProj[1]   = MassSignI[1]->ProjectionY("MassSignProI1",CutIndex+1,CutIndex+1);
      MassSignIProj[2]   = MassSignI[2]->ProjectionY("MassSignProI2",CutIndex+1,CutIndex+1);
      MassSignIProj[3]   = MassSignI[3]->ProjectionY("MassSignProI3",CutIndex+1,CutIndex+1);
      MassSignMProj[0]   = MassSignM[0]->ProjectionY("MassSignProM0",CutIndex+1,CutIndex+1);
      MassSignMProj[1]   = MassSignM[1]->ProjectionY("MassSignProM1",CutIndex+1,CutIndex+1);
      MassSignMProj[2]   = MassSignM[2]->ProjectionY("MassSignProM2",CutIndex+1,CutIndex+1);
      MassSignMProj[3]   = MassSignM[3]->ProjectionY("MassSignProM3",CutIndex+1,CutIndex+1);
      MassSignTProj[0]   = MassSignT[0]->ProjectionY("MassSignProT0",CutIndex+1,CutIndex+1);
      MassSignTProj[1]   = MassSignT[1]->ProjectionY("MassSignProT1",CutIndex+1,CutIndex+1);
      MassSignTProj[2]   = MassSignT[2]->ProjectionY("MassSignProT2",CutIndex+1,CutIndex+1);
      MassSignTProj[3]   = MassSignT[3]->ProjectionY("MassSignProT3",CutIndex+1,CutIndex+1);


      double NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
      double NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));
      double NPredErr    = pow(NPred*RescaleError,2);
      for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=pow(MassPredProj->GetBinError(i),2);}NPredErr=sqrt(NPredErr);

      if(isnan(NPred))continue;
      if(NPred<=0){continue;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
//    if(NPred<1E-4){continue;} //This will never be the selection which gives the best expected limit (cutting too much on signal) --> Slowdown computation for nothing...
      if(NPred>1000){continue;}  //When NPred is too big, expected limits just take an infinite time! 

      double Eff       = 0;
      double EffP      = 0;
      double EffI      = 0;
      double EffM      = 0;
      double EffT      = 0;
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
         }
      }
      if(Eff==0)continue;
      NPred*=RescaleFactor;

     

     //fprintf(pFile ,"CutIndex=%4i ManHSCPPerEvents = %6.2f %6.2f %6.2f %6.2f   NTracks = %6.3f %6.3f %6.3f %6.3f\n",CutIndex,signalsMeanHSCPPerEvent[0], signalsMeanHSCPPerEvent[1],signalsMeanHSCPPerEvent[2],signalsMeanHSCPPerEvent[3], MassSignProj[0]->Integral(), MassSignProj[1]->Integral(), MassSignProj[2]->Integral(), MassSignProj[3]->Integral());


     fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);
     fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.3f TOF>%6.3f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E+-%6.3E SignalEff=%6.3f",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred, NPredErr,Eff);fflush(stdout);


     double ExpLimit = 99999999;
     double ObsLimit = 99999999;
     double Significance = -1;     
     LimitResult CLMResults;
     double signalUncertainty=0.10;
     if (signals[JobIdToIndex(signal)].Mass<450) signalUncertainty=0.15;
     continue;
     //     CLMResults = roostats_clm(IntegratedLuminosity, IntegratedLuminosity*0.06, Eff, Eff*signalUncertainty,NPred, NPred*RescaleError, 1 , 1, "bayesian");   ExpLimit=CLMResults.GetExpectedLimit();  //1 Toy
     CLMResults = roostats_clm(IntegratedLuminosity, IntegratedLuminosity*0.06, Eff, Eff*signalUncertainty,NPred, NPred*RescaleError, 10, 1, "bayesian");   ExpLimit=CLMResults.GetExpectedLimit();  //10Toys

     fprintf(pFile ," --> %+7.2E expected",ExpLimit);
     fprintf(stdout," --> %+7.2E expected",ExpLimit);
     if(toReturn.XSec_Exp<=ExpLimit){fprintf(pFile  ,"\n"); printf("\n"); continue;}
     ObsLimit =  roostats_cl95(IntegratedLuminosity, IntegratedLuminosity*0.06, Eff, Eff*signalUncertainty,NPred, NPred*RescaleError              , NData, false, 1, "bayesian", "");
     Significance = getSignificance(NData, NPred, signalUncertainty, RescaleError, 0.06, outpath+"/"+modelName);
     fprintf(pFile ," (%+7.4E observed) --> Current Best Limit  Significance of %3.2f\n",ObsLimit,Significance);
     fprintf(stdout," (%+7.4E observed) --> Current Best Limit  Significance of %3.2f\n",ObsLimit,Significance);
     toReturn.Mass      = signals[JobIdToIndex(signal)].Mass;
     toReturn.MassMean  = Mean;
     toReturn.MassSigma = Width;
     toReturn.MassCut   = MinRange;
     toReturn.Index     = CutIndex;
     toReturn.WP_Pt     = HCuts_Pt ->GetBinContent(CutIndex+1);
     toReturn.WP_I      = HCuts_I  ->GetBinContent(CutIndex+1);
     toReturn.WP_TOF    = HCuts_TOF->GetBinContent(CutIndex+1);
     toReturn.XSec_Th   = signals[JobIdToIndex(signal)].XSec;
     toReturn.XSec_Err  = signals[JobIdToIndex(signal)].XSec * 0.15;
     toReturn.XSec_Exp  = ExpLimit;
     toReturn.XSec_Obs  = ObsLimit; 
     toReturn.Significance = Significance;
     toReturn.Eff       = Eff;
     toReturn.Eff_SYSTP = EffP;
     toReturn.Eff_SYSTI = EffI;
     toReturn.Eff_SYSTM = EffM;
     toReturn.Eff_SYSTT = EffT;
     toReturn.NData     = NData;
     toReturn.NPred     = NPred;
     toReturn.NPredErr  = NPredErr;
     toReturn.NSign     = Eff*(signals[CurrentSampleIndex].XSec*IntegratedLuminosity);


     FILE* pFile2 = fopen((outpath+"/"+modelName+".txt").c_str(),"w");
     if(!pFile2)printf("Can't open file : %s\n",(outpath+"/"+modelName+".txt").c_str());
     fprintf(pFile2,"Mass      : %f\n",signals[JobIdToIndex(signal)].Mass);
     fprintf(pFile2,"MassMean  : %f\n",toReturn.MassMean);
     fprintf(pFile2,"MassSigma : %f\n",toReturn.MassSigma);
     fprintf(pFile2,"MassCut   : %f\n",toReturn.MassCut);
     fprintf(pFile2,"Index     : %f\n",toReturn.Index);
     fprintf(pFile2,"WP_Pt     : %f\n",toReturn.WP_Pt);
     fprintf(pFile2,"WP_I      : %f\n",toReturn.WP_I);
     fprintf(pFile2,"WP_TOF    : %f\n",toReturn.WP_TOF);
     fprintf(pFile2,"Eff       : %f\n",toReturn.Eff);
     fprintf(pFile2,"Eff_SystP : %f\n",toReturn.Eff_SYSTP);
     fprintf(pFile2,"Eff_SystI : %f\n",toReturn.Eff_SYSTI);
     fprintf(pFile2,"Eff_SystM : %f\n",toReturn.Eff_SYSTM);
     fprintf(pFile2,"Eff_SystT : %f\n",toReturn.Eff_SYSTT);
     fprintf(pFile2,"Signif    : %f\n",toReturn.Significance);
     fprintf(pFile2,"XSec_Th   : %f\n",toReturn.XSec_Th);
     fprintf(pFile2,"XSec_Exp  : %f\n",toReturn.XSec_Exp);
     fprintf(pFile2,"XSec_Obs  : %f\n",toReturn.XSec_Obs);
     fprintf(pFile2,"NData     : %+6.2E\n",toReturn.NData);
     fprintf(pFile2,"NPred     : %+6.2E\n",toReturn.NPred);
     fprintf(pFile2,"NPredErr  : %+6.2E\n",toReturn.NPredErr);
     fprintf(pFile2,"NSign     : %+6.2E\n",toReturn.NSign);
     fclose(pFile2);
   }   
   fclose(pFile);   

  FILE* pFile2 = fopen((outpath+"/"+modelName+".txt").c_str(),"w");
  if(!pFile2)printf("Can't open file : %s\n",(outpath+"/"+modelName+".txt").c_str());
  fprintf(pFile2,"Mass      : %f\n",signals[JobIdToIndex(signal)].Mass);
  fprintf(pFile2,"MassMean  : %f\n",toReturn.MassMean);
  fprintf(pFile2,"MassSigma : %f\n",toReturn.MassSigma);
  fprintf(pFile2,"MassCut   : %f\n",toReturn.MassCut);
  fprintf(pFile2,"Index     : %f\n",toReturn.Index);
  fprintf(pFile2,"WP_Pt     : %f\n",toReturn.WP_Pt);
  fprintf(pFile2,"WP_I      : %f\n",toReturn.WP_I);
  fprintf(pFile2,"WP_TOF    : %f\n",toReturn.WP_TOF);
  fprintf(pFile2,"Eff       : %f\n",toReturn.Eff);
  fprintf(pFile2,"Eff_SystP : %f\n",toReturn.Eff_SYSTP);
  fprintf(pFile2,"Eff_SystI : %f\n",toReturn.Eff_SYSTI);
  fprintf(pFile2,"Eff_SystM : %f\n",toReturn.Eff_SYSTM);
  fprintf(pFile2,"Eff_SystT : %f\n",toReturn.Eff_SYSTT);
  fprintf(pFile2,"Signif    : %f\n",toReturn.Significance);
  fprintf(pFile2,"XSec_Th   : %f\n",toReturn.XSec_Th);
  fprintf(pFile2,"XSec_Exp  : %f\n",toReturn.XSec_Exp);
  fprintf(pFile2,"XSec_Obs  : %f\n",toReturn.XSec_Obs);
  fprintf(pFile2,"NData     : %+6.2E\n",toReturn.NData);
  fprintf(pFile2,"NPred     : %+6.2E\n",toReturn.NPred);
  fprintf(pFile2,"NPredErr  : %+6.2E\n",toReturn.NPredErr);
  fprintf(pFile2,"NSign     : %+6.2E\n",toReturn.NSign);
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


TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High)
{
   TCutG* cutg = new TCutG(name.c_str(),2*N);
   cutg->SetFillColor(kGreen-7);
   for(int i=0;i<N;i++){
      double Min = std::max(Low[i],PlotMinScale);
      cutg->SetPoint( i,Mass[i], Min);
   }
   for(int i=0;i<N;i++){
      double Max = std::min(High[N-1-i],PlotMaxScale);
      cutg->SetPoint(N+i,Mass[N-1-i], Max);
   }
   return cutg;
}

double getSignificance(double NData, double NPred, double signalUncertainty, double backgroundError, double luminosityError, string outpath)
{
  FILE* dataCard = fopen((outpath + "_temp_1.txt").c_str(),"w");

  fprintf(dataCard, "imax 1  number of channels\n");
  fprintf(dataCard, "jmax 1  number of backgrounds\n");
  fprintf(dataCard, "kmax 3  number of nuisance parameters (sources of systematical uncertainties)\n");
  fprintf(dataCard, "bin 1\n");
  fprintf(dataCard, "observation %f\n", NData);
  fprintf(dataCard, "bin              1     1\n");
  fprintf(dataCard, "process         signal   bckgd\n");
  fprintf(dataCard, "process          0     1\n");
  fprintf(dataCard, "rate             2    %f\n", NPred);
  fprintf(dataCard, "lumi    lnN    %f    -    lumi affects signal\n",1+luminosityError);
  fprintf(dataCard, "eff     lnN    %f    -   signal efficiency\n", 1+signalUncertainty);
  fprintf(dataCard, "backg lnN      -   %f  total background\n", 1+backgroundError);
  fclose(dataCard);

  system(("combine -M ProfileLikelihood --significance " + outpath + "_temp_1.txt > " + outpath + "_temp_2.txt").c_str());

  ifstream infile;
  infile.open ((outpath + "_temp_2.txt").c_str());
  bool loop=true;
  double significance=-1;

  if (infile.is_open())
    {
      while ( infile.good() && loop)
	{
          string word;
          infile >> word;
          if (word=="Significance:") loop=false;
        }
      if (infile.good()) infile >> significance;
      infile.close();
    }

  system(("rm " + outpath + "_temp_1.txt").c_str());
  system(("rm " + outpath + "_temp_2.txt").c_str());                                                                                                                                                       

  return significance;
}
