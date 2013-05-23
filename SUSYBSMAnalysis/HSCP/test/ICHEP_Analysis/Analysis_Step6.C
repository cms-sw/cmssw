
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
#include "CL95.h"

using namespace std;



struct stAllInfo{
   double Mass;
   double XSec_Th;
   double XSec_Err;
   double XSec_Exp;
   double XSec_Obs;
   double Eff;
   double Eff_SYSTA;
   double Eff_SYSTB;
   double WP_Pt;
   double WP_I;
   double WP_TOF;
   float  NData;
   float  NPred;
   float  NSign;

   stAllInfo(string path=""){
      Mass=-1; XSec_Th=-1; XSec_Err=-1; XSec_Exp=-1; XSec_Obs=-1; Eff=-1; Eff_SYSTA=-1; Eff_SYSTB=-1;
      if(path=="")return;
      FILE* pFile = fopen(path.c_str(),"r");
      fscanf(pFile,"Mass      : %lf\n",&Mass);
      fscanf(pFile,"WP_Pt     : %lf\n",&WP_Pt);
      fscanf(pFile,"WP_I      : %lf\n",&WP_I);
      fscanf(pFile,"WP_TOF    : %lf\n",&WP_TOF);
      fscanf(pFile,"Eff       : %lf\n",&Eff);
      fscanf(pFile,"Eff_SystA : %lf\n",&Eff_SYSTA);
      fscanf(pFile,"Eff_SystB : %lf\n",&Eff_SYSTB);
      fscanf(pFile,"XSec_Th   : %lf\n",&XSec_Th);
      fscanf(pFile,"XSec_Exp  : %lf\n",&XSec_Exp);
      fscanf(pFile,"XSec_Obs  : %lf\n",&XSec_Obs);
      fscanf(pFile,"NData     : %E\n" ,&NData);
      fscanf(pFile,"NPred     : %E\n" ,&NPred);
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

TGraph* MakePlot(FILE* pFile, string InputPattern, string ModelName, int XSectionType=2, string Mass0="", string Mass1="", string Mass2="", string Mass3="", string Mass4="", string Mass5="");


stAllInfo Exclusion(string pattern, string modelName, string signal, double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1);
int      JobIdToIndex(string JobId);

void GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex);
double FindIntersection(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty=0, bool debug=false);
int ReadXSection(string InputFile, double* Mass, double* XSec, double* Low, double* High,  double* ErrLow, double* ErrHigh);
TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High);


//double PlotMinScale = 0.1;
//double PlotMaxScale = 50000;

//double PlotMinScale = 2;
//double PlotMaxScale = 800;

double PlotMinScale = 0.1;
double PlotMaxScale = 100;



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
std::vector<double> signalsMeanHSCPPerEvent_SYSTA;
std::vector<double> signalsMeanHSCPPerEvent_SYSTB;

double RescaleFactor;
double RescaleError;
int Mode=0;
void Analysis_Step6(string MODE="COMPILE", string InputPattern="", string modelName="", string signal="", double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1){
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
      stAllInfo result = Exclusion(InputPattern, modelName, signal, Ratio_0C, Ratio_1C, Ratio_2C);
      return;
   }


   TCanvas* c1;

   string MuPattern  = "Results/dedxASmi/combined/Eta25/PtMin20/Type2/";
   string TkPattern  = "Results/dedxASmi/combined/Eta25/PtMin20/Type0/";

   FILE* pFile = fopen("Analysis_Step6_Result.txt","w");

   fprintf(pFile, "\\documentclass{article}\n");
   fprintf(pFile, "\\begin{document}\n\n");
   fprintf(pFile, "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");

   cout << "TMPA\n";
   TGraph* Tk_Obs_Gluino2C  = MakePlot(pFile,TkPattern,"Gluino  (2C)     ", 2, "Gluino200_2C" , "Gluino300_2C" , "Gluino400_2C" , "Gluino500_2C" , "Gluino600_2C" , "Gluino900_2C" );
   cout << "TMPB\n";
   TGraph* Tk_Obs_GluinoF0  = MakePlot(pFile,TkPattern,"Gluino  (f=00\\%)", 2, "Gluino200_f0" , "Gluino300_f0" , "Gluino400_f0" , "Gluino500_f0" , "Gluino600_f0" , "Gluino900_f0" );
   TGraph* Tk_Obs_GluinoF1  = MakePlot(pFile,TkPattern,"Gluino  (f=10\\%)", 2, "Gluino200_f1" , "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino900_f1" );
   TGraph* Tk_Obs_GluinoF5  = MakePlot(pFile,TkPattern,"Gluino  (f=50\\%)", 2, "Gluino200_f5" , "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino900_f5" );
   TGraph* Tk_Obs_GluinoNF0 = MakePlot(pFile,TkPattern,"GluinoN (f=00\\%)", 2, "Gluino200N_f0", "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino900N_f0");
   TGraph* Tk_Obs_GluinoNF1 = MakePlot(pFile,TkPattern,"GluinoN (f=10\\%)", 2, "Gluino200N_f1", "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino900N_f1");
   TGraph* Tk_Obs_GluinoNF5 = MakePlot(pFile,TkPattern,"GluinoN (f=50\\%)", 2, "Gluino200N_f5", "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino900N_f5");
   cout << "TMPC\n";
   TGraph* Tk_Obs_Stop2C    = MakePlot(pFile,TkPattern,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop500_2C"   , "Stop800_2C"                    );
   TGraph* Tk_Obs_Stop      = MakePlot(pFile,TkPattern,"Stop             ", 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop500"      , "Stop800"                       );
   TGraph* Tk_Obs_StopN     = MakePlot(pFile,TkPattern,"StopN            ", 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop500N"     , "Stop800N"                      );
   TGraph* Tk_Obs_GMStau    = MakePlot(pFile,TkPattern,"GMSB Stau        ", 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308"    );
   TGraph* Tk_Obs_PPStau    = MakePlot(pFile,TkPattern,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247"    , "PPStau308"    );
   TGraph* Tk_Obs_DCStau    = MakePlot(pFile,TkPattern,"DiChamp    Stau  ", 2, "DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302"                                      );
   cout << "TMPD\n";

   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile, "\\begin{table}\n   \\centering\n      \\begin{tabular}{|l|cccccc|}\n      \\hline\n");
   TGraph* Mu_Obs_Gluino2C  = MakePlot(pFile,MuPattern,"Gluino  (2C)     ", 2, "Gluino200_2C" , "Gluino300_2C" , "Gluino400_2C" , "Gluino500_2C" , "Gluino600_2C" , "Gluino900_2C" );
   TGraph* Mu_Obs_GluinoF0  = MakePlot(pFile,MuPattern,"Gluino  (f=00\\%)", 2, "Gluino200_f0" , "Gluino300_f0" , "Gluino400_f0" , "Gluino500_f0" , "Gluino600_f0" , "Gluino900_f0" );
   TGraph* Mu_Obs_GluinoF1  = MakePlot(pFile,MuPattern,"Gluino  (f=10\\%)", 2, "Gluino200_f1" , "Gluino300_f1" , "Gluino400_f1" , "Gluino500_f1" , "Gluino600_f1" , "Gluino900_f1" );
   TGraph* Mu_Obs_GluinoF5  = MakePlot(pFile,MuPattern,"Gluino  (f=50\\%)", 2, "Gluino200_f5" , "Gluino300_f5" , "Gluino400_f5" , "Gluino500_f5" , "Gluino600_f5" , "Gluino900_f5" );
   TGraph* Mu_Obs_GluinoNF0 = MakePlot(pFile,MuPattern,"GluinoN (f=00\\%)", 2, "Gluino200N_f0", "Gluino300N_f0", "Gluino400N_f0", "Gluino500N_f0", "Gluino600N_f0", "Gluino900N_f0");
   TGraph* Mu_Obs_GluinoNF1 = MakePlot(pFile,MuPattern,"GluinoN (f=10\\%)", 2, "Gluino200N_f1", "Gluino300N_f1", "Gluino400N_f1", "Gluino500N_f1", "Gluino600N_f1", "Gluino900N_f1");
   TGraph* Mu_Obs_GluinoNF5 = MakePlot(pFile,MuPattern,"GluinoN (f=50\\%)", 2, "Gluino200N_f5", "Gluino300N_f5", "Gluino400N_f5", "Gluino500N_f5", "Gluino600N_f5", "Gluino900N_f5");
   TGraph* Mu_Obs_Stop2C    = MakePlot(pFile,MuPattern,"Stop    (2C)     ", 2, "Stop130_2C"   , "Stop200_2C"   , "Stop300_2C"   , "Stop500_2C"   , "Stop800_2C"                    );
   TGraph* Mu_Obs_Stop      = MakePlot(pFile,MuPattern,"Stop             ", 2, "Stop130"      , "Stop200"      , "Stop300"      , "Stop500"      , "Stop800"                       );
   TGraph* Mu_Obs_StopN     = MakePlot(pFile,MuPattern,"StopN            ", 2, "Stop130N"     , "Stop200N"     , "Stop300N"     , "Stop500N"     , "Stop800N"                      );
   TGraph* Mu_Obs_GMStau    = MakePlot(pFile,MuPattern,"GMSB Stau        ", 2, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308"    );
   TGraph* Mu_Obs_PPStau    = MakePlot(pFile,MuPattern,"Pair Prod. Stau  ", 2, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247"    , "PPStau308"    );
   TGraph* Mu_Obs_DCStau    = MakePlot(pFile,MuPattern,"DiChamp    Stau  ", 2, "DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302"                                      );
   cout << "TMPE\n";

   fprintf(pFile,"      \\end{tabular}\n\\end{table}\n\n");
   fprintf(pFile,"\\end{document}\n\n");

   TGraph* GMStauXSec = MakePlot(NULL,MuPattern,"GMSB Stau        ", 0, "GMStau100"    , "GMStau126"    , "GMStau156"    , "GMStau200"    , "GMStau247"    , "GMStau308"    );
   TGraph* PPStauXSec = MakePlot(NULL,MuPattern,"Pair Prod. Stau  ", 0, "PPStau100"    , "PPStau126"    , "PPStau156"    , "PPStau200"    , "PPStau247"    , "PPStau308"    ); 
   TGraph* DCStauXSec = MakePlot(NULL,MuPattern,"DiChamp    Stau  ", 0, "DCStau121"    , "DCStau182"    , "DCStau242"    , "DCStau302"                                      ); 

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

   fprintf(pFile,"-----------------------\n0%% TK ONLY       \n-------------------------\n");
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Tk_Obs_Gluino2C,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Tk_Obs_GluinoF0,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Tk_Obs_GluinoF1,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Tk_Obs_GluinoF5,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Tk_Obs_GluinoNF0, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Tk_Obs_GluinoNF1, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Tk_Obs_GluinoNF5, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Tk_Obs_Stop2C   , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Tk_Obs_Stop     , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Tk_Obs_StopN    , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Tk_Obs_GMStau   , GMStauXSec   , 100, 808, 1, 0.15));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Tk_Obs_PPStau   , PPStauXSec   , 100, 308, 1, 0.15));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Tk_Obs_DCStau   , DCStauXSec   , 121, 302, 1, 0.15));

   fprintf(pFile,"-----------------------\n0%% TK TOF        \n-------------------------\n");
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Mu_Obs_Gluino2C,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Mu_Obs_GluinoF0,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Mu_Obs_GluinoF1,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Mu_Obs_GluinoF5,  GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Mu_Obs_GluinoNF0, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Mu_Obs_GluinoNF1, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Mu_Obs_GluinoNF5, GluinoXSecLow, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Mu_Obs_Stop2C   , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Mu_Obs_Stop     , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Mu_Obs_StopN    , StopXSecLow  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Mu_Obs_GMStau   , GMStauXSec   , 100, 808, 1, 0.15));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Mu_Obs_PPStau   , PPStauXSec   , 100, 308, 1, 0.15));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Mu_Obs_DCStau   , DCStauXSec   , 121, 302, 1, 0.15));


   fprintf(pFile,"-----------------------\nNO TH UNCERTAINTY ACCOUNTED FOR   \n-------------------------\n");

   fprintf(pFile,"-----------------------\n0%% TK ONLY       \n-------------------------\n");
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Tk_Obs_Gluino2C,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Tk_Obs_GluinoF0,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Tk_Obs_GluinoF1,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Tk_Obs_GluinoF5,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Tk_Obs_GluinoNF0, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Tk_Obs_GluinoNF1, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Tk_Obs_GluinoNF5, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Tk_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Tk_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Tk_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Tk_Obs_GMStau   , GMStauXSec   , 100, 808, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Tk_Obs_PPStau   , PPStauXSec   , 100, 308, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Tk_Obs_DCStau   , DCStauXSec   , 121, 302, 1, 0.0));
   
   fprintf(pFile,"-----------------------\n0%% TK TOF        \n-------------------------\n");
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(Mu_Obs_Gluino2C,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(Mu_Obs_GluinoF0,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(Mu_Obs_GluinoF1,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(Mu_Obs_GluinoF5,  GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(Mu_Obs_GluinoNF0, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(Mu_Obs_GluinoNF1, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(Mu_Obs_GluinoNF5, GluinoXSec, 200, 900, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(Mu_Obs_Stop2C   , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(Mu_Obs_Stop     , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(Mu_Obs_StopN    , StopXSec  , 130, 800, 1, 0.00));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for GMStau   \n", FindIntersection(Mu_Obs_GMStau   , GMStauXSec   , 100, 808, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for PPStau   \n", FindIntersection(Mu_Obs_PPStau   , PPStauXSec   , 100, 308, 1, 0.0));
   fprintf(pFile,"MASS EXCLUDED UP TO %8.3fGeV for DCStau   \n", FindIntersection(Mu_Obs_DCStau   , DCStauXSec   , 121, 302, 1, 0.0));

   fclose(pFile);


   string outpath = string("Results/EXCLUSION/");
   MakeDirectories(outpath);

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
   Mu_Obs_GMStau   ->SetLineColor(1);  Mu_Obs_GMStau   ->SetMarkerColor(1);   Mu_Obs_GMStau   ->SetLineWidth(2);   Mu_Obs_GMStau   ->SetLineStyle(1);  Mu_Obs_GMStau   ->SetMarkerStyle(20);
   Tk_Obs_GMStau   ->SetLineColor(1);  Tk_Obs_GMStau   ->SetMarkerColor(1);   Tk_Obs_GMStau   ->SetLineWidth(2);   Tk_Obs_GMStau   ->SetLineStyle(1);  Tk_Obs_GMStau   ->SetMarkerStyle(20);




   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* MGMu = new TMultiGraph();
   MGMu->Add(GluinoXSec      ,"L");
   MGMu->Add(StopXSec        ,"L");
   MGMu->Add(GMStauXSec      ,"L");
   MGMu->Add(Mu_Obs_GluinoF1      ,"LP");
   MGMu->Add(Mu_Obs_GluinoF5      ,"LP");
   MGMu->Add(Mu_Obs_GluinoNF1     ,"LP");
   MGMu->Add(Mu_Obs_Stop          ,"LP");
   MGMu->Add(Mu_Obs_StopN         ,"LP");
   MGMu->Add(Mu_Obs_GMStau        ,"LP");
   MGMu->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr  ->Draw("f");
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
   LEGMu->AddEntry(Mu_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(Mu_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGMu->AddEntry(Mu_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGMu->AddEntry(Mu_Obs_Stop     , "stop"            ,"LP");
   LEGMu->AddEntry(Mu_Obs_StopN    , "stop; ch. suppr.","LP");
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
   LEGTh->AddEntry(GMStauXSec   ,"GMSB stau   (LO)" ,"L");
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
   MGTk->Add(Tk_Obs_GluinoF1      ,"LP");
   MGTk->Add(Tk_Obs_GluinoF5      ,"LP");
   MGTk->Add(Tk_Obs_GluinoNF1     ,"LP");
   MGTk->Add(Tk_Obs_Stop          ,"LP");
   MGTk->Add(Tk_Obs_StopN         ,"LP");
   MGTk->Add(Tk_Obs_GMStau        ,"LP");
   MGTk->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr  ->Draw("f");
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
   LEGTk->AddEntry(Tk_Obs_GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   LEGTk->AddEntry(Tk_Obs_GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_Stop     , "stop"            ,"LP");
   LEGTk->AddEntry(Tk_Obs_StopN    , "stop; ch. suppr.","LP");
   LEGTk->AddEntry(Tk_Obs_GMStau   , "GMSB stau"       ,"LP");
   LEGTk->Draw();

   LEGTh->Draw();

//   c1->SetGridx(true);
//   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("TkExclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("TkExclusionLog"));
   delete c1;

   return; 
}


TGraph* MakePlot(FILE* pFile, string InputPattern, string ModelName, int XSectionType, string Mass0, string Mass1, string Mass2, string Mass3, string Mass4, string Mass5){
   unsigned int N=0;
   stAllInfo Infos[6];

   if(Mass0!=""){Infos[0] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass0+".txt"); N=1;}
   if(Mass1!=""){Infos[1] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass1+".txt"); N=2;}
   if(Mass2!=""){Infos[2] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass2+".txt"); N=3;}
   if(Mass3!=""){Infos[3] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass3+".txt"); N=4;}
   if(Mass4!=""){Infos[4] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass4+".txt"); N=5;}
   if(Mass5!=""){Infos[5] = stAllInfo(InputPattern+"/EXCLUSION/"+Mass5+".txt"); N=6;}

   double Mass   [6];for(unsigned int i=0;i<6;i++){Mass   [i]=Infos[i].Mass;    }
   double XSecTh [6];for(unsigned int i=0;i<6;i++){XSecTh [i]=Infos[i].XSec_Th; }
   double XSecObs[6];for(unsigned int i=0;i<6;i++){XSecObs[i]=Infos[i].XSec_Obs;}
   double XSecExp[6];for(unsigned int i=0;i<6;i++){XSecExp[i]=Infos[i].XSec_Exp;}

   if(pFile){
      fprintf(pFile,"%40s",(ModelName + " mass (GeV/$c^2$)").c_str());for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.0f ",Infos[i].Mass);}     for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\\hline\n");
      fprintf(pFile,"%40s","Total acceptance (\\%)");                 for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",100.*Infos[i].Eff);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Expected 95\\% C.L. limit (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Exp);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Observed 95\\% C.L. limit (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Obs);} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\n");
      fprintf(pFile,"%40s","Theoretical cross section (pb) ");        for(unsigned int i=0;i<N;i++){fprintf(pFile,"& %7.3f ",Infos[i].XSec_Th );} for(unsigned int i=N;i<6;i++){fprintf(pFile,"& ");}fprintf(pFile,"\\\\\\hline\\hline\n");
   }


   for(unsigned int i=0;i<N;i++)printf("%s %7.0f -->  WP_Pt = %7.2f  WP_I = %7.2f  WP_TOF=%7.2f\n",ModelName.c_str(),Infos[i].Mass,Infos[i].WP_Pt,Infos[i].WP_I,Infos[i].WP_TOF);
   
   TGraph* graph = NULL;
   if(XSectionType==0)graph = new TGraph(N,Mass,XSecTh);
   if(XSectionType==1)graph = new TGraph(N,Mass,XSecExp);
   if(XSectionType==2)graph = new TGraph(N,Mass,XSecObs);
   graph->SetTitle("");
   graph->GetYaxis()->SetTitle("CrossSection ( pb )");
   graph->GetYaxis()->SetTitleOffset(1.70);
   return graph;
}

stAllInfo Exclusion(string pattern, string modelName, string signal, double Ratio_0C, double Ratio_1C, double Ratio_2C){
   stAllInfo toReturn;
   toReturn.XSec_Obs  = 1E50;
   toReturn.XSec_Exp  = 1E50;

   double RescaleFactor = 1.0;
   double RescaleError  = 0.1;

   double RatioValue[] = {Ratio_0C, Ratio_1C, Ratio_2C};

   GetSignalDefinition(signals);
   CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  } 

   string outpath = pattern + "/EXCLUSION/";
   MakeDirectories(outpath);


   FILE* pFile = fopen((outpath+"/"+modelName+".info").c_str(),"w");
   if(!pFile)printf("Can't open file : %s\n",(outpath+"/"+modelName+".info").c_str());

   string InputPath     = pattern + "Histos.root";
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
   MassSign[0]          = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "/Mass");
   MassSign[1]          = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC0/Mass");
   MassSign[2]          = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC1/Mass");
   MassSign[3]          = (TH2D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC2/Mass");

   for(int CutIndex=0;CutIndex<MassData->GetNbinsX();CutIndex++){
      if(H_E->GetBinContent(CutIndex+1) >0 && (H_A->GetBinContent(CutIndex+1)<50 || H_F->GetBinContent(CutIndex+1)<50 || H_G->GetBinContent(CutIndex+1)<50))continue;  //Skip events where Prediction (AFG/EE) is not reliable
      if(H_E->GetBinContent(CutIndex+1)==0 && (H_C->GetBinContent(CutIndex+1)<50 || H_B->GetBinContent(CutIndex+1)<50))continue;  //Skip events where Prediction (CB/A) is not reliable

      GetSignalMeanHSCPPerEvent(pattern,CutIndex);
      TH1D* MassDataProj = MassData->ProjectionY("MassDataProj",CutIndex+1,CutIndex+1);
      TH1D* MassPredProj = MassPred->ProjectionY("MassPredProj",CutIndex+1,CutIndex+1);
      TH1D* MassSignProj[4];
      MassSignProj[0] = MassSign[0]->ProjectionY("MassSignProj0",CutIndex+1,CutIndex+1);
      MassSignProj[1] = MassSign[1]->ProjectionY("MassSignProj1",CutIndex+1,CutIndex+1);
      MassSignProj[2] = MassSign[2]->ProjectionY("MassSignProj2",CutIndex+1,CutIndex+1);
      MassSignProj[3] = MassSign[3]->ProjectionY("MassSignProj3",CutIndex+1,CutIndex+1);


      double Mean  = MassSignProj[0]->GetMean();
      double Width = MassSignProj[0]->GetRMS();
      if(RatioValue[0]<0 && RatioValue[1]<0 && RatioValue[2]<0){
         Mean  = (MassSignProj[1]->GetMean()*RatioValue[0] + MassSignProj[2]->GetMean()*RatioValue[1] + MassSignProj[3]->GetMean()*RatioValue[2]) / (RatioValue[0]+RatioValue[1]+RatioValue[2]);
         Width = (MassSignProj[1]->GetRMS() *RatioValue[0] + MassSignProj[2]->GetRMS() *RatioValue[1] + MassSignProj[3]->GetRMS() *RatioValue[2]) / (RatioValue[0]+RatioValue[1]+RatioValue[2]);
      }
      MinRange = std::max(0.0, Mean-2*Width);
//      MinRange = 0;

      //fprintf(pFile  ,"%10s: Signal Mean = %7.2f  Signal RMS = %7.2f --> MinRange=%7.2f\n",signal.c_str(),Mean,Width,MinRange);fflush(stdout);

      double NData       = MassDataProj->Integral(MassDataProj->GetXaxis()->FindBin(MinRange), MassDataProj->GetXaxis()->FindBin(MaxRange));
      double NPredErr    = 0;
      for(int i=MassPredProj->GetXaxis()->FindBin(MinRange); i<=MassPredProj->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=(MassPredProj->GetBinError(i)*MassPredProj->GetBinError(i));}NPredErr=sqrt(NPredErr);
      double NPred       = MassPredProj->Integral(MassPredProj->GetXaxis()->FindBin(MinRange), MassPredProj->GetXaxis()->FindBin(MaxRange));

      if(isnan(NPred))continue;
      if(NPred<=0){continue;} //Is <=0 only when prediction failed or is not meaningful (i.e. WP=(0,0,0) )
      if(NPred>10){continue;}  //When NPred is too big, expected limits just take an infinite time! 

      double Eff       = 0;
      if(RatioValue[0]<0 && RatioValue[1]<0 && RatioValue[2]<0){
            CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  } 
            double INTERN_ESign       = MassSignProj[0]->Integral(MassSignProj[0]            ->GetXaxis()->FindBin(MinRange), MassSignProj[0]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [4*CurrentSampleIndex]; 
            double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            Eff       = INTERN_Eff;
            //fprintf(pFile  ,"%10s: INTERN_ESign=%6.2E   INTERN_Eff=%6.E   XSec=%6.2E   Lumi=%6.2E\n",signal.c_str(),INTERN_ESign,INTERN_Eff,signals[CurrentSampleIndex].XSec, IntegratedLuminosity);fflush(stdout);
      }else{
         for(unsigned int i=0;i<3;i++){
            CurrentSampleIndex        = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  }
            double INTERN_ESign       = MassSignProj[i+1]->Integral(MassSignProj[i+1]            ->GetXaxis()->FindBin(MinRange), MassSignProj[i+1]      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [4*CurrentSampleIndex+1+i]; 
            double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
            Eff       += INTERN_Eff   * RatioValue[i];
         }
      }
      if(Eff==0){continue;}
      NPred*=RescaleFactor;

     //fprintf(pFile ,"CutIndex=%4i ManHSCPPerEvents = %6.2f %6.2f %6.2f %6.2f   NTracks = %6.3f %6.3f %6.3f %6.3f\n",CutIndex,signalsMeanHSCPPerEvent[4*CurrentSampleIndex], signalsMeanHSCPPerEvent[4*CurrentSampleIndex+1],signalsMeanHSCPPerEvent[4*CurrentSampleIndex+2],signalsMeanHSCPPerEvent[4*CurrentSampleIndex+3], MassSignProj[0]->Integral(), MassSignProj[1]->Integral(), MassSignProj[2]->Integral(), MassSignProj[3]->Integral());


     fprintf(pFile  ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.2f TOF>%6.2f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E SignalEff=%6.2f",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred,Eff);fflush(stdout);
     fprintf(stdout ,"%10s: Testing CutIndex=%4i (Pt>%6.2f I>%6.2f TOF>%6.2f) %3.0f<M<inf Ndata=%+6.2E NPred=%6.3E SignalEff=%6.2f",signal.c_str(),CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), MinRange,NData,NPred,Eff);fflush(stdout);

     double ExpLimit = 99999999;
     double ObsLimit = 99999999;
     if(NData<NPred){
	 ObsLimit =  CL95(IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15,NPred, NPred*RescaleError              , NData, false, 1);
         if(toReturn.XSec_Exp<ObsLimit){fprintf(pFile  ,"\n"); printf("\n"); continue;}
         ExpLimit =  CLA (IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15, NPred, NPred*RescaleError, 1);
         if(toReturn.XSec_Exp<ExpLimit){fprintf(pFile  ,"\n"); printf("\n"); continue;}
     }else{
         ExpLimit =  CLA (IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15, NPred, NPred*RescaleError, 1);
         if(toReturn.XSec_Exp<ExpLimit){fprintf(pFile  ,"\n"); printf("\n"); continue;}
         ObsLimit =  CL95(IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15,NPred, NPred*RescaleError              , NData, false, 1);
     }
     fprintf(pFile ," --> %+7.2E expected (%+7.2E observed) --> Current Best Limit\n",ExpLimit, ObsLimit);
     fprintf(stdout," --> %+7.2E expected (%+7.2E observed) --> Current Best Limit\n",ExpLimit, ObsLimit);

    

     toReturn.Mass      = signals[JobIdToIndex(signal)].Mass;
     toReturn.WP_Pt     = HCuts_Pt ->GetBinContent(CutIndex+1);
     toReturn.WP_I      = HCuts_I  ->GetBinContent(CutIndex+1);
     toReturn.WP_TOF    = HCuts_TOF->GetBinContent(CutIndex+1);
     toReturn.XSec_Th   = signals[JobIdToIndex(signal)].XSec;
     toReturn.XSec_Err  = signals[JobIdToIndex(signal)].XSec * 0.15;
     toReturn.XSec_Exp  = ExpLimit;
     toReturn.XSec_Obs  = ObsLimit; 
     toReturn.Eff       = Eff;
     toReturn.Eff_SYSTA = 0;//Eff_SYSTA;
     toReturn.Eff_SYSTB = 0;//Eff_SYSTB;
     toReturn.NData     = NData;
     toReturn.NPred     = NPred;
     toReturn.NSign     = Eff*(signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
   }
   fclose(pFile);    

   pFile = fopen((outpath+"/"+modelName+".txt").c_str(),"w");
   fprintf(pFile,"Mass      : %f\n",signals[JobIdToIndex(signal)].Mass);
   fprintf(pFile,"WP_Pt     : %f\n",toReturn.WP_Pt);
   fprintf(pFile,"WP_I      : %f\n",toReturn.WP_I);
   fprintf(pFile,"WP_TOF    : %f\n",toReturn.WP_TOF);
   fprintf(pFile,"Eff       : %f\n",toReturn.Eff);
   fprintf(pFile,"Eff_SystA : %f\n",toReturn.Eff_SYSTA);
   fprintf(pFile,"Eff_SystB : %f\n",toReturn.Eff_SYSTB);
   fprintf(pFile,"XSec_Th   : %f\n",toReturn.XSec_Th);
   fprintf(pFile,"XSec_Exp  : %f\n",toReturn.XSec_Exp);
   fprintf(pFile,"XSec_Obs  : %f\n",toReturn.XSec_Obs);
   fprintf(pFile,"NData     : %+6.2E\n",toReturn.NData);
   fprintf(pFile,"NPred     : %+6.2E\n",toReturn.NPred);
   fprintf(pFile,"NSign     : %+6.2E\n",toReturn.NSign);
   fclose(pFile);
   return toReturn;
}






int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}


void GetSignalMeanHSCPPerEvent(string InputPattern, unsigned int CutIndex)
{
   string InputPath     = InputPattern + "Histos.root";
   TFile* InputFile     = new TFile(InputPath.c_str());

   signalsMeanHSCPPerEvent.clear();
   signalsMeanHSCPPerEvent_SYSTA.clear();
   signalsMeanHSCPPerEvent_SYSTB.clear();
   for(unsigned int s=0;s<signals.size();s++){
   for(unsigned int n=0;n<4;n++){
      signalsMeanHSCPPerEvent.push_back(2.0);
      signalsMeanHSCPPerEvent_SYSTA.push_back(2.0);
      signalsMeanHSCPPerEvent_SYSTB.push_back(2.0); 
   }}

   for(unsigned int s=0;s<signals.size();s++){
      TH1D*  NTracksPassingSelection     = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "/TOF");
      TH1D*  NEventsPassingSelection     = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name          + "/HSCPE");
      TH1D*  NTracksPassingSelection_NC0 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC0" + "/TOF");
      TH1D*  NEventsPassingSelection_NC0 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC0" + "/HSCPE");
      TH1D*  NTracksPassingSelection_NC1 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC1" + "/TOF");
      TH1D*  NEventsPassingSelection_NC1 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC1" + "/HSCPE");
      TH1D*  NTracksPassingSelection_NC2 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC2" + "/TOF");
      TH1D*  NEventsPassingSelection_NC2 = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_NC2" + "/HSCPE");

      signalsMeanHSCPPerEvent    [4*s  ] = (float)std::min(1.0,NTracksPassingSelection    ->GetBinContent(CutIndex+1) / NEventsPassingSelection    ->GetBinContent(CutIndex+1));
      signalsMeanHSCPPerEvent    [4*s+1] = (float)std::min(1.0,NTracksPassingSelection_NC0->GetBinContent(CutIndex+1) / NEventsPassingSelection_NC0->GetBinContent(CutIndex+1));
      signalsMeanHSCPPerEvent    [4*s+2] = (float)std::min(1.0,NTracksPassingSelection_NC1->GetBinContent(CutIndex+1) / NEventsPassingSelection_NC1->GetBinContent(CutIndex+1));
      signalsMeanHSCPPerEvent    [4*s+3] = (float)std::min(1.0,NTracksPassingSelection_NC2->GetBinContent(CutIndex+1) / NEventsPassingSelection_NC2->GetBinContent(CutIndex+1));

//     signalsMeanHSCPPerEvent_SYSTA[4*Index+n] = (float)std::min(1.0f,weff_SYSTA);
//     signalsMeanHSCPPerEvent_SYSTB[4*Index+n] = (float)std::min(1.0f,weff_SYSTB);

      delete NTracksPassingSelection;
      delete NEventsPassingSelection;
      delete NTracksPassingSelection_NC0;
      delete NEventsPassingSelection_NC0;
      delete NTracksPassingSelection_NC1;
      delete NEventsPassingSelection_NC1;
      delete NTracksPassingSelection_NC2;
      delete NEventsPassingSelection_NC2;    
   }

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

