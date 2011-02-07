
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

//double PlotMinScale = 0.1;
//double PlotMaxScale = 50000;

//double PlotMinScale = 2;
//double PlotMaxScale = 800;

double PlotMinScale = 0.01;
double PlotMaxScale = 10;



struct stResult{
   double SignalCrossSection;
   double SignalMean;
   double SignalSigma;
   double SL3Sigma;
   double SL5Sigma;
   double SLMedian;
   double SLObs;
   double SLObsIntegral;
};


struct stAllInfo{
   double Mass;
   double XSec_Th;
   double XSec_Err;
   double XSec_Exp;
   double XSec_Obs;
   double Eff;
   double Eff_SYSTA;
   double Eff_SYSTB;

   stAllInfo(){Mass=-1; XSec_Th=-1; XSec_Err=-1; XSec_Exp=-1; XSec_Obs=-1; Eff=-1; Eff_SYSTA=-1; Eff_SYSTB=-1;}
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

using namespace std;

Double_t fitPred(Double_t *v, Double_t *par);
Double_t fitPredAndSignal(Double_t *v, Double_t *par);

TH1D*    GetPDF(TH1D* pdf);
TH1D*    GetPDF(TF1* pdf, string name, int NBins, int Xmin, int Xmax);
double   GetRandValue(TH1D* PDF);
TH1D*    MakePseudoExperiment(TH1D* PDF, double NEntries, int NBins=0);
double   GetMedian(TH1D* pdf);
double   SigmaFromProb(double p, string where="outside");
double   LogLikeliHood(TH1D* Histo1, TH1D* Histo2);
double   LogLikeliHood(TH1D* Histo1, TF1*  Histo2);
double   GetS(TH1D* Data, double* FitParams, double SignalMean, double SignalSigma, TCanvas* c0=NULL);
stGraph  Analysis_Step6_Core(string);
void     Analysis_Step6_Init(string signal, string Path);
void     Analysis_Step6_SLDistrib(stResult& results);
double   GetIntegralOnLeft(TH1D* pdf, double IntegralRatio);

stAllInfo   Exclusion(string signal, string pattern, double Ratio_0C=-1, double Ratio_1C=-1, double Ratio_2C=-1);
stAllInfo   Exclusion_LL(string signal, string pattern);
stAllInfo   Exclusion_Counting(string signal, string pattern, double Ratio_0C, double Ratio_1C, double Ratio_2C);

void     SimRecoCorrelation(string InputPattern);
int      JobIdToIndex(string JobId);

TGraph* PopulateTheGraph(TGraph* in, double Min, double Max, double Step);

void GetSignalMeanHSCPPerEvent(string InputPattern);
double FindIntersection(TGraph* obs, TGraph* th, double Min, double Max, double Step, double ThUncertainty=0, bool debug=false);
int ReadXSection(string InputFile, double* Mass, double* XSec, double* Low, double* High,  double* ErrLow, double* ErrHigh);
TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High);

double MinRange = 75;
double MaxRange = 999;

unsigned int NPseudoExperiment = 1000;

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
void Analysis_Step6(){
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
//   gStyle->SetNdivisions(509,"X");


   MinRange      = 75;
   Mode          = 0;   
   stGraph tkmu = Analysis_Step6_Core("Results/dedxASmi/dt/Eta25/Type2/SplitMode0/WPPt20/WPI20/WPTOF20/");
//   stGraph tkmu = Analysis_Step6_Core("Results/dedxASmi/dt/Eta25/Type1/SplitMode0/WPPt35/WPI35/WPTOF00/");
//   stGraph tkmu = Analysis_Step6_Core("Results/dedxASmi/dt/Eta25/Type0/SplitMode0/WPPt35/WPI35/WPTOF00/");
   return;

   MinRange = 75;
   Mode     = 0;
   stGraph tkonly =  Analysis_Step6_Core("Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-35/WPI-40/");//X


   string outpath = string("Results/EXCLUSION/");
   MakeDirectories(outpath);

   tkonly.GluinoTh ->SetLineColor(4);  tkonly.GluinoTh ->SetLineWidth(1);   tkonly.GluinoTh ->SetLineStyle(3);  tkonly.GluinoTh ->SetMarkerStyle(1);
   tkmu  .GluinoF1 ->SetLineColor(4);  tkmu  .GluinoF1 ->SetLineWidth(2);   tkmu  .GluinoF1 ->SetLineStyle(1);  tkmu  .GluinoF1 ->SetMarkerStyle(22);
   tkmu  .GluinoF5 ->SetLineColor(4);  tkmu  .GluinoF5 ->SetLineWidth(2);   tkmu  .GluinoF5 ->SetLineStyle(1);  tkmu  .GluinoF5 ->SetMarkerStyle(23);
   tkonly.GluinoNF1->SetLineColor(4);  tkonly.GluinoNF1->SetLineWidth(2);   tkonly.GluinoNF1->SetLineStyle(1);  tkonly.GluinoNF1->SetMarkerStyle(26);
   tkonly.StopTh   ->SetLineColor(2);  tkonly.StopTh   ->SetLineWidth(1);   tkonly.StopTh   ->SetLineStyle(2);  tkonly.StopTh   ->SetMarkerStyle(1);
   tkmu  .Stop     ->SetLineColor(2);  tkmu  .Stop     ->SetLineWidth(2);   tkmu  .Stop     ->SetLineStyle(1);  tkmu  .Stop     ->SetMarkerStyle(21);
   tkonly.StopN    ->SetLineColor(2);  tkonly.StopN    ->SetLineWidth(2);   tkonly.StopN    ->SetLineStyle(1);  tkonly.StopN    ->SetMarkerStyle(25);

   TCanvas* c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* mgA = new TMultiGraph();
   mgA->Add(tkonly.GluinoTh      ,"L");
   mgA->Add(tkonly.StopTh        ,"L");
   mgA->Add(tkmu  .GluinoF1      ,"LP");
   mgA->Add(tkmu  .GluinoF5      ,"LP");
   mgA->Add(tkonly.GluinoNF1     ,"LP");
   mgA->Add(tkmu  .Stop          ,"LP");
   mgA->Add(tkonly.StopN         ,"LP");
   mgA->Draw("A");
   tkonly.GluinoThErr->Draw("f");
   tkonly.StopThErr  ->Draw("f");
   mgA->Draw("same");
   mgA->SetTitle("");
   mgA->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   mgA->GetYaxis()->SetTitle("#sigma (pb)");
   mgA->GetYaxis()->SetTitleOffset(1.70);
   mgA->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   
   DrawPreliminary(IntegratedLuminosity);
   
   TLegend* legA = new TLegend(0.50,0.65,0.80,0.90);
   legA->SetHeader("95% C.L. Limits");
   legA->SetFillColor(0); 
   legA->SetBorderSize(0);
   legA->AddEntry(tkmu  .GluinoF1 , "gluino; 10% #tilde{g}g"    ,"LP");
   legA->AddEntry(tkmu  .GluinoF5 , "gluino; 50% #tilde{g}g"    ,"LP");
   legA->AddEntry(tkonly.GluinoNF1, "gluino; 10% #tilde{g}g; ch. suppr.","LP");
   legA->AddEntry(tkmu  .Stop     , "stop"            ,"LP");
   legA->AddEntry(tkonly.StopN    , "stop; ch. suppr.","LP");
   legA->Draw();
   
   TLegend* leg2A = new TLegend(0.15,0.70,0.50,0.90);
   leg2A->SetHeader("Theoretical Prediction");
   leg2A->SetFillColor(0);
   leg2A->SetBorderSize(0);
   TGraph* GlThLeg = (TGraph*) tkonly.GluinoTh->Clone("GluinoThLeg");
   GlThLeg->SetFillColor(tkonly.GluinoThErr->GetFillColor());
   leg2A->AddEntry(GlThLeg, "gluino (NLO+NLL)" ,"LF");
   TGraph* StThLeg = (TGraph*) tkonly.StopTh->Clone("StopThLeg");
   StThLeg->SetFillColor(tkonly.GluinoThErr->GetFillColor());
   leg2A->AddEntry(StThLeg   ,"stop   (NLO+NLL)" ,"LF");
//   leg2A->AddEntry(tkonly.StopThErr,"Th. Uncertainty"  ,"F");
   leg2A->Draw();

   c1->SetGridx(true);
   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("Exclusion"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("ExclusionLog"));
   delete c1;

}

stGraph Analysis_Step6_Core(string InputPattern){
   TCanvas* c1;


//   GetPredictionRescale(InputPattern,RescaleFactor, RescaleError, true);
//   RescaleError*=2.0;

   RescaleFactor = 1.0;
   RescaleError  = 0.1;


   GetSignalDefinition(signals);
   GetSignalMeanHSCPPerEvent(InputPattern);
//   SimRecoCorrelation(InputPattern);

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

   string outpath = InputPattern + "/EXCLUSION/"; 
   MakeDirectories(outpath);


//   double Gluino2000 = Exclusion("Gluino200",InputPattern);
//   double Gluino200B = Exclusion("Gluino200",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
//   double Gluino200A = Exclusion("Gluino200",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
//   printf("%f --> %f --> %f\n",Gluino2000,Gluino200B,Gluino200A);

   stAllInfo Gluino200_2C = Exclusion("Gluino200",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
   stAllInfo Gluino300_2C = Exclusion("Gluino300",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
   stAllInfo Gluino400_2C = Exclusion("Gluino400",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
   stAllInfo Gluino500_2C = Exclusion("Gluino500",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
   stAllInfo Gluino600_2C = Exclusion("Gluino600",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);
   stAllInfo Gluino900_2C = Exclusion("Gluino900",InputPattern, 0.0    / 0.3029 , 0.0    / 0.4955 , 1.0    / 0.2015);

   stAllInfo Gluino200_f0 = Exclusion("Gluino200",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino300_f0 = Exclusion("Gluino300",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino400_f0 = Exclusion("Gluino400",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino500_f0 = Exclusion("Gluino500",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino600_f0 = Exclusion("Gluino600",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino900_f0 = Exclusion("Gluino900",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);

   stAllInfo Gluino200_f1 = Exclusion("Gluino200",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino300_f1 = Exclusion("Gluino300",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino400_f1 = Exclusion("Gluino400",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino500_f1 = Exclusion("Gluino500",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino600_f1 = Exclusion("Gluino600",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino900_f1 = Exclusion("Gluino900",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);

   stAllInfo Gluino200_f5 = Exclusion("Gluino200",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino300_f5 = Exclusion("Gluino300",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino400_f5 = Exclusion("Gluino400",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino500_f5 = Exclusion("Gluino500",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino600_f5 = Exclusion("Gluino600",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino900_f5 = Exclusion("Gluino900",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);

   stAllInfo Gluino200N_f0 = Exclusion("Gluino200N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino300N_f0 = Exclusion("Gluino300N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino400N_f0 = Exclusion("Gluino400N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino500N_f0 = Exclusion("Gluino500N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino600N_f0 = Exclusion("Gluino600N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);
   stAllInfo Gluino900N_f0 = Exclusion("Gluino900N",InputPattern, 0.2524 / 0.3029 , 0.4893 / 0.4955 , 0.2583 / 0.2015);

   stAllInfo Gluino200N_f1= Exclusion("Gluino200N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino300N_f1= Exclusion("Gluino300N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino400N_f1= Exclusion("Gluino400N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino500N_f1= Exclusion("Gluino500N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino600N_f1= Exclusion("Gluino600N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);
   stAllInfo Gluino900N_f1= Exclusion("Gluino900N",InputPattern, 0.3029 / 0.3029 , 0.4955 / 0.4955 , 0.2015 / 0.2015);

   stAllInfo Gluino200N_f5 = Exclusion("Gluino200N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino300N_f5 = Exclusion("Gluino300N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino400N_f5 = Exclusion("Gluino400N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino500N_f5 = Exclusion("Gluino500N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino600N_f5 = Exclusion("Gluino600N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);
   stAllInfo Gluino900N_f5 = Exclusion("Gluino900N",InputPattern, 0.5739 / 0.3029 , 0.3704 / 0.4955 , 0.0557 / 0.2015);

   stAllInfo Stop130_2C   = Exclusion("Stop130",InputPattern, 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427);
   stAllInfo Stop200_2C   = Exclusion("Stop200",InputPattern, 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427);
   stAllInfo Stop300_2C   = Exclusion("Stop300",InputPattern, 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427);
   stAllInfo Stop500_2C   = Exclusion("Stop500",InputPattern, 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427);
   stAllInfo Stop800_2C   = Exclusion("Stop800",InputPattern, 0.0    / 0.1705 , 0.0    / 0.4868 , 1.0    / 0.3427);

   stAllInfo Stop130      = Exclusion("Stop130",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop200      = Exclusion("Stop200",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop300      = Exclusion("Stop300",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop500      = Exclusion("Stop500",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop800      = Exclusion("Stop800",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);

   stAllInfo Stop130N     = Exclusion("Stop130N",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop200N     = Exclusion("Stop200N",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop300N     = Exclusion("Stop300N",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop500N     = Exclusion("Stop500N",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);
   stAllInfo Stop800N     = Exclusion("Stop800N",InputPattern, 0.1705 / 0.1705 , 0.4868 / 0.4868 , 0.3427 / 0.3427);

   stAllInfo GMStau100      = Exclusion("GMStau100"  ,InputPattern);
   stAllInfo GMStau126      = Exclusion("GMStau126"  ,InputPattern);
   stAllInfo GMStau156      = Exclusion("GMStau156"  ,InputPattern);
   stAllInfo GMStau200      = Exclusion("GMStau200"  ,InputPattern);
   stAllInfo GMStau247      = Exclusion("GMStau247"  ,InputPattern);
   stAllInfo GMStau308      = Exclusion("GMStau308"  ,InputPattern);

   stAllInfo PPStau100      = Exclusion("PPStau100"  ,InputPattern);
   stAllInfo PPStau126      = Exclusion("PPStau126"  ,InputPattern);
   stAllInfo PPStau156      = Exclusion("PPStau156"  ,InputPattern);
   stAllInfo PPStau200      = Exclusion("PPStau200"  ,InputPattern);
   stAllInfo PPStau247      = Exclusion("PPStau247"  ,InputPattern);
   stAllInfo PPStau308      = Exclusion("PPStau308"  ,InputPattern);

   stAllInfo DCStau121      = Exclusion("DCStau121"  ,InputPattern);
   stAllInfo DCStau182      = Exclusion("DCStau182"  ,InputPattern);
   stAllInfo DCStau242      = Exclusion("DCStau242"  ,InputPattern);
   stAllInfo DCStau302      = Exclusion("DCStau302"  ,InputPattern);

   double Gluino2C_Mass   [] = {Gluino200_2C.Mass    , Gluino300_2C.Mass    , Gluino400_2C.Mass    , Gluino500_2C.Mass    , Gluino600_2C.Mass    , Gluino900_2C.Mass    };
   double Gluino2C_XSecTh [] = {Gluino200_2C.XSec_Th , Gluino300_2C.XSec_Th , Gluino400_2C.XSec_Th , Gluino500_2C.XSec_Th , Gluino600_2C.XSec_Th , Gluino900_2C.XSec_Th };
   double Gluino2C_XSecErr[] = {Gluino200_2C.XSec_Err, Gluino300_2C.XSec_Err, Gluino400_2C.XSec_Err, Gluino500_2C.XSec_Err, Gluino600_2C.XSec_Err, Gluino900_2C.XSec_Err};
   double Gluino2C_XSecObs[] = {Gluino200_2C.XSec_Obs, Gluino300_2C.XSec_Obs, Gluino400_2C.XSec_Obs, Gluino500_2C.XSec_Obs, Gluino600_2C.XSec_Obs, Gluino900_2C.XSec_Obs};
   double Gluino2C_XSecExp[] = {Gluino200_2C.XSec_Exp, Gluino300_2C.XSec_Exp, Gluino400_2C.XSec_Exp, Gluino500_2C.XSec_Exp, Gluino600_2C.XSec_Exp, Gluino900_2C.XSec_Exp};

   double GluinoF0_Mass   [] = {Gluino200_f0.Mass    , Gluino300_f0.Mass    , Gluino400_f0.Mass    , Gluino500_f0.Mass    , Gluino600_f0.Mass    , Gluino900_f0.Mass    };
   double GluinoF0_XSecTh [] = {Gluino200_f0.XSec_Th , Gluino300_f0.XSec_Th , Gluino400_f0.XSec_Th , Gluino500_f0.XSec_Th , Gluino600_f0.XSec_Th , Gluino900_f0.XSec_Th };
   double GluinoF0_XSecErr[] = {Gluino200_f0.XSec_Err, Gluino300_f0.XSec_Err, Gluino400_f0.XSec_Err, Gluino500_f0.XSec_Err, Gluino600_f0.XSec_Err, Gluino900_f0.XSec_Err};
   double GluinoF0_XSecObs[] = {Gluino200_f0.XSec_Obs, Gluino300_f0.XSec_Obs, Gluino400_f0.XSec_Obs, Gluino500_f0.XSec_Obs, Gluino600_f0.XSec_Obs, Gluino900_f0.XSec_Obs};
   double GluinoF0_XSecExp[] = {Gluino200_f0.XSec_Exp, Gluino300_f0.XSec_Exp, Gluino400_f0.XSec_Exp, Gluino500_f0.XSec_Exp, Gluino600_f0.XSec_Exp, Gluino900_f0.XSec_Exp};

   double GluinoF1_Mass   [] = {Gluino200_f1.Mass    , Gluino300_f1.Mass    , Gluino400_f1.Mass    , Gluino500_f1.Mass    , Gluino600_f1.Mass    , Gluino900_f1.Mass    };
   double GluinoF1_XSecTh [] = {Gluino200_f1.XSec_Th , Gluino300_f1.XSec_Th , Gluino400_f1.XSec_Th , Gluino500_f1.XSec_Th , Gluino600_f1.XSec_Th , Gluino900_f1.XSec_Th };
   double GluinoF1_XSecErr[] = {Gluino200_f1.XSec_Err, Gluino300_f1.XSec_Err, Gluino400_f1.XSec_Err, Gluino500_f1.XSec_Err, Gluino600_f1.XSec_Err, Gluino900_f1.XSec_Err};
   double GluinoF1_XSecObs[] = {Gluino200_f1.XSec_Obs, Gluino300_f1.XSec_Obs, Gluino400_f1.XSec_Obs, Gluino500_f1.XSec_Obs, Gluino600_f1.XSec_Obs, Gluino900_f1.XSec_Obs};
   double GluinoF1_XSecExp[] = {Gluino200_f1.XSec_Exp, Gluino300_f1.XSec_Exp, Gluino400_f1.XSec_Exp, Gluino500_f1.XSec_Exp, Gluino600_f1.XSec_Exp, Gluino900_f1.XSec_Exp};

   double GluinoF5_Mass   [] = {Gluino200_f5.Mass    , Gluino300_f5.Mass    , Gluino400_f5.Mass    , Gluino500_f5.Mass    , Gluino600_f5.Mass    , Gluino900_f5.Mass    };
   double GluinoF5_XSecTh [] = {Gluino200_f5.XSec_Th , Gluino300_f5.XSec_Th , Gluino400_f5.XSec_Th , Gluino500_f5.XSec_Th , Gluino600_f5.XSec_Th , Gluino900_f5.XSec_Th };
   double GluinoF5_XSecErr[] = {Gluino200_f5.XSec_Err, Gluino300_f5.XSec_Err, Gluino400_f5.XSec_Err, Gluino500_f5.XSec_Err, Gluino600_f5.XSec_Err, Gluino900_f5.XSec_Err};
   double GluinoF5_XSecObs[] = {Gluino200_f5.XSec_Obs, Gluino300_f5.XSec_Obs, Gluino400_f5.XSec_Obs, Gluino500_f5.XSec_Obs, Gluino600_f5.XSec_Obs, Gluino900_f5.XSec_Obs};
   double GluinoF5_XSecExp[] = {Gluino200_f5.XSec_Exp, Gluino300_f5.XSec_Exp, Gluino400_f5.XSec_Exp, Gluino500_f5.XSec_Exp, Gluino600_f5.XSec_Exp, Gluino900_f5.XSec_Exp};


   double GluinoNF0_Mass   [] = {Gluino200N_f0.Mass    , Gluino300N_f0.Mass    , Gluino400N_f0.Mass    , Gluino500N_f0.Mass    , Gluino600N_f0.Mass    , Gluino900N_f0.Mass    };
   double GluinoNF0_XSecTh [] = {Gluino200N_f0.XSec_Th , Gluino300N_f0.XSec_Th , Gluino400N_f0.XSec_Th , Gluino500N_f0.XSec_Th , Gluino600N_f0.XSec_Th , Gluino900N_f0.XSec_Th };
   double GluinoNF0_XSecErr[] = {Gluino200N_f0.XSec_Err, Gluino300N_f0.XSec_Err, Gluino400N_f0.XSec_Err, Gluino500N_f0.XSec_Err, Gluino600N_f0.XSec_Err, Gluino900N_f0.XSec_Err};
   double GluinoNF0_XSecObs[] = {Gluino200N_f0.XSec_Obs, Gluino300N_f0.XSec_Obs, Gluino400N_f0.XSec_Obs, Gluino500N_f0.XSec_Obs, Gluino600N_f0.XSec_Obs, Gluino900N_f0.XSec_Obs};
   double GluinoNF0_XSecExp[] = {Gluino200N_f0.XSec_Exp, Gluino300N_f0.XSec_Exp, Gluino400N_f0.XSec_Exp, Gluino500N_f0.XSec_Exp, Gluino600N_f0.XSec_Exp, Gluino900N_f0.XSec_Exp};

   double GluinoNF1_Mass   [] = {Gluino200N_f1.Mass    , Gluino300N_f1.Mass    , Gluino400N_f1.Mass    , Gluino500N_f1.Mass    , Gluino600N_f1.Mass    , Gluino900N_f1.Mass    };
   double GluinoNF1_XSecTh [] = {Gluino200N_f1.XSec_Th , Gluino300N_f1.XSec_Th , Gluino400N_f1.XSec_Th , Gluino500N_f1.XSec_Th , Gluino600N_f1.XSec_Th , Gluino900N_f1.XSec_Th };
   double GluinoNF1_XSecErr[] = {Gluino200N_f1.XSec_Err, Gluino300N_f1.XSec_Err, Gluino400N_f1.XSec_Err, Gluino500N_f1.XSec_Err, Gluino600N_f1.XSec_Err, Gluino900N_f1.XSec_Err};
   double GluinoNF1_XSecObs[] = {Gluino200N_f1.XSec_Obs, Gluino300N_f1.XSec_Obs, Gluino400N_f1.XSec_Obs, Gluino500N_f1.XSec_Obs, Gluino600N_f1.XSec_Obs, Gluino900N_f1.XSec_Obs};
   double GluinoNF1_XSecExp[] = {Gluino200N_f1.XSec_Exp, Gluino300N_f1.XSec_Exp, Gluino400N_f1.XSec_Exp, Gluino500N_f1.XSec_Exp, Gluino600N_f1.XSec_Exp, Gluino900N_f1.XSec_Exp};

   double GluinoNF5_Mass   [] = {Gluino200N_f5.Mass    , Gluino300N_f5.Mass    , Gluino400N_f5.Mass    , Gluino500N_f5.Mass    , Gluino600N_f5.Mass    , Gluino900N_f5.Mass    };
   double GluinoNF5_XSecTh [] = {Gluino200N_f5.XSec_Th , Gluino300N_f5.XSec_Th , Gluino400N_f5.XSec_Th , Gluino500N_f5.XSec_Th , Gluino600N_f5.XSec_Th , Gluino900N_f5.XSec_Th };
   double GluinoNF5_XSecErr[] = {Gluino200N_f5.XSec_Err, Gluino300N_f5.XSec_Err, Gluino400N_f5.XSec_Err, Gluino500N_f5.XSec_Err, Gluino600N_f5.XSec_Err, Gluino900N_f5.XSec_Err};
   double GluinoNF5_XSecObs[] = {Gluino200N_f5.XSec_Obs, Gluino300N_f5.XSec_Obs, Gluino400N_f5.XSec_Obs, Gluino500N_f5.XSec_Obs, Gluino600N_f5.XSec_Obs, Gluino900N_f5.XSec_Obs};
   double GluinoNF5_XSecExp[] = {Gluino200N_f5.XSec_Exp, Gluino300N_f5.XSec_Exp, Gluino400N_f5.XSec_Exp, Gluino500N_f5.XSec_Exp, Gluino600N_f5.XSec_Exp, Gluino900N_f5.XSec_Exp};

   double Stop2C_Mass     [] = {Stop130_2C.Mass      , Stop200_2C.Mass      , Stop300_2C.Mass      , Stop500_2C.Mass      , Stop800_2C.Mass    };
   double Stop2C_XSecTh   [] = {Stop130_2C.XSec_Th   , Stop200_2C.XSec_Th   , Stop300_2C.XSec_Th   , Stop500_2C.XSec_Th   , Stop800_2C.XSec_Th };
   double Stop2C_XSecErr  [] = {Stop130_2C.XSec_Err  , Stop200_2C.XSec_Err  , Stop300_2C.XSec_Err  , Stop500_2C.XSec_Err  , Stop800_2C.XSec_Err};
   double Stop2C_XSecObs  [] = {Stop130_2C.XSec_Obs  , Stop200_2C.XSec_Obs  , Stop300_2C.XSec_Obs  , Stop500_2C.XSec_Obs  , Stop800_2C.XSec_Obs};
   double Stop2C_XSecExp  [] = {Stop130_2C.XSec_Exp  , Stop200_2C.XSec_Exp  , Stop300_2C.XSec_Exp  , Stop500_2C.XSec_Exp  , Stop800_2C.XSec_Exp};

   double Stop_Mass       [] = {Stop130.Mass         , Stop200.Mass         , Stop300.Mass         , Stop500.Mass         , Stop800.Mass       };
   double Stop_XSecTh     [] = {Stop130.XSec_Th      , Stop200.XSec_Th      , Stop300.XSec_Th      , Stop500.XSec_Th      , Stop800.XSec_Th    };
   double Stop_XSecErr    [] = {Stop130.XSec_Err     , Stop200.XSec_Err     , Stop300.XSec_Err     , Stop500.XSec_Err     , Stop800.XSec_Err   };
   double Stop_XSecObs    [] = {Stop130.XSec_Obs     , Stop200.XSec_Obs     , Stop300.XSec_Obs     , Stop500.XSec_Obs     , Stop800.XSec_Obs   };
   double Stop_XSecExp    [] = {Stop130.XSec_Exp     , Stop200.XSec_Exp     , Stop300.XSec_Exp     , Stop500.XSec_Exp     , Stop800.XSec_Exp   };

   double StopN_Mass       [] = {Stop130N.Mass         , Stop200N.Mass         , Stop300N.Mass         , Stop500N.Mass         , Stop800N.Mass       };
   double StopN_XSecTh     [] = {Stop130N.XSec_Th      , Stop200N.XSec_Th      , Stop300N.XSec_Th      , Stop500N.XSec_Th      , Stop800N.XSec_Th    };
   double StopN_XSecErr    [] = {Stop130N.XSec_Err     , Stop200N.XSec_Err     , Stop300N.XSec_Err     , Stop500N.XSec_Err     , Stop800N.XSec_Err   };
   double StopN_XSecObs    [] = {Stop130N.XSec_Obs     , Stop200N.XSec_Obs     , Stop300N.XSec_Obs     , Stop500N.XSec_Obs     , Stop800N.XSec_Obs   };
   double StopN_XSecExp    [] = {Stop130N.XSec_Exp     , Stop200N.XSec_Exp     , Stop300N.XSec_Exp     , Stop500N.XSec_Exp     , Stop800N.XSec_Exp   };

   double GMStau_Mass       [] = {GMStau100.Mass         , GMStau126.Mass         , GMStau156.Mass         , GMStau200.Mass         , GMStau247.Mass         , GMStau308.Mass         };
   double GMStau_XSecTh     [] = {GMStau100.XSec_Th      , GMStau126.XSec_Th      , GMStau156.XSec_Th      , GMStau200.XSec_Th      , GMStau247.XSec_Th      , GMStau308.XSec_Th      };
   double GMStau_XSecErr    [] = {GMStau100.XSec_Err     , GMStau126.XSec_Err     , GMStau156.XSec_Err     , GMStau200.XSec_Err     , GMStau247.XSec_Err     , GMStau308.XSec_Err     };
   double GMStau_XSecObs    [] = {GMStau100.XSec_Obs     , GMStau126.XSec_Obs     , GMStau156.XSec_Obs     , GMStau200.XSec_Obs     , GMStau247.XSec_Obs     , GMStau308.XSec_Obs     };
   double GMStau_XSecExp    [] = {GMStau100.XSec_Exp     , GMStau126.XSec_Exp     , GMStau156.XSec_Exp     , GMStau200.XSec_Exp     , GMStau247.XSec_Exp     , GMStau308.XSec_Exp     };

   double PPStau_Mass       [] = {PPStau100.Mass         , PPStau126.Mass         , PPStau156.Mass         , PPStau200.Mass         , PPStau247.Mass         , PPStau308.Mass         };
   double PPStau_XSecTh     [] = {PPStau100.XSec_Th      , PPStau126.XSec_Th      , PPStau156.XSec_Th      , PPStau200.XSec_Th      , PPStau247.XSec_Th      , PPStau308.XSec_Th      };
   double PPStau_XSecErr    [] = {PPStau100.XSec_Err     , PPStau126.XSec_Err     , PPStau156.XSec_Err     , PPStau200.XSec_Err     , PPStau247.XSec_Err     , PPStau308.XSec_Err     };
   double PPStau_XSecObs    [] = {PPStau100.XSec_Obs     , PPStau126.XSec_Obs     , PPStau156.XSec_Obs     , PPStau200.XSec_Obs     , PPStau247.XSec_Obs     , PPStau308.XSec_Obs     };
   double PPStau_XSecExp    [] = {PPStau100.XSec_Exp     , PPStau126.XSec_Exp     , PPStau156.XSec_Exp     , PPStau200.XSec_Exp     , PPStau247.XSec_Exp     , PPStau308.XSec_Exp     };

   double DCStau_Mass       [] = {DCStau121.Mass         , DCStau182.Mass         , DCStau242.Mass         , DCStau302.Mass       };
   double DCStau_XSecTh     [] = {DCStau121.XSec_Th      , DCStau182.XSec_Th      , DCStau242.XSec_Th      , DCStau302.XSec_Th    };
   double DCStau_XSecErr    [] = {DCStau121.XSec_Err     , DCStau182.XSec_Err     , DCStau242.XSec_Err     , DCStau302.XSec_Err   };
   double DCStau_XSecObs    [] = {DCStau121.XSec_Obs     , DCStau182.XSec_Obs     , DCStau242.XSec_Obs     , DCStau302.XSec_Obs   };
   double DCStau_XSecExp    [] = {DCStau121.XSec_Exp     , DCStau182.XSec_Exp     , DCStau242.XSec_Exp     , DCStau302.XSec_Exp   };

   printf("gluino F0 mass (GeV/$c^2$)      & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200_f0.Mass, Gluino300_f0.Mass, Gluino400_f0.Mass, Gluino500_f0.Mass, Gluino600_f0.Mass, Gluino900_f0.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200_f0.Eff,100.*Gluino300_f0.Eff,100.*Gluino400_f0.Eff,100.*Gluino500_f0.Eff,100.*Gluino600_f0.Eff,100.*Gluino900_f0.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f0.XSec_Exp, Gluino300_f0.XSec_Exp, Gluino400_f0.XSec_Exp, Gluino500_f0.XSec_Exp, Gluino600_f0.XSec_Exp,Gluino900_f0.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f0.XSec_Obs, Gluino300_f0.XSec_Obs, Gluino400_f0.XSec_Obs, Gluino500_f0.XSec_Obs, Gluino600_f0.XSec_Obs,Gluino900_f0.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200_f0.XSec_Th,Gluino300_f0.XSec_Th, Gluino400_f0.XSec_Th, Gluino500_f0.XSec_Th, Gluino600_f0.XSec_Th,Gluino900_f0.XSec_Th);
   printf("\\hline\n");
   printf("gluino F1 mass (GeV/$c^2$)      & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200_f1.Mass, Gluino300_f1.Mass, Gluino400_f1.Mass, Gluino500_f1.Mass, Gluino600_f1.Mass, Gluino900_f1.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200_f1.Eff,100.*Gluino300_f1.Eff,100.*Gluino400_f1.Eff,100.*Gluino500_f1.Eff,100.*Gluino600_f1.Eff,100.*Gluino900_f1.Eff); 
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f1.XSec_Exp, Gluino300_f1.XSec_Exp, Gluino400_f1.XSec_Exp, Gluino500_f1.XSec_Exp, Gluino600_f1.XSec_Exp,Gluino900_f1.XSec_Exp); 
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f1.XSec_Obs, Gluino300_f1.XSec_Obs, Gluino400_f1.XSec_Obs, Gluino500_f1.XSec_Obs, Gluino600_f1.XSec_Obs,Gluino900_f1.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200_f1.XSec_Th,Gluino300_f1.XSec_Th, Gluino400_f1.XSec_Th, Gluino500_f1.XSec_Th, Gluino600_f1.XSec_Th,Gluino900_f1.XSec_Th);
   printf("\\hline\n");
   printf("gluino F5 mass (GeV/$c^2$)      & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200_f5.Mass, Gluino300_f5.Mass, Gluino400_f5.Mass, Gluino500_f5.Mass, Gluino600_f5.Mass, Gluino900_f5.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200_f5.Eff,100.*Gluino300_f5.Eff,100.*Gluino400_f5.Eff,100.*Gluino500_f5.Eff,100.*Gluino600_f5.Eff,100.*Gluino900_f5.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f5.XSec_Exp, Gluino300_f5.XSec_Exp, Gluino400_f5.XSec_Exp, Gluino500_f5.XSec_Exp, Gluino600_f5.XSec_Exp,Gluino900_f5.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200_f5.XSec_Obs, Gluino300_f5.XSec_Obs, Gluino400_f5.XSec_Obs, Gluino500_f5.XSec_Obs, Gluino600_f5.XSec_Obs,Gluino900_f5.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200_f5.XSec_Th,Gluino300_f5.XSec_Th, Gluino400_f5.XSec_Th, Gluino500_f5.XSec_Th, Gluino600_f5.XSec_Th,Gluino900_f5.XSec_Th);
   printf("\\hline\n");
   printf("gluino F0 cs mass (GeV/$c^2$)   & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200N_f0.Mass, Gluino300N_f0.Mass, Gluino400N_f0.Mass, Gluino500N_f0.Mass, Gluino600N_f0.Mass, Gluino900N_f0.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200N_f0.Eff,100.*Gluino300N_f0.Eff,100.*Gluino400N_f0.Eff,100.*Gluino500N_f0.Eff,100.*Gluino600N_f0.Eff,100.*Gluino900N_f0.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f0.XSec_Exp, Gluino300N_f0.XSec_Exp, Gluino400N_f0.XSec_Exp, Gluino500N_f0.XSec_Exp, Gluino600N_f0.XSec_Exp,Gluino900N_f0.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f0.XSec_Obs, Gluino300N_f0.XSec_Obs, Gluino400N_f0.XSec_Obs, Gluino500N_f0.XSec_Obs, Gluino600N_f0.XSec_Obs,Gluino900N_f0.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200N_f0.XSec_Th,Gluino300N_f0.XSec_Th, Gluino400N_f0.XSec_Th, Gluino500N_f0.XSec_Th, Gluino600N_f0.XSec_Th,Gluino900N_f0.XSec_Th);
   printf("\\hline\n");
   printf("gluino F1 cs mass (GeV/$c^2$)   & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200N_f1.Mass, Gluino300N_f1.Mass, Gluino400N_f1.Mass, Gluino500N_f1.Mass, Gluino600N_f1.Mass, Gluino900N_f1.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200N_f1.Eff,100.*Gluino300N_f1.Eff,100.*Gluino400N_f1.Eff,100.*Gluino500N_f1.Eff,100.*Gluino600N_f1.Eff,100.*Gluino900N_f1.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f1.XSec_Exp, Gluino300N_f1.XSec_Exp, Gluino400N_f1.XSec_Exp, Gluino500N_f1.XSec_Exp, Gluino600N_f1.XSec_Exp,Gluino900N_f1.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f1.XSec_Obs, Gluino300N_f1.XSec_Obs, Gluino400N_f1.XSec_Obs, Gluino500N_f1.XSec_Obs, Gluino600N_f1.XSec_Obs,Gluino900N_f1.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200N_f1.XSec_Th,Gluino300N_f1.XSec_Th, Gluino400N_f1.XSec_Th, Gluino500N_f1.XSec_Th, Gluino600N_f1.XSec_Th,Gluino900N_f1.XSec_Th);
   printf("\\hline\n");
   printf("gluino F5 cs mass (GeV/$c^2$)   & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",Gluino200N_f5.Mass, Gluino300N_f5.Mass, Gluino400N_f5.Mass, Gluino500N_f5.Mass, Gluino600N_f5.Mass, Gluino900N_f5.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*Gluino200N_f5.Eff,100.*Gluino300N_f5.Eff,100.*Gluino400N_f5.Eff,100.*Gluino500N_f5.Eff,100.*Gluino600N_f5.Eff,100.*Gluino900N_f5.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f5.XSec_Exp, Gluino300N_f5.XSec_Exp, Gluino400N_f5.XSec_Exp, Gluino500N_f5.XSec_Exp, Gluino600N_f5.XSec_Exp,Gluino900N_f5.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",Gluino200N_f5.XSec_Obs, Gluino300N_f5.XSec_Obs, Gluino400N_f5.XSec_Obs, Gluino500N_f5.XSec_Obs, Gluino600N_f5.XSec_Obs,Gluino900N_f5.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %4.2f & %4.2f & %5.3f \\\\ \\hline\n",Gluino200N_f5.XSec_Th,Gluino300N_f5.XSec_Th, Gluino400N_f5.XSec_Th, Gluino500N_f5.XSec_Th, Gluino600N_f5.XSec_Th,Gluino900N_f5.XSec_Th);
   printf("\\hline\n");
   printf("stop mass (GeV/$c^2$)           & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f &       \\\\ \\hline\n",Stop130.Mass, Stop200.Mass, Stop300.Mass, Stop500.Mass, Stop800.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",100.*Stop130.Eff,100.*Stop200.Eff,100.*Stop300.Eff,100.*Stop500.Eff,100.*Stop800.Eff);    
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",Stop130.XSec_Exp, Stop200.XSec_Exp, Stop300.XSec_Exp, Stop500.XSec_Exp, Stop800.XSec_Exp);      
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",Stop130.XSec_Obs, Stop200.XSec_Obs, Stop300.XSec_Obs, Stop500.XSec_Obs, Stop800.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %5.3f & %6.4f &       \\\\ \\hline\n",Stop130.XSec_Th, Stop200.XSec_Th, Stop300.XSec_Th, Stop500.XSec_Th, Stop800.XSec_Th);
   printf("\\hline\n");
   printf("stop cs mass (GeV/$c^2$)        & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f &       \\\\ \\hline\n",Stop130N.Mass, Stop200N.Mass, Stop300N.Mass, Stop500N.Mass, Stop800N.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",100.*Stop130N.Eff,100.*Stop200N.Eff,100.*Stop300N.Eff,100.*Stop500N.Eff,100.*Stop800N.Eff);
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",Stop130N.XSec_Exp, Stop200N.XSec_Exp, Stop300N.XSec_Exp, Stop500N.XSec_Exp, Stop800N.XSec_Exp);
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f &       \\\\\n",Stop130N.XSec_Obs, Stop200N.XSec_Obs, Stop300N.XSec_Obs, Stop500N.XSec_Obs, Stop800N.XSec_Obs);
   printf("Theoretical cross section (pb)  & %3.0f & %3.1f & %4.2f & %5.3f & %6.4f &       \\\\ \\hline\n",Stop130N.XSec_Th, Stop200N.XSec_Th, Stop300N.XSec_Th, Stop500N.XSec_Th, Stop800N.XSec_Th);
   printf("\\hline\n");
   printf("GMSB stau mass (GeV/$c^2$)      & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",GMStau100.Mass, GMStau126.Mass, GMStau156.Mass, GMStau200.Mass, GMStau247.Mass, GMStau308.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*GMStau100.Eff,100.*GMStau126.Eff,100.*GMStau156.Eff,100.*GMStau200.Eff,100.*GMStau247.Eff, 100.*GMStau308.Eff); 
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",GMStau100.XSec_Exp, GMStau126.XSec_Exp, GMStau156.XSec_Exp, GMStau200.XSec_Exp, GMStau247.XSec_Exp, GMStau308.XSec_Exp); 
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",GMStau100.XSec_Obs, GMStau126.XSec_Obs, GMStau156.XSec_Obs, GMStau200.XSec_Obs, GMStau247.XSec_Obs, GMStau308.XSec_Obs);
   printf("Theoretical cross section (pb)  & %4.2f & %4.2f & %5.3f & %5.3f & %5.3f & %5.3f \\\\ \\hline\n",GMStau100.XSec_Th, GMStau126.XSec_Th, GMStau156.XSec_Th, GMStau200.XSec_Th, GMStau247.XSec_Th,GMStau308.XSec_Th);
   printf("\\hline\n");   
   printf("PP stau mass (GeV/$c^2$)        & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f & %3.0f \\\\ \\hline\n",PPStau100.Mass, PPStau126.Mass, PPStau156.Mass, PPStau200.Mass, PPStau247.Mass, PPStau308.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",100.*PPStau100.Eff,100.*PPStau126.Eff,100.*PPStau156.Eff,100.*PPStau200.Eff,100.*PPStau247.Eff, 100.*PPStau308.Eff); 
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",PPStau100.XSec_Exp, PPStau126.XSec_Exp, PPStau156.XSec_Exp, PPStau200.XSec_Exp, PPStau247.XSec_Exp, PPStau308.XSec_Exp); 
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\\n",PPStau100.XSec_Obs, PPStau126.XSec_Obs, PPStau156.XSec_Obs, PPStau200.XSec_Obs, PPStau247.XSec_Obs, PPStau308.XSec_Obs);
   printf("Theoretical cross section (pb)  & %4.2f & %4.2f & %5.3f & %5.3f & %5.3f & %5.3f \\\\ \\hline\n",PPStau100.XSec_Th, PPStau126.XSec_Th, PPStau156.XSec_Th, PPStau200.XSec_Th, PPStau247.XSec_Th,PPStau308.XSec_Th);
   printf("\\hline\n");   
   printf("DICHAMP stau mass (GeV/$c^2$)   & %3.0f & %3.0f & %3.0f & %3.0f &       &       \\\\ \\hline\n",DCStau121.Mass, DCStau182.Mass, DCStau242.Mass, DCStau302.Mass);
   printf("Total acceptance (\\%%)         & %4.2f & %4.2f & %4.2f & %4.2f &       &       \\\\\n",100.*DCStau121.Eff,100.*DCStau182.Eff,100.*DCStau242.Eff,100.*DCStau302.Eff); 
   printf("Expected 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f &       &       \\\\\n",DCStau121.XSec_Exp, DCStau182.XSec_Exp, DCStau242.XSec_Exp, DCStau302.XSec_Exp); 
   printf("Observed 95\\%% C.L. limit (pb) & %4.2f & %4.2f & %4.2f & %4.2f &       &       \\\\\n",DCStau121.XSec_Obs, DCStau182.XSec_Obs, DCStau242.XSec_Obs, DCStau302.XSec_Obs);
   printf("Theoretical cross section (pb)  & %4.2f & %4.2f & %5.3f & %5.3f &       &       \\\\ \\hline\n",DCStau121.XSec_Th, DCStau182.XSec_Th, DCStau242.XSec_Th, DCStau302.XSec_Th);
   printf("\\hline\n");   

   double ThGluinoMass [100];
   double ThGluinoXSec [100];
   double ThGluinoLow  [100];
   double ThGluinoHigh [100];
   double ThGluinoErrLow  [100];
   double ThGluinoErrHigh [100];
   int ThGluinoN = ReadXSection("gluino_XSec.txt", ThGluinoMass,ThGluinoXSec,ThGluinoLow,ThGluinoHigh, ThGluinoErrLow, ThGluinoErrHigh);
   TGraph* GluinoXSec = new TGraph(ThGluinoN,ThGluinoMass,ThGluinoXSec);
   GluinoXSec->SetLineColor(4);
   GluinoXSec->SetLineStyle(1);
   GluinoXSec->SetLineWidth(1);
   GluinoXSec->SetMarkerColor(4);
   GluinoXSec->SetTitle("");
   GluinoXSec->GetXaxis()->SetTitle("Gluino HSCP Mass [ GeV/c^{2} ]");
   GluinoXSec->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoXSec->GetYaxis()->SetTitleOffset(1.70);

   TCutG* GluinoXSecErr = GetErrorBand("gluinoErr",ThGluinoN,ThGluinoMass,ThGluinoLow,ThGluinoHigh);
   TGraph* GluinoXSecLow = new TGraph(ThGluinoN,ThGluinoMass,ThGluinoLow);

   double ThStopMass [100];
   double ThStopXSec [100];
   double ThStopLow  [100];
   double ThStopHigh [100];
   double ThStopErrLow  [100];
   double ThStopErrHigh [100];
   int ThStopN = ReadXSection("stop_XSec.txt", ThStopMass,ThStopXSec,ThStopLow,ThStopHigh, ThStopErrLow, ThStopErrHigh);
   TGraph* StopXSec = new TGraph(ThStopN,ThStopMass,ThStopXSec);
   StopXSec->SetLineColor(2);
   StopXSec->SetLineStyle(3);
   StopXSec->SetLineWidth(1);
   StopXSec->SetMarkerColor(2);
   StopXSec->SetTitle("");
   StopXSec->GetXaxis()->SetTitle("Stop HSCP Mass [ GeV/c^{2} ]");
   StopXSec->GetYaxis()->SetTitle("CrossSection [ pb ]");
   StopXSec->GetYaxis()->SetTitleOffset(1.70);

   TCutG* StopXSecErr = GetErrorBand("StopErr", ThStopN,ThStopMass,ThStopLow,ThStopHigh);
   TGraph* StopXSecLow = new TGraph(ThStopN,ThStopMass,ThStopLow);

   TGraph* GMStauXSec = new TGraph(6,GMStau_Mass,GMStau_XSecTh);
   GMStauXSec->SetLineColor(1);
   GMStauXSec->SetLineStyle(7);
   GMStauXSec->SetLineWidth(1);
   GMStauXSec->SetMarkerColor(1);
   GMStauXSec->SetTitle("");
   GMStauXSec->GetXaxis()->SetTitle("GMStau HSCP Mass [ GeV/c^{2} ]");
   GMStauXSec->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GMStauXSec->GetYaxis()->SetTitleOffset(1.70);
   TGraphErrors* GMStauXSecErr = new TGraphErrors(6,GMStau_Mass,GMStau_XSecTh, NULL, GMStau_XSecErr);
   GMStauXSecErr->SetLineStyle(0);
   GMStauXSecErr->SetLineWidth(0);
   GMStauXSecErr->SetLineColor(0);
   GMStauXSecErr->SetFillColor(kGreen-7);



   TGraph* PPStauXSec = new TGraph(6,PPStau_Mass,PPStau_XSecTh);
   PPStauXSec->SetLineColor(12);
   PPStauXSec->SetLineStyle(6);
   PPStauXSec->SetLineWidth(1);
   PPStauXSec->SetMarkerColor(1);
   PPStauXSec->SetTitle("");
   PPStauXSec->GetXaxis()->SetTitle("PPStau HSCP Mass [ GeV/c^{2} ]");
   PPStauXSec->GetYaxis()->SetTitle("CrossSection [ pb ]");
   PPStauXSec->GetYaxis()->SetTitleOffset(1.70);
   TGraphErrors* PPStauXSecErr = new TGraphErrors(6,PPStau_Mass,PPStau_XSecTh, NULL, PPStau_XSecErr);
   PPStauXSecErr->SetLineStyle(0);
   PPStauXSecErr->SetLineWidth(0);
   PPStauXSecErr->SetLineColor(0);
   PPStauXSecErr->SetFillColor(kGreen-7);

   TGraph* DCStauXSec = new TGraph(4,DCStau_Mass,DCStau_XSecTh);
   DCStauXSec->SetLineColor(15);
   DCStauXSec->SetLineStyle(5);
   DCStauXSec->SetLineWidth(1);
   DCStauXSec->SetMarkerColor(1);
   DCStauXSec->SetTitle("");
   DCStauXSec->GetXaxis()->SetTitle("DCStau HSCP Mass [ GeV/c^{2} ]");
   DCStauXSec->GetYaxis()->SetTitle("CrossSection [ pb ]");
   DCStauXSec->GetYaxis()->SetTitleOffset(1.70);
   TGraphErrors* DCStauXSecErr = new TGraphErrors(4,DCStau_Mass,DCStau_XSecTh, NULL, DCStau_XSecErr);
   DCStauXSecErr->SetLineStyle(0);
   DCStauXSecErr->SetLineWidth(0);
   DCStauXSecErr->SetLineColor(0);
   DCStauXSecErr->SetFillColor(kGreen-7);



   /////////////////////////// OBSERVED /////////////////////////

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionF0 = new TGraph(6,GluinoF0_Mass,GluinoF0_XSecObs);
   GluinoExclusionF0->SetLineColor(kBlue-7);
   GluinoExclusionF0->SetFillColor(kBlue-7);
   GluinoExclusionF0->SetLineWidth(2);
   GluinoExclusionF0->SetMarkerColor(kBlue-7);
   GluinoExclusionF0->SetMarkerStyle(28);
   GluinoExclusionF0->Draw("ALP same");
   GluinoExclusionF0->SetTitle("");
   GluinoExclusionF0->GetXaxis()->SetTitle("Gluino (f=0.0) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionF0->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionF0->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoF0");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionF1 = new TGraph(6,GluinoF1_Mass,GluinoF1_XSecObs);
   GluinoExclusionF1->SetLineColor(4);
   GluinoExclusionF1->SetFillColor(4);
   GluinoExclusionF1->SetLineWidth(2);
   GluinoExclusionF1->SetMarkerColor(4);
   GluinoExclusionF1->SetMarkerStyle(20);
   GluinoExclusionF1->Draw("ALP same");
   GluinoExclusionF1->SetTitle("");
   GluinoExclusionF1->GetXaxis()->SetTitle("Gluino (f=0.1) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionF1->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionF1->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoF1");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionF5 = new TGraph(6,GluinoF5_Mass,GluinoF5_XSecObs);
   GluinoExclusionF5->SetLineColor(kBlue+2);
   GluinoExclusionF5->SetFillColor(kBlue+2);
   GluinoExclusionF5->SetLineWidth(2);
   GluinoExclusionF5->SetMarkerColor(kBlue+2);
   GluinoExclusionF5->SetMarkerStyle(25);
   GluinoExclusionF5->Draw("ALP same");
   GluinoExclusionF5->SetTitle("");
   GluinoExclusionF5->GetXaxis()->SetTitle("Gluino (f=0.5) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionF5->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionF5->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoF5");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusion2C = new TGraph(6,Gluino2C_Mass,Gluino2C_XSecObs);
   GluinoExclusion2C->SetLineColor(kBlue-9);
   GluinoExclusion2C->SetFillColor(kBlue-9);
   GluinoExclusion2C->SetLineWidth(2);
   GluinoExclusion2C->SetMarkerColor(kBlue-9);
   GluinoExclusion2C->SetMarkerStyle(26);
   GluinoExclusion2C->Draw("ALP same");
   GluinoExclusion2C->SetTitle("");
   GluinoExclusion2C->GetXaxis()->SetTitle("Gluino (2 charged HSCP/event) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusion2C->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusion2C->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_Gluino2C");
   delete c1;


   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionNF0 = new TGraph(6,GluinoNF0_Mass,GluinoNF0_XSecObs);
   GluinoExclusionNF0->SetLineColor(4);
   GluinoExclusionNF0->SetFillColor(4);
   GluinoExclusionNF0->SetLineWidth(2);
   GluinoExclusionNF0->SetMarkerColor(4);
   GluinoExclusionNF0->SetMarkerStyle(20);
   GluinoExclusionNF0->Draw("ALP same");
   GluinoExclusionNF0->SetTitle("");
   GluinoExclusionNF0->GetXaxis()->SetTitle("Gluino cs (f=0.1) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionNF0->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionNF0->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoNF0");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionNF1 = new TGraph(6,GluinoNF1_Mass,GluinoNF1_XSecObs);
   GluinoExclusionNF1->SetLineColor(4);
   GluinoExclusionNF1->SetFillColor(4);
   GluinoExclusionNF1->SetLineWidth(2);
   GluinoExclusionNF1->SetMarkerColor(4);
   GluinoExclusionNF1->SetMarkerStyle(20);
   GluinoExclusionNF1->Draw("ALP same");
   GluinoExclusionNF1->SetTitle("");
   GluinoExclusionNF1->GetXaxis()->SetTitle("Gluino cs (f=0.1) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionNF1->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionNF1->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoNF1");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GluinoExclusionNF5 = new TGraph(6,GluinoNF5_Mass,GluinoNF5_XSecObs);
   GluinoExclusionNF5->SetLineColor(4);
   GluinoExclusionNF5->SetFillColor(4);
   GluinoExclusionNF5->SetLineWidth(2);
   GluinoExclusionNF5->SetMarkerColor(4);
   GluinoExclusionNF5->SetMarkerStyle(20);
   GluinoExclusionNF5->Draw("ALP same");
   GluinoExclusionNF5->SetTitle("");
   GluinoExclusionNF5->GetXaxis()->SetTitle("Gluino cs (f=0.1) HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusionNF5->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GluinoExclusionNF5->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GluinoNF5");
   delete c1;


   c1 = new TCanvas("c1", "c1",600,600);
//   double Stop_MassNo130   [] = {Stop_Mass[1]   , Stop_Mass   [2], Stop_Mass   [3], Stop_Mass   [4]};
//   double Stop_XSecObsNo130[] = {Stop_XSecObs[1], Stop_XSecObs[2], Stop_XSecObs[3], Stop_XSecObs[4]};
//   TGraph* StopExclusion = new TGraph(4,Stop_MassNo130,Stop_XSecObsNo130);
   TGraph* StopExclusion = new TGraph(5,Stop_Mass,Stop_XSecObs);
   StopExclusion->SetLineColor(2);
   StopExclusion->SetLineWidth(2);
   StopExclusion->SetMarkerColor(2);
   StopExclusion->SetMarkerStyle(23);
   StopExclusion->Draw("ALP same");
   StopExclusion->SetTitle("");
   StopExclusion->GetXaxis()->SetTitle("Stop HSCP Mass [ GeV/c^{2} ]");
   StopExclusion->GetYaxis()->SetTitle("CrossSection [ pb ]");
   StopExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_Stop");
   delete c1;
   c1 = new TCanvas("c1", "c1",600,600);
//   double Stop2C_MassNo130   [] = {Stop2C_Mass[1]   , Stop2C_Mass   [2], Stop2C_Mass   [3], Stop2C_Mass   [4]};
//   double Stop2C_XSecObsNo130[] = {Stop2C_XSecObs[1], Stop2C_XSecObs[2], Stop2C_XSecObs[3], Stop2C_XSecObs[4]};
//   TGraph* StopExclusion2C = new TGraph(4,Stop2C_MassNo130,Stop2C_XSecObsNo130);
   TGraph* StopExclusion2C = new TGraph(5,Stop2C_Mass,Stop2C_XSecObs);
   StopExclusion2C->SetLineColor(kRed-4);
   StopExclusion2C->SetLineWidth(2);
   StopExclusion2C->SetMarkerColor(kRed-4);
   StopExclusion2C->SetMarkerStyle(22);
   StopExclusion2C->Draw("ALP same");
   StopExclusion2C->SetTitle("");
   StopExclusion2C->GetXaxis()->SetTitle("Stop (2 charged HSCP/event) HSCP Mass [ GeV/c^{2} ]");
   StopExclusion2C->GetYaxis()->SetTitle("CrossSection [ pb ]");
   StopExclusion2C->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_Stop2C");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
//   double StopN_MassNo130   [] = {StopN_Mass[1]   , StopN_Mass   [2], StopN_Mass   [3], StopN_Mass   [4]};
//   double StopN_XSecObsNo130[] = {StopN_XSecObs[1], StopN_XSecObs[2], StopN_XSecObs[3], StopN_XSecObs[4]};
//   TGraph* StopNExclusion = new TGraph(4,StopN_MassNo130,StopN_XSecObsNo130);
   TGraph* StopNExclusion = new TGraph(5,StopN_Mass,StopN_XSecObs);
   StopNExclusion->SetLineColor(2);
   StopNExclusion->SetLineWidth(2);
   StopNExclusion->SetMarkerColor(2);
   StopNExclusion->SetMarkerStyle(23);
   StopNExclusion->Draw("ALP same");
   StopNExclusion->SetTitle("");
   StopNExclusion->GetXaxis()->SetTitle("Stop HSCP Mass [ GeV/c^{2} ]");
   StopNExclusion->GetYaxis()->SetTitle("CrossSection [ pb ]");
   StopNExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_StopN");


   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* GMStauExclusion = new TGraph(6,GMStau_Mass,GMStau_XSecObs);
   GMStauExclusion->SetLineColor(1);
   GMStauExclusion->SetLineWidth(2);
   GMStauExclusion->SetFillColor(1);
   GMStauExclusion->SetMarkerColor(1);
   GMStauExclusion->SetMarkerStyle(21);
   GMStauExclusion->Draw("ALP same");
   GMStauExclusion->SetTitle("");
   GMStauExclusion->GetXaxis()->SetTitle("GMStau HSCP Mass [ GeV/c^{2} ]");
   GMStauExclusion->GetYaxis()->SetTitle("CrossSection [ pb ]");
   GMStauExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_GMStau");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* PPStauExclusion = new TGraph(6,PPStau_Mass,PPStau_XSecObs);
   PPStauExclusion->SetLineColor(12);
   PPStauExclusion->SetLineWidth(2);
   PPStauExclusion->SetFillColor(1);
   PPStauExclusion->SetMarkerColor(12);
   PPStauExclusion->SetMarkerStyle(22);
   PPStauExclusion->Draw("ALP same");
   PPStauExclusion->SetTitle("");
   PPStauExclusion->GetXaxis()->SetTitle("PPStau HSCP Mass [ GeV/c^{2} ]");
   PPStauExclusion->GetYaxis()->SetTitle("CrossSection [ pb ]");
   PPStauExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_PPStau");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   TGraph* DCStauExclusion = new TGraph(4,DCStau_Mass,DCStau_XSecObs);
   DCStauExclusion->SetLineColor(15);
   DCStauExclusion->SetLineWidth(2);
   DCStauExclusion->SetFillColor(1);
   DCStauExclusion->SetMarkerColor(15);
   DCStauExclusion->SetMarkerStyle(29);
   DCStauExclusion->Draw("ALP same");
   DCStauExclusion->SetTitle("");
   DCStauExclusion->GetXaxis()->SetTitle("DCStau HSCP Mass [ GeV/c^{2} ]");
   DCStauExclusion->GetYaxis()->SetTitle("CrossSection [ pb ]");
   DCStauExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, outpath, "ExclusionPlot_DCStau");
   delete c1;

   c1 = new TCanvas("c1", "c1",600,600);
   c1->SetLogy(true);
   TMultiGraph* mg = new TMultiGraph();
   mg->Add(GMStauXSec         ,"L");
   mg->Add(PPStauXSec         ,"L");
   mg->Add(DCStauXSec         ,"L");
   mg->Add(StopXSec         ,"L");
   mg->Add(GluinoXSec       ,"L");
   mg->Add(GMStauExclusion    ,"LP");
   mg->Add(PPStauExclusion    ,"LP");
   mg->Add(DCStauExclusion    ,"LP");
   mg->Add(StopExclusion    ,"LP");
   mg->Add(GluinoExclusionF1,"LP");
   mg->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr->Draw("f");
   mg->Draw("same");
   mg->SetTitle("");
   mg->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   mg->GetYaxis()->SetTitle("#sigma (pb)");
   mg->GetYaxis()->SetTitleOffset(1.70);
   mg->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);
   DrawPreliminary(IntegratedLuminosity);

   TLegend* leg = new TLegend(0.40,0.75,0.80,0.90);
   leg->SetHeader((string("95% C.L. Exclusion  (") + LegendFromType(InputPattern) + ")").c_str() );
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(GluinoExclusionF1, "Gluino"   ,"LP");
   leg->AddEntry(StopExclusion    , "Stop"     ,"LP");
   leg->AddEntry(GMStauExclusion    , "GMStau"     ,"LP");
   leg->AddEntry(PPStauExclusion    , "PPStau"     ,"LP");
   leg->AddEntry(DCStauExclusion    , "DCStau"     ,"LP");
   leg->Draw();

   TLegend* leg2 = new TLegend(0.15,0.75,0.40,0.90);
   leg2->SetHeader("Theoretical Prediction");
   leg2->SetFillColor(0);
   leg2->SetBorderSize(0);
   leg2->AddEntry(GMStauXSec  , "GMStau (LO)"  ,"L");
   leg2->AddEntry(PPStauXSec  , "PPStau (LO)"  ,"L");
   leg2->AddEntry(DCStauXSec  , "DCStau (LO)"  ,"L");
   leg2->AddEntry(StopXSec  , "Stop (NLO+NLL)"  ,"L");
   leg2->AddEntry(GluinoXSec, "Gluino (NLO+NLL)","L");
//   leg2->AddEntry(GluinoXSecErr, "Th. Uncertainty","F");
   leg2->Draw();

   c1->SetGridx(true);
   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("ExclusionPlotLog"));
   c1->SetLogy(false);
   SaveCanvas(c1, outpath, string("ExclusionPlot"));
   delete c1;

  std::cout<<"TESTF"<< endl;

   c1 = new TCanvas("c1", "c1",600,600);
   TMultiGraph* mgA = new TMultiGraph();
   mgA->Add(GMStauXSec         ,"L");
   mgA->Add(StopXSec         ,"L");
   mgA->Add(GluinoXSec       ,"L");
   mgA->Add(GMStauExclusion    ,"LP");
   mgA->Add(StopExclusion    ,"LP");
   mgA->Add(StopExclusion2C  ,"LP");
   mgA->Add(GluinoExclusionF0,"LP");
   mgA->Add(GluinoExclusionF1,"LP");
   mgA->Add(GluinoExclusionF5,"LP");
   mgA->Add(GluinoExclusion2C,"LP");
   mgA->Draw("A");
   GluinoXSecErr->Draw("f");
   StopXSecErr->Draw("f");
   mgA->Draw("same");
   mgA->SetTitle("");
   mgA->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
   mgA->GetYaxis()->SetTitle("#sigma (pb)");
   mgA->GetYaxis()->SetTitleOffset(1.70);
   mgA->GetYaxis()->SetRangeUser(PlotMinScale,PlotMaxScale);

   DrawPreliminary(IntegratedLuminosity);

   TLegend* legA = new TLegend(0.40,0.55,0.80,0.90);
   if(IsTrackerOnly){
      legA->SetHeader("95% C.L. Exclusion (Tracker - Only)");
   }else{
      legA->SetHeader("95% C.L. Exclusion (Tracker + Muon)");
   }
   legA->SetFillColor(0);
   legA->SetBorderSize(0);
   legA->AddEntry(GluinoExclusion2C, "Gluino (2Charged)","LP");
   legA->AddEntry(GluinoExclusionF0, "Gluino (f=0.0)"   ,"LP");
   legA->AddEntry(GluinoExclusionF1, "Gluino (f=0.1)"   ,"LP");
   legA->AddEntry(GluinoExclusionF5, "Gluino (f=0.5)"   ,"LP");
   legA->AddEntry(StopExclusion2C  , "Stop (2Charged)"  ,"LP");
   legA->AddEntry(StopExclusion    , "Stop (PYTHIA)"    ,"LP");
   legA->AddEntry(GMStauExclusion    , "GMStau"             ,"LP");
   legA->Draw();

   TLegend* leg2A = new TLegend(0.15,0.75,0.40,0.90);
   leg2A->SetHeader("Theoretical Prediction");
   leg2A->SetFillColor(0);
   leg2A->SetBorderSize(0);
   leg2A->AddEntry(GMStauXSec  , "GMStau (LO)"  ,"L");
   leg2A->AddEntry(StopXSec  , "Stop (NLO+NLL)"  ,"L");
   leg2A->AddEntry(GluinoXSec, "Gluino (NLO+NLL)","L");
//   leg2A->AddEntry(GluinoXSecErr, "Th. Uncertainty","F");
   leg2A->Draw();
   c1->SetGridx(true);
   c1->SetGridy(true);
   SaveCanvas(c1, outpath, string("ExclusionPlotaLL"));
   c1->SetLogy(true);
   SaveCanvas(c1, outpath, string("ExclusionPlotLogaLL"));
   delete c1;


   printf("-----------------------\n0%% TH Uncertainty\n-------------------------\n");
   printf("MASS EXCLUDED UP TO %8.3fGeV for Gluino2C \n", FindIntersection(GluinoExclusion2C, GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoF0 \n", FindIntersection(GluinoExclusionF0, GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoF1 \n", FindIntersection(GluinoExclusionF1, GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoF5 \n", FindIntersection(GluinoExclusionF5, GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoNF0\n", FindIntersection(GluinoExclusionNF0,GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoNF1\n", FindIntersection(GluinoExclusionNF1,GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GluinoNF5\n", FindIntersection(GluinoExclusionNF5,GluinoXSecLow, 200, 900, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for Stop2C   \n", FindIntersection(StopExclusion2C  , StopXSecLow  , 200, 500, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for Stop     \n", FindIntersection(StopExclusion    , StopXSecLow  , 200, 500, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for StopN    \n", FindIntersection(StopNExclusion   , StopXSecLow  , 200, 500, 1, 0.00));
   printf("MASS EXCLUDED UP TO %8.3fGeV for GMStau     \n", FindIntersection(GMStauExclusion    , GMStauXSec     , 100, 308, 1, 0.15));
   printf("MASS EXCLUDED UP TO %8.3fGeV for PPStau     \n", FindIntersection(PPStauExclusion    , PPStauXSec     , 100, 308, 1, 0.15));
   printf("MASS EXCLUDED UP TO %8.3fGeV for DCStau     \n", FindIntersection(DCStauExclusion    , DCStauXSec     , 121, 302, 1, 0.15));
   stGraph ToReturn;
   ToReturn.Stop        = StopExclusion;
   ToReturn.StopN       = StopNExclusion;
   ToReturn.GluinoF0    = GluinoExclusionF0;
   ToReturn.GluinoF1    = GluinoExclusionF1;
   ToReturn.GluinoF5    = GluinoExclusionF5;
   ToReturn.GluinoNF0   = GluinoExclusionNF0;
   ToReturn.GluinoNF1   = GluinoExclusionNF1;
   ToReturn.GluinoNF5   = GluinoExclusionNF5;
   ToReturn.GMStau        = GMStauExclusion;
   ToReturn.GluinoTh    = GluinoXSec;
   ToReturn.StopTh      = StopXSec;
   ToReturn.GMStauTh      = GMStauXSec;
   ToReturn.GluinoThErr = GluinoXSecErr;
   ToReturn.StopThErr   = StopXSecErr;
   return ToReturn;

}

stAllInfo Exclusion(string signal, string pattern, double Ratio_0C, double Ratio_1C, double Ratio_2C){
   if(Mode==0){
      return Exclusion_Counting(signal,pattern, Ratio_0C, Ratio_1C, Ratio_2C);
   }else{
      return Exclusion_LL(signal,pattern);
   }
}

stAllInfo Exclusion_Counting(string signal, string pattern, double Ratio_0C, double Ratio_1C, double Ratio_2C){
   stAllInfo toReturn;

   double RatioValue[] = {Ratio_0C, Ratio_1C, Ratio_2C};
   string RatioName [] = {"_NC0"  , "_NC1"    , "_NC2"   };

   InputPath            = pattern + "DumpHistos.root";
   TFile* InputFile     = new TFile(InputPath.c_str());
   MassData             = (TH1D*)GetObjectFromPath(InputFile, "Data_Mass");
   MassPred             = (TH1D*)GetObjectFromPath(InputFile, "Pred_Mass");

   double NPredErr    = 0;
   for(int i=MassPred->GetXaxis()->FindBin(MinRange); i<=MassPred->GetXaxis()->FindBin(MaxRange) ;i++){NPredErr+=(MassPred->GetBinError(i)*MassPred->GetBinError(i));}NPredErr=sqrt(NPredErr);
   double NPred       = MassPred->Integral(MassPred->GetXaxis()->FindBin(MinRange), MassPred->GetXaxis()->FindBin(MaxRange));
   double NData       = MassData->Integral(MassData->GetXaxis()->FindBin(MinRange), MassData->GetXaxis()->FindBin(MaxRange));
//   if(NData>0)printf("\n###############################\n BUG BUG Counter #Events is not 0 in data --> can not do exclusion!!! \n ###############################\n\n");

   double Eff       = 0;
   double Eff_SYSTA = 0;
   double Eff_SYSTB = 0;

   printf("%20s Total Eff = ",signal.c_str());

   if(RatioValue[0]<0 && RatioValue[1]<0 && RatioValue[2]<0){
      CurrentSampleIndex   = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  } 
      MassSign             = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_Mass" );
      TH1D* MassSign_SYSTA = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_Mass_Syst_PtLow");
      TH1D* MassSign_SYSTB = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_Mass_Syst_ILow");
      //signalsMeanHSCPPerEvent is there because we need #events and not #tracks, and NSIgn is at Track (and Not Event) Level.
      //double INTERN_NSign       = MassSign->Integral(MassSign            ->GetXaxis()->FindBin(MinRange), MassSign      ->GetXaxis()->FindBin(MaxRange));
      double INTERN_ESign       = MassSign->Integral(MassSign            ->GetXaxis()->FindBin(MinRange), MassSign      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [4*CurrentSampleIndex]; 
      double INTERN_ESign_SYSTA = MassSign_SYSTA->Integral(MassSign_SYSTA->GetXaxis()->FindBin(MinRange), MassSign_SYSTA->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent_SYSTA[4*CurrentSampleIndex];
      double INTERN_ESign_SYSTB = MassSign_SYSTB->Integral(MassSign_SYSTB->GetXaxis()->FindBin(MinRange), MassSign_SYSTB->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent_SYSTB[4*CurrentSampleIndex];
      double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      double INTERN_Eff_SYSTA   = INTERN_ESign_SYSTA / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      double INTERN_Eff_SYSTB   = INTERN_ESign_SYSTB / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      Eff       = INTERN_Eff;
      Eff_SYSTA = INTERN_Eff_SYSTA;
      Eff_SYSTB = INTERN_Eff_SYSTB;
   }else{
   for(unsigned int i=0;i<3;i++){
      CurrentSampleIndex   = JobIdToIndex(signal); if(CurrentSampleIndex<0){  printf("There is no signal corresponding to the JobId Given\n");  return toReturn;  }
      MassSign             = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + RatioName[i] + "_Mass");
      TH1D* MassSign_SYSTA = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + RatioName[i] + "_Mass_Syst_PtLow");
      TH1D* MassSign_SYSTB = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + RatioName[i] + "_Mass_Syst_ILow");

      //signalsMeanHSCPPerEvent is there because we need #events and not #tracks, and NSIgn is at Track (and Not Event) Level.
      //double INTERN_NSign       = MassSign->Integral(MassSign            ->GetXaxis()->FindBin(MinRange), MassSign      ->GetXaxis()->FindBin(MaxRange));
      double INTERN_ESign       = MassSign->Integral(MassSign            ->GetXaxis()->FindBin(MinRange), MassSign      ->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent      [4*CurrentSampleIndex+1+i]; 
      double INTERN_ESign_SYSTA = MassSign_SYSTA->Integral(MassSign_SYSTA->GetXaxis()->FindBin(MinRange), MassSign_SYSTA->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent_SYSTA[4*CurrentSampleIndex+1+i];
      double INTERN_ESign_SYSTB = MassSign_SYSTB->Integral(MassSign_SYSTB->GetXaxis()->FindBin(MinRange), MassSign_SYSTB->GetXaxis()->FindBin(MaxRange))/signalsMeanHSCPPerEvent_SYSTB[4*CurrentSampleIndex+1+i];
      double INTERN_Eff         = INTERN_ESign       / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      double INTERN_Eff_SYSTA   = INTERN_ESign_SYSTA / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      double INTERN_Eff_SYSTB   = INTERN_ESign_SYSTB / (signals[CurrentSampleIndex].XSec*IntegratedLuminosity);
      printf("%f X %f",INTERN_Eff, RatioValue[i]);
      if(i<2)printf(" + ");
      if(i==2)printf(" = ");

      Eff       += INTERN_Eff       * RatioValue[i];
      Eff_SYSTA += INTERN_Eff_SYSTA * RatioValue[i];
      Eff_SYSTB += INTERN_Eff_SYSTB * RatioValue[i];
   }
   }
   printf("%f\n",Eff);

   NPred*=RescaleFactor;
   toReturn.Mass      = signals[JobIdToIndex(signal)].Mass;
   toReturn.XSec_Th   = signals[JobIdToIndex(signal)].XSec;
   toReturn.XSec_Err  = signals[JobIdToIndex(signal)].XSec * 0.15;
   toReturn.XSec_Exp  = -1;
//   toReturn.XSec_Exp  = CLA (IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.20, NPred, NPred*RescaleError, 1);		//Last '1' is for logPrior integration
//   toReturn.XSec_Obs  = CL95(IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.20, 0.0  , 0.0               , 0, false, 1);   //Last '1' is for logPrior integration
//   toReturn.XSec_Exp  = CLA (IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15, NPred, NPred*RescaleError, 1);           //Last '1' is for logPrior integration
   toReturn.XSec_Obs  = 999999;
   if(Eff!=0)
   toReturn.XSec_Obs  = CL95(IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15,NPred, NPred*RescaleError              , NData, false, 1);   //Last '1' is for logPrior integration
//   toReturn.XSec_Obs  = CL95(IntegratedLuminosity, IntegratedLuminosity*0.11, Eff, Eff*0.15,NPred, NPred*RescaleError              , 0, false, 1);   //Last '1' is for logPrior integration
   toReturn.Eff       = Eff;
   toReturn.Eff_SYSTA = Eff_SYSTA;
   toReturn.Eff_SYSTB = Eff_SYSTB;

   //Not Using Greg Landsberg code:
//   double Alpha = 0.2;
//   double Rescale = 2.996 * (1+2.996*Alpha*Alpha*0.5) / ESign;
//   printf("Sample: %15s --> Total Efficiency = %f\n", signals[CurrentSampleIndex].Name.c_str(), Eff);
//   printf("Luminosity= %6.2E  XSec=%6.2Epb --> SignTrack=%6.2E SignalEvent=%6.2E ObservedInData=%6.2E\n", IntegratedLuminosity,signals[CurrentSampleIndex].XSec,NSign,ESign,NData);
//   printf("Luminosity= %6.2E  XSec=%6.2Epb --> SignTrack=%6.2E SignalEvent=%6.2E ObservedInData=%6.2E\n", IntegratedLuminosity,signals[CurrentSampleIndex].XSec*Rescale,NSign*Rescale,ESign*Rescale,NData);
//   printf("In [%4.0f,%4.0f]Observing %3f (data) while %3f+-%3f (Pred) and %3f (sign) are expected--> Probability = %6.3f%%\n",MinRange,MaxRange,NData,NPred,NPredErr,ESign*Rescale,100.0*TMath::Poisson(NData, ESign*Rescale));

//   printf("%15s: Event Eff = %7.3E (Normal) %7.3E --> %6.2f\%% (Pt*0.95) %7.3E --> %6.2f\%% (I*0.95)\n",signals[CurrentSampleIndex].Name.c_str(), Eff, Eff_SYSTA, (100.0*Eff_SYSTA)/Eff, Eff_SYSTB,(100.0*Eff_SYSTB)/Eff);
//   printf("%E | %E | %E --> %E %E %E\n", ESign, ESign_SYSTA, ESign_SYSTB, signalsMeanHSCPPerEvent[CurrentSampleIndex], signalsMeanHSCPPerEvent_SYSTA[CurrentSampleIndex], signalsMeanHSCPPerEvent_SYSTB[CurrentSampleIndex]);
//   return signals[CurrentSampleIndex].XSec*Rescale;

   return toReturn;
}

stAllInfo Exclusion_LL(string signal, string pattern){
   stAllInfo toReturn;

   Analysis_Step6_Init(signal, pattern);

   std::vector<double> TestCrossSection;
   TestCrossSection.push_back(50);
   TestCrossSection.push_back(100);
   TestCrossSection.push_back(200);
   TestCrossSection.push_back(500);
   TestCrossSection.push_back(1000);
   TestCrossSection.push_back(2000);
   TestCrossSection.push_back(3000);
   TestCrossSection.push_back(4000);
   TestCrossSection.push_back(5000);
   TestCrossSection.push_back(6000);
   TestCrossSection.push_back(7000);
   TestCrossSection.push_back(8000);
   TestCrossSection.push_back(9000);
   TestCrossSection.push_back(10000);
   TestCrossSection.push_back(15000);
   TestCrossSection.push_back(20000);
   TestCrossSection.push_back(25000);
   TestCrossSection.push_back(30000);
   TestCrossSection.push_back(35000);
   TestCrossSection.push_back(40000);

   int CurrentJobIndex = JobIdToIndex(signal);
   if(CurrentJobIndex<0){
      printf("There is no signal corresponding to the JobId Given\n");
      return toReturn;
   }

   int    Index=0;
   double* SLObsIntegral = new double[TestCrossSection.size()];
   double* SignalRescale = new double[TestCrossSection.size()];
   for(unsigned int i=0;i<TestCrossSection.size();i++){
      stResult results;
      results.SignalCrossSection=TestCrossSection[i];

      if(signals[CurrentJobIndex].Type=="Gluino"){
         results.SignalMean = Gluino_MMC_Fit->Eval(signals[CurrentJobIndex].Mass);
         results.SignalSigma= Gluino_SMC_Fit->Eval(results.SignalMean);
      }else if(signals[CurrentJobIndex].Type=="Stop"){
         results.SignalMean = Stop_MMC_Fit->Eval(signals[CurrentJobIndex].Mass);
         results.SignalSigma= Stop_SMC_Fit->Eval(results.SignalMean);
      }else if(signals[CurrentJobIndex].Type=="MGStop"){
         results.SignalMean = MGStop_MMC_Fit->Eval(signals[CurrentJobIndex].Mass);
         results.SignalSigma= MGStop_SMC_Fit->Eval(results.SignalMean);
      }else if(signals[CurrentJobIndex].Type=="Stau"){
         results.SignalMean = Stau_MMC_Fit->Eval(signals[CurrentJobIndex].Mass);
         results.SignalSigma= Stau_SMC_Fit->Eval(results.SignalMean);
      }else{
         printf("Unkown SampleType=%s\n",signals[CurrentJobIndex].Type.c_str());
	 return toReturn;
      }

      Analysis_Step6_SLDistrib(results);
      if(results.SLObsIntegral>0){
         SLObsIntegral[Index] = log10(results.SLObsIntegral);
         SignalRescale[Index] = results.SignalCrossSection;
         printf("CrossSection: %f  --> ObsIntegralProbability: %f  --> Log10: %f\n",SignalRescale[Index],results.SLObsIntegral, SLObsIntegral[Index]);
         Index++;
         if(SLObsIntegral[Index-1]<log10(0.05))break;
      }else{break;}
   }

   TCanvas* c1;
   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",600,600);
   TGraph* ExluSLIntPlot = new TGraph(Index,SLObsIntegral,SignalRescale);
   TF1*  ExluSLIntFit    = new TF1("ExluSLIntFit","pol3",-4, 0);
   ExluSLIntPlot->Fit("ExluSLIntFit","N","");
//   ExluSLIntPlot->SetRange(-2.5,0);
   ExluSLIntPlot->SetMinimum(0);
   ExluSLIntPlot->SetMaximum(ExluSLIntFit->Eval(-2.5));
   ExluSLIntFit->SetRange(-2.5,0);
   ExluSLIntFit->SetMinimum(0);
   ExluSLIntFit->SetMaximum(ExluSLIntFit->Eval(-2.5));
   ExluSLIntFit->SetLineWidth(2);
   ExluSLIntFit->SetLineColor(4);
   ExluSLIntFit->SetLineStyle(0);
   ExluSLIntFit->SetTitle("");
   ExluSLIntFit->GetXaxis()->SetTitle("Exclusion Integral Probability [ log10 ]");
   ExluSLIntFit->GetYaxis()->SetTitle("CrossSection [ pb ]");
   ExluSLIntFit->GetYaxis()->SetTitleOffset(1.70);
//   ExluSLIntFit->Draw("");
//   ExluSLIntPlot->Draw("* same");
   ExluSLIntPlot->Draw("A*");
   ExluSLIntFit->Draw("same");


   double X = log10(0.05);
   double Y = ExluSLIntFit->Eval(X);
   TBox*  b1 = new TBox(-2.5,0,X,Y                         ); b1->SetFillStyle(3004); b1->SetFillColor(kGreen-6); b1->Draw("same");
   TBox*  b2 = new TBox(-2.5,Y,0,ExluSLIntFit->GetMaximum()); b2->SetFillStyle(3004); b2->SetFillColor(kGreen-6); b2->Draw("same");

   SaveCanvas(c1, OutputPath, "Exclusion_Plot");
   delete c1;

   toReturn.XSec_Obs = Y;
   return toReturn;
}


void Analysis_Step6_Init(string signal, string pattern)
{
   CurrentSampleIndex = JobIdToIndex(signal);
   if(CurrentSampleIndex<0){
      printf("There is no signal corresponding to the JobId Given\n");
      return;
   }

   InputPath  = pattern + "DumpHistos.root";
   OutputPath = pattern + "/EXCLUSION/" + signals[CurrentSampleIndex].Name;
   MakeDirectories(OutputPath);


   TFile* InputFile = new TFile(InputPath.c_str());
   MassMCTr = (TH1D*)GetObjectFromPath(InputFile, "MCTr_Mass");
   MassSign = (TH1D*)GetObjectFromPath(InputFile, signals[CurrentSampleIndex].Name + "_Mass");
   MassPred = (TH1D*)GetObjectFromPath(InputFile, "Pred_Mass");
   MassData = (TH1D*)GetObjectFromPath(InputFile, "Data_Mass");
   MassMCTr->Rebin(4);
   MassSign->Rebin(4);
   MassPred->Rebin(4);
   MassData->Rebin(4);


   printf("Binning: MassMCTr=%3i MassSign=%3i MassPred=%3i MassData=%3i\n",MassMCTr->GetNbinsX(), MassSign->GetNbinsX(), MassPred->GetNbinsX(), MassData->GetNbinsX());

   printf("INTEGRALS = %f %f %f %f\n",MassMCTr->Integral(), MassSign->Integral(), MassPred->Integral(), MassData->Integral());
   MassPred->Scale(MassData->Integral()/MassPred->Integral());
   MassMCTr->Scale(MassData->Integral()/MassMCTr->Integral());

   printf("MCTrEvents in range [%f,%f] = %f\n",MinRange,MaxRange,MassMCTr->Integral(MassMCTr->GetXaxis()->FindBin(MinRange),MassMCTr->GetXaxis()->FindBin(MaxRange)));
   printf("SignEvents in range [%f,%f] = %f\n",MinRange,MaxRange,MassSign->Integral(MassSign->GetXaxis()->FindBin(MinRange),MassSign->GetXaxis()->FindBin(MaxRange)));
   printf("DataEvents in range [%f,%f] = %f\n",MinRange,MaxRange,MassData->Integral(MassData->GetXaxis()->FindBin(MinRange),MassData->GetXaxis()->FindBin(MaxRange)));

   MassMCTr->SetStats(kFALSE);
   TF1*  MassMCTrFit    = new TF1("MassMCTrFit",fitPred,0,1500,5);
   MassMCTrFit->SetLineWidth(2);
   MassMCTrFit->SetLineColor(4);
   MassMCTrFit->SetParameter (0,0.001*MassMCTr->Integral());
   MassMCTrFit->SetParLimits (0,0,2  *MassMCTr->Integral());
   MassMCTrFit->SetParameter (1,50);
   MassMCTrFit->SetParLimits (1,1,100);
   MassMCTrFit->SetParameter (2,0.5);
   MassMCTrFit->SetParLimits (2,0.005,5.00);
   MassMCTrFit->SetParameter (3,0.5);
   MassMCTrFit->SetParLimits (3,0.005,20);
   MassMCTrFit->SetParameter (4,1);
   MassMCTrFit->SetParLimits (4,0.005,20);

   MassSign->SetStats(kFALSE);
   TF1* MassSignFit = new TF1("MassSignFit","gaus(0)", 0, 1500);
   MassSignFit->SetParameter(0, 0.5*MassSign->Integral());
   MassSignFit->SetParLimits(0, 0,1*MassSign->Integral());
   MassSignFit->SetParameter(1, 400);
   MassSignFit->SetParLimits(1, 50,1000);
   MassSignFit->SetParameter(2, 100);
   MassSignFit->SetParLimits(2, 10,400);

   MassPred->SetStats(kFALSE);
   TF1*  MassPredFit    = new TF1("MassPredFit",fitPred,0,1500,5);
   MassPredFit->SetLineWidth(2);
   MassPredFit->SetLineColor(2);
   MassPredFit->SetParameter(0,0.001*MassPred->Integral());
   MassPredFit->SetParLimits(0,0,2  *MassPred->Integral());
   MassPredFit->SetParameter(1,50.0);
   MassPredFit->SetParLimits(1,1,100.0);
   MassPredFit->SetParameter(2,0.5);
   MassPredFit->SetParLimits(2,0.005,5.00);
   MassPredFit->SetParameter(3,0.5);
   MassPredFit->SetParLimits(3,0.005,20);
   MassPredFit->SetParameter(4,1.0);
   MassPredFit->SetParLimits(4,0.005,20);

   MassData->SetStats(kFALSE);
   TF1*  MassDataFit    = new TF1("MassDataFit",fitPred,0,1500,5);
   MassDataFit->SetLineWidth(2);
   MassDataFit->SetLineColor(4);
   MassDataFit->SetParameter(0,0.001*MassData->Integral());
   MassDataFit->SetParLimits(0,0,2  *MassData->Integral());
   MassDataFit->SetParameter(1,50);
   MassDataFit->SetParLimits(1,1,100);
   MassDataFit->SetParameter(2,0.5);
   MassDataFit->SetParLimits(2,0.005,5.00);
   MassDataFit->SetParameter(3,0.5);
   MassDataFit->SetParLimits(3,0.005,20);
   MassDataFit->SetParameter(4,1);
   MassDataFit->SetParLimits(4,0.005,20);

   MassMCTr     ->Fit("MassMCTrFit","LL R 0");
   MassSign     ->Fit("MassSignFit","LL R 0");
   MassPred     ->Fit("MassPredFit","LL R 0");
   MassData     ->Fit("MassDataFit","LL R 0");

   FitParam[0] = MassPredFit->GetParameter(0);
   FitParam[1] = MassPredFit->GetParameter(1);
   FitParam[2] = MassPredFit->GetParameter(2);
   FitParam[3] = MassPredFit->GetParameter(3);
   FitParam[4] = MassPredFit->GetParameter(4);

   MassSignPDF = GetPDF(MassSign);
 //MassSignPDF = GetPDF(MassSignFit, "MassSignPDF", round(MassSign->GetXaxis()->GetXmax()-MassSign->GetXaxis()->GetXmin()),MassSign->GetXaxis()->GetXmin(),MassSign->GetXaxis()->GetXmax());
   MassPredPDF = GetPDF(MassPredFit, "MassPredPDF", round(MassSign->GetXaxis()->GetXmax()-MassSign->GetXaxis()->GetXmin()),MassSign->GetXaxis()->GetXmin(),MassSign->GetXaxis()->GetXmax());


   TCanvas* c0 = new TCanvas("Prediction Fit", "Prediction Fit",600,600);
   c0->Divide(2,2);
   c0->cd(1);
   MassData   ->GetYaxis()->SetTitle("Data");
   MassData   ->SetTitle("Data");
   MassData   ->SetMinimum(0);
   MassData   ->Draw();
   MassDataFit->Draw("same");
   c0->cd(2);
   MassMCTr   ->GetYaxis()->SetTitle("MC");
   MassMCTr   ->SetTitle("MCTr");
   MassMCTr   ->SetMinimum(0);
   MassMCTr   ->Draw();
   MassMCTrFit->Draw("same");
   c0->cd(3);
   MassPred   ->GetYaxis()->SetTitle("Prediction");
   MassPred   ->SetTitle("Prediction");
   MassPred   ->SetMinimum(0);
   MassPred   ->Draw();
   MassPredFit->Draw("same");
   c0->cd(4);
   MassSign   ->GetYaxis()->SetTitle("Signal");
   MassSign   ->SetTitle("Signal");
   MassSign   ->SetMinimum(0);
   MassSign   ->Draw();
   MassSignFit->Draw("same");
   c0->cd(0);
   SaveCanvas(c0, OutputPath, "SeedFit");
   delete c0;

   c0 = new TCanvas("c0", "c0",600,600);
   MassSign->GetXaxis()->SetNdivisions(5+500);
   MassSign->SetTitle("");
   MassSign->SetStats(kFALSE);
   MassSign->GetXaxis()->SetTitle("Reconstructed Mass [ GeV/c^{2} ]");
   MassSign->GetYaxis()->SetTitle("Entries in 10pb^{-1}");
   MassSign->GetYaxis()->SetTitleOffset(1.50);
   MassSign   ->SetMinimum(0);
   MassSign   ->Draw();
   MassSignFit->Draw("same");
   TPaveText* st1 = new TPaveText(0.40,0.82,0.79,0.92, "NDC");
   st1->SetFillColor(0);
   st1->SetTextAlign(12);
   sprintf(Buffer,"Cst   %+6.2E #pm %6.2E",MassSignFit->GetParameter(0),MassSignFit->GetParError(0));        st1->AddText(Buffer);
   sprintf(Buffer,"Mean  %+6.2E #pm %6.2E",MassSignFit->GetParameter(1),MassSignFit->GetParError(1));        st1->AddText(Buffer);
   sprintf(Buffer,"Sigma %+6.2E #pm %6.2E",MassSignFit->GetParameter(2),MassSignFit->GetParError(2));        st1->AddText(Buffer);
   st1->Draw("same");
   SaveCanvas(c0, OutputPath, "SignFit");
   delete c0;

   c0 = new TCanvas("c0", "c0",600,600);
   MassData->GetXaxis()->SetNdivisions(5+500);
   MassData->SetTitle("");
   MassData->SetStats(kFALSE);
   MassData->GetXaxis()->SetTitle("Reconstructed Mass [ GeV/c^{2} ]");
   MassData->GetYaxis()->SetTitle("Entries in 10pb^{-1}");
   MassData->GetYaxis()->SetTitleOffset(1.50);
   MassData   ->SetMinimum(0);
   MassData   ->SetMaximum(1.1*std::max(MassPred->GetMaximum(),MassData->GetMaximum()));
   MassData   ->SetMinimum(0);
   MassData   ->SetFillColor(7);
   MassData   ->SetLineColor(11);
   MassData   ->SetMarkerColor(11);
   MassData   ->Draw("HIST");
   MassData   ->Draw("E1 same");
   MassPred   ->Draw("E1 same");
   MassPredFit->Draw("same");
   TPaveText* st2 = new TPaveText(0.40,0.72,0.79,0.92, "NDC");
   st2->SetFillColor(0);
   st2->SetTextAlign(12);
   sprintf(Buffer,"N %+6.2E  #pm %6.2E"     ,   MassPredFit->GetParameter(0),MassPredFit->GetParError(0));        st2->AddText(Buffer);
   sprintf(Buffer,"#Delta %+6.2E  #pm %6.2E",   MassPredFit->GetParameter(1),MassPredFit->GetParError(1));        st2->AddText(Buffer);
   sprintf(Buffer,"#alpha %+6.2E  #pm %6.2E",-1*MassPredFit->GetParameter(2),MassPredFit->GetParError(2));        st2->AddText(Buffer);
   sprintf(Buffer,"#beta %+6.2E  #pm %6.2E" ,   MassPredFit->GetParameter(3),MassPredFit->GetParError(3));        st2->AddText(Buffer);
   sprintf(Buffer,"#gamma %+6.2E  #pm %6.2E",   MassPredFit->GetParameter(4),MassPredFit->GetParError(4));        st2->AddText(Buffer);
   st2->Draw("same");
   SaveCanvas(c0, OutputPath, "PredFit");
   delete c0;
}


void Analysis_Step6_SLDistrib(stResult& results){
   TCanvas* c1 = NULL;

   char tmpbuffer[1024];
   sprintf(tmpbuffer,"_CS%05.0f_Mean%03.0f_Sigma%03.0f",results.SignalCrossSection, results.SignalMean, results.SignalSigma);
   string Path =  OutputPath + tmpbuffer;


   double Rescale     = results.SignalCrossSection/signals[CurrentSampleIndex].XSec;
   double Sign_NEvent = MassSign->Integral() * Rescale;
   double Bckg_NEvent = MassData->Integral() - Sign_NEvent;

   int MaxExpectedEvent;
   MaxExpectedEvent = std::max(10.0,2*Sign_NEvent);
   TH1D* SignPoissonPdf = new TH1D("SignPoissonPdf","SignPoissonPdf",MaxExpectedEvent,0,MaxExpectedEvent);
   for(int i=0;i<=SignPoissonPdf->GetNbinsX();i++){ SignPoissonPdf->SetBinContent(i, TMath::Poisson(i*1.0, Sign_NEvent) ); }
   TH1D* SignPoissonPDF = GetPDF(SignPoissonPdf);

   MaxExpectedEvent = std::max(10.0,2*Bckg_NEvent);
   TH1D* PredPoissonPdf = new TH1D("PredPoissonPdf","PredPoissonPdf",MaxExpectedEvent,0,MaxExpectedEvent);
   for(int i=0;i<=PredPoissonPdf->GetNbinsX();i++){ PredPoissonPdf->SetBinContent(i, TMath::Poisson(i*1.0, Bckg_NEvent) ); }
   TH1D* PredPoissonPDF = GetPDF(PredPoissonPdf);

   c1 = new TCanvas("c1", "c1",600,600);
   results.SLObs = GetS(MassData, FitParam,results.SignalMean,results.SignalSigma,c1);
   SaveCanvas(c1, Path, "SL_Obs");
   delete c1;

   TH1D* SDistrib = new TH1D("SDistrib", "SDistrib", 10000,0,5);
   SDistrib->SetBit(TH1::kCanRebin);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Making PsuedoExperiment      :");
   int CountStep = NPseudoExperiment/50;if(CountStep==0)CountStep=1;
   for(unsigned int i=0;i<NPseudoExperiment;i++){
      if(i%CountStep==0){printf(".");fflush(stdout);}
      TH1D* DataPE = MakePseudoExperiment(MassPredPDF,   GetRandValue(PredPoissonPDF), MassSignPDF->GetXaxis()->GetNbins() );
      TH1D* SignPE = MakePseudoExperiment(MassSignPDF,   GetRandValue(SignPoissonPDF), MassSignPDF->GetXaxis()->GetNbins() );
      DataPE->Add(SignPE,1.0);

      double S;
      if(i<5){
         TCanvas* c1 = new TCanvas("c1", "c1",600,600);
         S = GetS(DataPE,FitParam,results.SignalMean,results.SignalSigma,c1);           
         SaveCanvas(c1, Path, string(string("SL_PE")+(long)i), true);
         delete c1;
      }else{
         S = GetS(DataPE,FitParam,results.SignalMean,results.SignalSigma);
      }

//      if(S<0)S=0;
      SDistrib->Fill(S);
      delete SignPE;
      delete DataPE;
   }printf("\n");

   results.SL3Sigma      = GetIntegralOnLeft(SDistrib,1.0/742);
   results.SL5Sigma      = GetIntegralOnLeft(SDistrib,1.0-1.0/3489);
   results.SLMedian      = GetIntegralOnLeft(SDistrib,1.0/2);
   results.SLObsIntegral = SDistrib->Integral(0,SDistrib->GetXaxis()->FindBin(results.SLObs)) / NPseudoExperiment;

   c1 = new TCanvas("c1", "c1",600,600);
   SDistrib->Rebin(100);
   SDistrib->Draw();
//   TLine* l1 = new TLine(results.SL3Sigma,0,results.SL3Sigma,SDistrib->GetMaximum());
//   l1->SetLineWidth(3);   l1->SetLineColor(8);   l1->Draw("same");
//   TLine* l2 = new TLine(results.SL5Sigma,0,results.SL5Sigma,SDistrib->GetMaximum());
//   l2->SetLineWidth(3);   l2->SetLineColor(8);   l2->Draw("same");
   TLine* l3 = new TLine(results.SLMedian,0,results.SLMedian,SDistrib->GetMaximum());
   l3->SetLineWidth(3);   l3->SetLineColor(4);   l3->Draw("same");
   TLine* l4 = new TLine(results.SLObs,0,results.SLObs,SDistrib->GetMaximum());
   l4->SetLineWidth(3);   l4->SetLineColor(2);   l4->Draw("same");

   c1->SetLogy(true);
   SaveCanvas(c1, Path, "SL_Distrib", true);
   delete c1;

   delete PredPoissonPdf;
   delete PredPoissonPDF;
   delete SignPoissonPdf;
   delete SignPoissonPDF;
   delete SDistrib;
}

double GetS(TH1D* Data, double* FitParams, double SignalMean, double SignalSigma, TCanvas* c0){
   double FitScale = Data->Integral();

   TF1*  DataFit    = new TF1("bckgFit",fitPred,0,1500,5);
   DataFit->SetLineWidth(2);
   DataFit->SetLineColor(2);
   DataFit->SetParameter(0,0.001*FitScale);
   DataFit->SetParLimits (0,0.0,1*FitScale);
   DataFit->FixParameter(1,FitParams[1]);
   DataFit->FixParameter(2,FitParams[2]);
   DataFit->FixParameter(3,FitParams[3]);
   DataFit->FixParameter(4,FitParams[4]);
   Data->Fit("bckgFit","LL R Q0");

   TF1*  SignFit    = new TF1("signFit",fitPredAndSignal,0,1500,8);
   SignFit->SetLineWidth(2);
   SignFit->SetLineColor(4);
   SignFit->SetParameter(0,0.001*FitScale);
   SignFit->SetParLimits (0,0.0,1*FitScale);
   SignFit->FixParameter(1,FitParams[1]);
   SignFit->FixParameter(2,FitParams[2]);
   SignFit->FixParameter(3,FitParams[3]);
   SignFit->FixParameter(4,FitParams[4]);
   SignFit->FixParameter(5,SignalMean);
   SignFit->FixParameter(6,SignalSigma);
   SignFit->SetParameter(7,1*FitScale);
   SignFit->SetParLimits (7,0.0,100*FitScale);
   Data->Fit("signFit","LL R Q0");

   double Log1 = LogLikeliHood(Data,DataFit);
   double Log2 = LogLikeliHood(Data,SignFit);
   double Arg  = Log2 - Log1;
   double S = 0;   
   if(Arg>=0){S=sqrt(2)*sqrt(Arg);}else{S=-sqrt(2)*sqrt(-Arg);}

   if(c0){
      Data->SetTitle("");
      Data->SetStats(kFALSE);
      Data->GetXaxis()->SetTitle("Reconstructed Mass [ GeV/c^{2} ]");
      Data->GetYaxis()->SetTitle("Entries");
      Data->GetYaxis()->SetTitleOffset(1.70);
  
      Data   ->Draw();
      DataFit->Draw("same");
      SignFit->Draw("same");

      char Buffer[1024];
      TPaveText* st1 = new TPaveText(0.40,0.82,0.79,0.92, "NDC");
      st1->SetFillColor(0);
      st1->SetTextAlign(12);
      sprintf(Buffer,"log( L_{data only} ) = %+6.2E", Log1);        st1->AddText(Buffer);
      sprintf(Buffer,"log( L_{data+sign} ) = %+6.2E", Log2);        st1->AddText(Buffer);
      sprintf(Buffer,"S_{L} = %6.2E"                ,S);            st1->AddText(Buffer);
      st1->Draw("same");
   }

   return S;
}


Double_t fitPred(Double_t *v, Double_t *par){
   double x = v[0];

   if(x-par[1]<=0)return 0;
   double Expo = TMath::Exp(-1*par[2]*pow(x,par[3]));
   double Sqrt = pow(x-par[1],par[4]);
   return par[0] * Expo * Sqrt;
}

Double_t fitPredAndSignal(Double_t *v, Double_t *par){
   double x = v[0];
   double Data = fitPred(v,par);
   double Sign = TMath::Gaus(x, par[5],par[6],true);
   return Data + par[7]*Sign;
}


TH1D* GetPDF(TF1* pdf, string name, int NBins, int Xmin, int Xmax){
   string NewName = name + "_PDF";

   double funcRangeMin;
   double funcRangeMax;
   pdf->GetRange(funcRangeMin,funcRangeMax);

   TH1D* PDF = new TH1D(NewName.c_str(),NewName.c_str(),NBins,Xmin,Xmax);
   for(int i=0;i<=NBins;i++){
      double Value;
      if(PDF->GetBinLowEdge(i)>=funcRangeMin && PDF->GetBinLowEdge(i)<=funcRangeMax){
         Value = pdf->Eval(PDF->GetBinLowEdge(i));
      }else{
         Value = 0;
      }

      if(i==0){
         PDF->SetBinContent(i, Value );
      }else{
         PDF->SetBinContent(i, Value+PDF->GetBinContent(i-1) );
      }
   }
   PDF->Scale(1.0/PDF->GetBinContent(PDF->GetNbinsX()));
   return PDF;
}



TH1D* GetPDF(TH1D* pdf){
   char NewName[2048];
   sprintf(NewName,"%s_PDF", pdf->GetName());

   TH1D* PDF = new TH1D(NewName,NewName,pdf->GetNbinsX(),pdf->GetXaxis()->GetXmin(),pdf->GetXaxis()->GetXmax());
   for(int i=0;i<=pdf->GetNbinsX();i++){
      if(i==0){
         PDF->SetBinContent(i, pdf->GetBinContent(i) );
      }else{
         PDF->SetBinContent(i, pdf->GetBinContent(i)+PDF->GetBinContent(i-1) );
      }
   }
   PDF->Scale(1.0/PDF->GetBinContent(PDF->GetNbinsX()));
   return PDF;
}

double GetRandValue(TH1D* PDF){
   int randNumber = rand();
   double uniform = randNumber / (double)RAND_MAX;
   for(int i=0;i<=PDF->GetNbinsX();i++){
      if(PDF->GetBinContent(i)>uniform){
         return PDF->GetXaxis()->GetBinLowEdge(i);
      }
   }
   return PDF->GetXaxis()->GetBinLowEdge(PDF->GetNbinsX());
}

TH1D* MakePseudoExperiment(TH1D* PDF, double NEntries, int NBins){
   char NewName[2048];
   sprintf(NewName,"%s_PseudoExp", PDF->GetName());

   if(NBins<=0)NBins=PDF->GetNbinsX();
   TH1D* PseudoExp = new TH1D(NewName,NewName, NBins ,PDF->GetXaxis()->GetXmin(),PDF->GetXaxis()->GetXmax());
   for(unsigned int i=0;i<NEntries;i++){
      PseudoExp->Fill(GetRandValue(PDF));
   }
   return PseudoExp;
}



double GetMedian(TH1D* pdf){
   return GetIntegralOnLeft(pdf,0.5);
}

double GetIntegralOnLeft(TH1D* pdf, double IntegralRatio){
   double HalfIntegral = IntegralRatio*pdf->Integral(1, pdf->GetNbinsX());
   double Sum = 0;
   for(int i=1;i<=pdf->GetNbinsX();i++){
      Sum += pdf->GetBinContent(i);
      if(Sum>=HalfIntegral){
         return pdf->GetXaxis()->GetBinCenter(i);
      }
   }   
   return pdf->GetXaxis()->GetBinCenter(pdf->GetNbinsX());
}

double SigmaFromProb(double p, string where)
{
   //http://en.wikipedia.org/wiki/Standard_deviation
   if(where=="outside"){
      if(p==0)return 10;
      return TMath::ErfInverse(1-p)*sqrt(2);
   }else if(where=="inside"){
      return TMath::ErfInverse(p)*sqrt(2);
   }

   printf("unknown pobability region to compute Significance\n\"outside\" is assumed");
   return TMath::ErfInverse(1-p)*sqrt(2);
}


double LogLikeliHood(TH1D* Histo1, TH1D* Histo2){
   double LogLikeliHood = 0;
   for(int i=3;i<=Histo1->GetNbinsX();i++){
        double Expected = Histo2->GetBinContent(Histo2->GetXaxis()->FindBin(Histo1->GetXaxis()->GetBinLowEdge(i) ) );
        double Observed = Histo1->GetBinContent(i);
        if(Expected<=0){
           continue;
        }
        double Proba = TMath::Poisson(Observed,Expected);
        if(Proba<=1E-10)Proba=1E-10;
        LogLikeliHood += log(Proba);
   }
   return LogLikeliHood;
}

double LogLikeliHood(TH1D* Histo1, TF1* Histo2){
   double LogLikeliHood = 0;
   for(int i=3;i<=Histo1->GetNbinsX();i++){
	double Expected = Histo2->Eval(Histo1->GetXaxis()->GetBinCenter(i));
        double Observed = Histo1->GetBinContent(i);
	if(Expected<=0){
	   continue;
        }
        double Proba = TMath::Poisson(Observed,Expected);
        if(Proba<=1E-10)Proba=1E-10;
        LogLikeliHood += log(Proba);
   }
   return LogLikeliHood;
}




void SimRecoCorrelation(string InputPattern)
{
   std::vector<double> SampleMean;
   std::vector<double> SampleSigma;

   string Input = InputPattern + "DumpHistos.root";
   string outpath = InputPattern + "/EXCLUSION";
   MakeDirectories(outpath);

   TFile* InputFile = new TFile(Input.c_str());

   for(unsigned int s=0;s<signals.size();s++){
      TH1* Sign = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[s].Name);
      TF1* SignFit = new TF1("SignFit","gaus(0)", 0, 1500);
      SignFit->SetParameter(0, 0.5*Sign->Integral());
      SignFit->SetParLimits(0, 0,1*Sign->Integral());
      SignFit->SetParameter(1, 400);
      SignFit->SetParLimits(1, 50,1000);
      SignFit->SetParameter(2, 100);
      SignFit->SetParLimits(2, 10,400);
      Sign->SetStats(kFALSE);
      Sign->Fit("SignFit","LL M R 0Q");

      SampleMean .push_back(SignFit->GetParameter(1));
      SampleSigma.push_back(SignFit->GetParameter(2));      
     
      TCanvas* c1 = new TCanvas("c1", "c1",600,600);
      Sign->Draw();
      SignFit->Draw("same");
      SaveCanvas(c1, outpath + "MassFit", signals[s].Name,true);
      delete c1;

      delete SignFit;
      delete Sign;
   }
   delete InputFile;


   int    NSample = signals.size();
   int    NStau   = 0;   double StauMass  [NSample];   double StauMean  [NSample];   double StauSigma  [NSample];
   int    NStop   = 0;   double StopMass  [NSample];   double StopMean  [NSample];   double StopSigma  [NSample];
   int    NMGStop = 0;   double MGStopMass[NSample];   double MGStopMean[NSample];   double MGStopSigma[NSample];
   int    NGluino = 0;   double GluinoMass[NSample];   double GluinoMean[NSample];   double GluinoSigma[NSample];
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Type=="Gluino"){GluinoMass[NGluino] = signals[s].Mass; GluinoMean[NGluino]= SampleMean[s]; GluinoSigma[NGluino] = SampleSigma[s]; NGluino++;}
      if(signals[s].Type=="Stop"  ){StopMass  [NStop]   = signals[s].Mass; StopMean  [NStop]  = SampleMean[s]; StopSigma  [NStop]   = SampleSigma[s]; NStop++;  }
      if(signals[s].Type=="MGStop"){MGStopMass[NMGStop] = signals[s].Mass; MGStopMean[NMGStop]= SampleMean[s]; MGStopSigma[NMGStop] = SampleSigma[s]; NMGStop++;}
      if(signals[s].Type=="Stau"  ){StauMass  [NStau]   = signals[s].Mass; StauMean  [NStau]  = SampleMean[s]; StauSigma  [NStau]   = SampleSigma[s]; NStau++;  }
   }

   TGraph* Stop_MMC = new TGraph(NStop,StopMass,StopMean);
   Stop_MMC_Fit    = new TF1("Stop_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Stop_MMC_Fit->SetParLimits(0, 0,10);
   Stop_MMC_Fit->SetParameter(0,10);
   Stop_MMC->Fit("Stop_MMC_Fit","M NR");
   Stop_MMC_Fit->SetLineWidth(2);
   Stop_MMC_Fit->SetLineColor(4);
   Stop_MMC_Fit->SetLineStyle(2);
   Stop_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   Stop_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stop_MMC->SetMarkerColor(Stop_MMC_Fit->GetLineColor());
   Stop_MMC->SetLineColor  (Stop_MMC_Fit->GetLineColor());
   Stop_MMC->SetLineStyle  (Stop_MMC_Fit->GetLineStyle());

   TGraph* MGStop_MMC = new TGraph(NMGStop,MGStopMass,MGStopMean);
   MGStop_MMC_Fit    = new TF1("MGStop_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   MGStop_MMC_Fit->SetParLimits(0, 0,10);
   MGStop_MMC_Fit->SetParameter(0,10);
   MGStop_MMC->Fit("MGStop_MMC_Fit","M NR");
   MGStop_MMC_Fit->SetLineWidth(2);
   MGStop_MMC_Fit->SetLineColor(1);
   MGStop_MMC_Fit->SetLineStyle(2);
   MGStop_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   MGStop_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   MGStop_MMC->SetMarkerColor(MGStop_MMC_Fit->GetLineColor());
   MGStop_MMC->SetLineColor  (MGStop_MMC_Fit->GetLineColor());
   MGStop_MMC->SetLineStyle  (MGStop_MMC_Fit->GetLineStyle());

   TGraph* Gluino_MMC = new TGraph(NGluino,GluinoMass,GluinoMean);
   Gluino_MMC_Fit    = new TF1("Gluino_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Gluino_MMC_Fit->SetParLimits(0, 0,10);
   Gluino_MMC_Fit->SetParameter(0,10);
   Gluino_MMC->Fit("Gluino_MMC_Fit","M NR");
   Gluino_MMC_Fit->SetLineWidth(2);
   Gluino_MMC_Fit->SetLineColor(2);
   Gluino_MMC_Fit->SetLineStyle(2);
   Gluino_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   Gluino_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Gluino_MMC->SetMarkerColor(Gluino_MMC_Fit->GetLineColor());
   Gluino_MMC->SetLineColor  (Gluino_MMC_Fit->GetLineColor());
   Gluino_MMC->SetLineStyle  (Gluino_MMC_Fit->GetLineStyle());

   TGraph* Stau_MMC = new TGraph(NStau,StauMass,StauMean);
   Stau_MMC_Fit    = new TF1("Stau_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Stau_MMC_Fit->SetParLimits(0, 0,10);
   Stau_MMC_Fit->SetParameter(0,10);
   Stau_MMC->Fit("Stau_MMC_Fit","M NR");
   Stau_MMC_Fit->SetLineWidth(2);
   Stau_MMC_Fit->SetLineColor(8);
   Stau_MMC_Fit->SetLineStyle(2);
   Stau_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   Stau_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stau_MMC->SetMarkerColor(Stau_MMC_Fit->GetLineColor());
   Stau_MMC->SetLineColor  (Stau_MMC_Fit->GetLineColor());
   Stau_MMC->SetLineStyle  (Stau_MMC_Fit->GetLineStyle());

   TGraph* Stop_SMC = new TGraph(NStop,StopMean,StopSigma);
   Stop_SMC_Fit    = new TF1("Stop_SMC_Fit","[0]+[1]*x+[2]*x*x",0,800);
   Stop_SMC_Fit->SetParLimits(0, 0,20);
   Stop_SMC_Fit->SetParameter(0,10);
   Stop_SMC->Fit("Stop_SMC_Fit","M NR");
   Stop_SMC_Fit->SetLineWidth(Stop_MMC_Fit->GetLineWidth());
   Stop_SMC_Fit->SetLineColor(Stop_MMC_Fit->GetLineColor());
   Stop_SMC_Fit->SetLineStyle(Stop_MMC_Fit->GetLineStyle());
   Stop_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stop_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   Stop_SMC->SetMarkerColor(Stop_SMC_Fit->GetLineColor());
   Stop_SMC->SetLineColor  (Stop_SMC_Fit->GetLineColor());
   Stop_SMC->SetLineStyle  (Stop_SMC_Fit->GetLineStyle());

   TGraph* MGStop_SMC = new TGraph(NMGStop,MGStopMean,MGStopSigma);
   MGStop_SMC_Fit    = new TF1("MGStop_SMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   MGStop_SMC_Fit->SetParLimits(0, 0,20);
   MGStop_SMC_Fit->SetParameter(0,10);
   MGStop_SMC->Fit("MGStop_SMC_Fit","M NR");
   MGStop_SMC_Fit->SetLineWidth(MGStop_MMC_Fit->GetLineWidth());
   MGStop_SMC_Fit->SetLineColor(MGStop_MMC_Fit->GetLineColor());
   MGStop_SMC_Fit->SetLineStyle(MGStop_MMC_Fit->GetLineStyle());
   MGStop_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   MGStop_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   MGStop_SMC->SetMarkerColor(MGStop_SMC_Fit->GetLineColor());
   MGStop_SMC->SetLineColor  (MGStop_SMC_Fit->GetLineColor());
   MGStop_SMC->SetLineStyle  (MGStop_SMC_Fit->GetLineStyle());

   TGraph* Gluino_SMC = new TGraph(NGluino,GluinoMean,GluinoSigma);
   Gluino_SMC_Fit    = new TF1("Gluino_SMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Gluino_SMC_Fit->SetParLimits(0, 0,20);
   Gluino_SMC_Fit->SetParameter(0,10);
   Gluino_SMC->Fit("Gluino_SMC_Fit","M NR");
   Gluino_SMC_Fit->SetLineWidth(Gluino_MMC_Fit->GetLineWidth());
   Gluino_SMC_Fit->SetLineColor(Gluino_MMC_Fit->GetLineColor());
   Gluino_SMC_Fit->SetLineStyle(Gluino_MMC_Fit->GetLineStyle());
   Gluino_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Gluino_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   Gluino_SMC->SetMarkerColor(Gluino_SMC_Fit->GetLineColor());
   Gluino_SMC->SetLineColor  (Gluino_SMC_Fit->GetLineColor());
   Gluino_SMC->SetLineStyle  (Gluino_SMC_Fit->GetLineStyle());

   TGraph* Stau_SMC = new TGraph(NStau,StauMean,StauSigma);
   Stau_SMC_Fit    = new TF1("Stau_SMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Stau_SMC_Fit->SetParLimits(0, 0,20);
   Stau_SMC_Fit->SetParameter(0,10);
   Stau_SMC->Fit("Stau_SMC_Fit","M NR");
   Stau_SMC_Fit->SetLineWidth(Stau_MMC_Fit->GetLineWidth());
   Stau_SMC_Fit->SetLineColor(Stau_MMC_Fit->GetLineColor());
   Stau_SMC_Fit->SetLineStyle(Stau_MMC_Fit->GetLineStyle());
   Stau_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stau_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   Stau_SMC->SetMarkerColor(Stau_SMC_Fit->GetLineColor());
   Stau_SMC->SetLineColor  (Stau_SMC_Fit->GetLineColor());
   Stau_SMC->SetLineStyle  (Stau_SMC_Fit->GetLineStyle());


   TCanvas* c1;
   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",600,600);
   c1->SetGridx(true);
   c1->SetGridy(true);
   Stop_MMC_Fit->Draw("");
   Stop_MMC->Draw("* same");
   MGStop_MMC_Fit->Draw("same");
   MGStop_MMC->Draw("* same");
   Gluino_MMC_Fit->Draw("same");
   Gluino_MMC->Draw("* same");
   Stau_MMC_Fit->Draw("same");
   Stau_MMC->Draw("* same");

   TLegend* leg = new TLegend(0.15,0.93,0.35,0.73);
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stau_MMC  , "Stau"  ,"PL");
   leg->AddEntry(Stop_MMC  , "Stop"  ,"PL");
   leg->AddEntry(MGStop_MMC, "MGStop","PL");
   leg->AddEntry(Gluino_MMC, "Gluino","PL");
   leg->Draw();
   SaveCanvas(c1, outpath, "Correlation_MassMass");
   delete c1;

   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",600,600);
   c1->SetGridx(true);
   c1->SetGridy(true);
   Stop_SMC_Fit->Draw("");
   Stop_SMC->Draw("* same");
   MGStop_SMC_Fit->Draw("same");
   MGStop_SMC->Draw("* same");
   Gluino_SMC_Fit->Draw("same");
   Gluino_SMC->Draw("* same");
   Stau_SMC_Fit->Draw("same");
   Stau_SMC->Draw("* same");

   leg = new TLegend(0.15,0.93,0.35,0.73);
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stau_SMC  , "Stau"  ,"PL");
   leg->AddEntry(Stop_SMC  , "Stop"  ,"PL");
   leg->AddEntry(MGStop_SMC, "MGStop","PL");
   leg->AddEntry(Gluino_SMC, "Gluino","PL");
   leg->Draw();
   SaveCanvas(c1, outpath, "Correlation_SigmaMass");
   delete c1;
}

int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}


void GetSignalMeanHSCPPerEvent(string InputPattern)
{
   string Input = InputPattern + "Aeff.tmp";
   FILE* pFile = fopen(Input.c_str(),"r");
   if(!pFile){
      printf("Not Found: %s\n",Input.c_str());
      return;
   }

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
   for(unsigned int n=0;n<4;n++){
     char  sname[256];
     float weff, ueff;
     float weff_SYSTA, ueff_SYSTA;
     float weff_SYSTB, ueff_SYSTB;
     if(n==0)fscanf(pFile,"%s     Eff=%E (%E)SYSTA: Eff=%E (%E)SYSTB: Eff=%E (%E)\n",sname,&weff,&ueff,&weff_SYSTA,&ueff_SYSTA,&weff_SYSTB,&ueff_SYSTB);
     if(n==1)fscanf(pFile,"%s NC0 Eff=%E (%E)SYSTA: Eff=%E (%E)SYSTB: Eff=%E (%E)\n",sname,&weff,&ueff,&weff_SYSTA,&ueff_SYSTA,&weff_SYSTB,&ueff_SYSTB);
     if(n==2)fscanf(pFile,"%s NC1 Eff=%E (%E)SYSTA: Eff=%E (%E)SYSTB: Eff=%E (%E)\n",sname,&weff,&ueff,&weff_SYSTA,&ueff_SYSTA,&weff_SYSTB,&ueff_SYSTB);
     if(n==3)fscanf(pFile,"%s NC2 Eff=%E (%E)SYSTA: Eff=%E (%E)SYSTB: Eff=%E (%E)\n",sname,&weff,&ueff,&weff_SYSTA,&ueff_SYSTA,&weff_SYSTB,&ueff_SYSTB);

     int Index = JobIdToIndex(sname);
     if(Index<0){
        printf("BUG UNKNOWN SIGNAL (%s) WHEN READING AVERAGE SELECTED HSCP PER EVENT\n",sname);
     }else{
        //printf("%s n=%i Eff=%f\n",sname,n,weff);
        signalsMeanHSCPPerEvent      [4*Index+n] = (float)std::min(1.0f,weff);
        signalsMeanHSCPPerEvent_SYSTA[4*Index+n] = (float)std::min(1.0f,weff_SYSTA);
        signalsMeanHSCPPerEvent_SYSTB[4*Index+n] = (float)std::min(1.0f,weff_SYSTB);
     }
   }}

   fclose(pFile);

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


TGraph* PopulateTheGraph(TGraph* in, double Min, double Max, double Step){

   TGraph* out = new TGraph((Max-Min)/Step);

   unsigned int Index = 0;
   for(double x=Min;x<Max;x+=Step){
      out->SetPoint(Index,x,in->Eval(x, 0, ""));
      Index++;
   }

   return out;
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



