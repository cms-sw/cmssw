
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"
#include "tdrstyle.C"
#include "Analysis_PlotFunction.h"
#include "Analysis_Samples.h"

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
void     Analysis_Step6_Init(string signal, string Path);
void     Analysis_Step6_SLDistrib(stResult& results);
double   GetIntegralOnLeft(TH1D* pdf, double IntegralRatio);
double   Exclusion(string signal, string pattern);

void     SimRecoCorrelation(string InputPattern);
int      JobIdToIndex(string JobId);




double MinRange = 50;
double MaxRange = 300;

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
TF1* Gluino_MMC_Fit = NULL;
TF1* Stau_SMC_Fit   = NULL;
TF1* Stop_SMC_Fit   = NULL;
TF1* Gluino_SMC_Fit = NULL;

std::vector<stSignal> signals;

void Analysis_Step6(){
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.13);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505,"X");
   TCanvas* c1;

   GetSignalDefinition(signals);
   SimRecoCorrelation("SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");


   
   double Gluino200 = Exclusion("Gluino200","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Gluino300 = Exclusion("Gluino300","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Gluino400 = Exclusion("Gluino400","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Gluino500 = Exclusion("Gluino500","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Gluino600 = Exclusion("Gluino600","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Gluino900 = Exclusion("Gluino900","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");

   printf("200 --> Excluded Above %f\n",Gluino200);
   printf("300 --> Excluded Above %f\n",Gluino300);
   printf("400 --> Excluded Above %f\n",Gluino400);
   printf("500 --> Excluded Above %f\n",Gluino500);
   printf("600 --> Excluded Above %f\n",Gluino600);
   printf("900 --> Excluded Above %f\n",Gluino900);


   double MassGluino[] = {200,300,400,500,600,900};
   double XSecGluino[] = {Gluino200, Gluino300, Gluino400, Gluino500, Gluino600, Gluino900};

   c1 = new TCanvas("c1", "c1",800,600);
   TGraph* GluinoExclusion = new TGraph(6,MassGluino,XSecGluino);
   GluinoExclusion->SetLineColor(4);
   GluinoExclusion->SetFillColor(4);
   GluinoExclusion->SetLineWidth(2001);
   GluinoExclusion->SetFillStyle(3004);
   GluinoExclusion->Draw("AL* same");
   GluinoExclusion->SetTitle("");
   GluinoExclusion->GetXaxis()->SetTitle("Gluino HSCP Mass [ GeV/c^{2} ]");
   GluinoExclusion->GetYaxis()->SetTitle("CrossSection [ Pb ]");
   GluinoExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, "Results/EXCLUSION/ExclusionPlot", "Gluino");
   delete c1;


   double Stop130 = Exclusion("Stop130","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stop200 = Exclusion("Stop200","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stop300 = Exclusion("Stop300","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stop500 = Exclusion("Stop500","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stop800 = Exclusion("Stop800","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");

   printf("130 --> Excluded Above %f\n",Stop130);
   printf("200 --> Excluded Above %f\n",Stop200);
   printf("300 --> Excluded Above %f\n",Stop300);
   printf("500 --> Excluded Above %f\n",Stop500);
   printf("800 --> Excluded Above %f\n",Stop800);

   double MassStop[] = {130,200,300,500,800};
   double XSecStop[] = {Stop130, Stop200, Stop300, Stop500, Stop800};

   c1 = new TCanvas("c1", "c1",800,600);
   TGraph* StopExclusion = new TGraph(5,MassStop,XSecStop);
   StopExclusion->SetLineColor(2);
   StopExclusion->SetFillColor(2);
   StopExclusion->SetLineWidth(2001);
   StopExclusion->SetFillStyle(3004);
   StopExclusion->Draw("AL* same");
   StopExclusion->SetTitle("");
   StopExclusion->GetXaxis()->SetTitle("Stop HSCP Mass [ GeV/c^{2} ]");
   StopExclusion->GetYaxis()->SetTitle("CrossSection [ Pb ]");
   StopExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, "Results/EXCLUSION/ExclusionPlot", "Stop");
   delete c1;


   double Stau100 = Exclusion("Stau100","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stau126 = Exclusion("Stau126","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stau156 = Exclusion("Stau156","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stau200 = Exclusion("Stau200","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stau247 = Exclusion("Stau247","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");
   double Stau308 = Exclusion("Stau308","SplitMode2/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/WPPt-20/WPI-25/");

   printf("100 --> Excluded Above %f\n",Stau100);
   printf("126 --> Excluded Above %f\n",Stau126);
   printf("156 --> Excluded Above %f\n",Stau156);
   printf("200 --> Excluded Above %f\n",Stau200);
   printf("247 --> Excluded Above %f\n",Stau247);
   printf("308 --> Excluded Above %f\n",Stau308);

   double MassStau[] = {100, 126, 156, 200, 247, 308};
   double XSecStau[] = {Stau100, Stau126, Stau156, Stau200, Stau247, Stau308};

   c1 = new TCanvas("c1", "c1",800,600);
   TGraph* StauExclusion = new TGraph(6,MassStau,XSecStau);
   StauExclusion->SetLineColor(8);
   StauExclusion->SetFillColor(8);
   StauExclusion->SetLineWidth(2001);
   StauExclusion->SetFillStyle(3004);
   StauExclusion->Draw("AL* same");
   StauExclusion->SetTitle("");
   StauExclusion->GetXaxis()->SetTitle("Stau HSCP Mass [ GeV/c^{2} ]");
   StauExclusion->GetYaxis()->SetTitle("CrossSection [ Pb ]");
   StauExclusion->GetYaxis()->SetTitleOffset(1.70);
   SaveCanvas(c1, "Results/EXCLUSION/ExclusionPlot", "Stau");
   delete c1;
}


double Exclusion(string signal, string pattern){
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
      return -1;
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
      }else if(signals[CurrentJobIndex].Type=="Stau"){
         results.SignalMean = Stau_MMC_Fit->Eval(signals[CurrentJobIndex].Mass);
         results.SignalSigma= Stau_SMC_Fit->Eval(results.SignalMean);
      }else{
         printf("Unkown SampleType=%s\n",signals[CurrentJobIndex].Type.c_str());
	 return -1;
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
   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",800,600);
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
   ExluSLIntFit->GetYaxis()->SetTitle("CrossSection [ Pb ]");
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

   return Y;
}


void Analysis_Step6_Init(string signal, string pattern)
{
   CurrentSampleIndex = JobIdToIndex(signal);
   if(CurrentSampleIndex<0){
      printf("There is no signal corresponding to the JobId Given\n");
      return;
   }

   InputPath  = "Results/ANALYSE/" + pattern + "Histos.root";
   OutputPath = string("Results/EXCLUSION/") + pattern + signals[CurrentSampleIndex].Name;
   MakeDirectories(OutputPath);


   TFile* InputFile = new TFile(InputPath.c_str());
   MassMCTr = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");
   MassSign = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[CurrentSampleIndex].Name);
   MassPred = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");
   MassData = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");
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
   MassSignFit->SetParLimits(2, 10,500);

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
   MassSign->GetYaxis()->SetTitle("Entries in 10Pb^{-1}");
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
   MassData->GetYaxis()->SetTitle("Entries in 10Pb^{-1}");
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
   SaveCanvas(c1, Path, "SL_Distrib");
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

   string Input = "Results/ANALYSE/" + InputPattern + "Histos.root";
   TFile* InputFile = new TFile(Input.c_str());

   for(unsigned int s=0;s<signals.size();s++){
      TH1* Sign = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[s].Name);
      TF1* SignFit = new TF1("SignFit","gaus(0)", 0, 1500);
      SignFit->SetParameter(0, 0.5*Sign->Integral());
      SignFit->SetParLimits(0, 0,1*Sign->Integral());
      SignFit->SetParameter(1, 400);
      SignFit->SetParLimits(1, 50,1000);
      SignFit->SetParameter(2, 100);
      SignFit->SetParLimits(2, 10,500);
      Sign->SetStats(kFALSE);
      Sign->Fit("SignFit","LL R 0");

      SampleMean .push_back(SignFit->GetParameter(1));
      SampleSigma.push_back(SignFit->GetParameter(2));      
     
      TCanvas* c1 = new TCanvas("c1", "c1",600,600);
      Sign->Draw();
      SignFit->Draw("same");
      SaveCanvas(c1, "Results/EXCLUSION/MassFit", signals[s].Name,true);
      delete c1;

      delete SignFit;
      delete Sign;
   }
   delete InputFile;


   int    NSample = signals.size();
   int    NStau   = 0;   double StauMass  [NSample];   double StauMean  [NSample];   double StauSigma  [NSample];
   int    NStop   = 0;   double StopMass  [NSample];   double StopMean  [NSample];   double StopSigma  [NSample];
   int    NGluino = 0;   double GluinoMass[NSample];   double GluinoMean[NSample];   double GluinoSigma[NSample];
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Type=="Gluino"){GluinoMass[NGluino] = signals[s].Mass; GluinoMean[NGluino]= SampleMean[s]; GluinoSigma[NGluino] = SampleSigma[s]; NGluino++;}
      if(signals[s].Type=="Stop"  ){StopMass  [NStop]   = signals[s].Mass; StopMean  [NStop]  = SampleMean[s]; StopSigma  [NStop]   = SampleSigma[s]; NStop++;  }
      if(signals[s].Type=="Stau"  ){StauMass  [NStau]   = signals[s].Mass; StauMean  [NStau]  = SampleMean[s]; StauSigma  [NStau]   = SampleSigma[s]; NStau++;  }
   }

   TGraph* Stop_MMC = new TGraph(NStop,StopMass,StopMean);
   Stop_MMC_Fit    = new TF1("Stop_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Stop_MMC_Fit->SetParLimits(0, 0,10);
   Stop_MMC_Fit->SetParameter(0,10);
   Stop_MMC->Fit("Stop_MMC_Fit","NR");
   Stop_MMC_Fit->SetLineWidth(2);
   Stop_MMC_Fit->SetLineColor(4);
   Stop_MMC_Fit->SetLineStyle(2);
   Stop_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   Stop_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stop_MMC->SetMarkerColor(Stop_MMC_Fit->GetLineColor());
   Stop_MMC->SetLineColor  (Stop_MMC_Fit->GetLineColor());
   Stop_MMC->SetLineStyle  (Stop_MMC_Fit->GetLineStyle());

   TGraph* Gluino_MMC = new TGraph(NGluino,GluinoMass,GluinoMean);
   Gluino_MMC_Fit    = new TF1("Gluino_MMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Gluino_MMC_Fit->SetParLimits(0, 0,10);
   Gluino_MMC_Fit->SetParameter(0,10);
   Gluino_MMC->Fit("Gluino_MMC_Fit","NR");
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
   Stau_MMC->Fit("Stau_MMC_Fit","NR");
   Stau_MMC_Fit->SetLineWidth(2);
   Stau_MMC_Fit->SetLineColor(8);
   Stau_MMC_Fit->SetLineStyle(2);
   Stau_MMC_Fit->GetXaxis()->SetTitle("HSCP Simulated Mass [ GeV/c^{2} ]");
   Stau_MMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stau_MMC->SetMarkerColor(Stau_MMC_Fit->GetLineColor());
   Stau_MMC->SetLineColor  (Stau_MMC_Fit->GetLineColor());
   Stau_MMC->SetLineStyle  (Stau_MMC_Fit->GetLineStyle());

   TGraph* Stop_SMC = new TGraph(NStop,StopMean,StopSigma);
   Stop_SMC_Fit    = new TF1("Stop_SMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Stop_SMC_Fit->SetParLimits(0, 0,10);
   Stop_SMC_Fit->SetParameter(0,10);
   Stop_SMC->Fit("Stop_SMC_Fit","NR");
   Stop_SMC_Fit->SetLineWidth(Stop_MMC_Fit->GetLineWidth());
   Stop_SMC_Fit->SetLineColor(Stop_MMC_Fit->GetLineColor());
   Stop_SMC_Fit->SetLineStyle(Stop_MMC_Fit->GetLineStyle());
   Stop_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stop_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   Stop_SMC->SetMarkerColor(Stop_SMC_Fit->GetLineColor());
   Stop_SMC->SetLineColor  (Stop_SMC_Fit->GetLineColor());
   Stop_SMC->SetLineStyle  (Stop_SMC_Fit->GetLineStyle());

   TGraph* Gluino_SMC = new TGraph(NGluino,GluinoMean,GluinoSigma);
   Gluino_SMC_Fit    = new TF1("Gluino_SMC_Fit","[0]+[1]*x+[2]*x*x",0,1000);
   Gluino_SMC_Fit->SetParLimits(0, 0,10);
   Gluino_SMC_Fit->SetParameter(0,10);
   Gluino_SMC->Fit("Gluino_SMC_Fit","NR");
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
   Stau_SMC_Fit->SetParLimits(0, 0,10);
   Stau_SMC_Fit->SetParameter(0,10);
   Stau_SMC->Fit("Stau_SMC_Fit","NR");
   Stau_SMC_Fit->SetLineWidth(Stau_MMC_Fit->GetLineWidth());
   Stau_SMC_Fit->SetLineColor(Stau_MMC_Fit->GetLineColor());
   Stau_SMC_Fit->SetLineStyle(Stau_MMC_Fit->GetLineStyle());
   Stau_SMC_Fit->GetXaxis()->SetTitle("HSCP Reconstructed Mass [ GeV/c^{2} ]");
   Stau_SMC_Fit->GetYaxis()->SetTitle("HSCP Reconstructed Sigma [ GeV/c^{2} ]");
   Stau_SMC->SetMarkerColor(Stau_SMC_Fit->GetLineColor());
   Stau_SMC->SetLineColor  (Stau_SMC_Fit->GetLineColor());
   Stau_SMC->SetLineStyle  (Stau_SMC_Fit->GetLineStyle());


   TCanvas* c1;
   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",800,600);
   c1->SetGridx(true);
   c1->SetGridy(true);
   Stop_MMC_Fit->Draw("");
   Stop_MMC->Draw("* same");
   Gluino_MMC_Fit->Draw("same");
   Gluino_MMC->Draw("* same");
   Stau_MMC_Fit->Draw("same");
   Stau_MMC->Draw("* same");

   TLegend* leg = new TLegend(0.15,0.93,0.35,0.73);
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stau_MMC  , "Stau"  ,"PL");
   leg->AddEntry(Stop_MMC  , "Stop"  ,"PL");
   leg->AddEntry(Gluino_MMC, "Gluino","PL");
   leg->Draw();
   SaveCanvas(c1, "Results/EXCLUSION/Correlation", "MassMass",true);
   delete c1;

   c1 = new TCanvas("MassMassCorrelation", "MassMassCorrelation",800,600);
   c1->SetGridx(true);
   c1->SetGridy(true);
   Stop_SMC_Fit->Draw("");
   Stop_SMC->Draw("* same");
   Gluino_SMC_Fit->Draw("same");
   Gluino_SMC->Draw("* same");
   Stau_SMC_Fit->Draw("same");
   Stau_SMC->Draw("* same");

   leg = new TLegend(0.15,0.93,0.35,0.73);
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stau_SMC  , "Stau"  ,"PL");
   leg->AddEntry(Stop_SMC  , "Stop"  ,"PL");
   leg->AddEntry(Gluino_SMC, "Gluino","PL");
   leg->Draw();
   SaveCanvas(c1, "Results/EXCLUSION/Correlation", "SigmaMass",true);
   delete c1;
}

int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}



