
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
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"
#include "tdrstyle.C"


#include "Analysis_CommonFunction.h"
#include "Analysis_Global.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"


/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void WPMap(string InputPattern, double MinM, double MaxM,  bool Rescale=true);
//void WPMap1D(string InputPattern, double MinM, double MaxM, std::vector<double>& PtEff, std::vector<double>& IEff);
void WPMap1D(std::vector<string> InputPatterns, std::vector<string> Legends, double MinM, double MaxM, std::vector<double>& PtEff, std::vector<double>& IEff);
void CutEfficiency(std::vector<string> InputPatterns, std::vector<string> Legends, std::vector<double>& Cuts);
void MassPlot(string InputPattern);
void SelectionPlot(string InputPattern);
void PredictionAndControlPlot(string InputPattern);

void dEdxVsP_Plot_Core(string InputPattern);
void Make2DPlot_Core(string ResultPattern);
void MakeCompPlot(string DirName, string InputPattern1, string InputPattern2="");
void CheckPredictionRescale(string InputPattern, bool RecomputeRescale=false);
void MakeHitSplit_Plot(string InputPattern);
void CheckHitSplitSloap_Plot(string InputPattern);
double GetEventInRange(double min, double max, TH1D* hist);

int JobIdToIndex(string JobId);



std::vector<stSignal> signals;
std::vector<stMC>     MCsample;

string LegendTitle;

/////////////////////////// CODE PARAMETERS /////////////////////////////

void Analysis_Step5()
{
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.12);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   GetSignalDefinition(signals);
   GetMCDefinition(MCsample);

   string InputDir;
   dEdxSeleIndex = 11;
   std::vector<string> Legends;                 std::vector<string> Inputs;

//  WPMap("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/", 75,2000);
//  WPMap("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/", 75,2000);

//  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/", 75,2000);
//  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/", 75,2000);
//  return;


/*
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/",  0,2000);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/", 75,2000);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/", 0,2000, false);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/", 0,75, false);

  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/",  0,2000);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/", 75,2000);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/",  0,2000, false);
  WPMap("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/",  0,75, false);
*/

/*
   std::vector<double> CutEff;
   CutEff.push_back(pow(10,-0.25));
   CutEff.push_back(pow(10,-0.50));
   CutEff.push_back(pow(10,-0.75));
   CutEff.push_back(pow(10,-1.00));
   CutEff.push_back(pow(10,-1.25));
   CutEff.push_back(pow(10,-1.50));
   CutEff.push_back(pow(10,-1.75));
   CutEff.push_back(pow(10,-2.00));
   CutEff.push_back(pow(10,-2.25));
   CutEff.push_back(pow(10,-2.50));
   CutEff.push_back(pow(10,-2.75));
   CutEff.push_back(pow(10,-3.00));
   CutEff.push_back(pow(10,-3.25));
   CutEff.push_back(pow(10,-3.50));
   CutEff.push_back(pow(10,-3.75));
   CutEff.push_back(pow(10,-4.00));
   CutEff.push_back(pow(10,-4.25));
   CutEff.push_back(pow(10,-4.50));

   Legends.clear();                             Inputs.clear();
   Legends.push_back("Harmonic-2");             Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTCNPHarm2/Mass_dedxSTCNPHarm2/Type1/");
   Legends.push_back("Asymmetric Smirnov");     Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/");
   CutEfficiency(Inputs, Legends,  CutEff);

   Legends.clear();                             Inputs.clear();
   Legends.push_back("Harmonic-2");             Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTCNPHarm2/Mass_dedxSTCNPHarm2/Type0/");
   Legends.push_back("Asymmetric Smirnov");     Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/");
   CutEfficiency(Inputs, Legends,  CutEff);
*/

/*
   std::vector<double> PtEff; std::vector<double> IEff; 
   PtEff.push_back(-4.5);   IEff.push_back(-4.5);
   PtEff.push_back(-4.0);   IEff.push_back(-4.0);
   PtEff.push_back(-3.5);   IEff.push_back(-3.5); 
   PtEff.push_back(-3.0);   IEff.push_back(-3.0);
   PtEff.push_back(-2.5);   IEff.push_back(-2.5);
   PtEff.push_back(-2.0);   IEff.push_back(-2.0);
   PtEff.push_back(-1.5);   IEff.push_back(-1.5);
   PtEff.push_back(-1.0);   IEff.push_back(-1.0);
   PtEff.push_back(-0.5);   IEff.push_back(-0.5);

//   Legends.clear();                             Inputs.clear();
//   Legends.push_back("Harmonic-2");             Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTCNPHarm2/Mass_dedxSTCNPHarm2/Type1/");
//   Legends.push_back("Asymmetric Smirnov");     Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/");
//   WPMap1D(Inputs, Legends,  75,1000, PtEff, IEff);

//   Legends.clear();                             Inputs.clear();
//   Legends.push_back("Harmonic-2");             Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTCNPHarm2/Mass_dedxSTCNPHarm2/Type0/");
//   Legends.push_back("Asymmetric Smirnov");     Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/");
//   WPMap1D(Inputs, Legends,  75,1000, PtEff, IEff);

   Legends.clear();                             Inputs.clear();
//   Legends.push_back("#eta & hit splitting");   Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/");
   Legends.push_back("hit splitting");          Inputs.push_back("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/");
   Legends.push_back("no splitting");           Inputs.push_back("Results/Eta25/PtErr015/SplitMode0/MinHit09/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/");
   WPMap1D(Inputs, Legends,  75,2000, PtEff, IEff);

   Legends.clear();                             Inputs.clear();
//   Legends.push_back("#eta & hit splitting");   Inputs.push_back("Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/");
   Legends.push_back("hit splitting");          Inputs.push_back("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/");
   Legends.push_back("no splitting");           Inputs.push_back("Results/Eta25/PtErr015/SplitMode0/MinHit09/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/");
   WPMap1D(Inputs, Legends,  75,2000, PtEff, IEff);

//   CheckHitSplitSloap_Plot("SAVE_BUG/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt+00/WPI+00/");

   MakeHitSplit_Plot("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt+00/WPI+00/");
   MakeHitSplit_Plot("Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt+00/WPI+00/");
return;

   MakeCompPlot("TkOnlyClusterCleaning", "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt+00/WPI+00/", "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/Type0/WPPt+00/WPI+00/");
   MakeCompPlot("TkMuonClusterCleaning", "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt+00/WPI+00/", "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxASmi/Mass_dedxCNPHarm2/Type1/WPPt+00/WPI+00/");

   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt+00/WPI+00/";
   Make2DPlot_Core(InputDir);
   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt+00/WPI+00/";
*/
/*
   dEdxSeleIndex = 11; 
   bool RecomputeRecale = true;
   for(double Pt=0;Pt>-5.0;Pt-=0.5){
   for(double  I=0; I>-5.0; I-=0.5){      
      char tmp[2048];
      sprintf(tmp,"Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt%+03i/WPI%+03i/",(int)(10*Pt),(int)(10*I));
      CheckPredictionRescale(tmp, RecomputeRecale);
      sprintf(tmp,"Results/Eta25/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt%+03i/WPI%+03i/",(int)(10*Pt),(int)(10*I));
      CheckPredictionRescale(tmp, RecomputeRecale);
      RecomputeRecale = false;
   }}
*/




/*

//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-15/WPI-40/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-15/WPI-40/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-25/WPI-35/";
   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-30/WPI-40/"; //X
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-30/WPI-35/"; //X
   CheckPredictionRescale(InputDir, true); 
//   Make2DPlot_Core(InputDir);
//   SelectionPlot(InputDir);
//   PredictionAndControlPlot(InputDir);
//   dEdxVsP_Plot_Core(InputDir);

//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-25/WPI-45/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-15/WPI-50/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-25/WPI-45/";
   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-35/WPI-45/"; //X
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-35/WPI-40/"; //X
   CheckPredictionRescale(InputDir, true);
//   Make2DPlot_Core(InputDir);
//   SelectionPlot(InputDir);
//   PredictionAndControlPlot(InputDir);
//   dEdxVsP_Plot_Core(InputDir);
*/
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-10/WPI-15/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-10/WPI-20/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-10/WPI-15/";
   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-10/WPI-15/"; //X
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-05/WPI-10/"; //X
//   CheckPredictionRescale(InputDir, true);
//   Make2DPlot_Core(InputDir);
   SelectionPlot(InputDir);
//   PredictionAndControlPlot(InputDir);
//   dEdxVsP_Plot_Core(InputDir);

//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-20/WPI-20/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-20/WPI-25/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-20/WPI-25/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-15/WPI-20/";
//   InputDir = "Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-15/WPI-20/";
   InputDir = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-15/WPI-20/"; //X
//   InputDir = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-10/WPI-15/"; //X
//   CheckPredictionRescale(InputDir, true);
//   Make2DPlot_Core(InputDir);
   SelectionPlot(InputDir);
//   PredictionAndControlPlot(InputDir);
//   dEdxVsP_Plot_Core(InputDir);




/*
   string Input    = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-30/WPI-35/DumpHistos.root";
   TFile* InputFile = new TFile(Input.c_str());
   TH2D* Hist1    = (TH2D*)GetObjectFromPath(InputFile, "Gluino400_AS_EtaP");
   TH1D* Hist1Eta = (TH1D*)Hist1->ProjectionX();
   printf("Eta<10 --> %f Entries\n",Hist1Eta->Integral());
   printf("Eta<10 --> %f Entries\n",Hist1Eta->Integral(Hist1Eta->GetXaxis()->FindBin(-2.6), Hist1Eta->GetXaxis()->FindBin(2.6)));

   Input          = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-30/WPI-40/DumpHistos.root";
   InputFile      = new TFile(Input.c_str());
   TH2D* Hist2    = (TH2D*)GetObjectFromPath(InputFile, "Gluino400_AS_EtaP");
   TH1D* Hist2Eta = (TH1D*)Hist2->ProjectionX();
   printf("Eta<25 --> %f Entries\n",Hist2Eta->Integral());
   printf("Eta<25 --> %f Entries\n",Hist2Eta->Integral(Hist2Eta->GetXaxis()->FindBin(-2.6), Hist2Eta->GetXaxis()->FindBin(2.6)));

   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = Hist1Eta;                          legend.push_back("|#eta| < 1.0");
   Histos[1] = Hist2Eta;                          legend.push_back("|#eta| < 2.5");
   DrawSuperposedHistos((TH1**)Histos, legend, "PH",  "#eta", "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,"","PL");
//   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"./","EtaAS_TkMuon");
   delete c1;




   Input    = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-35/WPI-40/DumpHistos.root";
   InputFile = new TFile(Input.c_str());
   Hist1    = (TH2D*)GetObjectFromPath(InputFile, "Gluino400_AS_EtaP");
   Hist1Eta = (TH1D*)Hist1->ProjectionX();
   printf("Eta<10 --> %f Entries\n",Hist1Eta->Integral());
   printf("Eta<10 --> %f Entries\n",Hist1Eta->Integral(Hist1Eta->GetXaxis()->FindBin(-2.6), Hist1Eta->GetXaxis()->FindBin(2.6)));

   Input          = "Results/Eta25/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type0/WPPt-35/WPI-45/DumpHistos.root";
   InputFile      = new TFile(Input.c_str());
   Hist2    = (TH2D*)GetObjectFromPath(InputFile, "Gluino400_AS_EtaP");
   Hist2Eta = (TH1D*)Hist2->ProjectionX();
   printf("Eta<25 --> %f Entries\n",Hist2Eta->Integral());
   printf("Eta<25 --> %f Entries\n",Hist2Eta->Integral(Hist2Eta->GetXaxis()->FindBin(-2.6), Hist2Eta->GetXaxis()->FindBin(2.6)));

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = Hist1Eta;                          legend.push_back("|#eta| < 1.0");
   Histos[1] = Hist2Eta;                          legend.push_back("|#eta| < 2.5");
   DrawSuperposedHistos((TH1**)Histos, legend, "PH",  "#eta", "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,"","PL");
//   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"./","EtaAS_TkOnly");
   delete c1;
*/


/*
   string Input     = "Results/Eta10/PtErr015/SplitMode1/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt-30/WPI-35/DumpHistos.root";
   TFile* InputFile = new TFile(Input.c_str());
   TH2D* Hist1      = (TH2D*)GetObjectFromPath(InputFile, "Gluino400_AS_EtaIs");
   TH1D* Hist1Eta09 = (TH1D*)Hist1->ProjectionY("H1Eta09" ,Hist1->GetXaxis()->FindBin(-0.9), Hist1->GetXaxis()->FindBin( 0.0));
   TH1D* Hist1Eta14 = (TH1D*)Hist1->ProjectionY("H1Eta14" ,Hist1->GetXaxis()->FindBin(-1.4), Hist1->GetXaxis()->FindBin(-0.9));
 //     Hist1Eta14->Add((TH1D*)Hist1->ProjectionY("H1Eta14b",Hist1->GetXaxis()->FindBin( 0.9), Hist1->GetXaxis()->FindBin( 1.4)),1.0);
   TH1D* Hist1Eta25 = (TH1D*)Hist1->ProjectionY("H1Eta25" ,Hist1->GetXaxis()->FindBin(-2.5), Hist1->GetXaxis()->FindBin(-1.4));
 //     Hist1Eta25->Add((TH1D*)Hist1->ProjectionY("H1Eta25b",Hist1->GetXaxis()->FindBin( 1.4), Hist1->GetXaxis()->FindBin( 2.5)),1.0);

   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = Hist1Eta09;                        legend.push_back("0.0 < |#eta| < 0.9");
   Histos[1] = Hist1Eta14;                        legend.push_back("0.9 < |#eta| < 1.4");
   Histos[2] = Hist1Eta25;                        legend.push_back("1.4 < |#eta| < 2.5");
   DrawSuperposedHistos((TH1**)Histos, legend, "PH",  "I_{as}", "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,"","PL");
//   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,"./","EtaAS_TkMuon");
   delete c1;
*/




/*
   TFile* InputFile = new TFile("Results/Eta10/PtErr015/SplitMode2/MinHit01/Sele_dedxSTASmi/Mass_dedxSTCNPHarm2/Type1/WPPt+00/WPI+00/DumpHistos.root");

   TH1D*   hist  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data"   );
  int binMin = hist->GetXaxis()->FindBin(10);
  int binMax = hist->GetXaxis()->FindBin(1000);
  printf("PtCut 10 --> %6.2EEntries\n",hist->Integral(binMin,binMax));

  binMin = hist->GetXaxis()->FindBin(12.5);
  printf("PtCut 12.5 --> %6.2EEntries\n",hist->Integral(binMin,binMax));

  binMin = hist->GetXaxis()->FindBin(15);
  printf("PtCut 15 --> %6.2EEntries\n",hist->Integral(binMin,binMax));
*/

   return;
}


//////////////////////////////////////////////////     CREATE WORKING POINT MAPS


void WPMap(string InputPattern, double MinM, double MaxM, bool Rescale)
{
   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }

   double RescaleFactor, RescaleError;
   if(Rescale){
      GetPredictionRescale(InputPattern,RescaleFactor, RescaleError, true);
   }else{
      RescaleFactor=1.0;
      RescaleError =0.0;
   }
   RescaleError*=2.0;

   char Buffer[2048]; sprintf(Buffer,"Range_%04i_%04i_Rescale%i/",(int)MinM,(int)MaxM, (int) Rescale );
   string SavePath  = InputPattern + "MAP/";			   system((string("mkdir ") + SavePath).c_str());
          SavePath += Buffer;                                      system((string("mkdir ") + SavePath).c_str());
   MakeDirectories(SavePath);

   TH2D*  WP_D     = new TH2D("WP D" , "WP D"    , 11,-5.25,0.25,11,-5.25,0.25);
   TH2D*  WP_P     = new TH2D("WP P" , "WP P"    , 11,-5.25,0.25,11,-5.25,0.25);
   TH2D*  WP_M     = new TH2D("WP M" , "WP M"    , 11,-5.25,0.25,11,-5.25,0.25);
   TH2D*  WP_DP    = new TH2D("WP DP", "WP DP"   , 11,-5.25,0.25,11,-5.25,0.25);
   TH2D** WP_S     = new TH2D*[signals.size()];
   TH2D** WP_SD    = new TH2D*[signals.size()];
   TH2D** WP_SP    = new TH2D*[signals.size()];
   TH2D** WP_SM    = new TH2D*[signals.size()];
   for(unsigned int s=0;s<signals.size();s++){
      string Name;
      WP_S    [s] = new TH2D(Name.c_str(), Name.c_str()    , 11,-5.25,0.25,11,-5.25,0.25);
      Name = string("WP S ") + signals[s].Name + "Eff";
      Name = string("WP S/D ") + signals[s].Name ;
      WP_SD   [s] = new TH2D(Name.c_str(), Name.c_str()    , 11,-5.25,0.25,11,-5.25,0.25);
      Name = string("WP S/P ") + signals[s].Name ;
      WP_SP   [s] = new TH2D(Name.c_str(), Name.c_str()    , 11,-5.25,0.25,11,-5.25,0.25);
      Name = string("WP S/M ") + signals[s].Name ;
      WP_SM   [s] = new TH2D(Name.c_str(), Name.c_str()    , 11,-5.25,0.25,11,-5.25,0.25);
   }

   FILE* pDump = fopen( (SavePath + "Map.txt").c_str(),"w");

   for(float WP_Pt=0;WP_Pt>=-5;WP_Pt-=0.5f){
   for(float WP_I =0;WP_I >=-5;WP_I -=0.5f){
      int Bin_Pt = WP_D->GetXaxis()->FindBin(WP_Pt);
      int Bin_I  = WP_D->GetYaxis()->FindBin(WP_I );
      float d=0,p=0,m=0,s=0;
      
      sprintf(Buffer,"%sWPPt%+03i/WPI%+03i/DumpHistos.root",InputPattern.c_str(),(int)(10*WP_Pt),(int)(10*WP_I));
      TFile* InputFile = new TFile(Buffer); 
      if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )continue;

      TH1D* Hd = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");if(Hd){d=GetEventInRange(MinM,MaxM,Hd);delete Hd;}
      TH1D* Hp = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");if(Hp){p=GetEventInRange(MinM,MaxM,Hp);delete Hp;}
      TH1D* Hm = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");if(Hm){m=GetEventInRange(MinM,MaxM,Hm);delete Hm;}
      p*=RescaleFactor;


      WP_D->SetBinContent(Bin_Pt,Bin_I,d);
      WP_P->SetBinContent(Bin_Pt,Bin_I,p);
      WP_M->SetBinContent(Bin_Pt,Bin_I,m);
//      if(!(d!=d) && p>0)WP_DP->SetBinContent(Bin_Pt,Bin_I,d/p);
      if(!(d!=d) && p>0 && d>20 && (WP_Pt+WP_I)<=-2)WP_DP->SetBinContent(Bin_Pt,Bin_I,d/p);


      for(unsigned int S=0;S<signals.size();S++){
         fprintf(pDump ,"Signal=%10s WP=(%+6.2f,%+6.2f) --> Numbers: D=%3.2E P=%3.2E M=%3.2E S=%3.2E S/D=%4.3E S/P=%4.3E S/M=%4.3E\n",signals[S].Name.c_str(),WP_Pt,WP_I,d,p,m,s,s/d,s/p,s/m);
         fprintf(stdout,"Signal=%10s WP=(%+6.2f,%+6.2f) --> Numbers: D=%3.2E P=%3.2E M=%3.2E S=%3.2E S/D=%4.3E S/P=%4.3E S/M=%4.3E\n",signals[S].Name.c_str(),WP_Pt,WP_I,d,p,m,s,s/d,s/p,s/m);

         TH1D* Hs = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[S].Name);if(Hs){s=GetEventInRange(MinM,MaxM,Hs);delete Hs;}
         if(!(s!=s))WP_S[S]->SetBinContent(Bin_Pt,Bin_I,s);
         if(!(s!=s) && d>0)WP_SD[S]->SetBinContent(Bin_Pt,Bin_I,s/d);
         if(!(s!=s) && p>0)WP_SP[S]->SetBinContent(Bin_Pt,Bin_I,s/p);
         if(!(s!=s) && m>0)WP_SM[S]->SetBinContent(Bin_Pt,Bin_I,s/m);
      }
      InputFile->Close();
   }}
   fclose(pDump);

   printf("RESCALE FACTOR = %f\n",RescaleFactor);  
 
   TCanvas* c1;
   c1  = new TCanvas("D", "D", 600,600);
   c1->SetLogx(false);
   c1->SetLogy(false);
   c1->SetLogz(true);
   c1->SetGridx(true);
   c1->SetGridy(true);
   gStyle->SetPaintTextFormat("1.1E");
   WP_D->SetMarkerSize(1.0);
   WP_D->SetTitle("");
   WP_D->SetStats(kFALSE);
   WP_D->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
   WP_D->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
   WP_D->GetYaxis()->SetTitleOffset(1.60);
   WP_D->Draw("COLZ TEXT45");
   Smart_SetAxisRange(WP_D);
   WP_D->GetXaxis()->SetNdivisions(520);
   WP_D->GetYaxis()->SetNdivisions(520);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, SavePath, string("Map_D") );
   WP_D->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_D_NoText"));
   WP_D->SetAxisRange(1,1E6,"Z");
   WP_D->Draw("COLZ TEXT45");
   SaveCanvas(c1, SavePath, string("Map_D_Ranged") );
   WP_D->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_D_RangedNoText") );
   delete c1;

   c1  = new TCanvas("P", "P", 600,600);
   c1->SetLogx(false);
   c1->SetLogy(false);
   c1->SetLogz(true);
   c1->SetGridx(true);
   c1->SetGridy(true);
   gStyle->SetPaintTextFormat("1.1E");
   WP_P->SetMarkerSize(1.0);
   WP_P->SetTitle("");
   WP_P->SetStats(kFALSE);
   WP_P->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
   WP_P->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
   WP_P->GetYaxis()->SetTitleOffset(1.60);
   Smart_SetAxisRange(WP_P);
   WP_P->GetXaxis()->SetNdivisions(520);
   WP_P->GetYaxis()->SetNdivisions(520);
   WP_P->Draw("COLZ TEXT45");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, SavePath, string("Map_P") );
   WP_P->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_P_NoText"));
   WP_P->SetAxisRange(1E-4,1E3,"Z");
   WP_P->Draw("COLZ TEXT45");
   SaveCanvas(c1, SavePath, string("Map_P_Ranged") );
   WP_P->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_P_RangedNoText") );
   delete c1;

   c1  = new TCanvas("M", "M", 600,600);
   c1->SetLogx(false);
   c1->SetLogy(false);
   c1->SetLogz(true);
   c1->SetGridx(true);
   c1->SetGridy(true);
   gStyle->SetPaintTextFormat("1.1E");
   WP_M->SetMarkerSize(1.0);
   WP_M->SetTitle("");
   WP_M->SetStats(kFALSE);
   WP_M->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
   WP_M->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
   WP_M->GetYaxis()->SetTitleOffset(1.60);
   WP_M->GetXaxis()->SetNdivisions(520);
   WP_M->GetYaxis()->SetNdivisions(520);
   WP_M->Draw("COLZ TEXT45");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, SavePath, string("Map_M") );
   WP_M->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_M_NoText"));
   WP_M->SetAxisRange(1,1E6,"Z");
   WP_M->Draw("COLZ TEXT45");
   SaveCanvas(c1, SavePath, string("Map_M_Ranged") );
   WP_M->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_M_RangedNoText") );
   delete c1;

   c1  = new TCanvas("DP", "DP", 600,600);
   c1->SetLogx(false);
   c1->SetLogy(false);
   c1->SetLogz(true);
   c1->SetGridx(true);
   c1->SetGridy(true);
   WP_DP->SetMarkerSize(1.0);
   WP_DP->SetTitle("");
   WP_DP->SetStats(kFALSE);
   WP_DP->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
   WP_DP->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
   WP_DP->GetYaxis()->SetTitleOffset(1.60);
   WP_DP->GetXaxis()->SetNdivisions(520);
   WP_DP->GetYaxis()->SetNdivisions(520);
   WP_DP->Draw("COLZ TEXT45");
   Smart_SetAxisRange(WP_DP);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, SavePath, string("Map_DP"));
   WP_DP->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_DP_NoText"));
   WP_DP->SetAxisRange(1E-2,1E2,"Z");
   WP_DP->Draw("COLZ TEXT45");
   SaveCanvas(c1, SavePath, string("Map_DP_Ranged"));
   WP_DP->Draw("COLZ");
   SaveCanvas(c1, SavePath, string("Map_DP_RangedNoText"));
   delete c1;

   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      c1  = new TCanvas("S", "S", 600,600);
      c1->SetLogx(false);
      c1->SetLogy(false);
      c1->SetLogz(true);
      c1->SetGridx(true);
      c1->SetGridy(true);
      WP_S[s]->SetMarkerSize(1.0);
      WP_S[s]->SetTitle("");
      WP_S[s]->SetStats(kFALSE);
      WP_S[s]->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
      WP_S[s]->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
      WP_S[s]->GetYaxis()->SetTitleOffset(1.60);
      Smart_SetAxisRange(WP_S[s]);
      WP_S[s]->GetXaxis()->SetNdivisions(520);
      WP_S[s]->GetYaxis()->SetNdivisions(520);
      WP_S[s]->Draw("COLZ TEXT45");
      DrawPreliminary(IntegratedLuminosity);
      SaveCanvas(c1, SavePath, string("Map_S_") + signals[s].Name);
      WP_S[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_S_") + signals[s].Name + "_NoText");
      WP_S[s]->SetAxisRange(1E-3,1E2,"Z");
      WP_S[s]->Draw("COLZ TEXT45");
      SaveCanvas(c1, SavePath, string("Map_S_") + signals[s].Name + "_Ranged");
      WP_S[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_S_") + signals[s].Name + "_RangedNoText");
      delete c1;

      c1  = new TCanvas("SD", "SD", 600,600);
      c1->SetLogx(false);
      c1->SetLogy(false);
      c1->SetLogz(true);
      c1->SetGridx(true);
      c1->SetGridy(true);
      WP_SD[s]->SetMarkerSize(1.0);
      WP_SD[s]->SetTitle("");
      WP_SD[s]->SetStats(kFALSE);
      WP_SD[s]->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
      WP_SD[s]->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
      WP_SD[s]->GetYaxis()->SetTitleOffset(1.60);
      Smart_SetAxisRange(WP_SD[s]);
      WP_SD[s]->GetXaxis()->SetNdivisions(520);
      WP_SD[s]->GetYaxis()->SetNdivisions(520);
      WP_SD[s]->Draw("COLZ TEXT45");
      DrawPreliminary(IntegratedLuminosity);
      SaveCanvas(c1, SavePath, string("Map_SD_") + signals[s].Name);
      WP_SD[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SD_") + signals[s].Name + "_NoText");
      WP_SD[s]->SetAxisRange(1E-6,1E2,"Z");
      WP_SD[s]->Draw("COLZ TEXT45");
      SaveCanvas(c1, SavePath, string("Map_SD_") + signals[s].Name + "_Ranged" );
      WP_SD[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SD_") + signals[s].Name + "_RangedNoText" );
      delete c1;

      c1  = new TCanvas("SP", "SP", 600,600);
      c1->SetLogx(false);
      c1->SetLogy(false);
      c1->SetLogz(true);
      c1->SetGridx(true);
      c1->SetGridy(true);
      WP_SP[s]->SetMarkerSize(1.0);
      WP_SP[s]->SetTitle("");
      WP_SP[s]->SetStats(kFALSE);
      WP_SP[s]->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
      WP_SP[s]->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
      WP_SP[s]->GetYaxis()->SetTitleOffset(1.60);
      Smart_SetAxisRange(WP_SP[s]);
      WP_SP[s]->GetXaxis()->SetNdivisions(520);
      WP_SP[s]->GetYaxis()->SetNdivisions(520);
      WP_SP[s]->Draw("COLZ TEXT45");
      DrawPreliminary(IntegratedLuminosity);
      SaveCanvas(c1, SavePath, string("Map_SP_") + signals[s].Name);
      WP_SP[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SP_") + signals[s].Name + "_NoText");
      WP_SP[s]->SetAxisRange(1E-6,1E2,"Z");
      WP_SP[s]->Draw("COLZ TEXT45");
      SaveCanvas(c1, SavePath, string("Map_SP_") + signals[s].Name + "_Ranged" );
      WP_SP[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SP_") + signals[s].Name + "_RangedNoText" );
      delete c1;

      c1  = new TCanvas("SM", "SM", 600,600);
      c1->SetLogx(false);
      c1->SetLogy(false);
      c1->SetLogz(true);
      c1->SetGridx(true);
      c1->SetGridy(true);
      WP_SM[s]->SetMarkerSize(1.0);
      WP_SM[s]->SetTitle("");
      WP_SM[s]->SetStats(kFALSE);
      WP_SM[s]->GetXaxis()->SetTitle("Selection Efficiency on PT (log10)");
      WP_SM[s]->GetYaxis()->SetTitle("Selection Efficiency on I  (log10)");
      WP_SM[s]->GetYaxis()->SetTitleOffset(1.60);
      Smart_SetAxisRange(WP_SM[s]);
      WP_SM[s]->GetXaxis()->SetNdivisions(520);
      WP_SM[s]->GetYaxis()->SetNdivisions(520);
      WP_SM[s]->Draw("COLZ TEXT45");
      DrawPreliminary(IntegratedLuminosity);
      SaveCanvas(c1, SavePath, string("Map_SM_") + signals[s].Name);
      WP_SM[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SM_") + signals[s].Name + "_NoText");
      WP_SM[s]->SetAxisRange(1E-6,1E2,"Z");
      WP_SM[s]->Draw("COLZ TEXT45");
      SaveCanvas(c1, SavePath, string("Map_SM_") + signals[s].Name + "_Ranged" );
      WP_SM[s]->Draw("COLZ");
      SaveCanvas(c1, SavePath, string("Map_SM_") + signals[s].Name + "_RangedNoText" );
      delete c1;
   }
}

void WPMap1D(std::vector<string> InputPatterns, std::vector<string> Legends, double MinM, double MaxM, std::vector<double>& PtEff, std::vector<double>& IEff)
{
   TCanvas* c1;
   std::vector<string> legend;

   bool IsTrackerOnly = (InputPatterns[0].find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }


   char Buffer[2048]; sprintf(Buffer,"Range_%04i_%04i/",(int)MinM,(int)MaxM );
   string SavePath  = InputPatterns[0] + "MAP/";	   system((string("mkdir ") + SavePath).c_str());
          SavePath += Buffer;                              system((string("mkdir ") + SavePath).c_str());
   MakeDirectories(SavePath);


   TGraphErrors*** Graphs = new TGraphErrors**[InputPatterns.size()];
   for(unsigned int I=0;I<InputPatterns.size();I++){
      Graphs[I] = new TGraphErrors*[signals.size()];
   }


   for(unsigned int I=0;I<InputPatterns.size();I++){
      double RescaleFactor, RescaleError;
      GetPredictionRescale(InputPatterns[I],RescaleFactor, RescaleError, true);
      RescaleError*=2.0;

      unsigned int N  = 0;
      double*  P      = new double [PtEff.size()];
      double*  PErr   = new double [PtEff.size()];
      double** S      = new double*[signals.size()];
      double** SP     = new double*[signals.size()];
      for(unsigned int s=0;s<signals.size();s++){
          S[s] = new double [PtEff.size()];
         SP[s] = new double [PtEff.size()];
      }

      for(unsigned int i=0;i<PtEff.size();i++){
         float d=0,p=0,m=0,s=0;

         float WP_Pt = PtEff[i];
         float WP_I  = IEff[i];      

         sprintf(Buffer,"%sWPPt%+03i/WPI%+03i/DumpHistos.root",InputPatterns[I].c_str(),(int)(10*WP_Pt),(int)(10*WP_I));
         TFile* InputFile = new TFile(Buffer); 
         if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )continue;

         TH1D* Hd = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");if(Hd){d=GetEventInRange(MinM,MaxM,Hd);delete Hd;}
         TH1D* Hp = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");if(Hp){p=GetEventInRange(MinM,MaxM,Hp);delete Hp;}
         TH1D* Hm = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");if(Hm){m=GetEventInRange(MinM,MaxM,Hm);delete Hm;}

         P[N] = p * RescaleFactor; 
         PErr[N] = P[N] * RescaleError;
      
         printf("%f + %f --> %6.2E\n",WP_Pt,WP_I,P[N]);

         for(unsigned int j=0;j<signals.size();j++){
            TH1D* Hs = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[j].Name);if(Hs){s=GetEventInRange(MinM,MaxM,Hs);delete Hs;}
         
            (S[j]) [N] = s;
            (SP[j])[N] = s/p;
         }
         InputFile->Close();

         if(P[N]<1e-5 || P[N]>1e3)continue;
         N++; 
      }

      for(unsigned int j=0;j<signals.size();j++){
         Graphs[I][j] = new TGraphErrors(N, P, S[j], PErr, NULL);
         Graphs[I][j]->SetLineColor(Color[I]);
         Graphs[I][j]->SetLineStyle(1);
         Graphs[I][j]->SetLineWidth(1);
         Graphs[I][j]->SetMarkerColor(Color[I]);
         Graphs[I][j]->SetMarkerStyle(Marker[I]);
         Graphs[I][j]->SetTitle("");
         Graphs[I][j]->GetXaxis()->SetTitle("number of background tracks (predicted)");
         Graphs[I][j]->GetYaxis()->SetTitle("number of signal tracks (expected from MC)");
         Graphs[I][j]->GetYaxis()->SetTitleOffset(2.10);
         Graphs[I][j]->GetXaxis()->SetTitleOffset(1.25);
      }
   }

         c1 = new TCanvas("c1","c1,",600,600);          
         c1->SetRightMargin (0.12);
         c1->SetLeftMargin  (0.16);
         c1->SetLogx(true);
//         test->Draw("ALP E");
//         test->GetXaxis()->SetLimits(5e-5,1e3);

   for(unsigned int j=0;j<signals.size();j++){
      if(!signals[j].MakePlot)continue;
      c1 = new TCanvas("c1","c1,",600,600);          
      c1->SetRightMargin (0.12);
      c1->SetLeftMargin  (0.16);
      c1->SetLogx(true);

      int N = InputPatterns.size() -1; 
      for(unsigned int I=0;I<InputPatterns.size();I++){
         if(I==0){  (Graphs[N-I])[j]->Draw("ALP E");
         }else   {  (Graphs[N-I])[j]->Draw(" LP E"); }
         (Graphs[N-I])[j]->GetXaxis()->SetLimits(5e-5,1e3);

         Graphs[N-I][j]->GetXaxis()->SetTitle("number of background tracks (predicted)");
         Graphs[N-I][j]->GetYaxis()->SetTitle("number of signal tracks (expected from MC)");
         Graphs[N-I][j]->GetYaxis()->SetTitleOffset(2.10);
         Graphs[N-I][j]->GetXaxis()->SetTitleOffset(1.25);
      }


      TLegend* leg = new TLegend(0.18,0.90-0.05-InputPatterns.size()*0.05,0.60,0.90);
      leg->SetHeader((LegendTitle + "  (  " + signals[j].Legend + " )").c_str());
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      if(InputPatterns.size()>1){
      for(unsigned int I=0;I<InputPatterns.size();I++){
         leg->AddEntry((TObject*)(Graphs[I])[j], Legends[I].c_str(),"PE"); 
      }}
      leg->Draw();
      DrawPreliminary(IntegratedLuminosity);

      SaveCanvas(c1, SavePath, string("Map1D_SvsP_") + signals[j].Name );
      c1->SetLogy(true);
      SaveCanvas(c1, SavePath, string("Map1D_SvsP_") + signals[j].Name + "_Log");
      delete c1;
   }
}


void CutEfficiency(std::vector<string> InputPatterns, std::vector<string> Legends, std::vector<double>& Cuts)
{
   TCanvas* c1;
   std::vector<string> legend;

   bool IsTrackerOnly = (InputPatterns[0].find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }

   string SavePath  = InputPatterns[0] + "MAP/";	   system((string("mkdir ") + SavePath).c_str());
   MakeDirectories(SavePath);

   TGraphErrors*** Graphs = new TGraphErrors**[InputPatterns.size()];
   for(unsigned int I=0;I<InputPatterns.size();I++){
      Graphs[I] = new TGraphErrors*[signals.size()];
   }

   for(unsigned int I=0;I<InputPatterns.size();I++){

      unsigned int N  = 0;
      double*  B      = new double [Cuts.size()];
      double*  BErr   = new double [Cuts.size()];
      double** S      = new double*[signals.size()];
      double** SErr   = new double*[signals.size()];
      for(unsigned int s=0;s<signals.size();s++){
          S[s]   = new double [Cuts.size()];
         SErr[s] = new double [Cuts.size()];
      }

      for(unsigned int i=0;i<Cuts.size();i++){

         char Buffer[2048];
         sprintf(Buffer,"%sWPPt%+03i/WPI%+03i/DumpHistos.root",InputPatterns[I].c_str(),0,0);
         TFile* InputFile = new TFile(Buffer); 
         if(!InputFile || InputFile->IsZombie() || !InputFile->IsOpen() || InputFile->TestBit(TFile::kRecovered) )continue;

         TH1D* HistoData = (TH1D*)GetObjectFromPath(InputFile, "Data_BS_Is");

         double DataCutPt = CutFromEfficiency(HistoData,Cuts[i]);
         double DataEff   = EfficiencyAndError(HistoData,DataCutPt, BErr[N]); 

         B[N]    = DataEff; 
//         BErr[N] = 0;//sqrt(DataEff);
      
         printf("%6.2E --> %f --> %6.2E\n",Cuts[i],DataCutPt,DataEff);

         for(unsigned int j=0;j<signals.size();j++){
            TH1D* HistoSign = (TH1D*)GetObjectFromPath(InputFile, signals[j].Name + "_BS_Is" );
            double SignEff  = EfficiencyAndError(HistoSign,DataCutPt, (SErr[j])[N]);
         
            (S[j])   [N] = SignEff;
//            (SErr[j])[N] = 0;//sqrt(SignEff);
         }
         InputFile->Close();
         N++; 
      }

      for(unsigned int j=0;j<signals.size();j++){
         Graphs[I][j] = new TGraphErrors(N, B, S[j], BErr, SErr[j]);
         Graphs[I][j]->SetLineColor(Color[I]);
         Graphs[I][j]->SetLineStyle(1);
         Graphs[I][j]->SetLineWidth(1);
         Graphs[I][j]->SetMarkerColor(Color[I]);
         Graphs[I][j]->SetMarkerStyle(Marker[I]);
         Graphs[I][j]->SetTitle("");
         Graphs[I][j]->GetYaxis()->SetTitleOffset(2.10);
         Graphs[I][j]->GetXaxis()->SetTitleOffset(1.25);
      }
   }

   for(unsigned int j=0;j<signals.size();j++){
      if(!signals[j].MakePlot)continue;
      c1 = new TCanvas("c1","c1,",600,600);          
      c1->SetRightMargin (0.12);
      c1->SetLeftMargin  (0.16);
      c1->SetLogx(true);

      int N = InputPatterns.size() -1; 
      for(unsigned int I=0;I<InputPatterns.size();I++){
         if(I==0){  (Graphs[N-I])[j]->Draw("ALP E");
         }else   {  (Graphs[N-I])[j]->Draw(" LP E"); }

         Graphs[N-I][j]->GetXaxis()->SetTitle("data efficiency");
         Graphs[N-I][j]->GetYaxis()->SetTitle("signal efficiency");
         Graphs[N-I][j]->GetYaxis()->SetTitleOffset(2.10);
         Graphs[N-I][j]->GetXaxis()->SetTitleOffset(1.25);

         (Graphs[N-I])[j]->GetYaxis()->SetLimits(1E-5,1);
         (Graphs[N-I])[j]->GetXaxis()->SetLimits(1e-5,1);
      }


      TLegend* leg = new TLegend(0.18,0.90-0.05-InputPatterns.size()*0.05,0.60,0.90);
      leg->SetHeader((LegendTitle + "  (  " + signals[j].Legend + " )").c_str());
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      if(InputPatterns.size()>1){
      for(unsigned int I=0;I<InputPatterns.size();I++){
         leg->AddEntry((TObject*)(Graphs[I])[j], Legends[I].c_str(),"PE"); 
      }}
      leg->Draw();
      DrawPreliminary(IntegratedLuminosity);

      SaveCanvas(c1, SavePath, string("CutEff_dEdx_") + signals[j].Name );
      delete c1;
   }
}




//////////////////////////////////////////////////     CREATE PLOTS OF MASS DISTRIBUTION


void MassPlot(string InputPattern){
   TCanvas* c1;

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }


   string Input     = InputPattern + "DumpHistos.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
   TH1D* DATA = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");	DATA->Rebin(2);
   TH1D* PRED = (TH1D*)GetObjectFromPath(InputFile, "Mass_Pred");       PRED->Rebin(2);
   TH1D* MCTR = (TH1D*)GetObjectFromPath(InputFile, "Mass_MCTr");       MCTR->Rebin(2);

   if(DATA->Integral()>0){
      if(MCTR->Integral()>0)MCTR->Scale(DATA->Integral()/MCTR->Integral());
   }

   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      TH1D* SIGN = (TH1D*)GetObjectFromPath(InputFile, string("Mass_") + signals[s].Name);   SIGN->Rebin(2);

      double Max = std::max(DATA->GetMaximum(), PRED->GetMaximum());
      Max        = std::max(MCTR->GetMaximum(), Max);
      Max        = std::max(SIGN->GetMaximum(), Max);
      Max       *= 1.5;
      double Min = std::min(0.01,PRED->GetMaximum()*0.05);

      c1 = new TCanvas("c1","c1,",600,600);

      MCTR->GetXaxis()->SetNdivisions(505);
      MCTR->SetTitle("");
      MCTR->SetStats(kFALSE);
      MCTR->GetXaxis()->SetTitle("m (GeV/c^{2})");
      MCTR->GetYaxis()->SetTitle("#tracks");
      MCTR->GetYaxis()->SetTitleOffset(1.50);
      MCTR->SetLineColor(39);
      MCTR->SetFillColor(64);
      MCTR->SetMarkerStyle(1);
      MCTR->SetMarkerColor(39);
      MCTR->SetMaximum(Max);
      MCTR->SetMinimum(Min);
      MCTR->SetAxisRange(0,500,"X");
      MCTR->Draw("HIST");
      TH1D* MCTRErr = (TH1D*)MCTR->Clone("MCTR_Mass_Err");
      MCTRErr->SetLineColor(39);
      MCTRErr->Draw("E1 same");

      SIGN->GetXaxis()->SetNdivisions(505);
      SIGN->SetTitle("");
      SIGN->SetStats(kFALSE);
      SIGN->GetXaxis()->SetTitle("m (GeV/c^{2})");
      SIGN->GetYaxis()->SetTitle("#tracks");
      SIGN->GetYaxis()->SetTitleOffset(1.50);
      SIGN->SetLineColor(46);
      SIGN->SetFillColor(2);
      SIGN->SetFillStyle(3001);
      SIGN->SetMarkerStyle(1);
      SIGN->SetMarkerColor(46);
      SIGN->SetMaximum(Max);
      SIGN->SetMinimum(Min);
      SIGN->Draw("HIST same");
      TH1D* SIGNErr = (TH1D*)SIGN->Clone("SIGN_Mass_Err");
      SIGNErr->SetLineColor(46);
      SIGNErr->Draw("E1 same");

      PRED->SetMarkerStyle(21);
      PRED->SetMarkerColor(8);
      PRED->SetMarkerSize(1);
      PRED->SetLineColor(8);
      PRED->SetFillColor(0);
      PRED->Draw("E1 same");

      DATA->SetMarkerStyle(20);
      DATA->SetMarkerColor(1);
      DATA->SetMarkerSize(1);
      DATA->SetLineColor(1);
      DATA->SetFillColor(0);
      DATA->Draw("E1 same");

      TLegend* leg = new TLegend(0.79,0.93,0.59,0.73);
      leg->SetHeader(LegendTitle.c_str());
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      leg->AddEntry(MCTR, "MC - MB + QCD","F");
      char SignalLegEntry[256];sprintf(SignalLegEntry,"MC - %s",signals[s].Legend.c_str());
      leg->AddEntry(SIGN, SignalLegEntry     ,"F");
      leg->AddEntry(PRED, "Bgd Prediction"   ,"P");
      leg->AddEntry(DATA, "Data"             ,"P");
      leg->Draw();

      DrawPreliminary(IntegratedLuminosity);

      SaveCanvas(c1, SavePath, signals[s].Name + "_MassLinear");
      c1->SetLogy(true);
      SaveCanvas(c1, SavePath, signals[s].Name + "_Mass");

      if(DATA->Integral()>0){
         if(PRED->Integral()>0)PRED->Scale(DATA->Integral()/PRED->Integral());
      }
      c1->Update();
      c1->SetLogy(false);
      SaveCanvas(c1, SavePath, signals[s].Name + "_MassNormLinear");
      c1->SetLogy(true);
      SaveCanvas(c1, SavePath, signals[s].Name + "_MassNorm");

      delete c1;
   }
}


//////////////////////////////////////////////////     CREATE PLOTS OF SELECTION

void SelectionPlot(string InputPattern){

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }


//   TCanvas* c1;
   string Input     = InputPattern + "DumpHistos.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
 
   stPlots DataPlots, MCTrPlots, SignPlots[signals.size()];
   stPlots_InitFromFile(DataPlots,"Data", InputFile);

   stPlots_InitFromFile(MCTrPlots,"MCTr", InputFile);
   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      stPlots_InitFromFile(SignPlots[s],signals[s].Name, InputFile);

      printf("PLOT SIGNAL %i\n",s);
      stPlots_DrawComparison(SignPlots[s], MCTrPlots, DataPlots, signals[s], SavePath + "/Selection_Comp_" + signals[s].Name, LegendTitle);
      stPlots_Draw(SignPlots[s], SavePath + "/Selection_" +  signals[s].Name, LegendTitle);
   }

   stPlots_Draw(DataPlots, SavePath + "/Selection_Data", LegendTitle);
   stPlots_Draw(MCTrPlots, SavePath + "/Selection_MCTr", LegendTitle);

   stPlots_Clear(DataPlots);
   stPlots_Clear(MCTrPlots);
   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      stPlots_Clear(SignPlots[s]);
   }

}



 //////////////////////////////////////////////////     CREATE PLOTS OF CONTROLS AND PREDICTION

void PredictionAndControlPlot(string InputPattern){
   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<string> legend;

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }



   string Input     = InputPattern + "DumpHistos.root";
   string SavePath  = InputPattern;
   MakeDirectories(SavePath);

   TFile* InputFile = new TFile(Input.c_str());
   TH1D* Ctrl_BckgP            = (TH1D*)GetObjectFromPath(InputFile, "Ctrl_BckgP"   ); 	Ctrl_BckgP->Rebin(5);
   TH1D* CtrlPt_BckgIs         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_BckgIs");	CtrlPt_BckgIs->Rebin(5);
   TH1D* CtrlPt_BckgIm         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_BckgIm");	CtrlPt_BckgIm->Rebin(1);
   TH1D* CtrlP_BckgIs          = (TH1D*)GetObjectFromPath(InputFile, "CtrlP_BckgIs" );	CtrlP_BckgIs->Rebin(5);
   TH1D* CtrlP_BckgIm          = (TH1D*)GetObjectFromPath(InputFile, "CtrlP_BckgIm" );	CtrlP_BckgIm->Rebin(1);
   TH1D* Ctrl_SignP            = (TH1D*)GetObjectFromPath(InputFile, "Ctrl_SignP"   );	Ctrl_SignP->Rebin(5);
   TH1D* CtrlPt_SignIs         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_SignIs");	CtrlPt_SignIs->Rebin(5);
   TH1D* CtrlPt_SignIm         = (TH1D*)GetObjectFromPath(InputFile, "CtrlPt_SignIm");	CtrlPt_SignIm->Rebin(1);
   TH1D* CtrlP_SignIs          = (TH1D*)GetObjectFromPath(InputFile, "CtrlP_SignIs" );	CtrlP_SignIs->Rebin(5);
   TH1D* CtrlP_SignIm          = (TH1D*)GetObjectFromPath(InputFile, "CtrlP_SignIm" );	CtrlP_SignIm->Rebin(1);

   TH1D* Pred_Expected_Entries = (TH1D*)GetObjectFromPath(InputFile, "Pred_Expected_Entries");
   TH1D* Pred_Observed_Entries = (TH1D*)GetObjectFromPath(InputFile, "Pred_Observed_Entries");

   TH1D* Pred_Correlation_A    = (TH1D*)GetObjectFromPath(InputFile, "Pred_Correlation_A");
   TH1D* Pred_Correlation_B    = (TH1D*)GetObjectFromPath(InputFile, "Pred_Correlation_B");
   TH1D* Pred_Correlation_C    = (TH1D*)GetObjectFromPath(InputFile, "Pred_Correlation_C");
   TH1D* Pred_Correlation_D    = (TH1D*)GetObjectFromPath(InputFile, "Pred_Correlation_D");

   TH1D* Pred_P                = (TH1D*)GetObjectFromPath(InputFile, "P_Pred");
   TH1D* Pred_I                = (TH1D*)GetObjectFromPath(InputFile, "I_Pred");





   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(Ctrl_BckgP->Integral()>0)Ctrl_BckgP->Scale(1/Ctrl_BckgP->Integral());
   if(Ctrl_SignP->Integral()>0)Ctrl_SignP->Scale(1/Ctrl_SignP->Integral());
   Histos[0] = Ctrl_BckgP;                         legend.push_back("control sample");
   Histos[1] = Ctrl_SignP;                         legend.push_back("signal like sample");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "P (Gev/c)", "arbitrary units", 0,1000, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Control_PSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_BckgIs->Integral()>0)CtrlPt_BckgIs->Scale(1/CtrlPt_BckgIs->Integral());
   if(CtrlPt_SignIs->Integral()>0)CtrlPt_SignIs->Scale(1/CtrlPt_SignIs->Integral());
//   Histos[0] = CtrlPt_BckgIs;                     legend.push_back("7.5<p_{T}<20 GeV");
   Histos[0] = CtrlPt_BckgIs;                     legend.push_back("15<p_{T}<25 GeV");
   Histos[1] = CtrlPt_SignIs;                     legend.push_back("p_{T}>25 GeV");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxLegend[dEdxSeleIndex], "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Control_IsSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlPt_BckgIm->Integral()>0)CtrlPt_BckgIm->Scale(1/CtrlPt_BckgIm->Integral());
   if(CtrlPt_SignIm->Integral()>0)CtrlPt_SignIm->Scale(1/CtrlPt_SignIm->Integral());
//   Histos[0] = CtrlPt_BckgIm;                     legend.push_back("7.5<p_{T}<20 GeV");
   Histos[0] = CtrlPt_BckgIm;                     legend.push_back("15<p_{T}<25 GeV");
   Histos[1] = CtrlPt_SignIm;                     legend.push_back("p_{T}>25 GeV");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxLegend[dEdxMassIndex], "arbitrary units", 0,10, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Control_ImSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlP_BckgIs->Integral()>0)CtrlP_BckgIs->Scale(1/CtrlP_BckgIs->Integral());
   if(CtrlP_SignIs->Integral()>0)CtrlP_SignIs->Scale(1/CtrlP_SignIs->Integral());
//   Histos[0] = CtrlP_BckgIs;                      legend.push_back("7.5<p<20 GeV");
   Histos[0] = CtrlP_BckgIs;                      legend.push_back("15<p<25 GeV");
   Histos[1] = CtrlP_SignIs;                      legend.push_back("p>25 GeV");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxLegend[dEdxSeleIndex], "arbitrary units", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlP_IsSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   if(CtrlP_BckgIm->Integral()>0)CtrlP_BckgIm->Scale(1/CtrlP_BckgIm->Integral());
   if(CtrlP_SignIm->Integral()>0)CtrlP_SignIm->Scale(1/CtrlP_SignIm->Integral());
//   Histos[0] = CtrlP_BckgIm;                      legend.push_back("7.5<p<20 GeV");
   Histos[0] = CtrlP_BckgIm;                      legend.push_back("15<p<25 GeV");
   Histos[1] = CtrlP_SignIm;                      legend.push_back("p>25 GeV");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  dEdxLegend[dEdxMassIndex], "arbitrary units", 0,10, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   c1->SetLogy(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"ControlP_ImSpectrum");
   delete c1;



   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = Pred_Correlation_A;                legend.push_back("Region A");
   Histos[1] = Pred_Correlation_B;                legend.push_back("Region B");
   Histos[2] = Pred_Correlation_C;                legend.push_back("Region C");
   Histos[3] = Pred_Correlation_D;                legend.push_back("Region D");
   DrawSuperposedHistos((TH1**)Histos, legend, "P",  "Interval Index", "Correlation Factor", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Correlation");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = Pred_Expected_Entries;             legend.push_back("Predicted");
   Histos[1] = Pred_Observed_Entries;             legend.push_back("Observed");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Interval Index", "#tracks", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_Entries_Absolute");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   TH1D* Diff = (TH1D*)Pred_Observed_Entries->Clone();
   Diff->Add(Pred_Expected_Entries,-1);
   Histos[0] = Diff;                              legend.push_back("Observed-Predicted");
   DrawSuperposedHistos((TH1**)Histos, legend, "E1",  "Interval Index", "#tracks", 0,0, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_Entries_Difference");
   delete Diff;
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   if(Pred_P->Integral()>0)Pred_P->Scale(1/Pred_P->Integral());
   Histos[0] = Pred_P;                            legend.push_back("");
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  "p (Gev/c)", "u.a.", 0,1000, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_PSpectrum");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   c1->SetLogy(true);
   if(Pred_I->Integral()>0)Pred_I->Scale(1/Pred_I->Integral());
   Histos[0] = Pred_I;                            legend.push_back("");
   DrawSuperposedHistos((TH1**)Histos, legend, "Hist E1",  dEdxLegend[dEdxMassIndex], "u.a.", 0,10, 0,0);
   DrawLegend(Histos,legend,LegendTitle,"P");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,SavePath,"Prediction_ISpectrum");
   delete c1;
}


void Make2DPlot_Core(string InputPattern){
   TCanvas* c1;
   TLegend* leg;
 
//   GetSignalDefinition(signals);

   string Input = InputPattern + "DumpHistos.root";
//   string outpath = string("Results/PLOT/") + InputPattern;
   string outpath = InputPattern;
   MakeDirectories(outpath);

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

   TFile* InputFile = new TFile(Input.c_str());
   //TH1D* Data_Mass    = (TH1D*)GetObjectFromPath(InputFile, "Mass_Data");
   TH2D* Data_PIs     = (TH2D*)GetObjectFromPath(InputFile, "Data_AS_PIs");
   TH2D* Data_PIm     = (TH2D*)GetObjectFromPath(InputFile, "Data_AS_PIm");
   TH1D* Stop130_Mass = (TH1D*)GetObjectFromPath(InputFile, "Mass_Stop130");
   TH2D* Stop130_PIs  = (TH2D*)GetObjectFromPath(InputFile, "Stop130_AS_PIs");
   TH2D* Stop130_PIm  = (TH2D*)GetObjectFromPath(InputFile, "Stop130_AS_PIm");
   TH1D* Stop200_Mass = (TH1D*)GetObjectFromPath(InputFile, "Mass_Stop200");
   TH2D* Stop200_PIs  = (TH2D*)GetObjectFromPath(InputFile, "Stop200_AS_PIs");
   TH2D* Stop200_PIm  = (TH2D*)GetObjectFromPath(InputFile, "Stop200_AS_PIm");
   TH1D* Stop300_Mass = (TH1D*)GetObjectFromPath(InputFile, "Mass_Stop300");
   TH2D* Stop300_PIs  = (TH2D*)GetObjectFromPath(InputFile, "Stop300_AS_PIs");
   TH2D* Stop300_PIm  = (TH2D*)GetObjectFromPath(InputFile, "Stop300_AS_PIm");
   TH1D* Stop500_Mass = (TH1D*)GetObjectFromPath(InputFile, "Mass_Stop500");
   TH2D* Stop500_PIs  = (TH2D*)GetObjectFromPath(InputFile, "Stop500_AS_PIs");
   TH2D* Stop500_PIm  = (TH2D*)GetObjectFromPath(InputFile, "Stop500_AS_PIm");
   TH1D* Stop800_Mass = (TH1D*)GetObjectFromPath(InputFile, "Mass_Stop800");
   TH2D* Stop800_PIs  = (TH2D*)GetObjectFromPath(InputFile, "Stop800_AS_PIs");
   TH2D* Stop800_PIm  = (TH2D*)GetObjectFromPath(InputFile, "Stop800_AS_PIm");


//   Data_Mass    = (TH1D*) Data_Mass->Rebin(4);
   Stop130_Mass = (TH1D*) Stop130_Mass->Rebin(4);
   Stop200_Mass = (TH1D*) Stop200_Mass->Rebin(4);
   Stop300_Mass = (TH1D*) Stop300_Mass->Rebin(4);
   Stop500_Mass = (TH1D*) Stop500_Mass->Rebin(4);
   Stop800_Mass = (TH1D*) Stop800_Mass->Rebin(4);


   double Min = 1E-6;
   double Max = 1E2;

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"#tracks / %2.0f GeV/c^{2}",Stop130_Mass->GetXaxis()->GetBinWidth(1));


   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridx(true);
//   c1->SetGridy(true);
   Stop130_Mass->SetAxisRange(0,1250,"X");
   Stop130_Mass->SetAxisRange(Min,Max,"Y");
   Stop130_Mass->SetTitle("");
   Stop130_Mass->SetStats(kFALSE);
   Stop130_Mass->GetXaxis()->SetTitle("m (GeV/c^{2})");
   Stop130_Mass->GetYaxis()->SetTitle(YAxisLegend);
   Stop130_Mass->SetLineWidth(2);
   Stop130_Mass->SetLineColor(Color[0]);
   Stop130_Mass->SetMarkerColor(Color[0]);
   Stop130_Mass->SetMarkerStyle(Marker[0]);
//   Stop130_Mass->SetMarkerSize(0.3);
   Stop130_Mass->Draw("E1");
   Stop200_Mass->Draw("E1 same");
   Stop200_Mass->SetLineWidth(2);
   Stop200_Mass->SetLineColor(Color[1]);
   Stop200_Mass->SetMarkerColor(Color[1]);
   Stop200_Mass->SetMarkerStyle(Marker[1]);
//   Stop200_Mass->SetMarkerSize(0.3);
   Stop300_Mass->Draw("E1 same");
   Stop300_Mass->SetLineWidth(2);
   Stop300_Mass->SetLineColor(Color[2]);
   Stop300_Mass->SetMarkerColor(Color[2]);
   Stop300_Mass->SetMarkerStyle(Marker[2]);
//   Stop300_Mass->SetMarkerSize(0.3);
   Stop500_Mass->Draw("E1 same");
   Stop500_Mass->SetLineWidth(2);
   Stop500_Mass->SetLineColor(Color[3]);
   Stop500_Mass->SetMarkerColor(Color[3]);
   Stop500_Mass->SetMarkerStyle(Marker[3]);
//   Stop500_Mass->SetMarkerSize(0.3);
   Stop800_Mass->Draw("E1 same");
   Stop800_Mass->SetLineWidth(2);
   Stop800_Mass->SetLineColor(Color[4]);
   Stop800_Mass->SetMarkerColor(Color[4]);
   Stop800_Mass->SetMarkerStyle(Marker[4]);
//   Stop800_Mass->SetMarkerSize(0.3);
   c1->SetLogy(true);


   TLine* lineStop130 = new TLine(130, Min, 130, Max);
   lineStop130->SetLineWidth(2);
   lineStop130->SetLineColor(Color[0]);
   lineStop130->SetLineStyle(2);
   lineStop130->Draw("same");

   TLine* lineStop200 = new TLine(200, Min, 200, Max);
   lineStop200->SetLineWidth(2);
   lineStop200->SetLineColor(Color[1]);
   lineStop200->SetLineStyle(2);
   lineStop200->Draw("same");

   TLine* lineStop300 = new TLine(300, Min, 300, Max);
   lineStop300->SetLineWidth(2);
   lineStop300->SetLineColor(Color[2]);
   lineStop300->SetLineStyle(2);
   lineStop300->Draw("same");

   TLine* lineStop500 = new TLine(500, Min, 500, Max);
   lineStop500->SetLineWidth(2);
   lineStop500->SetLineColor(Color[3]);
   lineStop500->SetLineStyle(2);
   lineStop500->Draw("same");

   TLine* lineStop800 = new TLine(800, Min, 800, Max);
   lineStop800->SetLineWidth(2);
   lineStop800->SetLineColor(Color[4]);
   lineStop800->SetLineStyle(2);
   lineStop800->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stop130_Mass, "Stop130"   ,"P");
   leg->AddEntry(Stop200_Mass, "Stop200"   ,"P");
   leg->AddEntry(Stop300_Mass, "Stop300"   ,"P");
   leg->AddEntry(Stop500_Mass, "Stop500"   ,"P");
   leg->AddEntry(Stop800_Mass, "Stop800"   ,"P");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Stop_Mass");
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
//   c1->SetGridx(true);
//   c1->SetGridy(true);
   Data_PIs->SetTitle("");
   Data_PIs->SetStats(kFALSE);
   Data_PIs->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIs->GetYaxis()->SetTitle(DiscrLeg);
   Data_PIs->SetAxisRange(0,1250,"X");
   Data_PIs->SetMarkerSize (0.2);
   Data_PIs->SetMarkerColor(Color[4]);
   Data_PIs->SetFillColor(Color[4]);
   Data_PIs->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIs");
   delete c1;

   c1 = new TCanvas("c1","c1", 600, 600);
   c1->SetLogz(true);
//   c1->SetGridx(true);
//   c1->SetGridy(true);
   Data_PIm->SetTitle("");
   Data_PIm->SetStats(kFALSE);
   Data_PIm->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIm->GetYaxis()->SetTitle(EstimLeg);
   Data_PIm->SetAxisRange(0,1250,"X");
   Data_PIm->SetAxisRange(0,15,"Y");
   Data_PIm->SetMarkerSize (0.2);
   Data_PIm->SetMarkerColor(Color[4]);
   Data_PIm->SetFillColor(Color[4]);
   Data_PIm->Draw("COLZ");
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIm");
   delete c1;



   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridx(true);
//   c1->SetGridy(true);
   Stop800_PIs->SetTitle("");
   Stop800_PIs->SetStats(kFALSE);
   Stop800_PIs->GetXaxis()->SetTitle("p (GeV/c)");
   Stop800_PIs->GetYaxis()->SetTitle(DiscrLeg);
   Stop800_PIs->SetAxisRange(0,1250,"X");
   Stop800_PIs->Scale(1/Stop800_PIs->Integral());
   Stop800_PIs->SetMarkerSize (0.2);
   Stop800_PIs->SetMarkerColor(Color[4]);
   Stop800_PIs->SetFillColor(Color[4]);
   Stop800_PIs->Draw("");
   Stop500_PIs->Scale(1/Stop500_PIs->Integral());
   Stop500_PIs->SetMarkerSize (0.2);
   Stop500_PIs->SetMarkerColor(Color[3]);
   Stop500_PIs->SetFillColor(Color[3]);
   Stop500_PIs->Draw("same");
   Stop300_PIs->Scale(1/Stop300_PIs->Integral());
   Stop300_PIs->SetMarkerSize (0.2);
   Stop300_PIs->SetMarkerColor(Color[2]);
   Stop300_PIs->SetFillColor(Color[2]);
   Stop300_PIs->Draw("same");
   Stop200_PIs->Scale(1/Stop200_PIs->Integral());
   Stop200_PIs->SetMarkerSize (0.2);
   Stop200_PIs->SetMarkerColor(Color[1]);
   Stop200_PIs->SetFillColor(Color[1]);
   Stop200_PIs->Draw("same");
   Stop130_PIs->Scale(1/Stop130_PIs->Integral());
   Stop130_PIs->SetMarkerSize (0.2);
   Stop130_PIs->SetMarkerColor(Color[0]);
   Stop130_PIs->SetFillColor(Color[0]);
   Stop130_PIs->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stop130_PIs, "Stop130"   ,"F");
   leg->AddEntry(Stop200_PIs, "Stop200"   ,"F");
   leg->AddEntry(Stop300_PIs, "Stop300"   ,"F");
   leg->AddEntry(Stop500_PIs, "Stop500"   ,"F");
   leg->AddEntry(Stop800_PIs, "Stop800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Stop_PIs");
   delete c1;



   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridx(true);
//   c1->SetGridy(true);
   Stop800_PIm->SetTitle("");
   Stop800_PIm->SetStats(kFALSE);
   Stop800_PIm->GetXaxis()->SetTitle("p (GeV/c)");
   Stop800_PIm->GetYaxis()->SetTitle(EstimLeg);
   Stop800_PIm->SetAxisRange(0,1250,"X");
   Stop800_PIm->SetAxisRange(0,15,"Y");
   Stop800_PIm->Scale(1/Stop800_PIm->Integral());
   Stop800_PIm->SetMarkerSize (0.2);
   Stop800_PIm->SetMarkerColor(Color[4]);
   Stop800_PIm->SetFillColor(Color[4]);
   Stop800_PIm->Draw("");
   Stop500_PIm->Scale(1/Stop500_PIm->Integral());
   Stop500_PIm->SetMarkerSize (0.2);
   Stop500_PIm->SetMarkerColor(Color[3]);
   Stop500_PIm->SetFillColor(Color[3]);
   Stop500_PIm->Draw("same");
   Stop300_PIm->Scale(1/Stop300_PIm->Integral());
   Stop300_PIm->SetMarkerSize (0.2);
   Stop300_PIm->SetMarkerColor(Color[2]);
   Stop300_PIm->SetFillColor(Color[2]);
   Stop300_PIm->Draw("same");
   Stop200_PIm->Scale(1/Stop200_PIm->Integral());
   Stop200_PIm->SetMarkerSize (0.2);
   Stop200_PIm->SetMarkerColor(Color[1]);
   Stop200_PIm->SetFillColor(Color[1]);
   Stop200_PIm->Draw("same");
   Stop130_PIm->Scale(1/Stop130_PIm->Integral());
   Stop130_PIm->SetMarkerSize (0.2);
   Stop130_PIm->SetMarkerColor(Color[0]);
   Stop130_PIm->SetFillColor(Color[0]);
   Stop130_PIm->Draw("same");

   TF1* Stop800Line = GetMassLine(800, true);
   Stop800Line->SetLineColor(kMagenta-7);
   Stop800Line->SetLineWidth(2);
   Stop800Line->Draw("same");
   TF1* Stop500Line = GetMassLine(500, true);
   Stop500Line->SetLineColor(kGreen-7);
   Stop500Line->SetLineWidth(2);
   Stop500Line->Draw("same");
   TF1* Stop300Line = GetMassLine(300, true);
   Stop300Line->SetLineColor(kGray+3);
   Stop300Line->SetLineWidth(2);
   Stop300Line->Draw("same");
   TF1* Stop200Line = GetMassLine(200, true);
   Stop200Line->SetLineColor(kBlue-7);
   Stop200Line->SetLineWidth(2);
   Stop200Line->Draw("same");
   TF1* Stop130Line = GetMassLine(130, true);
   Stop130Line->SetLineColor(kRed-7);
   Stop130Line->SetLineWidth(2);
   Stop130Line->Draw("same");

   leg = new TLegend(0.80,0.93,0.80 - 0.20,0.93 - 6*0.03);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Stop130_PIm, "Stop130"   ,"F");
   leg->AddEntry(Stop200_PIm, "Stop200"   ,"F");
   leg->AddEntry(Stop300_PIm, "Stop300"   ,"F");
   leg->AddEntry(Stop500_PIm, "Stop500"   ,"F");
   leg->AddEntry(Stop800_PIm, "Stop800"   ,"F");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Stop_PIm");
   delete c1;
}


void dEdxVsP_Plot_Core(string InputPattern){
   TCanvas* c1;
   TLegend* leg;
 
//   GetSignalDefinition(signals);

   string Input = InputPattern + "DumpHistos.root";
   string outpath = InputPattern;
   MakeDirectories(outpath);

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

   TFile* InputFile    = new TFile(Input.c_str());
   TH2D* Data_PIm      = (TH2D*)GetObjectFromPath(InputFile, "Data_AS_PIm");
   TH2D* Data_PIm_075  = (TH2D*)Data_PIm->Clone();   Data_PIm_075->Reset(); 
   TH2D* Data_PIm_150  = (TH2D*)Data_PIm->Clone();   Data_PIm_150->Reset();
   TH2D* Data_PIm_300  = (TH2D*)Data_PIm->Clone();   Data_PIm_300->Reset();
   TH2D* Data_PIm_450  = (TH2D*)Data_PIm->Clone();   Data_PIm_450->Reset();
   TH2D* Data_PIm_All  = (TH2D*)Data_PIm->Clone();   Data_PIm_All->Reset();

   for(unsigned int i=0;i<(unsigned int)Data_PIm->GetNbinsX();i++){
   for(unsigned int j=0;j<(unsigned int)Data_PIm->GetNbinsY();j++){
      if(Data_PIm->GetBinContent(i,j)<=0)continue;
      double M = GetMass(Data_PIm->GetXaxis ()->GetBinCenter(i), Data_PIm->GetYaxis ()->GetBinCenter(j));
      if(isnan(M))continue;
      if     (M<100){ Data_PIm_075->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<200){ Data_PIm_150->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<300){ Data_PIm_300->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else if(M<395){ Data_PIm_450->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
      else          { Data_PIm_All->SetBinContent(i,j, Data_PIm->GetBinContent(i,j) ); }
   }}


   c1 = new TCanvas("c1","c1", 600, 600);
   Data_PIm_075->SetTitle("");
   Data_PIm_075->SetStats(kFALSE);
   Data_PIm_075->GetXaxis()->SetTitle("p (GeV/c)");
   Data_PIm_075->GetYaxis()->SetTitle(EstimLeg);
   Data_PIm_075->SetAxisRange(0,15,"Y");
   Data_PIm_075->SetAxisRange(0,1250,"X");
   Data_PIm_075->SetMarkerSize (0.6);
   Data_PIm_075->SetMarkerColor(Color[4]);
   Data_PIm_075->SetMarkerStyle(Marker[4]);
   Data_PIm_075->SetFillColor(Color[4]);
   Data_PIm_075->Draw("");
   Data_PIm_150->SetMarkerSize (0.8);
   Data_PIm_150->SetMarkerColor(Color[3]);
   Data_PIm_150->SetMarkerStyle(Marker[3]);
   Data_PIm_150->SetFillColor(Color[3]);
   Data_PIm_150->Draw("same");
   Data_PIm_300->SetMarkerSize (1.0);
   Data_PIm_300->SetMarkerColor(Color[2]);
   Data_PIm_300->SetMarkerStyle(Marker[2]);
   Data_PIm_300->SetFillColor(Color[2]);
   Data_PIm_300->Draw("same");
   Data_PIm_450->SetMarkerSize (1.2);
   Data_PIm_450->SetMarkerColor(Color[1]);
   Data_PIm_450->SetMarkerStyle(Marker[1]);
   Data_PIm_450->SetFillColor(Color[1]);
   Data_PIm_450->Draw("same");
   Data_PIm_All->SetMarkerSize (1.4);
   Data_PIm_All->SetMarkerColor(Color[0]);
//   Data_PIm_All->SetMarkerStyle(Marker[0]);
   Data_PIm_All->SetFillColor(Color[0]);
   Data_PIm_All->Draw("same");
/*
   TF1* Stop800Line = GetMassLine(800, true);
   Stop800Line->SetLineColor(kMagenta-7);
   Stop800Line->SetLineWidth(2);
   Stop800Line->Draw("same");
   TF1* Stop500Line = GetMassLine(500, true);
   Stop500Line->SetLineColor(kGreen-7);
   Stop500Line->SetLineWidth(2);
   Stop500Line->Draw("same");
   TF1* Stop300Line = GetMassLine(300, true);
   Stop300Line->SetLineColor(kGray+3);
   Stop300Line->SetLineWidth(2);
   Stop300Line->Draw("same");
   TF1* Stop200Line = GetMassLine(200, true);
   Stop200Line->SetLineColor(kBlue-7);
   Stop200Line->SetLineWidth(2);
   Stop200Line->Draw("same");
   TF1* Stop130Line = GetMassLine(130, true);
   Stop130Line->SetLineColor(kRed-7);
   Stop130Line->SetLineWidth(2);
   Stop130Line->Draw("same");
*/
   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.03);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_PIm_075, "M < 100 GeV","P");
   leg->AddEntry(Data_PIm_150, "100 < M < 200 GeV","P");
   leg->AddEntry(Data_PIm_300, "200 < M < 300 GeV","P");
   leg->AddEntry(Data_PIm_450, "300 < M < 400 GeV","P");
   leg->AddEntry(Data_PIm_All, "400 < M GeV"      ,"P");
   leg->Draw();
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "Data_PIm_Colored");
   delete c1;
}




void MakeCompPlot(string DirName, string InputPattern1, string InputPattern2){
   string outpath = string("Results/PLOT/") + DirName;
   MakeDirectories(outpath);

   bool IsTrackerOnly = (InputPattern1.find("Type0",0)<string::npos);
   string LegendTitle;
   if(IsTrackerOnly){
      LegendTitle = "Tracker - Only";
   }else{
      LegendTitle = "Tracker + Muon";
   }


   TH1D* Histo1;
   TH1D* Histo2;
   TFile* InputFile1;
   TFile* InputFile2;
   string Input;

   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;
   TPaveText* pave;

   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Data_BS_Is"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);
   printf("BIN WIDTH=%f\n",Histo1->GetBinWidth(10));

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Data_BS_Is"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  DiscrLeg, "#tracks / 0.05", 0,0, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("Data");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Is_Data");
   delete c1;

   printf("INTEGRAL OF DATA from 0.6 to 1 --> %6.2E\n",Histo1->Integral(Histo1->GetXaxis()->FindBin(0.6), Histo1->GetXaxis()->FindBin(1.0)));
   printf("INTEGRAL OF DATA from 0.6 to 1 --> %6.2E\n",Histo2->Integral(Histo2->GetXaxis()->FindBin(0.6), Histo2->GetXaxis()->FindBin(1.0)));







   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "MCTr_BS_Is"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);
   printf("BIN WIDTH=%f\n",Histo1->GetBinWidth(10));

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "MCTr_BS_Is"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  DiscrLeg, "#tracks / 0.05", 0,0, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("MC - QCD");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Is_MC");
   delete c1;

   printf("INTEGRAL OF MC from 0.6 to 1 --> %6.2E\n",Histo1->Integral(Histo1->GetXaxis()->FindBin(0.6), Histo1->GetXaxis()->FindBin(1.0)));
   printf("INTEGRAL OF MC from 0.6 to 1 --> %6.2E\n",Histo2->Integral(Histo2->GetXaxis()->FindBin(0.6), Histo2->GetXaxis()->FindBin(1.0)));








   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Gluino200_BS_Is"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Gluino200_BS_Is"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  DiscrLeg, "#tracks / 0.05", 0,0, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("MC - #tilde{g} 200");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Is_Gluino200");
   delete c1;

   printf("INTEGRAL OF S from 0.6 to 1 --> %6.2E\n",Histo1->Integral(Histo1->GetXaxis()->FindBin(0.6), Histo1->GetXaxis()->FindBin(1.0)));
   printf("INTEGRAL OF S from 0.6 to 1 --> %6.2E\n",Histo2->Integral(Histo2->GetXaxis()->FindBin(0.6), Histo2->GetXaxis()->FindBin(1.0)));


   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Gluino900_BS_Is"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Gluino900_BS_Is"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   Histos[0]->SetMaximum(Histos[0]->GetMaximum()*4);
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  DiscrLeg, "#tracks / 0.05", 0,0, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("MC - #tilde{g} 900");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Is_Gluino900");
   delete c1;





   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Data_BS_Im"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Data_BS_Im"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  EstimLeg, "#tracks", 0,25, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("Data");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Im_Data");
   delete c1;


   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Gluino200_BS_Im"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Gluino200_BS_Im"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  EstimLeg, "#tracks", 0,25, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("MC - #tilde{g} 200");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Im_Gluino200");
   delete c1;


   Input = InputPattern1 + "DumpHistos.root";
   InputFile1 = new TFile(Input.c_str());
   Histo1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Gluino900_BS_Im"))->Clone("Hist1");
   Histo1 = (TH1D*)Histo1->Rebin(500);

   Input = InputPattern2 + "DumpHistos.root";
   InputFile2 = new TFile(Input.c_str());
   Histo2 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile2, "Gluino900_BS_Im"))->Clone("Hist2");
   Histo2 = (TH1D*)Histo2->Rebin(500);

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)Histo1;                      legend.push_back("With ClusterCleaning");
   Histos[1] = (TH1*)Histo2;                      legend.push_back("Without Cluster Cleaning");
   Histo1->SetMaximum(Histo1->GetMaximum()*1.1);
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  EstimLeg, "#tracks", 0,25, 0,0);
   Histo2->SetLineStyle(2);
   DrawLegend((TObject**)Histos,legend,LegendTitle,"LP",0.79, 0.90, 0.45, 0.07);
   c1->SetLogy(true);
   c1->Modified();

   pave = new TPaveText(0.80, 0.93, 0.60, 0.88, "NDC");
   pave->SetFillColor(0);
   pave->SetBorderSize(0);
   pave->AddText("MC - #tilde{g} 900");
   pave->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,outpath,"Im_Gluino900");
   delete c1;
}




void CheckPredictionRescale(string InputPattern, bool RecomputeRescale)
{
   double Rescale, RMS;
   GetPredictionRescale(InputPattern,Rescale, RMS, RecomputeRescale);

   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

//   string outpath = string("Results/PLOT/") + DirName;
   string outpath = InputPattern;
   MakeDirectories(outpath);

//   TH1D* Histo1;
   TFile* InputFile1;
   string Input;

//   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;

   char Buffer[2048];
   sprintf(Buffer,"%s/DumpHistos.root",InputPattern.c_str());
   InputFile1 = new TFile(Buffer);
   if(!InputFile1 || InputFile1->IsZombie() || !InputFile1->IsOpen() || InputFile1->TestBit(TFile::kRecovered) )return;
   TH1D* Pred1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Mass_Pred"))->Clone("Pred1");
   TH1D* Data1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Mass_Data"))->Clone("Data1");
   TH1D* MCTr1 = (TH1D*)((TH1D*)GetObjectFromPath(InputFile1, "Mass_MCTr"))->Clone("MCTr1");
   MCTr1->Scale(Data1->Integral()/MCTr1->Integral());
   TH1D* Resc1 = (TH1D*)(Pred1->Clone("Resc1"));
   Resc1->Scale(Rescale);




   printf("%s\n",InputPattern.c_str());
   double M,D,P,R, Rerr;
   M = MCTr1->Integral(MCTr1->GetXaxis()->FindBin( 0.0),  MCTr1->GetXaxis()->FindBin(1999.0));
   D = Data1->Integral(Data1->GetXaxis()->FindBin( 0.0),  Data1->GetXaxis()->FindBin(1999.0));  
   P = Pred1->Integral(Pred1->GetXaxis()->FindBin( 0.0),  Pred1->GetXaxis()->FindBin(1999.0));
   R = Resc1->Integral(Resc1->GetXaxis()->FindBin( 0.0),  Resc1->GetXaxis()->FindBin(1999.0));   
   Rerr = 0; for(int i=Resc1->GetXaxis()->FindBin( 0.0);i<Resc1->GetXaxis()->FindBin(1999.0);i++){ Rerr += pow(Resc1->GetBinError(i),2); }  Rerr = sqrt(Rerr);
   printf("INTEGRAL in [  0,1999] --> M = %9.3f P = %9.3f R = %9.3f +- %9.3f(stat) +- %9.3f(syst) (=%9.3f) D = %9.3f\n", M, P, R, Rerr,R*(2*RMS),sqrt(Rerr*Rerr + pow(R*(2*RMS),2)), D);

   M = MCTr1->Integral(MCTr1->GetXaxis()->FindBin(75.0),  MCTr1->GetXaxis()->FindBin(1999.0));
   D = Data1->Integral(Data1->GetXaxis()->FindBin(75.0),  Data1->GetXaxis()->FindBin(1999.0));
   P = Pred1->Integral(Pred1->GetXaxis()->FindBin(75.0),  Pred1->GetXaxis()->FindBin(1999.0));
   R = Resc1->Integral(Resc1->GetXaxis()->FindBin(75.0),  Resc1->GetXaxis()->FindBin(1999.0));
   Rerr = 0; for(int i=Resc1->GetXaxis()->FindBin(75.0);i<Resc1->GetXaxis()->FindBin(1999.0);i++){ Rerr += pow(Resc1->GetBinError(i),2); }  Rerr = sqrt(Rerr);
   printf("INTEGRAL in [ 75,1999] --> M = %9.3f P = %9.3f R = %9.3f +- %9.3f(stat) +- %9.3f(syst) (=%9.3f) D = %9.3f\n", M, P, R, Rerr,R*(2*RMS),sqrt(Rerr*Rerr + pow(R*(2*RMS),2)), D);

   M = MCTr1->Integral(MCTr1->GetXaxis()->FindBin(75.0),  MCTr1->GetXaxis()->FindBin(100.0));
   D = Data1->Integral(Data1->GetXaxis()->FindBin(75.0),  Data1->GetXaxis()->FindBin(100.0));
   P = Pred1->Integral(Pred1->GetXaxis()->FindBin(75.0),  Pred1->GetXaxis()->FindBin(100.0));
   R = Resc1->Integral(Resc1->GetXaxis()->FindBin(75.0),  Resc1->GetXaxis()->FindBin(100.0));
   Rerr = 0; for(int i=Resc1->GetXaxis()->FindBin(75.0);i<Resc1->GetXaxis()->FindBin(100.0);i++){ Rerr += pow(Resc1->GetBinError(i),2); }  Rerr = sqrt(Rerr);
   printf("INTEGRAL in [ 75, 100] --> M = %9.3f P = %9.3f R = %9.3f +- %9.3f(stat) +- %9.3f(syst) (=%9.3f) D = %9.3f\n", M, P, R, Rerr,R*(2*RMS),sqrt(Rerr*Rerr + pow(R*(2*RMS),2)), D);

   M = MCTr1->Integral(MCTr1->GetXaxis()->FindBin(100.0),  MCTr1->GetXaxis()->FindBin(1999.0));
   D = Data1->Integral(Data1->GetXaxis()->FindBin(100.0),  Data1->GetXaxis()->FindBin(1999.0));
   P = Pred1->Integral(Pred1->GetXaxis()->FindBin(100.0),  Pred1->GetXaxis()->FindBin(1999.0));
   R = Resc1->Integral(Resc1->GetXaxis()->FindBin(100.0),  Resc1->GetXaxis()->FindBin(1999.0));
   Rerr = 0; for(int i=Resc1->GetXaxis()->FindBin(100.0);i<Resc1->GetXaxis()->FindBin(1999.0);i++){ Rerr += pow(Resc1->GetBinError(i),2); }  Rerr = sqrt(Rerr);
   printf("INTEGRAL in [100,1999] --> M = %9.3f P = %9.3f R = %9.3f +- %9.3f(stat) +- %9.3f(syst) (=%9.3f) D = %9.3f\n", M, P, R, Rerr,R*(2*RMS),sqrt(Rerr*Rerr + pow(R*(2*RMS),2)), D);

   M = MCTr1->Integral(MCTr1->GetXaxis()->FindBin(300.0),  MCTr1->GetXaxis()->FindBin(1999.0));
   D = Data1->Integral(Data1->GetXaxis()->FindBin(300.0),  Data1->GetXaxis()->FindBin(1999.0));
   P = Pred1->Integral(Pred1->GetXaxis()->FindBin(300.0),  Pred1->GetXaxis()->FindBin(1999.0));
   R = Resc1->Integral(Resc1->GetXaxis()->FindBin(300.0),  Resc1->GetXaxis()->FindBin(1999.0));
   Rerr = 0; for(int i=Resc1->GetXaxis()->FindBin(300.0);i<Resc1->GetXaxis()->FindBin(1999.0);i++){ Rerr += pow(Resc1->GetBinError(i),2); }  Rerr = sqrt(Rerr);
   printf("INTEGRAL in [300,1999] --> M = %9.3f P = %9.3f R = %9.3f +- %9.3f(stat) +- %9.3f(syst) (=%9.3f) D = %9.3f\n", M, P, R, Rerr,R*(2*RMS),sqrt(Rerr*Rerr + pow(R*(2*RMS),2)), D);


/*
   unsigned int CountBin=1;
   double BinWidth = Data1->GetBinWidth(1);
   double* NewBinning = new double[Data1->GetNbinsX()+2];
   NewBinning[0] = Data1->GetBinLowEdge(0);
   while(NewBinning[CountBin-1]<250){NewBinning[CountBin] = NewBinning[CountBin-1] + 1*BinWidth; CountBin++;}
   while(NewBinning[CountBin-1]<300){NewBinning[CountBin] = NewBinning[CountBin-1] + 2*BinWidth; CountBin++;}
   while(NewBinning[CountBin-1]<800){NewBinning[CountBin] = NewBinning[CountBin-1] + 3*BinWidth; CountBin++;}
//   while(NewBinning[CountBin-1]<500){NewBinning[CountBin] = NewBinning[CountBin-1] + 4*BinWidth; CountBin++;}
//   for(unsigned int i=0;i<CountBin;i++){ printf("Bin %3i --> %8.2f\n", i, NewBinning[i]);  }

   MCTr1 = (TH1D*) MCTr1->Rebin(CountBin-1, "MCTr1Rebinned", NewBinning);
   Pred1 = (TH1D*) Pred1->Rebin(CountBin-1, "Pred1Rebinned", NewBinning);
   Resc1 = (TH1D*) Resc1->Rebin(CountBin-1, "Resc1Rebinned", NewBinning);
   Data1 = (TH1D*) Data1->Rebin(CountBin-1, "Data1Rebinned", NewBinning);
*/

   MCTr1->Rebin(10);
   Pred1->Rebin(10);
   Resc1->Rebin(10);
   Data1->Rebin(10);

   Resc1->Reset();
   MassPredictionFromABCD(InputPattern,Resc1);
   Resc1->Scale(Rescale);

   double Max = std::max(Data1->GetMaximum(), Resc1->GetMaximum());
   Max        = std::max(MCTr1->GetMaximum(), Max);
   Max       *= 2.0;
   double Min = std::min(0.01,Pred1->GetMaximum());
//   Min       *= 0.5;
   Min = 0.03;

   TLegend* leg;
   c1 = new TCanvas("c1","c1,",600,600);

   char YAxisLegend[1024];
   sprintf(YAxisLegend,"#tracks / %2.0f GeV/c^{2}",MCTr1->GetXaxis()->GetBinWidth(1));

   MCTr1->GetXaxis()->SetNdivisions(505);
   MCTr1->SetTitle("");
   MCTr1->SetStats(kFALSE);
   MCTr1->GetXaxis()->SetTitle("m (GeV/c^{2})");
   MCTr1->GetYaxis()->SetTitle(YAxisLegend);
   MCTr1->GetYaxis()->SetTitleOffset(1.50);
   MCTr1->SetLineColor(39);
   MCTr1->SetFillColor(64);
   MCTr1->SetMarkerSize(1);
   MCTr1->SetMarkerStyle(1);
   MCTr1->SetMarkerColor(39);
   MCTr1->SetMaximum(Max);
   MCTr1->SetMinimum(Min);
   MCTr1->SetAxisRange(0,1400,"X");
   MCTr1->Draw("HIST");
   TH1D* MCTr1Err = (TH1D*)MCTr1->Clone("MCTr1_Mass_Err");
   MCTr1Err->SetLineColor(39);
   MCTr1Err->Draw("E1 same");


/*
   Pred1->SetMarkerStyle(21);
   Pred1->SetMarkerColor(8);
   Pred1->SetMarkerSize(1.0);
   Pred1->SetLineColor(8);
   Pred1->SetFillColor(0);
   Pred1->Draw("E1 same");
*/


   TH1D* Resc1Err = (TH1D*) Resc1->Clone("Resc1Err");
   for(unsigned int i=0;i<(unsigned int)Resc1->GetNbinsX();i++){
      double error2 = pow(Resc1Err->GetBinError(i),2);
      error2 += pow(Resc1->GetBinContent(i)*2*RMS,2);
      Resc1Err->SetBinError(i,sqrt(error2));
      if(Resc1Err->GetBinContent(i)<=Min)Resc1Err->SetBinContent(i,0);
   }
   Resc1Err->SetLineColor(8);
   Resc1Err->SetFillColor(8);
   Resc1Err->SetFillStyle(3001);
   Resc1Err->SetMarkerStyle(22);
   Resc1Err->SetMarkerColor(2);
   Resc1Err->SetMarkerSize(1.0);
   Resc1Err->Draw("E5 same");

   Resc1->SetMarkerStyle(22);
   Resc1->SetMarkerColor(2);
   Resc1->SetMarkerSize(1.0);
   Resc1->SetLineColor(2);
   Resc1->SetFillColor(0);
   Resc1->Draw("same HIST P");

   Data1->SetMarkerStyle(20);
   Data1->SetMarkerColor(1);
   Data1->SetMarkerSize(1.0);
   Data1->SetLineColor(1);
   Data1->SetFillColor(0);
   Data1->Draw("E1 same");


   leg = new TLegend(0.79,0.93,0.44,0.73);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(MCTr1, "MC - QCD"          ,"F");
//   leg->AddEntry(Pred1, "Absolute Prediction"  ,"P");

   TH1D* RescLeg = (TH1D*) Resc1->Clone("RescLeg");
   RescLeg->SetFillColor(Resc1Err->GetFillColor());
   RescLeg->SetFillStyle(Resc1Err->GetFillStyle());
   leg->AddEntry(RescLeg, "Data-driven prediction"  ,"PF");
//   leg->AddEntry(Resc1Err, "Prediction Syst. Err."  ,"F");
   leg->AddEntry(Data1, "Data"        ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   c1->SetLogy(true);
//   SaveCanvas(c1, outpath, Histo);
   SaveCanvas(c1, outpath, "Rescale_Mass");

//   c1->SetLogy(true);
//   SaveCanvas(c1, outpath, Histo+"B");
   delete c1;


/*
   for(unsigned int s=0;s<signals.size();s++){
      if(!signals[s].MakePlot)continue;
      TH1D* SIGN = (TH1D*)GetObjectFromPath(InputFile1, string("Mass_") + signals[s].Name);
      SIGN->Rebin(10);

      c1 = new TCanvas("c1","c1,",600,600);

      double MaxSign = std::max(Max, SIGN->GetMaximum()*2);
      MCTr1->SetMaximum(MaxSign);
      MCTr1->SetMinimum(Min);
      MCTr1->Draw("HIST");
      MCTr1Err->Draw("E1 same");
      Resc1Err->Draw("E5 same");
      Resc1->Draw("same");
      Data1->Draw("E1 same");
      SIGN->SetLineWidth(1);
      SIGN->SetLineColor(6);
      SIGN->Draw("same HIST");

      leg = new TLegend(0.79,0.93,0.44,0.73);
      if(IsTrackerOnly){
         leg->SetHeader("Tracker - Only");
      }else{
         leg->SetHeader("Tracker + Muon");
      }
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      leg->AddEntry(MCTr1, "MC - QCD"          ,"F");
      leg->AddEntry(SIGN,  (string("MC - ") + signals[s].Legend).c_str()        ,"L");
      leg->AddEntry(RescLeg, "Data-driven prediction"  ,"PF");
      leg->AddEntry(Data1, "Data"        ,"P");
      leg->Draw();

      DrawPreliminary(IntegratedLuminosity);
      c1->SetLogy(true);
      SaveCanvas(c1, outpath, string("Rescale_Mass_") + signals[s].Name);

      delete c1;
   }
*/








  TH1D* Ratio1       = (TH1D*)Pred1->Clone("Ratio1");
  TH1D* Ratio2       = (TH1D*)Resc1->Clone("Ratio2");
  TH1D* Ratio3       = (TH1D*)Resc1->Clone("Ratio3");
  TH1D* DataWithStat = (TH1D*)Data1->Clone("DataWithStat");
/*  for(unsigned int i=0;i<Ratio1->GetNbinsX();i++){
     if(Data1->GetBinContent(i)<2){
        DataWithStat->SetBinContent(i,0);
        DataWithStat->SetBinError(i,0);
     }
  }*/
  Ratio1->Divide(DataWithStat);
  Ratio2->Divide(DataWithStat);
  Ratio3->Divide(MCTr1);

  
//  TH1D* Ratio1 = (TH1D*)Data1->Clone("Ratio1");
//  for(unsigned int i=0;i<Ratio1->GetNbinsX();i++){
//     if(Resc1->GetBinContent(i)>0 && Data1->GetBinContent(i)>1){
//        Ratio1->SetBinContent(i,Data1->GetBinContent(i)/Resc1->GetBinContent(i));
//        Ratio1->SetBinError(i,Ratio1->GetBinContent(i)*0.5);
//     }else{
//        Ratio1->SetBinContent(i,0);
//        Ratio1->SetBinError(i,0);
//     }
//  }
  

   c1 = new TCanvas("c1","c1,",600,600);
   Ratio2->SetAxisRange(0,1400,"X");
   Ratio2->SetAxisRange(0,2,"Y");
   Ratio2->SetTitle("");
   Ratio2->SetStats(kFALSE);
   Ratio2->GetXaxis()->SetTitle("m (GeV/c^{2})");
//   Ratio2->GetYaxis()->SetTitle("#Predicted / #Observed Tracks");
   Ratio2->GetYaxis()->SetTitle("Ratio");
   Ratio2->GetYaxis()->SetTitleOffset(1.50);
   Ratio2->SetMarkerStyle(21);
   Ratio2->SetMarkerColor(4);
   Ratio2->SetMarkerSize(1);
   Ratio2->SetLineColor(4);
   Ratio2->SetFillColor(0);
   Ratio2->Draw("E1");

  
   TBox* b = new TBox(0,1.0-2*RMS,510,1.0+2*RMS);
   b->SetFillStyle(3003);
   b->SetFillColor(8);
   b->Draw("same");

   TLine* l = new TLine(0,1.0,510,1.0);
   l->Draw("same");

   TLine* lm = new TLine(0,1.0-RMS,510,1.0-RMS);
   lm->SetLineStyle(2);
   lm->Draw("same");

   TLine* lp = new TLine(0,1.0+RMS,510,1.0+RMS);
   lp->SetLineStyle(2);
   lp->Draw("same");


   Ratio2->Draw("same E1");

   Ratio3->SetAxisRange(0,1400,"X");
   Ratio3->SetTitle("");
   Ratio3->SetStats(kFALSE);
   Ratio3->GetXaxis()->SetTitle("m (GeV/c^{2})");
   Ratio3->GetYaxis()->SetTitle("#Predicted / #Observed Tracks");
   Ratio3->GetYaxis()->SetTitleOffset(1.50);
   Ratio3->SetMarkerStyle(22);
   Ratio3->SetMarkerColor(2);
   Ratio3->SetMarkerSize(1);
   Ratio3->SetLineColor(2);
   Ratio3->SetFillColor(0);
   Ratio3->Draw("E1 same");

//   leg = new TLegend(0.79,0.93,0.50,0.75);
   leg = new TLegend(0.79,0.93,0.40,0.75);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Ratio2, "Rescaled Prediction / Data"     ,"P");
   leg->AddEntry(Ratio3, "Rescaled Prediction / MC"       ,"P");
   leg->Draw();


   DrawPreliminary(IntegratedLuminosity);  
//   SaveCanvas(c1, outpath, Histo + "Rescale1");
   SaveCanvas(c1, outpath, "Rescale_Ratio");
   delete c1;


   InputFile1->Close();
}



void MakeHitSplit_Plot(string InputPattern){
   TCanvas* c1;
   TLegend* leg;
 
   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

//   GetSignalDefinition(signals);

   string Input = InputPattern + "DumpHistos.root";
   string outpath = InputPattern;
   MakeDirectories(outpath);


   TFile* InputFile = new TFile(Input.c_str());
   TH1D* Data_05_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit08");
   TH1D* Data_10_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit10");
   TH1D* Data_15_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15");
   TH1D* Data_20_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit17");
//   TH1D* Data_05_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta10to15");
//   TH1D* Data_10_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit10_Eta10to15");
//   TH1D* Data_15_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta10to15");
//   TH1D* Data_20_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit20_Eta10to15");
   Data_05_I = (TH1D*) Data_05_I->Rebin(100);
   Data_10_I = (TH1D*) Data_10_I->Rebin(100);
   Data_15_I = (TH1D*) Data_15_I->Rebin(100);
   Data_20_I = (TH1D*) Data_20_I->Rebin(100);
   Data_05_I->Scale(1.0/Data_05_I->Integral());
   Data_10_I->Scale(1.0/Data_10_I->Integral());
   Data_15_I->Scale(1.0/Data_15_I->Integral());
   Data_20_I->Scale(1.0/Data_20_I->Integral());

   TH1D* Data_05_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit08");
   TH1D* Data_10_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit10");
   TH1D* Data_15_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15");
   TH1D* Data_20_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit17");
//   TH1D* Data_05_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit05_Eta10to15");
//   TH1D* Data_10_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit10_Eta10to15");
//   TH1D* Data_15_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta10to15");
//   TH1D* Data_20_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit20_Eta10to15");
   Data_05_Pt = (TH1D*) Data_05_Pt->Rebin(50);
   Data_10_Pt = (TH1D*) Data_10_Pt->Rebin(50);
   Data_15_Pt = (TH1D*) Data_15_Pt->Rebin(50);
   Data_20_Pt = (TH1D*) Data_20_Pt->Rebin(50);
   Data_05_Pt->Scale(1.0/Data_05_Pt->Integral());
   Data_10_Pt->Scale(1.0/Data_10_Pt->Integral());
   Data_15_Pt->Scale(1.0/Data_15_Pt->Integral());
   Data_20_Pt->Scale(1.0/Data_20_Pt->Integral());

//   TH1D* Data_E1_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_Eta00to05");
//   TH1D* Data_E2_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_Eta05to10");
//   TH1D* Data_E3_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_Eta10to15");
//   TH1D* Data_E4_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_Eta15to20");
//   TH1D* Data_E5_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_Eta20to25");
   TH1D* Data_E1_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta00to05");
   TH1D* Data_E2_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta05to10");
   TH1D* Data_E3_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta10to15");
   TH1D* Data_E4_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta15to20");
   TH1D* Data_E5_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit15_Eta20to25");
   Data_E1_I = (TH1D*) Data_E1_I->Rebin(100);
   Data_E2_I = (TH1D*) Data_E2_I->Rebin(100);
   Data_E3_I = (TH1D*) Data_E3_I->Rebin(100);
   Data_E4_I = (TH1D*) Data_E4_I->Rebin(100);
   Data_E5_I = (TH1D*) Data_E5_I->Rebin(100);
   Data_E1_I->Scale(1.0/Data_E1_I->Integral());
   Data_E2_I->Scale(1.0/Data_E2_I->Integral());
   Data_E3_I->Scale(1.0/Data_E3_I->Integral());
   Data_E4_I->Scale(1.0/Data_E4_I->Integral());
   Data_E5_I->Scale(1.0/Data_E5_I->Integral());

//   TH1D* Data_E1_Pt  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_Eta00to05");
//   TH1D* Data_E2_Pt  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_Eta05to10");
//   TH1D* Data_E3_Pt  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_Eta10to15");
//   TH1D* Data_E4_Pt  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_Eta15to20");
//   TH1D* Data_E5_Pt  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_Eta20to25");
   TH1D* Data_E1_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta00to05");
   TH1D* Data_E2_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta05to10");
   TH1D* Data_E3_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta10to15");
   TH1D* Data_E4_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta15to20");
   TH1D* Data_E5_Pt = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_Pt_Data_SSHit15_Eta20to25");
   Data_E1_Pt = (TH1D*) Data_E1_Pt->Rebin(25);
   Data_E2_Pt = (TH1D*) Data_E2_Pt->Rebin(25);
   Data_E3_Pt = (TH1D*) Data_E3_Pt->Rebin(25);
   Data_E4_Pt = (TH1D*) Data_E4_Pt->Rebin(25);
   Data_E5_Pt = (TH1D*) Data_E5_Pt->Rebin(25);
   Data_E1_Pt->Scale(1.0/Data_E1_Pt->Integral());
   Data_E2_Pt->Scale(1.0/Data_E2_Pt->Integral());
   Data_E3_Pt->Scale(1.0/Data_E3_Pt->Integral());
   Data_E4_Pt->Scale(1.0/Data_E4_Pt->Integral());
   Data_E5_Pt->Scale(1.0/Data_E5_Pt->Integral());


   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridy(true);
   Data_05_I->SetTitle("");
   Data_05_I->SetStats(kFALSE);
   Data_05_I->GetXaxis()->SetTitle(DiscrLeg);
   Data_05_I->GetYaxis()->SetTitle("arbitrary units");
   Data_05_I->SetLineWidth(2);
   Data_05_I->SetLineColor(Color[0]);
   Data_05_I->SetMarkerColor(Color[0]);
   Data_05_I->SetMarkerStyle(Marker[0]);
   Data_05_I->Draw("E1");
   Data_05_I->Draw("E1 same");
   Data_10_I->SetLineWidth(2);
   Data_10_I->SetLineColor(Color[1]);
   Data_10_I->SetMarkerColor(Color[1]);
   Data_10_I->SetMarkerStyle(Marker[1]);
   Data_10_I->Draw("E1 same");
   Data_15_I->SetLineWidth(2);
   Data_15_I->SetLineColor(Color[2]);
   Data_15_I->SetMarkerColor(Color[2]);
   Data_15_I->SetMarkerStyle(Marker[2]);
   Data_15_I->Draw("E1 same");
   Data_20_I->SetLineWidth(2);
   Data_20_I->SetLineColor(Color[3]);
   Data_20_I->SetMarkerColor(Color[3]);
   Data_20_I->SetMarkerStyle(Marker[3]);
   Data_20_I->Draw("E1 same");
   c1->SetLogy(true);

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.05);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_05_I, "<09 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_10_I, " 10 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_15_I, " 15 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_20_I, ">16 dE/dx Hits"   ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "HitSplit_IDistribution");
   delete c1;


   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridy(true);
   Data_05_Pt->SetAxisRange(0,200,"X");
   Data_05_Pt->SetTitle("");
   Data_05_Pt->SetStats(kFALSE);
   Data_05_Pt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   Data_05_Pt->GetYaxis()->SetTitle("arbitrary units");
   Data_05_Pt->SetLineWidth(2);
   Data_05_Pt->SetLineColor(Color[0]);
   Data_05_Pt->SetMarkerColor(Color[0]);
   Data_05_Pt->SetMarkerStyle(Marker[0]);
   Data_05_Pt->Draw("E1");
   Data_05_Pt->Draw("E1 same");
   Data_10_Pt->SetLineWidth(2);
   Data_10_Pt->SetLineColor(Color[1]);
   Data_10_Pt->SetMarkerColor(Color[1]);
   Data_10_Pt->SetMarkerStyle(Marker[1]);
   Data_10_Pt->Draw("E1 same");
   Data_15_Pt->SetLineWidth(2);
   Data_15_Pt->SetLineColor(Color[2]);
   Data_15_Pt->SetMarkerColor(Color[2]);
   Data_15_Pt->SetMarkerStyle(Marker[2]);
   Data_15_Pt->Draw("E1 same");
   Data_20_Pt->SetLineWidth(2);
   Data_20_Pt->SetLineColor(Color[3]);
   Data_20_Pt->SetMarkerColor(Color[3]);
   Data_20_Pt->SetMarkerStyle(Marker[3]);
   Data_20_Pt->Draw("E1 same");
   c1->SetLogy(true);

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.05);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_05_Pt, "<09 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_10_Pt, " 10 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_15_Pt, " 15 dE/dx Hits"   ,"P");
   leg->AddEntry(Data_20_Pt, ">16 dE/dx Hits"   ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "HitSplit_PtDistribution");
   delete c1;



   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridy(true);
   Data_E1_I->SetTitle("");
   Data_E1_I->SetStats(kFALSE);
   Data_E1_I->GetXaxis()->SetTitle(DiscrLeg);
   Data_E1_I->GetYaxis()->SetTitle("arbitrary units");
   Data_E1_I->SetLineWidth(2);
   Data_E1_I->SetLineColor(Color[0]);
   Data_E1_I->SetMarkerColor(Color[0]);
   Data_E1_I->SetMarkerStyle(Marker[0]);
   Data_E1_I->SetAxisRange(0,0.35,"X");
   Data_E1_I->Draw("E1");
   Data_E1_I->Draw("E1 same");
   Data_E2_I->SetLineWidth(2);
   Data_E2_I->SetLineColor(Color[1]);
   Data_E2_I->SetMarkerColor(Color[1]);
   Data_E2_I->SetMarkerStyle(Marker[1]);
   Data_E2_I->Draw("E1 same");
   Data_E3_I->SetLineWidth(2);
   Data_E3_I->SetLineColor(Color[2]);
   Data_E3_I->SetMarkerColor(Color[2]);
   Data_E3_I->SetMarkerStyle(Marker[2]);
   Data_E3_I->Draw("E1 same");
   Data_E4_I->SetLineWidth(2);
   Data_E4_I->SetLineColor(Color[3]);
   Data_E4_I->SetMarkerColor(Color[3]);
   Data_E4_I->SetMarkerStyle(Marker[3]);
   Data_E4_I->Draw("E1 same");
   Data_E5_I->SetLineWidth(2);
   Data_E5_I->SetLineColor(Color[4]);
   Data_E5_I->SetMarkerColor(Color[4]);
   Data_E5_I->SetMarkerStyle(Marker[4]);
   Data_E5_I->Draw("E1 same");
   c1->SetLogy(true);

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.05);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_E1_I, "0.0 < |#eta| < 0.5"   ,"P");
   leg->AddEntry(Data_E2_I, "0.5 < |#eta| < 1.0"   ,"P");
   leg->AddEntry(Data_E3_I, "1.0 < |#eta| < 1.5"   ,"P");
   leg->AddEntry(Data_E4_I, "1.5 < |#eta| < 2.0"   ,"P");
   leg->AddEntry(Data_E5_I, "2.0 < |#eta| < 2.5"   ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "EtaSplit_IDistribution");
   delete c1;


   c1 = new TCanvas("c1","c1", 600, 600);
//   c1->SetGridy(true);
   Data_E1_Pt->SetAxisRange(0,200,"X");
   Data_E1_Pt->SetTitle("");
   Data_E1_Pt->SetStats(kFALSE);
   Data_E1_Pt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   Data_E1_Pt->GetYaxis()->SetTitle("arbitrary units");
   Data_E1_Pt->SetLineWidth(2);
   Data_E1_Pt->SetLineColor(Color[0]);
   Data_E1_Pt->SetMarkerColor(Color[0]);
   Data_E1_Pt->SetMarkerStyle(Marker[0]);
   Data_E1_Pt->Draw("E1");
   Data_E1_Pt->Draw("E1 same");
   Data_E2_Pt->SetLineWidth(2);
   Data_E2_Pt->SetLineColor(Color[1]);
   Data_E2_Pt->SetMarkerColor(Color[1]);
   Data_E2_Pt->SetMarkerStyle(Marker[1]);
   Data_E2_Pt->Draw("E1 same");
   Data_E3_Pt->SetLineWidth(2);
   Data_E3_Pt->SetLineColor(Color[2]);
   Data_E3_Pt->SetMarkerColor(Color[2]);
   Data_E3_Pt->SetMarkerStyle(Marker[2]);
   Data_E3_Pt->Draw("E1 same");
   Data_E4_Pt->SetLineWidth(2);
   Data_E4_Pt->SetLineColor(Color[3]);
   Data_E4_Pt->SetMarkerColor(Color[3]);
   Data_E4_Pt->SetMarkerStyle(Marker[3]);
   Data_E4_Pt->Draw("E1 same");
   Data_E5_Pt->SetLineWidth(2);
   Data_E5_Pt->SetLineColor(Color[4]);
   Data_E5_Pt->SetMarkerColor(Color[4]);
   Data_E5_Pt->SetMarkerStyle(Marker[4]);
   Data_E5_Pt->Draw("E1 same");
   c1->SetLogy(true);

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.05);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_E1_Pt, "0.0 < |#eta| < 0.5"   ,"P");
   leg->AddEntry(Data_E2_Pt, "0.5 < |#eta| < 1.0"   ,"P");
   leg->AddEntry(Data_E3_Pt, "1.0 < |#eta| < 1.5"   ,"P");
   leg->AddEntry(Data_E4_Pt, "1.5 < |#eta| < 2.0"   ,"P");
   leg->AddEntry(Data_E5_Pt, "2.0 < |#eta| < 2.5"   ,"P");
   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "EtaSplit_PtDistribution");
   delete c1;

}



void CheckHitSplitSloap_Plot(string InputPattern){
   TCanvas* c1;
   TLegend* leg;
 
   bool IsTrackerOnly = (InputPattern.find("Type0",0)<string::npos);

//   GetSignalDefinition(signals);

   string Input = InputPattern + "DumpHistos.root";
   string outpath = InputPattern;
   MakeDirectories(outpath);


   TFile* InputFile = new TFile(Input.c_str());
   TH1D* Data_05_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05");
   Data_05_I = (TH1D*) Data_05_I->Rebin(200);

   TH1D* Data_E1_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta00to05");
   TH1D* Data_E2_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta05to10");
   TH1D* Data_E3_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta10to15");
   TH1D* Data_E4_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta15to20");
   TH1D* Data_E5_I  = (TH1D*)GetObjectFromPath(InputFile, "CutFinder_I_Data_SSHit05_Eta20to25");
   Data_E1_I = (TH1D*) Data_E1_I->Rebin(200);
   Data_E2_I = (TH1D*) Data_E2_I->Rebin(200);
   Data_E3_I = (TH1D*) Data_E3_I->Rebin(200);
   Data_E4_I = (TH1D*) Data_E4_I->Rebin(200);
   Data_E5_I = (TH1D*) Data_E5_I->Rebin(200);

   c1 = new TCanvas("c1","c1", 600, 600);
   Data_05_I->SetTitle("");
   Data_05_I->SetStats(kFALSE);
   Data_05_I->GetXaxis()->SetTitle(DiscrLeg);
   Data_05_I->GetYaxis()->SetTitle("arbitrary units");
   Data_05_I->SetLineWidth(2);
   Data_05_I->SetLineColor(Color[5]);
   Data_05_I->SetMarkerColor(Color[5]);
   Data_05_I->SetMarkerStyle(Marker[5]);
   Data_05_I->Draw("HIST E1");
   Data_05_I->Draw("E1 same");
   Data_E1_I->SetTitle("");
   Data_E1_I->SetStats(kFALSE);
   Data_E1_I->GetXaxis()->SetTitle(DiscrLeg);
   Data_E1_I->GetYaxis()->SetTitle("arbitrary units");
   Data_E1_I->SetLineWidth(2);
   Data_E1_I->SetLineColor(Color[0]);
   Data_E1_I->SetMarkerColor(Color[0]);
   Data_E1_I->SetMarkerStyle(Marker[0]);
   Data_E1_I->SetAxisRange(0,0.35,"X");
   Data_E1_I->Draw("E1 same");
   Data_E2_I->SetLineWidth(2);
   Data_E2_I->SetLineColor(Color[1]);
   Data_E2_I->SetMarkerColor(Color[1]);
   Data_E2_I->SetMarkerStyle(Marker[1]);
   Data_E2_I->Draw("E1 same");
   Data_E3_I->SetLineWidth(2);
   Data_E3_I->SetLineColor(Color[2]);
   Data_E3_I->SetMarkerColor(Color[2]);
   Data_E3_I->SetMarkerStyle(Marker[2]);
   Data_E3_I->Draw("E1 same");
   Data_E4_I->SetLineWidth(2);
   Data_E4_I->SetLineColor(Color[3]);
   Data_E4_I->SetMarkerColor(Color[3]);
   Data_E4_I->SetMarkerStyle(Marker[3]);
   Data_E4_I->Draw("E1 same");
   Data_E5_I->SetLineWidth(2);
   Data_E5_I->SetLineColor(Color[4]);
   Data_E5_I->SetMarkerColor(Color[4]);
   Data_E5_I->SetMarkerStyle(Marker[4]);
   Data_E5_I->Draw("E1 same");

   c1->SetLogy(true);

   leg = new TLegend(0.80,0.93,0.80 - 0.30,0.93 - 6*0.05);
   if(IsTrackerOnly){ 
      leg->SetHeader("Tracker - Only");
   }else{
      leg->SetHeader("Tracker + Muon");
   }
   leg->SetFillColor(0);
   leg->SetBorderSize(0);
   leg->AddEntry(Data_05_I, "05 dE/dx Hits"   ,"LP");
   leg->AddEntry(Data_E1_I, "0.0 < |#eta| < 0.5"   ,"P");
   leg->AddEntry(Data_E2_I, "0.5 < |#eta| < 1.0"   ,"P");
   leg->AddEntry(Data_E3_I, "1.0 < |#eta| < 1.5"   ,"P");
   leg->AddEntry(Data_E4_I, "1.5 < |#eta| < 2.0"   ,"P");
   leg->AddEntry(Data_E5_I, "2.0 < |#eta| < 2.5"   ,"P");

   leg->Draw();

   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1, outpath, "CheckHitSplitSloap_IDistribution");
   delete c1;

}




int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}




