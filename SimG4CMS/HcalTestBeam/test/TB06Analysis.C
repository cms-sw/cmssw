///////////////////////////////////////////////////////////////////////////////
//
// Analysis script to compare energy distribution of TB06 data with MC
//
// TB06Analysis.C        Class to run over the tree created by TB06Analysis
//                       within the framewok of CMSSW. This class produces
//                       histogram file for trigger scintillators as well
//                       as for energy in EB, HB and combined energy for
//                       all tracks and tracks which are MIP in ECAL. To
//                       run this, us the 2 following steps:
// TB06Analysis t(infile, outfile, particle, energy, model, corrEB, corrHB);
// t.Loop()
//
// where
//   infile    string    Name of the input ROOT tree file 
//   outfile   string    Name of the output ROOT histogram file
//   particle  int       particle number (0: pi-, 1: pi+, 2: K-, 3: K+,
//                                        4: p, 5: pbar)
//   energy    int       signifies the beam momentum (a value between 0 and
//                       14 to get momenta 2, 3, 4, 5, 6, 7, 8, 9, 20, 30,
//                       50, 100, 150, 200, 300, 350)
//   model     int       signifies Geant4 Physics list + Geant4 version 
//                       (a value between 0 and 4 for the following lists
//                        0 10.0.p2 QGSP-FTFP-BERT-EML
//                        1 10.0.p2 FTFP-BERT-EML
//                        2 10.0.p2 QGSP-FTFP-BERT-EMM
//                        3 10.0.p2 FTFP-BERT-EMM
//                        4 10.0.p2 FTFP-BERT-ATL-EMM)
//   corrEB    double     Correction to noise factor for EB (1.0)
//   corrHB    double     Correction to noise factor for HB (1.0)
//
//  There are several auxiliary functions to make fits/plots:
//
//  DrawTrigger(std::string infile)  
//                        Plots energy distribution in the trigger counters
//                        present in the histogram file *infile*
//  FitTrigger(std::string infile, int type)
//                        Fits Landau function to the energy distribution in
//                        the trigger counter indicated by *type* (0 S1, 1 S2,
//                        2 S3, 3 S4) from the histogram file *infile*
//  FitEBHB(std::string infile, int type)
//                        Fits Crystal Ball (Gaussian) function to energy
//                        deposit in EB (HB) as indicated by *type* : 0 (1)
//                        from the histogram file *infile*
//  plotDataMC(int particle, int energy, int models, int type)
//                        Compares energy measured for all (MIPs in EB) 
//                        tracks as indicated by *type* : 1 (2) for 
//                        *particle* (a value between 0 and 5), *energy*
//                        (a value between 0 and 14), number of models
//                        (packed word up to 5 digit with values 0/1 in
//                        each digit stating a given model is included
//                        or not. For example 1011 means model 0, 1, 3
//                        are selected). It assumes the data files are
//                        in directory TB06, MC files are in directory
//                        modelX (X having a value between 0 and 4)
//  DrawHist(std::string dirName, int particle, int energy)
//                        Makes plot of energy distribution for particle
//                        as given by *particle* with beam momentum as
//                        given by *energy* from a file in directory
//                        *dirName*
///////////////////////////////////////////////////////////////////////////////
#include "TCanvas.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TH1D.h"
#include "TH2.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMinuit.h"
#include "TMath.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TStyle.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Header file for the classes stored in the TTree if any.

class TB06Analysis {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Double_t        eBeam_;
  Double_t        etaBeam_;
  Double_t        phiBeam_;
  Double_t        edepEC_;
  Double_t        edepHB_;
  Double_t        edepHO_;
  Double_t        noiseEC_;
  Double_t        noiseHB_;
  Double_t        noiseHO_;
  Double_t        edepS1_;
  Double_t        edepS2_;
  Double_t        edepS3_;
  Double_t        edepS4_;
  Double_t        edepVC_;

  // List of branches
  TBranch        *b_eBeam_;   //!
  TBranch        *b_etaBeam_;   //!
  TBranch        *b_phiBeam_;   //!
  TBranch        *b_edepEC_;   //!
  TBranch        *b_edepHB_;   //!
  TBranch        *b_edepHO_;   //!
  TBranch        *b_noiseEC_;   //!
  TBranch        *b_noiseHB_;   //!
  TBranch        *b_noiseHO_;   //!
  TBranch        *b_edepS1_;   //!
  TBranch        *b_edepS2_;   //!
  TBranch        *b_edepS3_;   //!
  TBranch        *b_edepS4_;   //!
  TBranch        *b_edepVC_;   //!

  TB06Analysis(std::string infname, std::string outfname, int ipar, int ien,
	       int model, double corrEB=1, double corrHB=1);
  virtual ~TB06Analysis();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  std::string      outName_, modName_, partNam_, partFil_;
  double           scaleEB_, scaleHB_, mipCut_, cutS4_, cutVC_;
  double           xmin_, xmax_, delx_, corrEB_, corrHB_;
  int              model_, nbin_, iene_;
};

TB06Analysis::TB06Analysis(std::string inName, std::string outName, int ipar,
			   int ien, int model, double corrEB,
			   double corrHB) : fChain(0), outName_(outName),
					    corrEB_(corrEB), corrHB_(corrHB) {

  TFile      *file = new TFile(inName.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny("testbeam");
  TTree      *tree = (TTree*)dir->Get("TB06Sim");
  Init(tree);
  double scaleEB[5] = { 1.010,  1.011,  1.011,  1.011,  1.011};
  double scaleHB[5] = {114.13, 114.29, 106.61, 106.54, 106.33};
  double cutS4[5]   = {0.0028, 0.0028, 0.0028, 0.0028, 0.0028};
  double cutVC[5]   = {0.0014, 0.0014, 0.0014, 0.0014, 0.0014};
  std::string modelNames[5] = {"10.0.2 QGSP_FTFP_BERT_EML",
			       "10.0.2 FTFP_BERT_EML",
			       "10.2.2 QGSP_FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_ATL_EMM"};
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","p","pbar"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  double      xmaxh[16] = {12,20,20,25,25,25,30,35,50,60,100,200,250,350,450,500};
  double      delxs[16] = {.1,.1,.1,.1,.1,.25,.25,.25,.5,.5,1.,1.,2.,2.,2.,2.};

  if (model < 0 || model >= 5) model_ = 0;
  else                         model_ = model;
  if (ien < 0 || ien >= 16)    ien    = 0;
  if (ipar < 0 || ipar >= 6)   ipar   = 0;
  scaleEB_ = scaleEB[model_];
  scaleHB_ = scaleHB[model_];
  modName_ = modelNames[model_];
  mipCut_  = 1.20;
  cutS4_   = cutS4[model_];
  cutVC_   = cutVC[model_];
  xmin_    = -5.0;
  xmax_    = xmaxh[ien];
  delx_    = delxs[ien];
  nbin_    = (int)((xmax_-xmin_)/delx_);
  iene_    = iens[ien];
  partFil_ = partsF[ipar];
  partNam_ = partsN[ipar];
}

TB06Analysis::~TB06Analysis() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t TB06Analysis::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t TB06Analysis::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void TB06Analysis::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("eBeam_", &eBeam_, &b_eBeam_);
  fChain->SetBranchAddress("etaBeam_", &etaBeam_, &b_etaBeam_);
  fChain->SetBranchAddress("phiBeam_", &phiBeam_, &b_phiBeam_);
  fChain->SetBranchAddress("edepEC_", &edepEC_, &b_edepEC_);
  fChain->SetBranchAddress("edepHB_", &edepHB_, &b_edepHB_);
  fChain->SetBranchAddress("edepHO_", &edepHO_, &b_edepHO_);
  fChain->SetBranchAddress("noiseEC_", &noiseEC_, &b_noiseEC_);
  fChain->SetBranchAddress("noiseHB_", &noiseHB_, &b_noiseHB_);
  fChain->SetBranchAddress("noiseHO_", &noiseHO_, &b_noiseHO_);
  fChain->SetBranchAddress("edepS1_", &edepS1_, &b_edepS1_);
  fChain->SetBranchAddress("edepS2_", &edepS2_, &b_edepS2_);
  fChain->SetBranchAddress("edepS3_", &edepS3_, &b_edepS3_);
  fChain->SetBranchAddress("edepS4_", &edepS4_, &b_edepS4_);
  fChain->SetBranchAddress("edepVC_", &edepVC_, &b_edepVC_);
  Notify();

}

Bool_t TB06Analysis::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void TB06Analysis::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t TB06Analysis::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void TB06Analysis::Loop() {
  //   In a ROOT session, you can do:
  //      root> .L TB06Analysis.C
  //      root> TB06Analysis t
  //      root> t.GetEntry(12); // Fill t data members with entry number 12
  //      root> t.Show();       // Show values of entry 12
  //      root> t.Show(16);     // Read and show values of entry 16
  //      root> t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch

  if (fChain == 0) return;
  TFile *fout = new TFile(outName_.c_str(), "RECREATE");
  TH1D  *h_es[5], *h_EC, *h_HB, *h_EN1, *h_EN2;
  std::string names[5] = {"h1", "h2", "h3", "h4", "h5"};
  std::string title[5] = {"Energy in S1", "Energy in S2", "Energy in S3",
			  "Energy in S4", "Energy in VC"};
  for (int i=0; i<5; ++i) {
    h_es[i] = new TH1D(names[i].c_str(), title[i].c_str(), 1000, 0, 0.02);
    h_es[i]->GetXaxis()->SetTitle("Energy (GeV)");
    h_es[i]->GetYaxis()->SetTitle("Events");
  }
  h_EC = new TH1D("EC", "Energy in EB", 2000, 0, 100.0);
  h_EC->GetXaxis()->SetTitle("Energy (GeV)");
  h_EC->GetYaxis()->SetTitle("Events");
  h_HB = new TH1D("HB", "Energy in HB", 2000, 0, 1.0);
  h_HB->GetXaxis()->SetTitle("Energy (GeV)");
  h_HB->GetYaxis()->SetTitle("Events");
  char nameh[100], titlh[100];
  sprintf (nameh, "EN1%s%d", partFil_.c_str(), model_);
  sprintf (titlh, "Total Energy (%d GeV/c %s)", iene_, partNam_.c_str());
  h_EN1 = new TH1D(nameh, titlh, nbin_, xmin_, xmax_);
  h_EN1->GetXaxis()->SetTitle("Energy (GeV)");
  h_EN1->GetYaxis()->SetTitle("Events");
  sprintf (nameh, "EN2%s%d", partFil_.c_str(), model_);
  sprintf (titlh, "Total Energy (MIP in ECAL %d GeV/c %s)", iene_, partNam_.c_str());
  h_EN2 = new TH1D(nameh, titlh, nbin_, xmin_, xmax_);
  h_EN2->GetXaxis()->SetTitle("Energy (GeV)");
  h_EN2->GetYaxis()->SetTitle("Events");

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    h_es[0]->Fill(edepS1_);
    h_es[1]->Fill(edepS2_);
    h_es[2]->Fill(edepS3_);
    h_es[3]->Fill(edepS4_);
    h_es[4]->Fill(edepVC_);
    if (edepS4_ < cutS4_ && edepVC_ < cutVC_) {
      h_EC->Fill(edepEC_);
      h_HB->Fill(edepHB_);
      double enEB = scaleEB_*(edepEC_+corrEB_*noiseEC_);
      double enHB = scaleHB_*edepHB_+corrHB_*noiseHB_;
      double eTot = enEB+enHB;
      h_EN1->Fill(eTot);
      if (enEB < mipCut_) h_EN2->Fill(eTot);
    }
  }

  fout->cd(); fout->Write(); fout->Close();
}

TCanvas* DrawTrigger(std::string fileName) {
  std::string names[5]  = {"h1", "h2", "h3", "h4", "h5"};
  std::string title[5]  = {"S1", "S2", "S3", "S4", "Veto Counter"};
  int         colors[6] = {2,6,4,1,7,9};
  int         mtype[6]  = {20,21,22,23,24,33};

  TCanvas* pad(0);
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(0);
  TFile* file = new TFile(fileName.c_str());
  if (file) {  
    bool first(true);
    double dy  = 0.06;
    double yh  = 0.90;
    double yh1 = yh-5*dy - 0.01;
    pad = new TCanvas("Trigger","TriggerCounter", 700, 500);
    pad->SetLogy();
    TLegend *legend = new TLegend(0.65, yh1-0.20, 0.90, yh1);
    legend->SetFillColor(kWhite);
    for (int i=0; i<5; ++i) {
      TH1D *hist = (TH1D*)file->FindObjectAny(names[i].c_str());
      if (hist) {
	hist->SetLineColor(colors[i]);
	hist->SetMarkerColor(colors[i]);
	hist->SetMarkerStyle(mtype[i]);
	hist->GetXaxis()->SetTitle("Energy (GeV)");
	hist->GetYaxis()->SetTitle("Events");
	hist->GetYaxis()->SetLabelOffset(0.005);
	hist->GetYaxis()->SetTitleOffset(1.20);
	if (first) hist->Draw("");
	else       hist->Draw("sames");
	first = false;
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetFillColor(kWhite);
	  st1->SetLineColor(colors[i]);
	  st1->SetTextColor(colors[i]);
	  st1->SetY1NDC(yh-dy); st1->SetY2NDC(yh);
	  st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
	  yh -= dy;
	}
	legend->AddEntry(hist,title[i].c_str(),"lp");
      }
    }
    legend->Draw("same");
    pad->Update();
  }
  return pad;
}

Double_t langaufun(Double_t *x, Double_t *par) {

  //Fit parameters:
  //par[0]=Width (scale) parameter of Landau density
  //par[1]=Most Probable (MP, location) parameter of Landau density
  //par[2]=Total area (integral -inf to inf, normalization constant)
  //par[3]=Width (sigma) of convoluted Gaussian function
  //
  //In the Landau distribution (represented by the CERNLIB approximation), 
  //the maximum is located at x=-0.22278298 with the location parameter=0.
  //This shift is corrected within this function, so that the actual
  //maximum is identical to the MP parameter.

  // Numeric constants
  Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
  Double_t mpshift  = -0.22278298;       // Landau maximum location
  
  // Control constants
  Double_t np = 100.0;      // number of convolution steps
  Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

  // Variables
  Double_t xx;
  Double_t mpc;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow,xupp;
  Double_t step;
  Double_t i;
  Double_t par0=0.2;

  // MP shift correction
  mpc = par[1] - mpshift * par0 * par[1]; 

  // Range of convolution integral
  xlow = x[0] - sc * par[3];
  xupp = x[0] + sc * par[3];

  step = (xupp-xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for(i=1.0; i<=np/2; i++) {
    xx = xlow + (i-.5) * step;
    fland = TMath::Landau(xx,mpc,par0*par[1], kTRUE); // / par[0];
    sum += fland * TMath::Gaus(x[0],xx,par[3]);

    xx = xupp - (i-.5) * step;
    fland = TMath::Landau(xx,mpc,par0*par[1], kTRUE); // / par[0];
    sum += fland * TMath::Gaus(x[0],xx,par[3]);
  }

  return (par[2] * step * sum * invsq2pi / par[3]);
}

Double_t crystalfun(Double_t *x, Double_t *par) {

  // Variables
  Double_t xx    = x[0] - par[1];
  Double_t sigma = par[2];
  Double_t an    = par[3]; // 6.45
  Double_t alpha = par[4]; // 0.91
  
  double crystal = 0;
  if (xx > -alpha*sigma) {
    crystal = par[0]*std::exp(-0.5*(xx/sigma)*(xx/sigma));
  } else {
    Double_t den = (1.0-(alpha/an)*(xx/sigma) - (alpha*alpha)/an);
    if (den > 0) {
      Double_t den1 = an*std::log(den);
      crystal       = par[0]*std::exp(-0.5*alpha*alpha)/std::exp(den1);
    }
  }
  return crystal;
}

TF1 *functionFit(TH1D *his, double *fitrange, double *startvalues, 
		 double *parlimitslo, double *parlimitshi, int mode) {

  char FunName[100];
  std::string fname("LanGau");
  if (mode < 0) fname = "CrysBall";
  sprintf(FunName,"%s_%s",fname.c_str(), his->GetName());
  
  TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
  if (ffitold) delete ffitold;

  TF1 *ffit;
  int npar=4;
  if (mode >=0) {
    ffit = new TF1(FunName,langaufun,fitrange[0],fitrange[1],npar);
  } else {
    npar = 5;
    ffit = new TF1(FunName,crystalfun,fitrange[0],fitrange[1],npar);
  }
  ffit->SetParameters(startvalues);
  if (mode >=0) {
    ffit->SetParNames("Width","MP","Area","GSigma");
  } else {
    ffit->SetParNames("Area","Mean","Width","AN","Alpha");
  }
  if (mode == 0) {
    for (int i=0; i<npar; i++) 
      ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
  }
  his->Fit(FunName,"wwqRB0");   // fit within specified range, use ParLimits, do not plot
  
  return (ffit);              // return fit function
}

TCanvas* FitTrigger(std::string fileName, int type) {
  std::string names[5]  = {"h1", "h2", "h3", "h4", "h5"};
  std::string title[5]  = {"S1", "S2", "S3", "S4", "Veto Counter"};
  int         colors[6] = {2,6,4,1,7,9};
  int         mtype[6]  = {20,21,22,23,24,33};

  TCanvas* pad(0);
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(10);
  TFile* file = new TFile(fileName.c_str());
  if (file) {
    double dy  = 0.15;
    double yh  = 0.90;
    double yh1 = yh-dy-0.01;
    TH1D *hist = (TH1D*)file->FindObjectAny(names[type].c_str());
    if (hist) {
      char name[100];
      sprintf(name,"TriggerFit%d",type);
      pad = new TCanvas(name,name,700,500);
      TLegend *legend = new TLegend(0.65, yh1-0.05, 0.90, yh1);
      legend->SetFillColor(kWhite);
      hist->SetLineColor(colors[type]);
      hist->SetMarkerColor(colors[type]);
      hist->SetMarkerStyle(mtype[type]);
      hist->GetXaxis()->SetTitle("Energy (GeV)");
      hist->GetYaxis()->SetTitle("Events");
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      double LowEdge(0.001);
      double HighEdge(0.005);
      TF1 *Fitfun = new TF1(name,"landau",LowEdge,HighEdge);
      TFitResultPtr Fit = hist->Fit(Fitfun, "+0wwqR");
      hist->GetXaxis()->SetRangeUser(0, 0.005);
      hist->Draw("");
      Fitfun->Draw("same");
      //      hist->Fit("landau", "wwqRS", "", LowEdge, HighEdge);
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetFillColor(kWhite);
	st1->SetLineColor(colors[type]);
	st1->SetTextColor(colors[type]);
	st1->SetY1NDC(yh-dy); st1->SetY2NDC(yh);
	st1->SetX1NDC(0.60); st1->SetX2NDC(0.90);
      }
      legend->AddEntry(hist,title[type].c_str(),"lp");
      legend->Draw("same");
      pad->Update();
    }
  }
  return pad;
}

TCanvas* FitEBHB(std::string fileName, int type) {
  std::string names[2]  = {"EC", "HB"};
  std::string title[2]  = {"Energy in EB", "Energy in HB"};
  int         colors[6] = {2,6,4,1,7,9};
  int         mtype[6]  = {20,21,22,23,24,33};

  TCanvas* pad(0);
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(10);
  TFile* file = new TFile(fileName.c_str());
  if (file) {
    double dy  = 0.20;
    double yh  = 0.90;
    double yh1 = yh-dy-0.01;
    double xh  = (type == 0) ? 0.40 : 0.90;
    TH1D *hist = (TH1D*)file->FindObjectAny(names[type].c_str());
    if (hist) {
      char name[100];
      sprintf(name,"%sFit",names[type].c_str());
      pad = new TCanvas(name,name,700,500);
      hist->SetLineColor(colors[type]);
      hist->SetMarkerColor(colors[type]);
      hist->SetMarkerStyle(mtype[type]);
      hist->GetXaxis()->SetTitle("Energy (GeV)");
      hist->GetYaxis()->SetTitle("Events");
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      double LowEdge  = hist->GetMean() - 0.8*hist->GetRMS();
      double HighEdge = hist->GetMean() + 2.*hist->GetRMS();
      TFitResultPtr Fit = hist->Fit("gaus", "+0wwqRS", "", LowEdge, HighEdge);
      std::cout << Fit->Value(0) << " 1 " << Fit->Value(1) << " 2 " << Fit->Value(2) << std::endl;
      double startvalues[5], fitrange[2], lowValue[5], highValue[5];
      startvalues[0] = Fit->Value(0); lowValue[0] = 0.0; highValue[0] = 10.0*startvalues[0];
      startvalues[1] = Fit->Value(1); lowValue[1] = 0.5*startvalues[1]; highValue[1] = 2.*startvalues[1];
      startvalues[2] = Fit->Value(2); lowValue[2] = 0.5*startvalues[2]; highValue[2] = 2.*startvalues[2];
      startvalues[3] = 6.45;              lowValue[3] = 0.5*startvalues[3]; highValue[3] = 2.*startvalues[3];
      startvalues[4] = 0.91;              lowValue[4] = 0.5*startvalues[4]; highValue[4] = 2.*startvalues[4];
      TF1 *Fitfun;
      if (type == 0) {
	fitrange[0] = startvalues[1]-10.*startvalues[2];
	fitrange[1] = startvalues[1]+2.*startvalues[2];
	Fitfun = functionFit(hist, fitrange, startvalues, lowValue,highValue,-1);
      } else {
	fitrange[0] = startvalues[1]-10.*startvalues[2];
	fitrange[1] = startvalues[1]+10.*startvalues[2];
	Fitfun = new TF1(name,"gaus",fitrange[0],fitrange[1]);
	Fit = hist->Fit(Fitfun, "+0wwqR");
      }
      std::cout << LowEdge << " " << startvalues[1]-2.*startvalues[3] << " " << HighEdge << " " << fitrange[1] << std::endl;
      double low  = fitrange[0]-2.*startvalues[2]; 
      double high = fitrange[1]+2.*startvalues[2];
      hist->GetXaxis()->SetRangeUser(low, high);
      hist->Draw("");
      Fitfun->Draw("same");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetFillColor(kWhite);
	st1->SetLineColor(colors[type]);
	st1->SetTextColor(colors[type]);
	st1->SetY1NDC(yh-dy);  st1->SetY2NDC(yh);
	st1->SetX1NDC(xh-0.3); st1->SetX2NDC(xh);
      }
      TLegend *legend = new TLegend(xh-0.25, yh1-0.05, xh, yh1);
      legend->SetFillColor(kWhite);
      legend->AddEntry(hist,title[type].c_str(),"lp");
      legend->Draw("same");
      pad->Update();
    }
  }
  return pad;
}


TCanvas* plotDataMC(int ipar, int ien, int models, int type) {

  std::string modelNames[5] = {"10.0.2 QGSP_FTFP_BERT_EML",
			       "10.0.2 FTFP_BERT_EML",
			       "10.2.2 QGSP_FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_ATL_EMM"};
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsM[6] = {"pim","pip","km","kp","prop","prom"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","p","pbar"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  double      xmaxh[16] = {12,20,20,25,25,25,30,35,50,60,100,200,250,350,450,500};
  int         colors[6] = {2,4,6,7,9,1};
  int         mtype[6]  = {21,22,23,24,33,20};
  std::string titlty[2] = {"All events", "MIP in ECAL"};

  TCanvas* pad(0);
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(10);
  if (type != 2) type = 1;
  std::vector<int>   icol;
  std::vector<TH1D*> hists;

  //Get some model information
  char infile[120], title[100], name[120];
  double xmin(-5.0), xmax(xmaxh[ien]), ymax(0);
  int    nbin(0), nmod(0), model(models);
  for (int i=0; i<5; ++i) {
    if (model%10 > 0) {
      sprintf (infile,"model%d/%s%d.root",i,partsM[ipar].c_str(),iens[ien]);
      TFile *file = new TFile(infile);
      if (file) {
	sprintf (name,"EN%d%s1",type,partsF[ipar].c_str());
	TH1D* h1 = (TH1D*)file->FindObjectAny(name);
	if (h1) {
	  nmod++;
	  if (nbin == 0) {
	    nbin = h1->GetNbinsX();
	    xmin = h1->GetBinLowEdge(1);
	    xmax = h1->GetBinLowEdge(nbin) + h1->GetBinWidth(nbin);
	  }
	}
	file->Close();
      }
    }
    model /= 10;
  }

  //Data first
  TH1D *hist(0);
  std::cout << nbin << ":" << xmin << ":" << xmax << std::endl;
  sprintf (infile, "TB06/%s%d.txt", partsF[ipar].c_str(), iens[ien]);
  sprintf (title, "%d GeV %s", iens[ien], partsN[ipar].c_str());
  sprintf (name, "Data%d%s%d", type, partsF[ipar].c_str(), iens[ien]);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    hist = new TH1D(name, title, nbin, xmin, xmax);
    hist->Sumw2();
    hist->GetXaxis()->SetTitle("Energy (GeV)");
    hist->GetYaxis()->SetTitle("Events");
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->SetMarkerColor(colors[5]);
    hist->SetMarkerStyle(mtype[5]);
    hist->SetLineColor(colors[5]);
    char buffer [1024];
    fInput.getline(buffer, 1024);
    std::cout << buffer << std::endl;
    double eEB, eHB, eHO;
    while (1) {
      fInput >> eEB >> eHB >> eHO;
      if (!fInput.good()) break;
      double eTot = eEB+eHB;
      if      (type == 1) hist->Fill(eTot);
      else if (eEB < 1.2) hist->Fill(eTot);
    }
    fInput.close();
    for (int k=1; k<=nbin; ++k) {
      if (hist->GetBinContent(k)+hist->GetBinError(k) > ymax) {
	ymax = hist->GetBinContent(k)+hist->GetBinError(k);
      }
    }
    hists.push_back(hist);
    icol.push_back(colors[5]);
  }

  //Now loop of MC files
  if (hist) {
    double total(0);
    for (int k=1; k<=nbin; ++k) total += hist->GetBinContent(k);
    sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    pad = new TCanvas(name, name, 700, 700);
    double dy  = 0.08;
    double xh(0.90), yh(0.90);
    double yh1 = yh-1.5*dy*(nmod+1)- 0.01;
    TLegend *legend = new TLegend(xh-0.30, yh1-0.04*(nmod+1), xh, yh1);
    legend->SetFillColor(kWhite);

    sprintf (name,"ENStack1%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    THStack *Hs = new THStack(name,"");
    Hs->Add(hist,"pe sames");
    sprintf (title, "%d GeV %s (%s)", iens[ien], partsN[ipar].c_str(), titlty[type-1].c_str());
    legend->AddEntry(hist,title,"lp");

    int model(models);
    for (int i=0; i<5; ++i) {
      if (model%10 > 0) {
	sprintf (infile,"model%d/%s%d.root",i,partsM[ipar].c_str(),iens[ien]);
	TFile *file = new TFile(infile);
	if (file) {
	  sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),i);
	  TH1D* h1 = (TH1D*)file->FindObjectAny(name);
	  if (h1) {
	    sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),i);
	    TH1D *h2 = (TH1D*)h1->Clone(name);
	    double totm(0);
	    for (int k=1; k<=h2->GetNbinsX(); ++k) totm += h2->GetBinContent(k);
	    double scale = total/totm;
	    for (int k=1; k<=h2->GetNbinsX(); ++k) {
	      double cont = scale*h2->GetBinContent(k);
	      h2->SetBinContent(k,cont);
	      if (cont > ymax) {
		ymax = cont;
	      }
	    }
	    h2->SetMarkerColor(colors[i]);
	    h2->SetMarkerStyle(mtype[i]);
	    h2->SetLineColor(colors[i]);
	    std::cout << i << " max " << ymax << std::endl;
	    Hs->Add(h2,"sames");
	    legend->AddEntry(h2,modelNames[i].c_str(),"lp");
	    icol.push_back(colors[i]);
	    hists.push_back(h2);
	  }
	}
      }
      model /= 10;
    }

    int    imax = (int)(0.01*ymax);
    double ymx  = 100*(imax+1);
    std::cout << "Ymax " << ymax << " " << imax << " " << ymx << std::endl;
    Hs->SetMinimum(0.0); Hs->SetMaximum(ymx);
    sprintf (name,"ENPad1%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    TPad *pad1 = new TPad(name,"pad1",0,0.3,1,1);
    pad1->SetBottomMargin(0.01); pad1->SetTopMargin(0.10);
    pad1->SetRightMargin(0.10);
    pad1->Draw(); pad1->cd();
    Hs->Draw("nostack");
    Hs->GetHistogram()->GetYaxis()->SetTitle("Events");
    pad1->Update(); pad1->Modified();

    for (unsigned int k=0; k<hists.size(); ++k) {
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetFillColor(kWhite);
	st1->SetLineColor(icol[k]); st1->SetTextColor(icol[k]);
	st1->SetY1NDC(yh-dy);  st1->SetY2NDC(yh);
	st1->SetX1NDC(xh-0.3); st1->SetX2NDC(xh);
	yh -= dy;
      }
    }
    legend->Draw("same");

    pad->cd();
    sprintf (name,"ENStack2%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    THStack *Hsr = new THStack(name,"");
    Hsr->SetMinimum(0.0); Hsr->SetMaximum(3.99);
    sprintf (name,"ENPad2%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    TPad *pad2  = new TPad(name,"pad2",0,0,1,0.3);
    pad2->SetBottomMargin(0.20); pad2->SetTopMargin(0.005);
    pad2->SetRightMargin(0.10);
    pad2->Draw(); pad2->cd();
    double yh2(0.99);
    TLegend *legend1 = new TLegend(xh-0.30, yh2-0.16*nmod, xh, yh2);
    legend1->SetFillColor(kWhite);
    TH1D *h_ref =  (TH1D*)hists[0];
    for (unsigned int i=1; i<hists.size(); i++) {
      sprintf (name,"ENRatio%d%s%d",i,partsF[ipar].c_str(),iens[ien]);
      TH1D *ratio = new TH1D(name, "Ratio", nbin, xmin, xmax);
      double sumNum(0), sumDen(0);
      for (int k=1; k<=nbin; ++k) {
	if (h_ref->GetBinContent(k) > 10 && hists[i]->GetBinContent(k) > 10) {
	  double rat = h_ref->GetBinContent(k)/hists[i]->GetBinContent(k);
	  double drt = h_ref->GetBinError(k)/hists[i]->GetBinContent(k);
	  ratio->SetBinContent(k,rat); ratio->SetBinError(k,drt);
	  if (rat > 1.) {
	    rat = 1./rat; drt *= (rat*rat);
	  }
	  sumNum += (fabs(1.0-rat)/(drt*drt));
	  sumDen += (1.0/(drt*drt));
	}
      }
      double mean  = (sumDen>0) ? (sumNum/sumDen) : 0;
      double error = (sumDen>0) ? 1.0/sqrt(sumDen) : 0;
      std::cout << "Delta " << mean << " +- " << error << std::endl;
      sprintf (name, "#delta = %6.3f #pm %6.3f", mean, error);
      legend1->AddEntry(ratio,name,"lp");
      ratio->SetStats(0);
      ratio->SetLineColor(icol[i]);
      ratio->SetMarkerColor(icol[i]);
      ratio->SetMarkerStyle(mtype[i]);
      ratio->SetLineWidth(2);
      Hsr->Add(ratio,"pe same");
    }
    Hsr->Draw("nostack");
    pad2->Update();
    Hsr->GetHistogram()->SetStats(0);
    Hsr->GetHistogram()->GetYaxis()->SetTitle("Ratio");
    Hsr->GetHistogram()->GetYaxis()->SetTitleSize(0.1);
    Hsr->GetHistogram()->GetYaxis()->SetTitleOffset(0.45);
    Hsr->GetHistogram()->GetYaxis()->SetLabelSize(0.10);
    Hsr->GetHistogram()->GetXaxis()->SetTitle("Energy (GeV)");
    Hsr->GetHistogram()->GetXaxis()->SetTitleSize(0.10);
    Hsr->GetHistogram()->GetXaxis()->SetLabelSize(0.10);
    Hsr->GetHistogram()->GetYaxis()->SetTitleOffset(0.35);
    TLine *line = new TLine(xmin,1.0,xmax,1.0);
    line->SetLineStyle(2); line->SetLineWidth(2);
    line->SetLineColor(kBlack); line->Draw();
    legend1->Draw("same");
    pad2->Update();
    pad->Update();
  }
  return pad;
}

bool FillHist(char* infile, TH1D* h_eEB, TH1D* h_eHB, TH1D* h_eTot) {

  bool flag(false);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(1), good(0);
    fInput.getline(buffer, 1024);
    std::cout << buffer << std::endl;
    double eEB, eHB, eHO;
    double eEBl(9999), eEBh(-9999), eHBl(9999), eHBh(-9999), eTotl(9999), eToth(-9999);
    while (1) {
      fInput >> eEB >> eHB >> eHO;
      if (!fInput.good()) break;
      all++; good++;
      double eTot = eEB+eHB;
      if (h_eEB  != 0) h_eEB->Fill(eEB);
      if (h_eHB  != 0) h_eHB->Fill(eHB);
      if (h_eTot != 0) h_eTot->Fill(eTot);
      if (eEB  < eEBl)  eEBl  = eEB;  if (eEB  > eEBh)  eEBh  = eEB;
      if (eHB  < eHBl)  eHBl  = eHB;  if (eHB  > eHBh)  eHBh  = eHB;
      if (eTot < eTotl) eTotl = eTot; if (eTot > eToth) eToth = eTot;
    }
    std::cout << "Reads " << all << " (" << good << ") records from "
	      << infile << std::endl << "Minimum/maximum for EB " << eEBl
	      << ":" << eEBh << "  HB " << eHBl << ":" << eHBh << "  Total "
	      << eTotl << ":" << eToth << std::endl;
    fInput.close();
    flag = (good>0);
  }
  return flag;
}

std::vector<TCanvas*> DrawHist(std::string dirName, int type, int ie) {
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","p","pbar"};
  std::string names[3]  = {"ECAL", "HCAL", "Total"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  double      xmine(-2), xminh(-5);
  double      xmaxe[16] = {3,4,4,6,6,8,8,10,45,45,100,200,200,350,350,500};
  double      xmaxh[16] = {12,20,20,25,25,25,30,35,50,60,100,200,250,350,450,500};
  double      delxs[16] = {.1,.1,.1,.1,.1,.25,.25,.25,.5,.5,1.,1.,2.,2.,2.,2.};

  std::vector<TCanvas*> pads;
  char infile[200], title[50], name[50];
  if (dirName == "") sprintf(infile,"%s%d.txt", partsF[type].c_str(), iens[ie]);
  else               sprintf(infile,"%s/%s%d.txt", dirName.c_str(), partsF[type].c_str(), iens[ie]);
  sprintf (title, "%d GeV %s", iens[ie], partsN[type].c_str());
  TH1D* hist[3];
  double xmin(xmine), xmax, delx;
  int    nbin(0);
  xmax = xmaxe[ie];
  delx = delxs[ie];
  nbin = (int)((xmax-xmin)/delx);
  sprintf (name, "%s%s%d", names[0].c_str(), partsF[type].c_str(), iens[ie]);
  hist[0] = new TH1D(name, title, nbin, xmin, xmax);
  xmin = xminh;
  xmax = xmaxh[ie];
  nbin = (int)((xmax-xmin)/delx);
  sprintf (name, "%s%s%d", names[1].c_str(), partsF[type].c_str(), iens[ie]);
  hist[1] = new TH1D(name, title, nbin, xmin, xmax);
  sprintf (name, "%s%s%d", names[2].c_str(), partsF[type].c_str(), iens[ie]);
  hist[2] = new TH1D(name, title, nbin, xmin, xmax);
  bool ok = FillHist(infile, hist[0], hist[1], hist[2]);

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(0);
  if (ok) {
    for (int k=0; k<3; ++k) {
      sprintf(name, "%s%s%d", names[k].c_str(), partsF[type].c_str(), iens[ie]);
      TCanvas* pad = new TCanvas(name, name, 700, 500);
      pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
      sprintf(name, "%s Energy (GeV)", names[k].c_str());
      hist[k]->GetXaxis()->SetTitle(name); 
      hist[k]->GetYaxis()->SetTitle("Events");
      hist[k]->GetYaxis()->SetTitleOffset(1.2);
      hist[k]->Draw(); pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetFillColor(0);
	st1->SetY1NDC(0.75); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.70); st1->SetX2NDC(0.90);
      }
      TPaveText *text = new TPaveText(0.70,0.70,0.90,0.745,"blNDC");
      text->SetFillColor(0); text->AddText(title);   text->Draw("same");
      pad->Modified();       pad->Update();
      pads.push_back(pad);
    }
  }
  return pads;
}

void DrawHistAll(std::string dirName) {
  char filename[100];
  for (int k1=0; k1<6; ++k1) {
    for (int k2=0; k2<16; ++k2) {
      std::vector<TCanvas*> pads = DrawHist(dirName,k1,k2);
      for (unsigned int k=0; k<pads.size(); ++k) {
	sprintf (filename, "%s.pdf", pads[k]->GetName());
	pads[k]->Print(filename);
      }
    }
  }
}
