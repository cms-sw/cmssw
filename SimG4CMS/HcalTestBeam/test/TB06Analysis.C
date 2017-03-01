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
//  DrawHistData(int particle, int energy, std::string dirName, bool rescale)
//                        Makes plots of energy distribution (in ECAL, HCAL
//                        and combined) for particle as given by *particle* 
//                        with beam momentum as given by *energy* from a file 
//                        in directory *dirName*. Possibility of rescaling
//                        energy to match with the mean energy for the given
//                        data point
//  DrawHistDataAll(std::string dirName, bool rescale, bool save)
//                        Plots energy distributions (in ECAL, HCAL, combined)
//                        for all available particles and energies from files
//                        in directory *dirName*. Possibility of rescaling
//                        energy to match with the mean energy for the given
//                        data point and saving the plots as pdf file
//  plotDataMC(int particle, int energy, int models, int type, int rebin,
//             int irtype, std::string prefix, bool approve, bool stat,
//             double xmin0, double xmax0, bool rescale)
//                        Compares energy measured for all (MIPs in EB) 
//                        tracks as indicated by *type* : 1 (2) for 
//                        *particle* (a value between 0 and 5), *energy*
//                        (a value between 0 and 14), number of models
//                        (packed word *models* up to 5 digit with values 
//                        0/1 in each digit stating a given model to be
//                        included or not. For example 1011 means model 0, 
//                        1, 3 are selected). It assumes the data files are
//                        in directory TB06, MC files are in directory
//                        *prefix*X (X having a value between 0 and 4).
//                        It produces raw comparison (irtype=1), ratio of
//                        MC/Data (irtype=2) or both (irtype=3) in ths same
//                        canvas. It can rebin the default histograms using
//                        *rebin* and ranges from *xmin0* to *xmax0* (if 
//                        xmax0 is negative, the original range of histograms
//                        is kept). It can rescale the energy in the data 
//                        histogram to match the mean. The legend and stat
//                        box are controlled using the flags *approve*, *stat*
//  plotDataMCDist(int particle, int energy, int models, int rebin,
//		   std::string prefix, bool approve, bool stat, double xmin0,
//                 double xmax0, int save)
//                        Makes comparison plots for approval and save them
//                        as pdf file if needed (*save* > 0). The definition
//                        of other parameters as for plotDataMC
//  plotDataMC(int particle, int models, bool ratio, std::string dirName, 
//             bool approve)
//                        Plots mean response for data and MC or the ratio of
//                        response (MC/Data if *ratio* is true) as a function
//                        of beam momentum for particle *particle*. The mean
//                        values are read out from directory *dirName* and 
//                        models to be plotted are given by *models* (the
//                        usage is described earlier). *approve* decides the
//                        content of legends in the plot
//  plotDataMCResp(std::string dirName, bool approve, int save)
//                        Plots mean response for data and MC and also the
//                        ratio MC/Data for all available particles. The
//                        legends are controlled by *approve* and saving
//                        the plots as pdf file is done if *save* > 0
///////////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TGraphAsymmErrors.h"
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
  Double_t        edepS7_;
  Double_t        edepS8_;

  // List of branches
  TBranch        *b_eBeam_;    //!
  TBranch        *b_etaBeam_;  //!
  TBranch        *b_phiBeam_;  //!
  TBranch        *b_edepEC_;   //!
  TBranch        *b_edepHB_;   //!
  TBranch        *b_edepHO_;   //!
  TBranch        *b_noiseEC_;  //!
  TBranch        *b_noiseHB_;  //!
  TBranch        *b_noiseHO_;  //!
  TBranch        *b_edepS1_;   //!
  TBranch        *b_edepS2_;   //!
  TBranch        *b_edepS3_;   //!
  TBranch        *b_edepS4_;   //!
  TBranch        *b_edepVC_;   //!
  TBranch        *b_edepS7_;   //!
  TBranch        *b_edepS8_;   //!

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
  double           scaleEB_, scaleHB_, mipCut_, cutS4_, cutVC_, cutS8_;
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
  double cutS8[5]   = {0.0014, 0.0014, 0.0014, 0.0014, 0.0014};
  std::string modelNames[5] = {"10.0.2 QGSP_FTFP_BERT_EML",
			       "10.0.2 FTFP_BERT_EML",
			       "10.2.2 QGSP_FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_EMM",
			       "10.2.2 FTFP_BERT_ATL_EMM"};
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","p","pbar"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  /*
  double      xminh[16] = {-5,-5,-5,-5,-5,-5,-5,-5,5,10,10,10,10,10,10,10};
  double      xmaxh[16] = {12,20,20,25,25,25,30,35,50,60,100,200,250,350,450,500};
  double      delxs[16] = {.1,.1,.1,.1,.1,.25,.25,.25,.5,.5,1.,1.,2.,2.,2.,2.};
  */
  double      xminh[16] = {-2,-2,-1,-1,-1,-1,-1,-1,2,6,10,40,70,90,120,140};
  double      xmaxh[16] = {6,9,11,14,15,17,18,20,30,40,70,130,170,250,370,420};
  double      delxs[16] = {.2,.2,.2,.2,.2,.2,.2,.2,.5,.5,1.,1.,2.,2.,2.,2.};

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
  cutS8_   = cutS8[model_];
  xmin_    = xminh[ien];
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

  fChain->SetBranchAddress("eBeam_",   &eBeam_,   &b_eBeam_);
  fChain->SetBranchAddress("etaBeam_", &etaBeam_, &b_etaBeam_);
  fChain->SetBranchAddress("phiBeam_", &phiBeam_, &b_phiBeam_);
  fChain->SetBranchAddress("edepEC_",  &edepEC_,  &b_edepEC_);
  fChain->SetBranchAddress("edepHB_",  &edepHB_,  &b_edepHB_);
  fChain->SetBranchAddress("edepHO_",  &edepHO_,  &b_edepHO_);
  fChain->SetBranchAddress("noiseEC_", &noiseEC_, &b_noiseEC_);
  fChain->SetBranchAddress("noiseHB_", &noiseHB_, &b_noiseHB_);
  fChain->SetBranchAddress("noiseHO_", &noiseHO_, &b_noiseHO_);
  fChain->SetBranchAddress("edepS1_",  &edepS1_,  &b_edepS1_);
  fChain->SetBranchAddress("edepS2_",  &edepS2_,  &b_edepS2_);
  fChain->SetBranchAddress("edepS3_",  &edepS3_,  &b_edepS3_);
  fChain->SetBranchAddress("edepS4_",  &edepS4_,  &b_edepS4_);
  fChain->SetBranchAddress("edepVC_",  &edepVC_,  &b_edepVC_);
  fChain->SetBranchAddress("edepS7_",  &edepS7_,  &b_edepS7_);
  fChain->SetBranchAddress("edepS8_",  &edepS8_,  &b_edepS8_);
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
  TH1D  *h_es[7], *h_EC, *h_HB, *h_EN1, *h_EN2;
  std::string names[7] = {"h1", "h2", "h3", "h4", "h5", "h6", "h7"};
  std::string title[7] = {"Energy in S1", "Energy in S2", "Energy in S3",
			  "Energy in S4", "Energy in VC", "Energy in S7",
			  "Energy in S8"};
  for (int i=0; i<7; ++i) {
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
  h_EN1->Sumw2();
  h_EN1->GetXaxis()->SetTitle("Energy (GeV)");
  h_EN1->GetYaxis()->SetTitle("Events");
  sprintf (nameh, "EN2%s%d", partFil_.c_str(), model_);
  sprintf (titlh, "Total Energy (MIP in ECAL %d GeV/c %s)", iene_, partNam_.c_str());
  h_EN2 = new TH1D(nameh, titlh, nbin_, xmin_, xmax_);
  h_EN2->Sumw2();
  h_EN2->GetXaxis()->SetTitle("Energy (GeV)");
  h_EN2->GetYaxis()->SetTitle("Events");

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  std::cout << "Correction Factors " << corrEB_ << ":" << corrHB_ << std::endl;

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
    h_es[5]->Fill(edepS7_);
    h_es[6]->Fill(edepS8_);
    if (edepS4_ < cutS4_ && edepVC_ < cutVC_ && edepS8_ < cutS8_) {
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
  std::string names[7]  = {"h1", "h2", "h3", "h4", "h5", "h6", "h7"};
  std::string title[7]  = {"S1", "S2", "S3", "S4", "Veto Counter", "MC1", "MC2"};
  int         colors[7] = {2,6,4,1,7,9,3};
  int         mtype[7]  = {20,21,22,23,24,33,25};

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
    for (int i=0; i<7; ++i) {
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
  std::string names[7]  = {"h1", "h2", "h3", "h4", "h5", "h6", "h7"};
  std::string title[7]  = {"S1", "S2", "S3", "S4", "Veto Counter", "MC1", "MC2"};
  int         colors[7] = {2,6,4,1,7,9,3};
  int         mtype[7]  = {20,21,22,23,24,33,25};

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

double GetScaleFactor(int type, int ie, std::string dirName="RespAll") {
  char infil1[200], infil2[200];
  std::string partsF[6] = {"pim","pip","km","kp","prop","prom"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  if (dirName == "") {
    sprintf(infil1,"%s.txt",  partsF[type].c_str());
    sprintf(infil2,"%sn.txt", partsF[type].c_str());
  } else {
    sprintf(infil1,"%s/%s.txt",  dirName.c_str(), partsF[type].c_str());
    sprintf(infil2,"%s/%sn.txt", dirName.c_str(), partsF[type].c_str());
  }
  double pbeam, resp, reso, scale(1.0), resp1(0), resp2(0);
  bool   ok(false);
  std::ifstream fInput1(infil1);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << infil1 << std::endl;
  } else {
    while (1) {
      fInput1 >> pbeam >> resp >> reso;
      if (!fInput1.good()) break;
      int ip = (int)(pbeam+0.01);
      if (ip == iens[ie]) {
	resp1 = resp; ok = true; break;
      }
    }
    fInput1.close();
  }
  if (ok) {
    std::ifstream fInput2(infil2);
    ok = false;
    if (!fInput2.good()) {
      std::cout << "Cannot open file " << infil2 << std::endl;
    } else {
      while (1) {
	fInput2 >> pbeam >> resp >> reso;
	if (!fInput2.good()) break;
	int ip = (int)(pbeam+0.01);
	if (ip == iens[ie]) {
	  resp2 = resp; ok = true; break;
	}
      }
      fInput2.close();
    }
  }
  if (ok) scale = resp1/resp2;
  std::cout << type << ":" << ie << ":" << iens[ie] << " response " << resp1 
	    << ":" << resp2 << ":" << scale << std::endl;
  return scale;
}

bool FillHistData(char* infile, TH1D* h_eEB, TH1D* h_eHB, TH1D* h_eTot,
		  double scale=1.0) {

  bool flag(false);
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(1), good(0);
    fInput.getline(buffer, 1024);
 // std::cout << buffer << std::endl;
    double eEB, eHB, eHO;
    double eEBl(9999), eEBh(-9999), eHBl(9999), eHBh(-9999), eTotl(9999), eToth(-9999);
    while (1) {
      fInput >> eEB >> eHB >> eHO;
      if (!fInput.good()) break;
      all++; good++;
      double eTot = eEB+eHB;
      if (h_eEB  != 0) h_eEB->Fill(scale*eEB);
      if (h_eHB  != 0) h_eHB->Fill(scale*eHB);
      if (h_eTot != 0) h_eTot->Fill(scale*eTot);
      if (eEB  < eEBl)  eEBl  = eEB;  if (eEB  > eEBh)  eEBh  = eEB;
      if (eHB  < eHBl)  eHBl  = eHB;  if (eHB  > eHBh)  eHBh  = eHB;
      if (eTot < eTotl) eTotl = eTot; if (eTot > eToth) eToth = eTot;
    }
    /*
    std::cout << "Reads " << all << " (" << good << ") records from "
	      << infile << std::endl << "Minimum/maximum for EB " << eEBl
	      << ":" << eEBh << "  HB " << eHBl << ":" << eHBh << "  Total "
	      << eTotl << ":" << eToth << std::endl;
    */
    fInput.close();
    flag = (good>0);
  }
  return flag;
}

std::vector<TCanvas*> DrawHistData(int type, int ie, std::string dirName="TB06",
				   bool rescale=false) {
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","p","pbar"};
  std::string names[3]  = {"ECAL", "HCAL", "Total"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  double      xmine(-2);
  double      xmaxe[16] = {3,4,4,6,6,8,8,10,45,45,100,200,200,350,350,500};
  /*
  double      xminh(-5);
  double      xmaxh[16] = {12,20,20,25,25,25,30,35,50,60,100,200,250,350,450,500};
  double      delxs[16] = {.1,.1,.1,.1,.1,.25,.25,.25,.5,.5,1.,1.,2.,2.,2.,2.};
  */
  double      xminh[16] = {-2,-2,-1,-1,-1,-1,-1,-1,2,6,10,40,70,90,120,140};
  double      xmaxh[16] = {6,9,11,14,15,17,18,20,30,40,70,130,170,250,370,420};
  double      delxs[16] = {.2,.2,.2,.2,.2,.2,.2,.2,.5,.5,1.,1.,2.,2.,2.,2.};

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
  hist[0]->Sumw2();
  xmin = xminh[ie];
  xmax = xmaxh[ie];
  nbin = (int)((xmax-xmin)/delx);
  sprintf (name, "%s%s%d", names[1].c_str(), partsF[type].c_str(), iens[ie]);
  hist[1] = new TH1D(name, title, nbin, xmin, xmax);
  hist[1]->Sumw2();
  sprintf (name, "%s%s%d", names[2].c_str(), partsF[type].c_str(), iens[ie]);
  hist[2] = new TH1D(name, title, nbin, xmin, xmax);
  hist[2]->Sumw2();
  double scale = rescale ? GetScaleFactor(type, ie) : 1.0;
  bool ok = FillHistData(infile, hist[0], hist[1], hist[2], scale);

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(0);
  if (ok) {
    double mean[3], rms[3];
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
      mean[k] = hist[k]->GetMean(); rms[k] = hist[k]->GetRMS();
      TPaveText *text = new TPaveText(0.70,0.70,0.90,0.745,"blNDC");
      text->SetFillColor(0); text->AddText(title);   text->Draw("same");
      pad->Modified();       pad->Update();
      pads.push_back(pad);
    }
    std::cout << partsF[type] << " " << iens[ie] << " GeV";
    for (int k=0; k<3; ++k) std::cout << " " << mean[k] << " " << rms[k];
    std::cout << " " << mean[2]/iens[ie] << " " << rms[2]/mean[2] << std::endl;
  }
  return pads;
}

void DrawHistDataAll(std::string dirName="TB06", bool rescale=false,
		     bool save=false) {
  char filename[100];
  for (int k1=0; k1<6; ++k1) {
    for (int k2=0; k2<16; ++k2) {
      std::vector<TCanvas*> pads = DrawHistData(k1,k2,dirName,rescale);
      if (save) {
	for (unsigned int k=0; k<pads.size(); ++k) {
	  sprintf (filename, "%s.pdf", pads[k]->GetName());
	  pads[k]->Print(filename);
	}
      }
    }
  }
}


TCanvas* plotDataMC(int ipar, int ien, int models, int type, int rebin=1,
		    int irtype=1, std::string prefix="model", 
		    bool approve=false, bool stat=true, double xmin0=-1, 
		    double xmax0=-1, bool rescale=true) {

  std::string modelNames[5] = {"G4 10.0.p02 QGSP_FTFP_BERT_EML",
			       "G4 10.0.p02 FTFP_BERT_EML",
			       "G4 10.2.p02 QGSP_FTFP_BERT_EMM",
			       "G4 10.2.p02 FTFP_BERT_EMM",
			       "G4 10.2.p02 FTFP_BERT_ATL_EMM"};
  std::string partsF[6] = {"pi-","pi+","k-","k+","pro+","pro-"};
  std::string partsM[6] = {"pim","pip","km","kp","prop","prom"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","proton","antiproton"};
  int         iens[16]  = {2,3,4,5,6,7,8,9,20,30,50,100,150,200,300,350};
  int         types[16] = {0,0,0,0,0,0,0,0, 1, 1, 1,  1,  1,  2,  2,  2};
  int         colors[6] = {2,7,6,4,9,1};
  int         mtype[6] = {21,22,23,24,33,20};
  std::string titlty[2] = {"All events", "MIP in ECAL"};

  std::cout << "plotDataMC " << ipar << ", " << ien << ", " << models << ", " << type << ", " << rebin << ", " << irtype << ", " << prefix << ", " << approve << ", " << xmin0 << ", " << xmax0 << ", " << rescale << std::endl;
  TCanvas* pad(0);
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);         gStyle->SetOptFit(10);
  if (stat)   gStyle->SetOptStat(1110);
  else        gStyle->SetOptStat(0);
  if (type != 2) type = 1;
  std::vector<int>   icol;
  std::vector<TH1D*> hists;
  if (irtype < 0 || irtype > 3) irtype = 1;

  //Get some model information
  char infile[120], title[100], name[120];
  double xmin(-5.0), xmax(500.0), ymax(0);
  int    nbin(0), nmod(0), model(models);
  for (int i=0; i<5; ++i) {
    if (model%10 > 0) {
      sprintf (infile,"%s%d/%s%d.root",prefix.c_str(),i,partsM[ipar].c_str(),iens[ien]);
      TFile *file = new TFile(infile);
      if (file) {
	sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),i);
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
  if (xmax0 > 0) {
    nbin = (int)((xmax0-xmin0)/((xmax-xmin)*rebin));
    xmax = xmax0;
    xmin = xmin0;
  } else {
    nbin /= rebin;
  }

  //Data first
  TH1D *hist(0);
  sprintf (infile, "TB06/%s%d.txt", partsF[ipar].c_str(), iens[ien]);
  sprintf (title, "%d GeV %s", iens[ien], partsN[ipar].c_str());
  sprintf (name, "Data%d%d%s%d", type, irtype, partsF[ipar].c_str(), iens[ien]);
  double scalef = rescale ? GetScaleFactor(ipar, ien) : 1.0;
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
    double eEB, eHB, eHO;
    while (1) {
      fInput >> eEB >> eHB >> eHO;
      if (!fInput.good()) break;
      eEB *= scalef; eHB *= scalef;
      double eTot = (eEB+eHB);
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
    double legsize(0.04);
    for (int k=1; k<=nbin; ++k) total += hist->GetBinContent(k);
    double yext(0);
    if (irtype == 1) {
      yext = 500;
      sprintf (name,"EN%d%s%dH",type,partsF[ipar].c_str(),iens[ien]);
    } else if (irtype == 2) {
      yext = 300;
      sprintf (name,"EN%d%s%dR",type,partsF[ipar].c_str(),iens[ien]);
    } else {
      yext = 700;
      sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    }
    pad = new TCanvas(name, name, 700, yext);
    double dy  = 0.08;
    double xh(0.89), yh(0.89);
    double yh1 = yh-dy*(nmod+1)- 0.02;
    TLegend *legend(0);
    if (!stat) {
      if (approve) legend = new TLegend(xh-0.5, yh-legsize*(nmod+1)-0.01, xh, yh-0.01);
      else legend = new TLegend(xh-0.35, yh-legsize*(nmod+1)-0.01, xh, yh-0.01);
    } else if (types[ien] == 2) {
      legend = new TLegend(xh-0.35, yh-legsize*(nmod+1)-0.01, xh, yh-0.01);
      xh     = 0.4;
    } else if (types[ien] == 1) {
      legend = new TLegend(0.11, yh-legsize*(nmod+1), 0.45, yh-0.01);
    } else {
      legend = new TLegend(xh-0.35, yh1-legsize*(nmod+1), xh, yh1);
    }
    legend->SetFillColor(kWhite); legend->SetBorderSize(0); 
    legend->SetMargin(0.2);

    sprintf (name,"ENStack1%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
    THStack *Hs = new THStack(name,"");
    Hs->Add(hist,"pe sames");
    sprintf (title, "%d GeV %s (%s)", iens[ien], partsN[ipar].c_str(), titlty[type-1].c_str());
    legend->AddEntry(hist,title,"lp");

    int model(models);
    for (int i=0; i<5; ++i) {
      if (model%10 > 0) {
	sprintf (infile,"%s%d/%s%d.root",prefix.c_str(),i,partsM[ipar].c_str(),iens[ien]);
	TFile *file = new TFile(infile);
	if (file) {
	  sprintf (name,"EN%d%s%d",type,partsF[ipar].c_str(),i);
	  TH1D* h1 = (TH1D*)file->FindObjectAny(name);
	  if (h1) {
	    sprintf (name,"EN%d%s%d1",type,partsF[ipar].c_str(),i);
	    TH1D *h2 = new TH1D(name, "", nbin, xmin, xmax);
	    h2->Sumw2();
	    double totm(0);
	    for (int k=1; k<=h1->GetNbinsX(); ++k) totm += h1->GetBinContent(k);
	    double scale = total/totm;
	    int ibin = (int)((xmin-h1->GetBinLowEdge(1))/h1->GetBinWidth(1));
	    for (int k=1; k<=nbin; ++k) {
	      double cont(0);
	      for (int k1=0; k1<rebin; ++k1) {
		++ibin;
		double cv = h1->GetBinContent(ibin);
		cont += scale*cv;
	      }
	      h2->SetBinContent(k,cont);
	      if (cont > ymax) ymax = cont;
	      
	    }
	    h2->SetMarkerColor(colors[i]);
	    h2->SetMarkerStyle(mtype[i]);
	    h2->SetLineColor(colors[i]);
	    Hs->Add(h2,"hist sames");
	    legend->AddEntry(h2,modelNames[i].c_str(),"lp");
	    icol.push_back(colors[i]);
	    hists.push_back(h2);
	  }
	}
      }
      model /= 10;
    }

    int    imax = (ymax > 100) ? (int)(0.01*ymax) : (int)(0.1*ymax);
    double ymx  = (ymax > 100) ? 100*(imax+2) : 10*(imax+2);
    Hs->SetMinimum(0.0); Hs->SetMaximum(ymx);
    if (irtype != 2) {
      TPad *pad1(pad);
      if (irtype != 1) {
	sprintf (name,"ENPad1%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
	pad1 = new TPad(name,"pad1",0,0.3,1,1);
	pad1->SetBottomMargin(0.01);
      } 
      pad1->SetTopMargin(0.10); pad1->SetRightMargin(0.10);
      pad1->Draw(); pad1->cd();
      Hs->Draw("nostack");
      Hs->GetHistogram()->GetYaxis()->SetTitle("Events");
      if (irtype == 1) {
	Hs->GetHistogram()->GetXaxis()->SetTitle("Energy (GeV)");
	Hs->GetHistogram()->GetXaxis()->SetTitleOffset(0.90);
	Hs->GetHistogram()->GetYaxis()->SetTitleOffset(1.20);
      }
      pad1->Update(); pad1->Modified();
      if (stat) {
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
      }
      legend->Draw("same");
      if (approve) {
	double yht = ((types[ien] == 2) || (!stat)) ? (0.88-legsize*(nmod+1)) : 
	  ((types[ien] == 1) ? (yh1-0.02) : (yh1-legsize*(nmod+1)-0.02));
	TPaveText* text = new TPaveText(0.60, yht-0.04, 0.89, yht, "brNDC");
	text->AddText("CMS Preliminary");
	text->Draw("same");
	TPaveText* txt2 = new TPaveText(0.60, yht-0.08, 0.89,yht-0.04, "brNDC");
	txt2->AddText("2006 Test Beam Data");
	txt2->Draw("same");
      }
    }

    pad->cd();
    if (irtype != 1) {
      sprintf (name,"ENStack2%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
      THStack *Hsr = new THStack(name,"");
      Hsr->SetMinimum(0.0); Hsr->SetMaximum(3.99);
      TPad *pad2(pad);
      if (irtype != 2) {
	sprintf (name,"ENPad2%d%s%d",type,partsF[ipar].c_str(),iens[ien]);
	pad2  = new TPad(name,"pad2",0,0,1,0.3);
	pad2->SetTopMargin(0.005);
      }
      pad2->SetBottomMargin(0.20); pad2->SetRightMargin(0.10);
      pad2->Draw(); pad2->cd();
      double xh2 = (types[ien] == 2) ? 0.90 : xh;
      double yh2 = (irtype == 2) ? 0.88 : 0.98;
      double xh1 = (approve) ? (xh2-0.40) : ((irtype == 2) ? (xh2-0.60) : (xh2-0.30));
      double yh1 = (irtype == 2) ? (yh2-0.05*nmod) : (yh2-0.10*nmod);
      TLegend *legend1 = new TLegend(xh1, yh1, xh2, yh2);
      std::cout << "xh " << xh1 << ":" << xh2 << std::endl;
      legend1->SetFillColor(kWhite); legend1->SetBorderSize(0); 
      legend1->SetMargin(0.2);
      TH1D *h_ref =  (TH1D*)hists[0];
      int imx1 = (ymax > 100) ? 10 : 1;
      int imx2 = (ymax > 100) ? 100 : 5;
      for (unsigned int i=1; i<hists.size(); i++) {
	sprintf (name,"ENRatio%d%s%d",i,partsF[ipar].c_str(),iens[ien]);
	TH1D *ratio = new TH1D(name, "Ratio", nbin, xmin, xmax);
	double sumNum(0), sumDen(0);
	for (int k=1; k<=nbin; ++k) {
	  if (h_ref->GetBinContent(k) > imx1 && hists[i]->GetBinContent(k) > imx1) {
	    double rat = hists[i]->GetBinContent(k)/h_ref->GetBinContent(k);
	    double drt = rat*(h_ref->GetBinError(k)/h_ref->GetBinContent(k));
	    ratio->SetBinContent(k,rat); ratio->SetBinError(k,drt);
	    if (h_ref->GetBinContent(k) > imx2) {
	      if (rat > 1.) {
		rat = 1./rat; drt *= (rat*rat);
	      }
	      sumNum += (fabs(1.0-rat)/(drt*drt));
	      sumDen += (1.0/(drt*drt));
	    }
	  }
	}
	double mean  = (sumDen>0) ? (sumNum/sumDen) : 0;
	double error = (sumDen>0) ? 1.0/sqrt(sumDen) : 0;
	std::cout << "Model " << i << " Delta " << mean << " +- " << error <<"\n";
	if (approve) {
	  sprintf (name, "%s",modelNames[i-1].c_str());
	} else if (irtype == 2) {
	  sprintf (name, "#delta = %6.3f #pm %6.3f  %s",mean,error,modelNames[i-1].c_str());
	} else {
	  sprintf (name, "#delta = %6.3f #pm %6.3f",mean,error);
	}
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
      Hsr->GetHistogram()->GetYaxis()->SetTitle("MC/Data");
      Hsr->GetHistogram()->GetXaxis()->SetTitle("Energy (GeV)");
      if (irtype == 2) {
	Hsr->GetHistogram()->GetXaxis()->SetTitleSize(0.06);
	Hsr->GetHistogram()->GetYaxis()->SetTitleSize(0.06);
	Hsr->GetHistogram()->GetXaxis()->SetTitleOffset(1.00);
	Hsr->GetHistogram()->GetYaxis()->SetTitleOffset(0.60);
	Hsr->GetHistogram()->GetXaxis()->SetLabelSize(0.07);
	Hsr->GetHistogram()->GetYaxis()->SetLabelSize(0.07);
      } else {
	Hsr->GetHistogram()->GetXaxis()->SetTitleSize(0.10);
	Hsr->GetHistogram()->GetYaxis()->SetTitleSize(0.10);
	Hsr->GetHistogram()->GetXaxis()->SetTitleOffset(0.90);
	Hsr->GetHistogram()->GetYaxis()->SetTitleOffset(0.35);
	Hsr->GetHistogram()->GetXaxis()->SetLabelSize(0.10);
	Hsr->GetHistogram()->GetYaxis()->SetLabelSize(0.10);
      }
      TLine *line = new TLine(xmin,1.0,xmax,1.0);
      line->SetLineStyle(2); line->SetLineWidth(2);
      line->SetLineColor(kBlack); line->Draw();
      if (approve) {
	double dyt = (irtype == 3) ? 0.08 : 0.08;
	TPaveText* text = new TPaveText(0.11,yh2-dyt,xh2-0.29,yh2,"brNDC");
	sprintf (name, "2006 Test Beam Data (%d GeV %s %s)", iens[ien], 
		 partsN[ipar].c_str(), titlty[type-1].c_str());
	text->AddText(name);
	text->Draw("same");
	std::cout << "xh2 " << xh2 << std::endl;
	TPaveText* txt2 = new TPaveText(xh2-0.25, yh2-dyt, xh2, yh2, "brNDC");
	txt2->AddText("CMS Preliminary");
	txt2->Draw("same");
      } else {
	legend1->Draw("same");
      }
      pad2->Update();
    }
    pad->Update();
  }
  return pad;
}

void plotDataMCDist(int ipar, int ien, int models=1001, int rebin=2,
		    std::string prefix="model", bool approve=true, 
		    bool stat=true, double xmin0=-1, double xmax0=-1, 
		    int save=0) {

  for (int type=1; type<3; ++type) {
    char filename[100];
    TCanvas* c1 = plotDataMC(ipar, ien, models, type, rebin, 1, prefix, approve, stat, xmin0, xmax0, true);
    if (save != 0) {
      if (save < 0) sprintf (filename, "%s.jpg", c1->GetName());
      else          sprintf (filename, "%s.pdf", c1->GetName());
      c1->Print(filename);
    }
    TCanvas* c2 = plotDataMC(ipar, ien, models, type, rebin, 2, prefix, approve, stat, xmin0, xmax0, true);
    if (save != 0) {
      if (save < 0) sprintf (filename, "%s.jpg", c2->GetName());
      else          sprintf (filename, "%s.pdf", c2->GetName());
      c2->Print(filename);
    }
  }
}

TCanvas* plotDataMC(int ipar, int models, bool ratio=false,
		    std::string dirName="RespAll", bool approve=false) {

  std::string names[6] = {"Test Beam Data",
			  "G4 10.0.p02 QGSP_FTFP_BERT_EML",
			  "G4 10.0.p02 FTFP_BERT_EML",
			  "G4 10.2.p02 QGSP_FTFP_BERT_EMM",
			  "G4 10.2.p02 FTFP_BERT_EMM",
			  "G4 10.2.p02 FTFP_BERT_ATL_EMM"};
  std::string partsM[6] = {"pim","pip","km","kp","prop","prom"};
  std::string partsN[6] = {"#pi^{-}","#pi^{+}","K^{-}","K^{+}","proton","antiproton"};
  double      xmax[6]   = {400.0,30.0,15.0,15.0,400.0,15.0};
  double      ylow[6]   = {0.4,0.4,0.2,0.2,0.2,0.6};
  double      ylowr[6]  = {0.9,0.9,0.7,0.5,0.9,0.9};
  double      ymaxr[6]  = {1.2,1.2,1.2,1.4,1.2,1.2};
  int         colors[6] = {1,2,7,6,4,9};
  int         styles[6] = {20,21,22,23,24,33};

  TCanvas* canvas(0);
  char     cname[100];
  int      nm(1), model(models);
  for (int i=0; i<5; ++i) {
    if (model%10 > 0) ++nm;
    model /= 10;
  }
  double ymax = 0.948;
  double ymin = ymax-0.04*nm;
  TLegend*  legend(0);
  if (ratio) {
    legend = new TLegend(0.175, ymin, 0.970, ymax);
  } else {
    legend = new TLegend(0.175, ymin, 0.973, ymax);
  }
  legend->SetBorderSize(0); legend->SetFillColor(kWhite);
  legend->SetMargin(0.2);
  std::vector<TGraphAsymmErrors*> graphs;
  const int NPT=20;
  double    mom[NPT], dmom[NPT], momd[NPT], meand[NPT], dmeand[NPT], mean[NPT], dmean[NPT];

  // First data
  char                infile[100];
  if (dirName == "") {
    sprintf(infile,"%s.txt",  partsM[ipar].c_str());
  } else {
    sprintf(infile,"%s/%s.txt",  dirName.c_str(), partsM[ipar].c_str());
  }
  double    pb, resp, errsp;
  bool      ok(false);
  int       nptd(0);
  //First data
  std::ifstream fInput1(infile);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    ok = true;
    while (1) {
      fInput1 >> pb >> resp >> errsp;
      if (!fInput1.good()) break;
      momd[nptd] = pb; dmom[nptd] = 0; meand[nptd] = resp; dmeand[nptd] = errsp; ++nptd;
    }
    fInput1.close();
    ok = (nptd > 0);
    /*
    std::cout << "Reads " << npt << " points from " << infile << std::endl;
    for (int k=0; k<npt; ++k)
      std::cout << "[" << k << "] " << momd[k] << " +- " << dmom[k] << "   "
		<< meand[k] << " +- " << dmeand[k] << std::endl;
    */
  }
  if (ok) {
    if (!ratio) {
      TGraphAsymmErrors *graph = new TGraphAsymmErrors(nptd,momd,meand,dmom,dmom,dmeand,dmeand);
      graph->SetMarkerStyle(styles[0]);
      graph->SetMarkerColor(colors[0]);
      graph->SetMarkerSize(1.4);
      graph->SetLineColor(colors[0]);
      graphs.push_back(graph);
      sprintf(cname, "2006 %s (%s)", names[0].c_str(), partsN[ipar].c_str());
      legend->AddEntry(graph, cname, "lp");
    }
    int model(models);
    for (int i=0; i<5; ++i) {
      int npt(0);
      double sumNum(0), sumDen(0);
      if (model%10 > 0) {
	if (dirName == "") {
	  sprintf(infile,"%sm%d.txt",partsM[ipar].c_str(),i);
	} else {
	  sprintf(infile,"%s/%sm%d.txt",dirName.c_str(),partsM[ipar].c_str(),i);
	}
	std::ifstream fInput2(infile);
	if (!fInput2.good()) {
	  std::cout << "Cannot open file " << infile << std::endl;
	} else {
	  while (1) {
	    fInput2 >> pb >> resp;
	    if (!fInput2.good()) break;
	    if (ratio) {
	      for (int k=0; k<nptd; ++k) {
		if (std::abs(momd[k]-pb) < 0.1) {
		  double rat =  resp/meand[k]; double drt = dmeand[k]/meand[k];
		  mom[npt] = pb; dmom[npt] = 0; mean[npt] = rat; dmean[npt] = drt;
		  if (rat > 1.) {
		    rat = 1./rat; drt *= (rat*rat);
		  }
		  sumNum += (fabs(1.0-rat)/(drt*drt));
		  sumDen += (1.0/(drt*drt));
		  ++npt; break;
		}
	      }
	    } else {
	      mom[npt] = pb; dmom[npt] = 0; mean[npt] = resp; dmean[npt] = 0.001; ++npt;
	    }
	  }
	}
	fInput2.close();
	if (npt > 0) {
	  /*
	  std::cout << "Reads " << npt << " points from " << infile <<std::endl;
	  for (int k=0; k<npt; ++k)
	    std::cout << "[" << k << "] " << mom[k] << " +- " << dmom[k]<< "   "
		      << mean[k] << " +- " << dmean[k] << std::endl;
	  */
	  TGraphAsymmErrors *graph = new TGraphAsymmErrors(npt,mom,mean,dmom,dmom,dmean,dmean);
	  graph->SetMarkerStyle(styles[i+1]);
	  graph->SetMarkerColor(colors[i+1]);
	  graph->SetMarkerSize(1.2);
	  graph->SetLineColor(colors[i+1]);
	  graph->SetLineWidth(2);
	  graphs.push_back(graph);
	  if (ratio) {
	    double rat = (sumDen>0) ? (sumNum/sumDen) : 0;
	    double err = (sumDen>0) ? sqrt(1./sumDen) : 0;
	    std::cout << "Particle " << partsM[ipar] << " Model "
		      << names[i+1] << " ratio " << rat << " +- " 
		      << err << std::endl;
	    if (approve) sprintf (cname, "%s", names[i+1].c_str());
	    else         sprintf (cname, "(#delta = %5.3f) %s", rat, names[i+1].c_str());
	  } else {
	    sprintf (cname, "%s", names[i+1].c_str());
	  }
	  legend->AddEntry(graph, cname, "lp");
	}
      }
      model /= 10;
    }
    // Now prepare plot
    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(kFALSE);    gStyle->SetPadBorderMode(0);
    gStyle->SetCanvasBorderMode(0); gStyle->SetOptStat(0);
    if (ratio) {
      sprintf(cname, "c_%sRespR", partsM[ipar].c_str());
      canvas = new TCanvas(cname, cname, 500, 300);
    } else {
      sprintf(cname, "c_%sResp", partsM[ipar].c_str());
      canvas = new TCanvas(cname, cname, 500, 500);
    }
    canvas->SetTopMargin(0.05);       canvas->SetLogx();
    canvas->SetLeftMargin(0.15);      canvas->SetRightMargin(0.025);
    canvas->SetBottomMargin(0.14);
    TH1F *vFrame = canvas->DrawFrame(1.0, 0.01, xmax[ipar], 2.0);
    vFrame->GetXaxis()->SetRangeUser(1.0,xmax[ipar]);
    if (ratio) {
      if (approve) vFrame->GetYaxis()->SetRangeUser(ylowr[ipar],ymaxr[ipar]);
      else         vFrame->GetYaxis()->SetRangeUser(ylowr[ipar],1.4);
    } else {
      vFrame->GetYaxis()->SetRangeUser(ylow[ipar],1.0);
    }
    if (ratio) {
      vFrame->GetXaxis()->SetLabelSize(0.06);
      vFrame->GetYaxis()->SetLabelSize(0.06);
      vFrame->GetXaxis()->SetTitleSize(0.06);
      vFrame->GetYaxis()->SetTitleSize(0.06);
      vFrame->GetXaxis()->SetTitleOffset(0.9);
      vFrame->GetYaxis()->SetTitleOffset(1.0);
    } else {
      vFrame->GetXaxis()->SetLabelSize(0.04);
      vFrame->GetYaxis()->SetLabelSize(0.04);
      vFrame->GetXaxis()->SetTitleSize(0.04);
      vFrame->GetYaxis()->SetTitleSize(0.04);
      vFrame->GetXaxis()->SetTitleOffset(1.2);
      vFrame->GetYaxis()->SetTitleOffset(1.6);
    }
    vFrame->GetXaxis()->SetLabelOffset(0.0);
    vFrame->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
    if (ratio) {
      vFrame->GetYaxis()->SetTitle("MC/Data (Response)");
    } else {
      if (approve) vFrame->GetYaxis()->SetTitle("Mean of E_{Measured}/p_{Beam}");
      else         vFrame->GetYaxis()->SetTitle("<E_{Measured}>/p_{Beam}");
    }
    for (unsigned int ii=0; ii<graphs.size(); ++ii) {
      if (ii == 0 && !ratio) graphs[ii]->Draw("P");
      else                   graphs[ii]->Draw("LP");
    }
    if ((!approve) || (!ratio)) legend->Draw();
    double yvl = (approve && ratio) ? ymax : ymin;
    if ((!approve) || ratio) {
      TPaveText* text = new TPaveText(0.16,yvl-0.08, 0.70,yvl-0.01,"brNDC");
      sprintf(cname, "2006 %s (%s)", names[0].c_str(), partsN[ipar].c_str());
      text->AddText(cname);
      text->Draw("same");
    }
    TPaveText* txt2 = new TPaveText(0.70, yvl-0.08, 0.948, yvl-0.01, "brNDC");
    sprintf (cname, "CMS Preliminary");
    txt2->AddText(cname);
    txt2->Draw("same");
    if (ratio) {
      TLine *line = new TLine(1.0,1.0,xmax[ipar],1.0);
      line->SetLineStyle(2); line->SetLineWidth(2);
      line->SetLineColor(kBlack); line->Draw();
    }
  }
  return canvas;

}

void plotDataMCResp(std::string dirName="RespAll", bool approve=true, int save=0) {

  for (int k=0; k<6; ++k) {
    char filename[100];
    TCanvas* c1 = plotDataMC(k, 1001, false, dirName, approve);
    if (save != 0) {
      if (save < 0) sprintf (filename, "%s.jpg", c1->GetName());
      else          sprintf (filename, "%s.pdf", c1->GetName());
      c1->Print(filename);
    }
    TCanvas* c2 = plotDataMC(k, 1001, true, dirName, approve);
    if (save != 0) {
      if (save < 0) sprintf (filename, "%s.jpg", c2->GetName());
      else          sprintf (filename, "%s.pdf", c2->GetName());
      c2->Print(filename);
    }
  }
}

void convert(std::string indir="nmodel0", std::string outdir="model0", int md=0) {

  TB06Analysis p01((indir+"/km4.root"),  (outdir+"/km4.root"),    2, 3, md, 1.0, 1.0);
  p01.Loop();
  TB06Analysis p02((indir+"/km5.root"),	 (outdir+"/km5.root"),    2, 4, md, 1.0, 1.0);
  p02.Loop();
  TB06Analysis p03((indir+"/km6.root"),	 (outdir+"/km6.root"),    2, 5, md, 1.0, 1.0);
  p03.Loop();
  TB06Analysis p04((indir+"/km7.root"),	 (outdir+"/km7.root"),    2, 6, md, 1.0, 1.0);
  p04.Loop();
  TB06Analysis p05((indir+"/kp5.root"),	 (outdir+"/kp5.root"),    3, 4, md, 1.0, 1.0);
  p05.Loop();
  TB06Analysis p06((indir+"/kp6.root"),	 (outdir+"/kp6.root"),    3, 5, md, 1.0, 1.0);
  p06.Loop();
  TB06Analysis p07((indir+"/kp7.root"),	 (outdir+"/kp7.root"),    3, 6, md, 1.0, 1.0);
  p07.Loop();
  TB06Analysis p08((indir+"/pim2.root"),  (outdir+"/pim2.root"),   0, 0, md, 1.0, 1.0);
  p08.Loop();
  TB06Analysis p09((indir+"/pim3.root"),  (outdir+"/pim3.root"),   0, 1, md, 1.0, 1.0);
  p09.Loop();
  TB06Analysis p10((indir+"/pim4.root"),  (outdir+"/pim4.root"),   0, 2, md, 1.0, 1.0);
  p10.Loop();
  TB06Analysis p11((indir+"/pim5.root"),  (outdir+"/pim5.root"),   0, 3, md, 1.0, 1.0);
  p11.Loop();
  TB06Analysis p12((indir+"/pim6.root"),  (outdir+"/pim6.root"),   0, 4, md, 1.0, 1.0);
  p12.Loop();
  TB06Analysis p13((indir+"/pim7.root"),  (outdir+"/pim7.root"),   0, 5, md, 1.0, 1.0);
  p13.Loop();
  TB06Analysis p14((indir+"/pim8.root"),  (outdir+"/pim8.root"),   0, 6, md, 1.0, 1.0);
  p14.Loop();
  TB06Analysis p15((indir+"/pim9.root"),  (outdir+"/pim9.root"),   0, 7, md, 1.0, 1.0);
  p15.Loop();
  TB06Analysis p16((indir+"/pim20.root"),  (outdir+"/pim20.root"),  0, 8, md, 1.0, 1.0);
  p16.Loop();
  TB06Analysis p17((indir+"/pim30.root"),  (outdir+"/pim30.root"),  0, 9, md, 1.0, 1.0);
  p17.Loop();
  TB06Analysis p18((indir+"/pim50.root"),  (outdir+"/pim50.root"),  0,10, md, 1.0, 1.0);
  p18.Loop();
  TB06Analysis p19((indir+"/pim100.root"), (outdir+"/pim100.root"), 0,11, md, 1.0, 1.0);
  p19.Loop();
  TB06Analysis p20((indir+"/pim150.root"), (outdir+"/pim150.root"), 0,12, md, 1.0, 1.0);
  p20.Loop();
  TB06Analysis p21((indir+"/pim200.root"), (outdir+"/pim200.root"), 0,13, md, 1.0, 1.0);
  p21.Loop();
  TB06Analysis p22((indir+"/pim300.root"), (outdir+"/pim300.root"), 0,14, md, 1.0, 1.0);
  p22.Loop();
  TB06Analysis p23((indir+"/pip2.root"),  (outdir+"/pip2.root"),   1, 0, md, 1.0, 1.0);
  p23.Loop();
  TB06Analysis p24((indir+"/pip3.root"),  (outdir+"/pip3.root"),   1, 1, md, 1.0, 1.0);
  p24.Loop();
  TB06Analysis p25((indir+"/pip4.root"),  (outdir+"/pip4.root"),   1, 2, md, 1.0, 1.0);
  p25.Loop();
  TB06Analysis p26((indir+"/pip5.root"),  (outdir+"/pip5.root"),   1, 3, md, 1.0, 1.0);
  p26.Loop();
  TB06Analysis p27((indir+"/pip6.root"),  (outdir+"/pip6.root"),   1, 4, md, 1.0, 1.0);
  p27.Loop();
  TB06Analysis p28((indir+"/pip7.root"),  (outdir+"/pip7.root"),   1, 5, md, 1.0, 1.0);
  p28.Loop();
  TB06Analysis p29((indir+"/pip8.root"),  (outdir+"/pip8.root"),   1, 6, md, 1.0, 1.0);
  p29.Loop();
  TB06Analysis p30((indir+"/pip9.root"),  (outdir+"/pip9.root"),   1, 7, md, 1.0, 1.0);
  p30.Loop();
  TB06Analysis p31((indir+"/prom2.root"),  (outdir+"/prom2.root"),  5, 0, md, 1.0, 1.0);
  p31.Loop();
  TB06Analysis p32((indir+"/prom3.root"),  (outdir+"/prom3.root"),  5, 1, md, 1.0, 1.0);
  p32.Loop();
  TB06Analysis p33((indir+"/prom4.root"),  (outdir+"/prom4.root"),  5, 2, md, 1.0, 1.0);
  p33.Loop();
  TB06Analysis p34((indir+"/prom5.root"),  (outdir+"/prom5.root"),  5, 3, md, 1.0, 1.0);
  p34.Loop();
  TB06Analysis p35((indir+"/prom6.root"),  (outdir+"/prom6.root"),  5, 4, md, 1.0, 1.0);
  p35.Loop();
  TB06Analysis p36((indir+"/prom7.root"),  (outdir+"/prom7.root"),  5, 5, md, 1.0, 1.0);
  p36.Loop();
  TB06Analysis p37((indir+"/prom8.root"),  (outdir+"/prom8.root"),  5, 6, md, 1.0, 1.0);
  p37.Loop();
  TB06Analysis p38((indir+"/prom9.root"),  (outdir+"/prom9.root"),  5, 7, md, 1.0, 1.0);
  p38.Loop();
  TB06Analysis p39((indir+"/prop2.root"),  (outdir+"/prop2.root"),  4, 0, md, 1.0, 1.0);
  p39.Loop();
  TB06Analysis p40((indir+"/prop3.root"),  (outdir+"/prop3.root"),  4, 1, md, 1.0, 1.0);
  p40.Loop();
  TB06Analysis p41((indir+"/prop4.root"),  (outdir+"/prop4.root"),  4, 2, md, 1.0, 1.0);
  p41.Loop();
  TB06Analysis p42((indir+"/prop5.root"),  (outdir+"/prop5.root"),  4, 3, md, 1.0, 1.0);
  p42.Loop();
  TB06Analysis p43((indir+"/prop6.root"),  (outdir+"/prop6.root"),  4, 4, md, 1.0, 1.0);
  p43.Loop();
  TB06Analysis p44((indir+"/prop7.root"),  (outdir+"/prop7.root"),  4, 5, md, 1.0, 1.0);
  p44.Loop();
  TB06Analysis p45((indir+"/prop8.root"),  (outdir+"/prop8.root"),  4, 6, md, 1.0, 1.0);
  p45.Loop();
  TB06Analysis p46((indir+"/prop9.root"),  (outdir+"/prop9.root"),  4, 7, md, 1.0, 1.0);
  p46.Loop();
  TB06Analysis p47((indir+"/prop20.root"), (outdir+"/prop20.root"), 4, 8, md, 1.0, 1.0);
  p47.Loop();
  TB06Analysis p48((indir+"/prop30.root"), (outdir+"/prop30.root"), 4, 9, md, 1.0, 1.0);
  p48.Loop();
  TB06Analysis p49((indir+"/prop350.root"),(outdir+"/prop350.root"),4,15, md, 1.0, 1.0);
  p49.Loop();
}
