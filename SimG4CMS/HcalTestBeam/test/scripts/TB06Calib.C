
///////////////////////////////////////////////////////////////////////////////
//
// Analysis script to compare energy distribution of TB06 data with MC
//
// TB06Calib.C        Class to run over histograms created by TB06Analysis
//                    within the framewok of CMSSW.
//
//
///////////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TGraphErrors.h"
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

void TB06Calib() {
  gROOT->SetStyle("Plain");

  std::string fname = "FTFP_BERT_EMH_e-_RR_50gev.root";
  auto ff = new TFile(fname.c_str());
  if (!ff) {
    cout << "Error open file <" << fname << ">" << endl;
    exit(1);
  }
  TH1F* h1 = (TH1F*)ff->Get("testbeam/edepN");
  if (!h1) {
    cout << "Error open testbeam/edepN" << endl;
    exit(1);
  }
  double x = h1->GetMean();
  double d = h1->GetRMS();
  cout << "===== Mean=" << x << " RMS=" << d << " Calib=" << 100. / x << "  ======" << endl;
}
