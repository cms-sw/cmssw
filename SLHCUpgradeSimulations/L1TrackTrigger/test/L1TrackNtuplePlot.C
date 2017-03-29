// ----------------------------------------------------------------------------------------------------------------
// Basic example script for making tracking performance plots using the ntuples produced by L1TrackNtupleMaker.cc
// By Louise Skinnari, June 2013  
// ----------------------------------------------------------------------------------------------------------------

#include "TROOT.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TMath.h"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

void SetPlotStyle();
void mySmallText(Double_t x,Double_t y,Color_t color,char *text); 


// ----------------------------------------------------------------------------------------------------------------
// Main script
void L1TrackNtuplePlot(TString type, bool doPixelTrack) {

  SetPlotStyle();

  bool doDetailedPlots = false; //turn on to make full set of plots
  bool makeCanvas = false; //make PDF file with all the plots
  bool useTight = false; //use tight quality cut selection
  bool doHighD0 = false; //restrict to looking only at tracking particles with 0.01 < |d0| < 0.1cm
  bool doGausFit = false; //do gaussian fit for resolution vs eta/pt plots


  //some counters for integrated efficiencies
  int n_all_eta2p5 = 0;
  int n_all_eta1p75 = 0;
  int n_all_eta1p0 = 0;
  int n_match_eta2p5 = 0;
  int n_match_eta1p75 = 0;
  int n_match_eta1p0 = 0;

  int n10_all_eta2p5 = 0;
  int n10_all_eta1p75 = 0;
  int n10_all_eta1p0 = 0;
  int n10_match_eta2p5 = 0;
  int n10_match_eta1p75 = 0;
  int n10_match_eta1p0 = 0;


  // ----------------------------------------------------------------------------------------------------------------
  // read ntuples
  TChain* tree = new TChain("L1TrackNtuple/eventTree");

  if (type=="test") tree->Add("test_TrkPerf.root");
  else tree->Add(type+"_TrkPerf.root");

  if (tree->GetEntries() == 0) {
    cout << "File doesn't exist or is empty, returning..." << endl;
    return;
  }


  // ----------------------------------------------------------------------------------------------------------------
  // define leafs & branches

  // L1 tracks
  /*
  vector<float>* trk_pt;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_z0;
  vector<float>* trk_chi2; 
  vector<int>*   trk_nstub;
  vector<int>*   trk_genuine;
  vector<int>*   trk_unknown;
  vector<int>*   trk_combinatoric;
  */

  // tracking particles
  vector<float>* tp_pt;
  vector<float>* tp_eta;
  vector<float>* tp_phi;
  vector<float>* tp_z0;
  vector<float>* tp_d0;
  vector<float>* tp_dxy;
  vector<int>*   tp_pdgid;
  vector<int>*   tp_nmatch;
  vector<int>*   tp_npixmatch;
  vector<int>*   tp_nstub;

  // *L1 track* properties, for tracking particles matched to a L1 track
  vector<float>* matchtrk_pt;
  vector<float>* matchtrk_eta;
  vector<float>* matchtrk_phi;
  vector<float>* matchtrk_z0;
  vector<float>* matchtrk_chi2; 
  vector<float>* matchtrk_consistency; 
  vector<int>*   matchtrk_nstub;

  // *L1 track* properties, for tracking particles matched to a L1 track *** USING 5-PARAMETER TRACK FIT ***
  vector<float>* matchtrk5p_pt;
  vector<float>* matchtrk5p_eta;
  vector<float>* matchtrk5p_phi;
  vector<float>* matchtrk5p_z0;
  vector<float>* matchtrk5p_d0;
  vector<float>* matchtrk5p_chi2; 

  /*
  TBranch* b_trk_pt;
  TBranch* b_trk_eta;
  TBranch* b_trk_phi;
  TBranch* b_trk_z0;
  TBranch* b_trk_chi2; 
  TBranch* b_trk_nstub;
  TBranch* b_trk_genuine;
  TBranch* b_trk_unknown;
  TBranch* b_trk_combinatoric;
  */

  TBranch* b_tp_pt;
  TBranch* b_tp_eta;
  TBranch* b_tp_phi;
  TBranch* b_tp_z0;
  TBranch* b_tp_d0;
  TBranch* b_tp_dxy;
  TBranch* b_tp_pdgid;
  TBranch* b_tp_nmatch;
  TBranch* b_tp_npixmatch;
  TBranch* b_tp_nstub;

  TBranch* b_matchtrk_pt;
  TBranch* b_matchtrk_eta;
  TBranch* b_matchtrk_phi;
  TBranch* b_matchtrk_z0;
  TBranch* b_matchtrk_chi2; 
  TBranch* b_matchtrk_consistency; 
  TBranch* b_matchtrk_nstub;

  TBranch* b_matchtrk5p_pt;
  TBranch* b_matchtrk5p_eta;
  TBranch* b_matchtrk5p_phi;
  TBranch* b_matchtrk5p_z0;
  TBranch* b_matchtrk5p_d0;
  TBranch* b_matchtrk5p_chi2; 

  /*
  trk_pt    = 0;
  trk_eta   = 0;
  trk_phi   = 0;
  trk_z0    = 0;
  trk_chi2  = 0; 
  trk_consistency = 0; 
  trk_nstub = 0;
  trk_genuine = 0;
  trk_unknown = 0;
  trk_combinatoric = 0;
  */

  tp_pt  = 0;
  tp_eta = 0;
  tp_phi = 0;
  tp_z0  = 0;
  tp_d0  = 0;
  tp_dxy  = 0;
  tp_pdgid = 0;
  tp_nmatch = 0;
  tp_npixmatch = 0;
  tp_nstub = 0;

  matchtrk_pt  = 0;
  matchtrk_eta = 0;
  matchtrk_phi = 0;
  matchtrk_z0  = 0;
  matchtrk_chi2  = 0; 
  matchtrk_consistency  = 0; 
  matchtrk_nstub = 0;
  
  matchtrk5p_pt  = 0;
  matchtrk5p_eta = 0;
  matchtrk5p_phi = 0;
  matchtrk5p_z0  = 0;
  matchtrk5p_d0  = 0;
  matchtrk5p_chi2  = 0; 

  /*
  tree->SetBranchAddress("trk_pt",    &trk_pt,    &b_trk_pt);
  tree->SetBranchAddress("trk_eta",   &trk_eta,   &b_trk_eta);
  tree->SetBranchAddress("trk_phi",   &trk_phi,   &b_trk_phi);
  tree->SetBranchAddress("trk_z0",    &trk_z0,    &b_trk_z0);
  tree->SetBranchAddress("trk_chi2",  &trk_chi2,  &b_trk_chi2);
  tree->SetBranchAddress("trk_consistency",  &trk_consistency,  &b_trk_consistency);
  tree->SetBranchAddress("trk_nstub", &trk_nstub, &b_trk_nstub);
  tree->SetBranchAddress("trk_genuine",      &trk_genuine,      &b_trk_genuine);
  tree->SetBranchAddress("trk_unknown",      &trk_unknown,      &b_trk_unknown);
  tree->SetBranchAddress("trk_combinatoric", &trk_combinatoric, &b_trk_combinatoric);
  */

  tree->SetBranchAddress("tp_pt",     &tp_pt,     &b_tp_pt);
  tree->SetBranchAddress("tp_eta",    &tp_eta,    &b_tp_eta);
  tree->SetBranchAddress("tp_phi",    &tp_phi,    &b_tp_phi);
  tree->SetBranchAddress("tp_z0",     &tp_z0,     &b_tp_z0);
  tree->SetBranchAddress("tp_d0",     &tp_d0,     &b_tp_d0);
  tree->SetBranchAddress("tp_dxy",    &tp_dxy,    &b_tp_dxy);
  tree->SetBranchAddress("tp_pdgid",  &tp_pdgid,  &b_tp_pdgid);
  tree->SetBranchAddress("tp_nmatch", &tp_nmatch, &b_tp_nmatch);
  tree->SetBranchAddress("tp_npixmatch", &tp_npixmatch, &b_tp_npixmatch);
  tree->SetBranchAddress("tp_nstub", &tp_nstub, &b_tp_nstub);

  tree->SetBranchAddress("matchtrk_pt",    &matchtrk_pt,    &b_matchtrk_pt);
  tree->SetBranchAddress("matchtrk_eta",   &matchtrk_eta,   &b_matchtrk_eta);
  tree->SetBranchAddress("matchtrk_phi",   &matchtrk_phi,   &b_matchtrk_phi);
  tree->SetBranchAddress("matchtrk_z0",    &matchtrk_z0,    &b_matchtrk_z0);
  tree->SetBranchAddress("matchtrk_chi2",  &matchtrk_chi2,  &b_matchtrk_chi2);
  tree->SetBranchAddress("matchtrk_consistency", &matchtrk_consistency, &b_matchtrk_consistency);
  tree->SetBranchAddress("matchtrk_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
  
  if (doPixelTrack) {
    tree->SetBranchAddress("matchpixtrk_pt",    &matchtrk5p_pt,    &b_matchtrk5p_pt);
    tree->SetBranchAddress("matchpixtrk_eta",   &matchtrk5p_eta,   &b_matchtrk5p_eta);
    tree->SetBranchAddress("matchpixtrk_phi",   &matchtrk5p_phi,   &b_matchtrk5p_phi);
    tree->SetBranchAddress("matchpixtrk_z0",    &matchtrk5p_z0,    &b_matchtrk5p_z0);
    tree->SetBranchAddress("matchpixtrk_d0",    &matchtrk5p_d0,    &b_matchtrk5p_d0);
    tree->SetBranchAddress("matchpixtrk_chi2",  &matchtrk5p_chi2,  &b_matchtrk5p_chi2);
  }
  else {
    tree->SetBranchAddress("matchtrk5p_pt",    &matchtrk5p_pt,    &b_matchtrk5p_pt);
    tree->SetBranchAddress("matchtrk5p_eta",   &matchtrk5p_eta,   &b_matchtrk5p_eta);
    tree->SetBranchAddress("matchtrk5p_phi",   &matchtrk5p_phi,   &b_matchtrk5p_phi);
    tree->SetBranchAddress("matchtrk5p_z0",    &matchtrk5p_z0,    &b_matchtrk5p_z0);
    tree->SetBranchAddress("matchtrk5p_d0",    &matchtrk5p_d0,    &b_matchtrk5p_d0);
    tree->SetBranchAddress("matchtrk5p_chi2",  &matchtrk5p_chi2,  &b_matchtrk5p_chi2);
  }


  // ----------------------------------------------------------------------------------------------------------------
  // histograms
  // ----------------------------------------------------------------------------------------------------------------

  /////////////////////////////////////////////////
  // NOTATION:                                   //
  // 'C' - Central eta range, |eta|<0.8          //
  // 'I' - Intermediate eta range, 0.8<|eta|<1.6 //
  // 'F' - Forward eta range, |eta|>1.6          //
  //                                             //
  // 'L' - Low pt range, pt<5 GeV                //
  // 'M' - Middle pt range, 5<pt<15 GeV          //
  // 'H' - High pt range, pt>15 GeV              //
  /////////////////////////////////////////////////

  TString title = "L1 tracks";
  if (doPixelTrack) title = "L1 pixel tracks";

  TH1F* h_tp_pt   = new TH1F("tp_pt",   ";Tracking particle p_{T} [GeV]; Tracking particles / 1.0 GeV", 100,  0,   100.0);
  TH1F* h_tp_pt_L = new TH1F("tp_pt_L", ";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  50,  0,     5.0);
  TH1F* h_tp_eta  = new TH1F("tp_eta",  ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_tp_eta_H = new TH1F("tp_eta_H", ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_tp_phi  = new TH1F("tp_phi",  ";Tracking particle #phi [rad]; Tracking particles / 0.1",       64, -3.2,   3.2);
  TH1F* h_tp_z0   = new TH1F("tp_z0",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_tp_d0   = new TH1F("tp_d0",   ";Tracking particle d_{0} [cm]; Tracking particles / 0.0004 cm", 100, -0.02, 0.02);

  TH1F* h_match_tp_pt   = new TH1F("match_tp_pt",   ";Tracking particle p_{T} [GeV]; Tracking particles / 1.0 GeV", 100,  0,   100.0);
  TH1F* h_match_tp_pt_L = new TH1F("match_tp_pt_L", ";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  50,  0,     5.0);
  TH1F* h_match_tp_eta  = new TH1F("match_tp_eta",  ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_match_tp_eta_H = new TH1F("match_tp_eta_H", ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_match_tp_phi  = new TH1F("match_tp_phi",  ";Tracking particle #phi [rad]; Tracking particles / 0.1",       64, -3.2,   3.2);
  TH1F* h_match_tp_z0   = new TH1F("match_tp_z0",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_match_tp_d0   = new TH1F("match_tp_d0",   ";Tracking particle d_{0} [cm]; Tracking particles / 0.0004 cm", 100, -0.02,   0.02);

  TH1F* h_match_trk_nstub   = new TH1F("match_trk_nstub",   ";Number of stubs; "+title+" / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_C = new TH1F("match_trk_nstub_C", ";Number of stubs; "+title+" / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_I = new TH1F("match_trk_nstub_I", ";Number of stubs; "+title+" / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_F = new TH1F("match_trk_nstub_F", ";Number of stubs; "+title+" / 1.0", 15, 0, 15);

  // chi2 histograms
  // note: last bin is an overflow bin
  TH1F* h_match_trk_chi2     = new TH1F("match_trk_chi2",     ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match5p_trk_chi2   = new TH1F("match5p_trk_chi2",   ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_C_L = new TH1F("match_trk_chi2_C_L", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_I_L = new TH1F("match_trk_chi2_I_L", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_F_L = new TH1F("match_trk_chi2_F_L", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_C_M = new TH1F("match_trk_chi2_C_M", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_I_M = new TH1F("match_trk_chi2_I_M", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_F_M = new TH1F("match_trk_chi2_F_M", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_C_H = new TH1F("match_trk_chi2_C_H", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_I_H = new TH1F("match_trk_chi2_I_H", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_F_H = new TH1F("match_trk_chi2_F_H", ";#chi^{2}; "+title+" / 1.0", 100, 0, 100);

  // chi2/dof histograms
  // note: lastbin is an overflow bin
  TH1F* h_match_trk_chi2_dof     = new TH1F("match_trk_chi2_dof",     ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match5p_trk_chi2_dof   = new TH1F("match5p_trk_chi2_dof",   ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_C_L = new TH1F("match_trk_chi2_dof_C_L", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_I_L = new TH1F("match_trk_chi2_dof_I_L", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_F_L = new TH1F("match_trk_chi2_dof_F_L", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_C_M = new TH1F("match_trk_chi2_dof_C_M", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_I_M = new TH1F("match_trk_chi2_dof_I_M", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_F_M = new TH1F("match_trk_chi2_dof_F_M", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_C_H = new TH1F("match_trk_chi2_dof_C_H", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_I_H = new TH1F("match_trk_chi2_dof_I_H", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);
  TH1F* h_match_trk_chi2_dof_F_H = new TH1F("match_trk_chi2_dof_F_H", ";#chi^{2} / D.O.F.; "+title+" / 0.1", 150, 0, 15);


  // resolution histograms
  TH1F* h_res_pt    = new TH1F("res_pt",    ";p_{T} residual (L1 - sim) [GeV]; "+title+" / 0.05",   200,-5.0,   5.0);
  TH1F* h_res_ptRel = new TH1F("res_ptRel", ";p_{T} residual (L1 - sim) / p_{T}; "+title+" / 0.01", 200,-1.0,   1.0);
  TH1F* h_res_eta   = new TH1F("res_eta",   ";#eta residual (L1 - sim); "+title+" / 0.0002",        100,-0.01,  0.01);
  TH1F* h_res_phi   = new TH1F("res_phi",   ";#phi residual (L1 - sim) [rad]; "+title+" / 0.0001",  100,-0.005, 0.005);

  TH1F* h_res_z0;
  TH1F* h_res_z0_C;
  TH1F* h_res_z0_I;
  TH1F* h_res_z0_F;

  TH1F* h_res_z0_C_low;
  TH1F* h_res_z0_I_low;
  TH1F* h_res_z0_F_low;
  TH1F* h_res_z0_C_high;
  TH1F* h_res_z0_I_high;
  TH1F* h_res_z0_F_high;

  TH1F* h_res_z0_L;
  TH1F* h_res_z0_M;
  TH1F* h_res_z0_H;

  float maxz0res = 1.0;
  if (doPixelTrack) maxz0res = 0.1;
  TString titlez0res = "0.02";
  if (doPixelTrack) titlez0res = "0.002";

  h_res_z0    = new TH1F("res_z0",    ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_C  = new TH1F("res_z0_C",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_I  = new TH1F("res_z0_I",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_F  = new TH1F("res_z0_F",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_L  = new TH1F("res_z0_L",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_M  = new TH1F("res_z0_M",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_H  = new TH1F("res_z0_H",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  
  h_res_z0_C_low  = new TH1F("res_z0_C_low",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_I_low  = new TH1F("res_z0_I_low",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_F_low  = new TH1F("res_z0_F_low",  ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_C_high  = new TH1F("res_z0_C_high",";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_I_high  = new TH1F("res_z0_I_high",";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  h_res_z0_F_high  = new TH1F("res_z0_F_high",";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);


  TH1F* h_res5p_pt    = new TH1F("res5p_pt",    ";p_{T} residual (L1 - sim) [GeV]; "+title+" / 0.05",   200,-5.0,   5.0);
  TH1F* h_res5p_ptRel = new TH1F("res5p_ptRel", ";p_{T} residual (L1 - sim) / p_{T}; "+title+" / 0.01", 200,-1.0,   1.0);
  TH1F* h_res5p_eta   = new TH1F("res5p_eta",   ";#eta residual (L1 - sim); "+title+" / 0.0002",        100,-0.01,  0.01);
  TH1F* h_res5p_phi   = new TH1F("res5p_phi",   ";#phi residual (L1 - sim) [rad]; "+title+" / 0.0001",  100,-0.005, 0.005);
  TH1F* h_res5p_z0    = new TH1F("res5p_z0",    ";z_{0} residual (L1 - sim) [cm]; "+title+" / "+titlez0res, 100,(-1)*maxz0res, maxz0res);
  TH1F* h_res5p_d0    = new TH1F("res5p_d0",    ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0002 cm", 200,-0.02,0.02);

  TH1F* h_res5p_d0_C  = new TH1F("res5p_d0_C",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_I  = new TH1F("res5p_d0_I",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_F  = new TH1F("res5p_d0_F",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_L  = new TH1F("res5p_d0_L",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_M  = new TH1F("res5p_d0_M",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_H  = new TH1F("res5p_d0_H",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);

  TH1F* h_res5p_d0_C_low  = new TH1F("res5p_d0_C_low",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_I_low  = new TH1F("res5p_d0_I_low",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_F_low  = new TH1F("res5p_d0_F_low",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_C_high  = new TH1F("res5p_d0_C_high",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_I_high  = new TH1F("res5p_d0_I_high",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res5p_d0_F_high  = new TH1F("res5p_d0_F_high",  ";d_{0} residual (L1 - sim) [cm]; "+title+" / 0.0001 cm", 200,-0.05,0.05);


  // resolution vs. pt histograms
  const int nRANGE = 20;
  TString ptrange[nRANGE] = {"0-5","5-10", "10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55",
  			     "55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95","95-100"};

  TH1F* h_resVsPt_pt[nRANGE];
  TH1F* h_resVsPt_pt_C[nRANGE];
  TH1F* h_resVsPt_pt_I[nRANGE];
  TH1F* h_resVsPt_pt_F[nRANGE];

  TH1F* h_resVsPt_ptRel[nRANGE];
  TH1F* h_resVsPt_ptRel_C[nRANGE];
  TH1F* h_resVsPt_ptRel_I[nRANGE];
  TH1F* h_resVsPt_ptRel_F[nRANGE];

  TH1F* h_resVsPt_z0[nRANGE];
  TH1F* h_resVsPt_z0_C[nRANGE];
  TH1F* h_resVsPt_z0_I[nRANGE];
  TH1F* h_resVsPt_z0_F[nRANGE];

  TH1F* h_resVsPt_phi[nRANGE];
  TH1F* h_resVsPt_phi_C[nRANGE];
  TH1F* h_resVsPt_phi_I[nRANGE];
  TH1F* h_resVsPt_phi_F[nRANGE];

  TH1F* h_resVsPt_eta[nRANGE];

  // *** 5-parameter fit versions ***
  TH1F* h_res5pVsPt_pt[nRANGE];
  TH1F* h_res5pVsPt_ptRel[nRANGE];
  TH1F* h_res5pVsPt_z0[nRANGE];
  TH1F* h_res5pVsPt_d0[nRANGE];
  TH1F* h_res5pVsPt_phi[nRANGE];
  TH1F* h_res5pVsPt_eta[nRANGE];


  for (int i=0; i<nRANGE; i++) {
    h_resVsPt_pt[i]   = new TH1F("resVsPt_pt_"+ptrange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_C[i] = new TH1F("resVsPt_pt_C_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_I[i] = new TH1F("resVsPt_pt_I_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_F[i] = new TH1F("resVsPt_pt_F_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);

    // restictive range: -0.15 to 0.15
    h_resVsPt_ptRel[i]   = new TH1F("resVsPt_ptRel_"+ptrange[i],   ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_C[i] = new TH1F("resVsPt_ptRel_c_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_I[i] = new TH1F("resVsPt_ptRel_I_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_F[i] = new TH1F("resVsPt_ptRel_F_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);

    h_resVsPt_z0[i]   = new TH1F("resVsPt_z0_"+ptrange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_C[i] = new TH1F("resVsPt_z0_C_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_I[i] = new TH1F("resVsPt_z0_I_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_F[i] = new TH1F("resVsPt_z0_F_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);

    h_resVsPt_phi[i]   = new TH1F("resVsPt_phi_"+ptrange[i],   ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_C[i] = new TH1F("resVsPt_phi_C_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_I[i] = new TH1F("resVsPt_phi_I_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_F[i] = new TH1F("resVsPt_phi_F_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);

    h_resVsPt_eta[i] = new TH1F("resVsPt_eta_"+ptrange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100, -0.01, 0.01);

    // *** 5-parameter fit versions ***
    h_res5pVsPt_pt[i]    = new TH1F("res5pVsPt_pt_"+ptrange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   100, -5.0, 5.0);
    h_res5pVsPt_ptRel[i] = new TH1F("res5pVsPt_ptRel_"+ptrange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02",300, -0.15, 0.15);
    h_res5pVsPt_z0[i]    = new TH1F("res5pVsPt_z0_"+ptrange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02",   100, -1, 1);
    h_res5pVsPt_d0[i]    = new TH1F("res5pVsPt_d0_"+ptrange[i],   ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0004",   100, -0.02, 0.02);
    h_res5pVsPt_phi[i]   = new TH1F("res5pVsPt_phi_"+ptrange[i],  ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_res5pVsPt_eta[i]   = new TH1F("res5pVsPt_eta_"+ptrange[i],  ";#eta residual (L1 - sim); L1 tracks / 0.0002",       100, -0.01, 0.01);
  }

  // resolution vs. eta histograms
  const int nETARANGE = 25;
  TString etarange[nETARANGE] = {"0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0",
				 "1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0",
				 "2.1","2.2","2.3","2.4","2.5"};

  TH1F* h_resVsEta_eta[nETARANGE];
  TH1F* h_resVsEta_eta_L[nETARANGE];
  TH1F* h_resVsEta_eta_M[nETARANGE];
  TH1F* h_resVsEta_eta_H[nETARANGE];

  TH1F* h_resVsEta_z0[nETARANGE];
  TH1F* h_resVsEta_z0_L[nETARANGE];
  TH1F* h_resVsEta_z0_M[nETARANGE];
  TH1F* h_resVsEta_z0_H[nETARANGE];

  TH1F* h_resVsEta_phi[nETARANGE];
  TH1F* h_resVsEta_phi_L[nETARANGE];
  TH1F* h_resVsEta_phi_M[nETARANGE];
  TH1F* h_resVsEta_phi_H[nETARANGE];

  TH1F* h_resVsEta_pt[nETARANGE];

  TH1F* h_resVsEta_ptRel[nETARANGE];
  TH1F* h_resVsEta_ptRel_L[nETARANGE];
  TH1F* h_resVsEta_ptRel_M[nETARANGE];
  TH1F* h_resVsEta_ptRel_H[nETARANGE];

  // *** 5-parameter fit versions ***
  TH1F* h_res5pVsEta_eta[nETARANGE];
  TH1F* h_res5pVsEta_z0[nETARANGE];
  TH1F* h_res5pVsEta_d0[nETARANGE];
  TH1F* h_res5pVsEta_phi[nETARANGE];
  TH1F* h_res5pVsEta_pt[nETARANGE];
  TH1F* h_res5pVsEta_ptRel[nETARANGE];

  for (int i=0; i<nETARANGE; i++) {
    h_resVsEta_eta[i]   = new TH1F("resVsEta_eta_"+etarange[i],   ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_eta_L[i] = new TH1F("resVsEta_eta_L_"+etarange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_eta_M[i] = new TH1F("resVsEta_eta_M_"+etarange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_eta_H[i] = new TH1F("resVsEta_eta_H_"+etarange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);

    h_resVsEta_z0[i]   = new TH1F("resVsEta2_z0_"+etarange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);
    h_resVsEta_z0_L[i] = new TH1F("resVsEta2_z0_L_"+etarange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);
    h_resVsEta_z0_M[i] = new TH1F("resVsEta2_z0_M_"+etarange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);
    h_resVsEta_z0_H[i] = new TH1F("resVsEta2_z0_H_"+etarange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);

    h_resVsEta_phi[i]   = new TH1F("resVsEta2_phi_"+etarange[i],   ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_phi_L[i] = new TH1F("resVsEta2_phi_L_"+etarange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_phi_M[i] = new TH1F("resVsEta2_phi_M_"+etarange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_phi_H[i] = new TH1F("resVsEta2_phi_H_"+etarange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);

    h_resVsEta_pt[i] = new TH1F("resVsEta2_pt_"+etarange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",    100,-5.0,5.0);

    h_resVsEta_ptRel[i]   = new TH1F("resVsEta2_ptRel_"+etarange[i],  ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.01", 100,-0.5,0.5);
    h_resVsEta_ptRel_L[i] = new TH1F("resVsEta2_ptRel_L_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-0.1,0.1);
    h_resVsEta_ptRel_M[i] = new TH1F("resVsEta2_ptRel_M_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-0.1,0.1);
    h_resVsEta_ptRel_H[i] = new TH1F("resVsEta2_ptRel_H_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-0.25,0.25);

    // *** 5-parameter fit versions ***
    h_res5pVsEta_eta[i]   = new TH1F("res5pVsEta_eta_"+etarange[i],  ";#eta residual (L1 - sim); L1 tracks / 0.0002",        100,-0.01,0.01);
    h_res5pVsEta_z0[i]    = new TH1F("res5pVsEta_z0_"+etarange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02",    100,-1,    1);
    h_res5pVsEta_d0[i]    = new TH1F("res5pVsEta_d0_"+etarange[i],   ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.004",    100,-0.02, 0.02);
    h_res5pVsEta_phi[i]   = new TH1F("res5pVsEta_phi_"+etarange[i],  ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001",  100,-0.005,0.005);
    h_res5pVsEta_pt[i]    = new TH1F("res5pVsEta_pt_"+etarange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",    100,-5.0,5.0);
    h_res5pVsEta_ptRel[i] = new TH1F("res5pVsEta_ptRel_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-1.0,1.0);
  }

  // 2D histograms
  TH2F* h_2d_logchi2_eta     = new TH2F("2d_logchi2_eta",     ";Tracking particle #eta; log(#chi^{2})",                    50,-2.5,2.5, 100,-4.0,8.0);
  TH2F* h_2d_logchi2_dof_eta = new TH2F("2d_logchi2_cof_eta", ";Tracking particle #eta; log(#chi^{2} / D.O.F.)",           50,-2.5,2.5, 100,-4.0,8.0);
  TH2F* h_2d_dz0_eta         = new TH2F("2d_dz0_eta",         ";Tracking particle #eta; |#Deltaz_{0}| (L1 - sim) [cm]",    50,-2.5,2.5, 100, 0,  1.2);
  TH2F* h_2d_deta_eta        = new TH2F("2d_deta_eta",        ";Tracking particle #eta; |#Delta#eta| (L1 - sim)",          50,-2.5,2.5, 100, 0,  0.012);
  TH2F* h_2d_dphi_eta        = new TH2F("2d_dphi_eta",        ";Tracking particle #eta; |#Delta#phi| (L1 - sim) [rad]",    50,-2.5,2.5, 100, 0,  0.007);
  TH2F* h_2d_dptRel_eta      = new TH2F("2d_dptRel_eta",      ";Tracking particle #eta; |#Deltap_{T} / p_{T}| (L1 - sim)", 50,-2.5,2.5, 100, 0,  1.2);


  
  // ----------------------------------------------------------------------------------------------------------------
  //        * * * * *     S T A R T   O F   A C T U A L   R U N N I N G   O N   E V E N T S     * * * * *
  // ----------------------------------------------------------------------------------------------------------------
  
  int nevt = tree->GetEntries();
  cout << "number of events = " << nevt << endl;


  // ----------------------------------------------------------------------------------------------------------------
  // event loop
  for (int i=0; i<nevt; i++) {

    tree->GetEntry(i,0);
  
    
    // ----------------------------------------------------------------------------------------------------------------
    // tracking particle loop
    for (int it=0; it<(int)tp_pt->size(); it++) {
      
      if (tp_pt->at(it) < 0.2) continue;
      if (tp_pt->at(it) > 100.) continue; //to have same kinematic range as before for muons!
      if (fabs(tp_eta->at(it)) > 2.5) continue;
      if (fabs(tp_z0->at(it)) > 30.0) continue;

      if (doHighD0 && (fabs(tp_d0->at(it)) < 0.010 || fabs(tp_d0->at(it)) > 0.10) ) continue;

      h_tp_pt->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 5.0) h_tp_pt_L->Fill(tp_pt->at(it));
      
      if (tp_pt->at(it) > 2.0) {

	if (fabs(tp_eta->at(it)) < 1.0) n_all_eta1p0++;
	else if (fabs(tp_eta->at(it)) < 1.75) n_all_eta1p75++;
	else n_all_eta2p5++;

	if (tp_pt->at(it) > 10.0) {
	  if (fabs(tp_eta->at(it)) < 1.0) n10_all_eta1p0++;
	  else if (fabs(tp_eta->at(it)) < 1.75) n10_all_eta1p75++;
	  else n10_all_eta2p5++;
	}

	h_tp_eta->Fill(tp_eta->at(it));
	h_tp_phi->Fill(tp_phi->at(it));
	h_tp_z0->Fill(tp_z0->at(it));
	h_tp_d0->Fill(tp_d0->at(it));
      }
      if (tp_pt->at(it) > 10.0) {
	h_tp_eta_H->Fill(tp_eta->at(it));
      }	
    

      // ----------------------------------------------------------------------------------------------------------------
      // was the tracking particle matched to a L1 track?
      if (tp_nmatch->at(it) < 1) continue;

      if (doPixelTrack && tp_npixmatch->at(it) < 1) continue; //match with pixel track!

      
      // use only tracks with min 4 stubs
      if (!type.Contains("Electron") && matchtrk_nstub->at(it) < 4) continue;
      
      // fill chi2 & chi2/dof histograms before making chi2 cut
      h_2d_logchi2_eta    ->Fill(tp_eta->at(it), log(matchtrk_chi2->at(it)));
      h_2d_logchi2_dof_eta->Fill(tp_eta->at(it), log(matchtrk_chi2->at(it)/(2*matchtrk_nstub->at(it)-4)));
    
      float chi2 = matchtrk_chi2->at(it);
      int ndof = 2*matchtrk_nstub->at(it)-4;
      float chi2dof = (float)chi2/ndof;
      if (chi2 > 100) chi2 = 99.9; //for overflow bin
      if (chi2dof > 15) chi2dof = 14.99; //for overflow bin
  
      h_match_trk_chi2->Fill(chi2);
      h_match_trk_chi2_dof->Fill(chi2dof);

      float chi2_5p = matchtrk5p_chi2->at(it);
      float chi2dof_5p = (float)chi2_5p/ndof;
      if (chi2_5p > 100) chi2_5p = 99.9; //for overflow bin
      if (chi2dof_5p > 15) chi2dof_5p = 14.99; //for overflow bin

      h_match5p_trk_chi2->Fill(chi2_5p);
      h_match5p_trk_chi2_dof->Fill(chi2dof_5p);
      

      // central eta
      if (fabs(matchtrk_eta->at(it)) < 0.8) {
	if (matchtrk_pt->at(it) < 5) {
	  h_match_trk_chi2_C_L->Fill(chi2);
          h_match_trk_chi2_dof_C_L->Fill(chi2dof);
	} 
	else if (matchtrk_pt->at(it) < 15 && matchtrk_pt->at(it) >= 5) {
	  h_match_trk_chi2_C_M->Fill(chi2);
	  h_match_trk_chi2_dof_C_M->Fill(chi2dof);
	}
	else {
	  h_match_trk_chi2_C_H->Fill(chi2);
	  h_match_trk_chi2_dof_C_H->Fill(chi2dof);
	}
      }
      // intermediate eta
      else if (fabs(matchtrk_eta->at(it)) < 1.6 && fabs(matchtrk_eta->at(it)) >= 0.8) {
	if (matchtrk_pt->at(it) < 5) {
          h_match_trk_chi2_I_L->Fill(chi2);
          h_match_trk_chi2_dof_I_L->Fill(chi2dof);
	}
	else if (matchtrk_pt->at(it) < 15 && matchtrk_pt->at(it) >= 5) {
          h_match_trk_chi2_I_M->Fill(chi2);
          h_match_trk_chi2_dof_I_M->Fill(chi2dof);
	} 
	else {
          h_match_trk_chi2_I_H->Fill(chi2);
          h_match_trk_chi2_dof_I_H->Fill(chi2dof);
	}
      }
      // forward eta
      else if (fabs(matchtrk_eta->at(it)) >= 1.6) {
        if (matchtrk_pt->at(it) < 5) {
          h_match_trk_chi2_F_L->Fill(chi2);
          h_match_trk_chi2_dof_F_L->Fill(chi2dof);
	} 
	else if (matchtrk_pt->at(it) < 15 && matchtrk_pt->at(it) >= 5) {
          h_match_trk_chi2_F_M->Fill(chi2);
          h_match_trk_chi2_dof_F_M->Fill(chi2dof);
       	} 
	else {
          h_match_trk_chi2_F_H->Fill(chi2);
          h_match_trk_chi2_dof_F_H->Fill(chi2dof);
	}
      }
      

      // ----------------------------------------------------------------------------------------------------------------
      // cut on chi2
      if (!type.Contains("Electron") && matchtrk_chi2->at(it) > 100.0) continue;
     
      // use tight quality cut selection?
      if (useTight) {
	if (matchtrk_nstub->at(it)==4) {
	  if (fabs(matchtrk_eta->at(it))<2.2 && matchtrk_consistency->at(it)>10) continue;
	  else if (fabs(matchtrk_eta->at(it))>2.2 && chi2dof>5.0) continue;
	}
	if (matchtrk_pt->at(it)>10.0 && chi2dof>5.0) continue;
      }

      
      // ----------------------------------------------------------------------------------------------------------------
      // more plots

      // fill matched track histograms
      h_match_tp_pt->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 5) h_match_tp_pt_L->Fill(tp_pt->at(it));

      if (tp_pt->at(it) > 2.0) {
	h_match_tp_eta->Fill(tp_eta->at(it));
	h_match_tp_phi->Fill(tp_phi->at(it));
	h_match_tp_z0->Fill(tp_z0->at(it));
	h_match_tp_d0->Fill(tp_d0->at(it));

	if (fabs(tp_eta->at(it)) < 1.0) n_match_eta1p0++;
	else if (fabs(tp_eta->at(it)) < 1.75) n_match_eta1p75++;
	else n_match_eta2p5++;

	if (tp_pt->at(it) > 10.0) {
	  if (fabs(tp_eta->at(it)) < 1.0) n10_match_eta1p0++;
	  else if (fabs(tp_eta->at(it)) < 1.75) n10_match_eta1p75++;
	  else n10_match_eta2p5++;
	}

      }
      if (tp_pt->at(it) > 10) {
	h_match_tp_eta_H->Fill(tp_eta->at(it));
      }
      
      // fill nstub histograms
      h_match_trk_nstub->Fill(matchtrk_nstub->at(it));
      if (fabs(matchtrk_eta->at(it)) < 0.8) h_match_trk_nstub_C->Fill(matchtrk_nstub->at(it));
      else if (fabs(matchtrk_eta->at(it)) < 1.6 && fabs(matchtrk_eta->at(it)) >= 0.8) h_match_trk_nstub_I->Fill(matchtrk_nstub->at(it));
      else if (fabs(matchtrk_eta->at(it)) >= 1.6) h_match_trk_nstub_F->Fill(matchtrk_nstub->at(it));
      

      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution histograms
      h_res_pt   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
      h_res_ptRel->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
      h_res_eta  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
      h_res_phi  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
      h_res_z0   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
      
      h_res5p_pt   ->Fill(matchtrk5p_pt->at(it)  - tp_pt->at(it));
      h_res5p_ptRel->Fill((matchtrk5p_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
      h_res5p_eta  ->Fill(matchtrk5p_eta->at(it) - tp_eta->at(it));
      h_res5p_phi  ->Fill(matchtrk5p_phi->at(it) - tp_phi->at(it));
      h_res5p_z0   ->Fill(matchtrk5p_z0->at(it)  - tp_z0->at(it));
      h_res5p_d0->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));

      
      if (fabs(matchtrk_eta->at(it)) < 0.8) h_res_z0_C->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else if (fabs(matchtrk_eta->at(it)) < 1.6 && fabs(matchtrk_eta->at(it)) >= 0.8) h_res_z0_I->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else if (fabs(matchtrk_eta->at(it)) >= 1.6) h_res_z0_F->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      

      if (tp_pt->at(it) < 5.0) {
	if (fabs(matchtrk_eta->at(it)) < 1.0) h_res_z0_C_low->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	else  h_res_z0_F_low->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      }
      else if (tp_pt->at(it) > 10.0) {
	if (fabs(matchtrk_eta->at(it)) < 1.0) h_res_z0_C_high->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	else h_res_z0_F_high->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      }

      if (matchtrk_pt->at(it) < 5.0) h_res_z0_L->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else if (matchtrk_pt->at(it) < 15.0) h_res_z0_M->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else h_res_z0_H->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      

      if (fabs(matchtrk_eta->at(it)) < 0.8) h_res5p_d0_C->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      else if (fabs(matchtrk_eta->at(it)) < 1.6 && fabs(matchtrk_eta->at(it)) >= 0.8) h_res5p_d0_I->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      else if (fabs(matchtrk_eta->at(it)) >= 1.6) h_res5p_d0_F->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));

      if (tp_pt->at(it) < 5.0) {
	if (fabs(matchtrk_eta->at(it)) < 1.0) h_res5p_d0_C_low->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
	else h_res5p_d0_F_low->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      }
      else if (tp_pt->at(it) > 10.0) {
	if (fabs(matchtrk_eta->at(it)) < 1.0) h_res5p_d0_C_high->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
	else h_res5p_d0_F_high->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      }

      if (matchtrk_pt->at(it) < 5.0) h_res5p_d0_L->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      else if (matchtrk_pt->at(it) < 15.0) h_res5p_d0_M->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));
      else h_res5p_d0_H->Fill(matchtrk5p_d0->at(it) - tp_d0->at(it));


      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution vs. pt histograms    
      for (int im=0; im<nRANGE; im++) {
       if ( (tp_pt->at(it) > (float)im*5.0) && (tp_pt->at(it) < (float)im*5.0+5.0) ) {
	  h_resVsPt_pt[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	  h_resVsPt_ptRel[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	  h_resVsPt_eta[im]  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	  h_resVsPt_phi[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  h_resVsPt_z0[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));

	  if (fabs(matchtrk_eta->at(it)) < 0.8) {
	    h_resVsPt_pt_C[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_C[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_C[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_C[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }
	  else if (fabs(matchtrk_eta->at(it)) < 1.6 && fabs(matchtrk_eta->at(it)) >= 0.8) {
	    h_resVsPt_pt_I[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_I[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_I[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_I[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }
	  else if (fabs(matchtrk_eta->at(it)) >= 1.6) {
	    h_resVsPt_pt_F[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_F[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_F[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_F[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }

	  // ** 5-parameter fit versions ***
	  h_res5pVsPt_pt[im]   ->Fill(matchtrk5p_pt->at(it)  - tp_pt->at(it));
	  h_res5pVsPt_ptRel[im]->Fill((matchtrk5p_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	  h_res5pVsPt_eta[im]  ->Fill(matchtrk5p_eta->at(it) - tp_eta->at(it));
	  h_res5pVsPt_phi[im]  ->Fill(matchtrk5p_phi->at(it) - tp_phi->at(it));
	  h_res5pVsPt_z0[im]   ->Fill(matchtrk5p_z0->at(it)  - tp_z0->at(it));
	  h_res5pVsPt_d0[im]   ->Fill(matchtrk5p_d0->at(it)  - tp_d0->at(it));

	}
      }

      // fill resolution vs. eta histograms
      for (int im=0; im<nETARANGE; im++) {
       if ( (fabs(tp_eta->at(it)) > (float)im*0.1) && (fabs(tp_eta->at(it)) < (float)im*0.1+0.1) ) {
	 h_resVsEta_pt[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	 h_resVsEta_ptRel[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	 h_resVsEta_eta[im]  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	 h_resVsEta_phi[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	 h_resVsEta_z0[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));

	 if (matchtrk_pt->at(it)<5) {
	   h_resVsEta_ptRel_L[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	   h_resVsEta_eta_L[im]->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	   h_resVsEta_z0_L[im]->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	   h_resVsEta_phi_L[im]->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	 }
	 else if (matchtrk_pt->at(it)<15 && matchtrk_pt->at(it)>=5) {
	   h_resVsEta_ptRel_M[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	   h_resVsEta_eta_M[im]->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	   h_resVsEta_z0_M[im]->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	   h_resVsEta_phi_M[im]->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	 }
	 else {
	   h_resVsEta_ptRel_H[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	   h_resVsEta_eta_H[im]->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	   h_resVsEta_z0_H[im]->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	   h_resVsEta_phi_H[im]->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	 }


	 // *** 5-parameter fit versions ***
	 h_res5pVsEta_pt[im]   ->Fill(matchtrk5p_pt->at(it)  - tp_pt->at(it));
	 h_res5pVsEta_ptRel[im]->Fill((matchtrk5p_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	 h_res5pVsEta_eta[im]  ->Fill(matchtrk5p_eta->at(it) - tp_eta->at(it));
	 h_res5pVsEta_phi[im]  ->Fill(matchtrk5p_phi->at(it) - tp_phi->at(it));
	 h_res5pVsEta_z0[im]   ->Fill(matchtrk5p_z0->at(it)  - tp_z0->at(it));
	 h_res5pVsEta_d0[im]   ->Fill(matchtrk5p_d0->at(it)  - tp_d0->at(it));

       }
      }
      
      // fill 2D histograms
      h_2d_dz0_eta   ->Fill(tp_eta->at(it), fabs(matchtrk_z0->at(it)  - tp_z0->at(it)));
      h_2d_deta_eta  ->Fill(tp_eta->at(it), fabs(matchtrk_eta->at(it) - tp_eta->at(it)));
      h_2d_dphi_eta  ->Fill(tp_eta->at(it), fabs(matchtrk_phi->at(it) - tp_phi->at(it)));
      h_2d_dptRel_eta->Fill(tp_eta->at(it), fabs((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it)));
     
    } // end of matched track loop
  
  } // end of event loop
  // ----------------------------------------------------------------------------------------------------------------
  

  //some printouts
  float k = (float)n_match_eta1p0;
  float N = (float)n_all_eta1p0;
  if (fabs(N)>0) cout << endl << "efficiency for |eta| < 1.0 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n_match_eta1p75;
  N = (float)n_all_eta1p75;
  if (fabs(N)>0) cout << "efficiency for 1.0 < |eta| < 1.75 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n_match_eta2p5;
  N = (float)n_all_eta2p5;
  if (fabs(N)>0) cout << "efficiency for 1.75 < |eta| < 2.5 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  N = (float) n_all_eta1p0 + n_all_eta1p75 + n_all_eta2p5;
  k = (float) n_match_eta1p0 + n_match_eta1p75 + n_match_eta2p5;
  if (fabs(N)>0) cout << "combined efficiency for |eta| < 2.5 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl << endl;

  k = (float)n10_match_eta1p0;
  N = (float)n10_all_eta1p0;
  if (fabs(N)>0) cout << endl << "efficiency for |eta| < 1.0, pt>10 GeV = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n10_match_eta1p75;
  N = (float)n10_all_eta1p75;
  if (fabs(N)>0) cout << "efficiency for 1.0 < |eta| < 1.75, pt>10 GeV = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n10_match_eta2p5;
  N = (float)n10_all_eta2p5;
  if (fabs(N)>0) cout << "efficiency for 1.75 < |eta| < 2.5, pt>10 GeV = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  N = (float) n10_all_eta1p0 + n10_all_eta1p75 + n10_all_eta2p5;
  k = (float) n10_match_eta1p0 + n10_match_eta1p75 + n10_match_eta2p5;
  if (fabs(N)>0) cout << "combined efficiency for |eta| < 2.5, pt>10 GeV = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl << endl;



  // ----------------------------------------------------------------------------------------------------------------
  // 2D plots  
  // ----------------------------------------------------------------------------------------------------------------

  TH1F* h2_resVsPt_pt   = new TH1F("resVsPt2_pt",   ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_C = new TH1F("resVsPt2_pt_C", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_I = new TH1F("resVsPt2_pt_I", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_F = new TH1F("resVsPt2_pt_F", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);

  TH1F* h2_resVsPt_ptRel = new TH1F("resVsPt2_ptRel",     ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_C = new TH1F("resVsPt2_ptRel_C", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_I = new TH1F("resVsPt2_ptRel_I", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_F = new TH1F("resVsPt2_ptRel_F", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);

  TH1F* h2_mresVsPt_pt   = new TH1F("mresVsPt2_pt",   ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_C = new TH1F("mresVsPt2_pt_C", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_I = new TH1F("mresVsPt2_pt_I", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_F = new TH1F("mresVsPt2_pt_F", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);

  TH1F* h2_resVsPt_z0   = new TH1F("resVsPt2_z0",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_C = new TH1F("resVsPt2_z0_C", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_I = new TH1F("resVsPt2_z0_I", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_F = new TH1F("resVsPt2_z0_F", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);

  TH1F* h2_resVsPt_phi   = new TH1F("resVsPt2_phi",   ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_C = new TH1F("resVsPt2_phi_C", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_I = new TH1F("resVsPt2_phi_I", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_F = new TH1F("resVsPt2_phi_F", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);

  TH1F* h2_resVsPt_eta   = new TH1F("resVsPt2_eta",  ";Tracking particle p_{T} [GeV]; #eta resolution", 20,0,100);

  // *** 5-parameter fit versions ***
  TH1F* h2_res5pVsPt_pt   = new TH1F("res5pVsPt2_pt",   ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_res5pVsPt_ptRel = new TH1F("res5pVsPt2_ptRel",     ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_res5pVsPt_z0   = new TH1F("res5pVsPt2_z0",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_res5pVsPt_d0   = new TH1F("res5pVsPt2_d0",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [cm]", 20,0,100);
  TH1F* h2_res5pVsPt_phi   = new TH1F("res5pVsPt2_phi",   ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_res5pVsPt_eta   = new TH1F("res5pVsPt2_eta",  ";Tracking particle p_{T} [GeV]; #eta resolution", 20,0,100);


  for (int i=0; i<nRANGE; i++) {
    // set bin content and error
    h2_resVsPt_pt  ->SetBinContent(i+1, h_resVsPt_pt[i]  ->GetRMS());
    h2_resVsPt_pt  ->SetBinError(  i+1, h_resVsPt_pt[i]  ->GetRMSError());
    h2_resVsPt_pt_C->SetBinContent(i+1, h_resVsPt_pt_C[i]->GetRMS());
    h2_resVsPt_pt_C->SetBinError(  i+1, h_resVsPt_pt_C[i]->GetRMSError());
    h2_resVsPt_pt_I->SetBinContent(i+1, h_resVsPt_pt_I[i]->GetRMS());
    h2_resVsPt_pt_I->SetBinError(  i+1, h_resVsPt_pt_I[i]->GetRMSError());
    h2_resVsPt_pt_F->SetBinContent(i+1, h_resVsPt_pt_F[i]->GetRMS());
    h2_resVsPt_pt_F->SetBinError(  i+1, h_resVsPt_pt_F[i]->GetRMSError());

    h2_resVsPt_ptRel  ->SetBinContent(i+1, h_resVsPt_ptRel[i]  ->GetRMS());
    h2_resVsPt_ptRel  ->SetBinError(  i+1, h_resVsPt_ptRel[i]  ->GetRMSError());
    h2_resVsPt_ptRel_C->SetBinContent(i+1, h_resVsPt_ptRel_C[i]->GetRMS());
    h2_resVsPt_ptRel_C->SetBinError(  i+1, h_resVsPt_ptRel_C[i]->GetRMSError());
    h2_resVsPt_ptRel_I->SetBinContent(i+1, h_resVsPt_ptRel_I[i]->GetRMS());
    h2_resVsPt_ptRel_I->SetBinError(  i+1, h_resVsPt_ptRel_I[i]->GetRMSError());
    h2_resVsPt_ptRel_F->SetBinContent(i+1, h_resVsPt_ptRel_F[i]->GetRMS());
    h2_resVsPt_ptRel_F->SetBinError(  i+1, h_resVsPt_ptRel_F[i]->GetRMSError());

    h2_mresVsPt_pt  ->SetBinContent(i+1, h_resVsPt_pt[i]  ->GetMean());
    h2_mresVsPt_pt  ->SetBinError(  i+1, h_resVsPt_pt[i]  ->GetMeanError());
    h2_mresVsPt_pt_C->SetBinContent(i+1, h_resVsPt_pt_C[i]->GetMean());
    h2_mresVsPt_pt_C->SetBinError(  i+1, h_resVsPt_pt_C[i]->GetMeanError());
    h2_mresVsPt_pt_I->SetBinContent(i+1, h_resVsPt_pt_I[i]->GetMean());
    h2_mresVsPt_pt_I->SetBinError(  i+1, h_resVsPt_pt_I[i]->GetMeanError());
    h2_mresVsPt_pt_F->SetBinContent(i+1, h_resVsPt_pt_F[i]->GetMean());
    h2_mresVsPt_pt_F->SetBinError(  i+1, h_resVsPt_pt_F[i]->GetMeanError());

    h2_resVsPt_z0  ->SetBinContent(i+1, h_resVsPt_z0[i]  ->GetRMS());
    h2_resVsPt_z0  ->SetBinError(  i+1, h_resVsPt_z0[i]  ->GetRMSError());
    h2_resVsPt_z0_C->SetBinContent(i+1, h_resVsPt_z0_C[i]->GetRMS());
    h2_resVsPt_z0_C->SetBinError(  i+1, h_resVsPt_z0_C[i]->GetRMSError());
    h2_resVsPt_z0_I->SetBinContent(i+1, h_resVsPt_z0_I[i]->GetRMS());
    h2_resVsPt_z0_I->SetBinError(  i+1, h_resVsPt_z0_I[i]->GetRMSError());
    h2_resVsPt_z0_F->SetBinContent(i+1, h_resVsPt_z0_F[i]->GetRMS());
    h2_resVsPt_z0_F->SetBinError(  i+1, h_resVsPt_z0_F[i]->GetRMSError());

    h2_resVsPt_phi  ->SetBinContent(i+1, h_resVsPt_phi[i]  ->GetRMS());
    h2_resVsPt_phi  ->SetBinError(  i+1, h_resVsPt_phi[i]  ->GetRMSError());
    h2_resVsPt_phi_C->SetBinContent(i+1, h_resVsPt_phi_C[i]->GetRMS());
    h2_resVsPt_phi_C->SetBinError(  i+1, h_resVsPt_phi_C[i]->GetRMSError());
    h2_resVsPt_phi_I->SetBinContent(i+1, h_resVsPt_phi_I[i]->GetRMS());
    h2_resVsPt_phi_I->SetBinError(  i+1, h_resVsPt_phi_I[i]->GetRMSError());
    h2_resVsPt_phi_F->SetBinContent(i+1, h_resVsPt_phi_F[i]->GetRMS());
    h2_resVsPt_phi_F->SetBinError(  i+1, h_resVsPt_phi_F[i]->GetRMSError());

    h2_resVsPt_eta->SetBinContent(i+1, h_resVsPt_eta[i]->GetRMS());
    h2_resVsPt_eta->SetBinError(  i+1, h_resVsPt_eta[i]->GetRMSError());

    // *** 5-parameter fit versions ***
    // set bin content and error
    h2_res5pVsPt_pt  ->SetBinContent(i+1, h_res5pVsPt_pt[i]  ->GetRMS());
    h2_res5pVsPt_pt  ->SetBinError(  i+1, h_res5pVsPt_pt[i]  ->GetRMSError());
    h2_res5pVsPt_ptRel  ->SetBinContent(i+1, h_res5pVsPt_ptRel[i]  ->GetRMS());
    h2_res5pVsPt_ptRel  ->SetBinError(  i+1, h_res5pVsPt_ptRel[i]  ->GetRMSError());
    h2_res5pVsPt_z0  ->SetBinContent(i+1, h_res5pVsPt_z0[i]  ->GetRMS());
    h2_res5pVsPt_z0  ->SetBinError(  i+1, h_res5pVsPt_z0[i]  ->GetRMSError());
    h2_res5pVsPt_d0  ->SetBinContent(i+1, h_res5pVsPt_d0[i]  ->GetRMS());
    h2_res5pVsPt_d0  ->SetBinError(  i+1, h_res5pVsPt_d0[i]  ->GetRMSError());
    h2_res5pVsPt_phi  ->SetBinContent(i+1, h_res5pVsPt_phi[i]  ->GetRMS());
    h2_res5pVsPt_phi  ->SetBinError(  i+1, h_res5pVsPt_phi[i]  ->GetRMSError());
    h2_res5pVsPt_eta->SetBinContent(i+1, h_res5pVsPt_eta[i]->GetRMS());
    h2_res5pVsPt_eta->SetBinError(  i+1, h_res5pVsPt_eta[i]->GetRMSError());

  }


  // resolution vs. eta histograms
  TH1F* h2_resVsEta_eta   = new TH1F("resVsEta2_eta",   ";Tracking particle |#eta|; #eta resolution", 25,0,2.5);
  TH1F* h2_resVsEta_eta_L = new TH1F("resVsEta2_eta_L", ";Tracking particle |#eta|; #eta resolution", 25,0,2.5);
  TH1F* h2_resVsEta_eta_M = new TH1F("resVsEta2_eta_M", ";Tracking particle |#eta|; #eta resolution", 25,0,2.5);
  TH1F* h2_resVsEta_eta_H = new TH1F("resVsEta2_eta_H", ";Tracking particle |#eta|; #eta resolution", 25,0,2.5);

  TH1F* h2_mresVsEta_eta   = new TH1F("mresVsEta2_eta",   ";Tracking particle |#eta|; Mean(#eta residual)", 25,0,2.5);
  TH1F* h2_mresVsEta_eta_L = new TH1F("mresVsEta2_eta_L", ";Tracking particle |#eta|; Mean(#eta residual)", 25,0,2.5);
  TH1F* h2_mresVsEta_eta_M = new TH1F("mresVsEta2_eta_M", ";Tracking particle |#eta|; Mean(#eta residual)", 25,0,2.5);
  TH1F* h2_mresVsEta_eta_H = new TH1F("mresVsEta2_eta_H", ";Tracking particle |#eta|; Mean(#eta residual)", 25,0,2.5);

  TH1F* h2_resVsEta_z0   = new TH1F("resVsEta_z0",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 25,0,2.5);
  TH1F* h2_resVsEta_z0_L = new TH1F("resVsEta_z0_L", ";Tracking particle |#eta|; z_{0} resolution [cm]", 25,0,2.5);
  TH1F* h2_resVsEta_z0_M = new TH1F("resVsEta_z0_M", ";Tracking particle |#eta|; z_{0} resolution [cm]", 25,0,2.5);
  TH1F* h2_resVsEta_z0_H = new TH1F("resVsEta_z0_H", ";Tracking particle |#eta|; z_{0} resolution [cm]", 25,0,2.5);

  TH1F* h2_resVsEta_phi   = new TH1F("resVsEta_phi",   ";Tracking particle |#eta|; #phi resolution [rad]", 25,0,2.5);
  TH1F* h2_resVsEta_phi_L = new TH1F("resVsEta_phi_L", ";Tracking particle |#eta|; #phi resolution [rad]", 25,0,2.5);
  TH1F* h2_resVsEta_phi_M = new TH1F("resVsEta_phi_M", ";Tracking particle |#eta|; #phi resolution [rad]", 25,0,2.5);
  TH1F* h2_resVsEta_phi_H = new TH1F("resVsEta_phi_H", ";Tracking particle |#eta|; #phi resolution [rad]", 25,0,2.5);

  TH1F* h2_resVsEta_pt = new TH1F("resVsEta_pt",    ";Tracking particle |#eta|; p_{T} resolution [GeV]",   25,0,2.5);

  TH1F* h2_resVsEta_ptRel = new TH1F("resVsEta_ptRel", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 25,0,2.5);
  TH1F* h2_resVsEta_ptRel_L = new TH1F("resVsEta_ptRel_L", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 25,0,2.5);
  TH1F* h2_resVsEta_ptRel_M = new TH1F("resVsEta_ptRel_M", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 25,0,2.5);
  TH1F* h2_resVsEta_ptRel_H = new TH1F("resVsEta_ptRel_H", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 25,0,2.5);

  // *** 5-parameter fit versions ***
  TH1F* h2_res5pVsEta_eta   = new TH1F("res5pVsEta2_eta",   ";Tracking particle |#eta|; #eta resolution", 25,0,2.5);
  TH1F* h2_res5pVsEta_z0  = new TH1F("res5pVsEta2_z0",  ";Tracking particle |#eta|; z_{0} resolution [cm]", 25,0,2.5);
  TH1F* h2_res5pVsEta_d0  = new TH1F("res5pVsEta2_d0",  ";Tracking particle |#eta|; d_{0} resolution [cm]", 25,0,2.5);
  TH1F* h2_res5pVsEta_phi = new TH1F("res5pVsEta2_phi", ";Tracking particle |#eta|; #phi resolution [rad]", 25,0,2.5);
  TH1F* h2_res5pVsEta_pt    = new TH1F("res5pVsEta2_pt",    ";Tracking particle |#eta|; p_{T} resolution [GeV]",   25,0,2.5);
  TH1F* h2_res5pVsEta_ptRel = new TH1F("res5pVsEta2_ptRel", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 25,0,2.5);


  // resolution vs. eta histograms (gaussian fit)
  TH1F* h3_resVsEta_eta_L = new TH1F("resVsEta_eta_L_gaus", ";|#eta|; #sigma(#eta)", 25,0,2.5);
  TH1F* h3_resVsEta_eta_M = new TH1F("resVsEta_eta_M_gaus", ";|#eta|; #sigma(#eta)", 25,0,2.5);
  TH1F* h3_resVsEta_eta_H = new TH1F("resVsEta_eta_H_gaus", ";|#eta|; #sigma(#eta)", 25,0,2.5);

  TH1F* h3_resVsEta_z0_L = new TH1F("resVsEta_z0_L_gaus", ";|#eta|; #sigma(z_{0}) [cm]", 25,0,2.5);
  TH1F* h3_resVsEta_z0_M = new TH1F("resVsEta_z0_M_gaus", ";|#eta|; #sigma(z_{0}) [cm]", 25,0,2.5);
  TH1F* h3_resVsEta_z0_H = new TH1F("resVsEta_z0_H_gaus", ";|#eta|; #sigma(z_{0}) [cm]", 25,0,2.5);

  TH1F* h3_resVsEta_phi_L = new TH1F("resVsEta_phi_L_gaus", ";|#eta|; #sigma(#phi) [rad]", 25,0,2.5);
  TH1F* h3_resVsEta_phi_M = new TH1F("resVsEta_phi_M_gaus", ";|#eta|; #sigma(#phi) [rad]", 25,0,2.5);
  TH1F* h3_resVsEta_phi_H = new TH1F("resVsEta_phi_H_gaus", ";|#eta|; #sigma(#phi) [rad]", 25,0,2.5);

  TH1F* h3_resVsEta_ptRel_L = new TH1F("resVsEta_ptRel_L_gaus", ";|#eta|; #sigma(p_{T}) / p_{T}", 25,0,2.5);
  TH1F* h3_resVsEta_ptRel_M = new TH1F("resVsEta_ptRel_M_gaus", ";|#eta|; #sigma(p_{T}) / p_{T}", 25,0,2.5);
  TH1F* h3_resVsEta_ptRel_H = new TH1F("resVsEta_ptRel_H_gaus", ";|#eta|; #sigma(p_{T}) / p_{T}", 25,0,2.5);

  TString fitdir = "FitResults/";


  for (int i=0; i<nETARANGE; i++) {
    // set bin content and error
    h2_resVsEta_eta  ->SetBinContent(i+1, h_resVsEta_eta[i]  ->GetRMS());
    h2_resVsEta_eta  ->SetBinError(  i+1, h_resVsEta_eta[i]  ->GetRMSError());
    h2_resVsEta_eta_L->SetBinContent(i+1, h_resVsEta_eta_L[i]->GetRMS());
    h2_resVsEta_eta_L->SetBinError(  i+1, h_resVsEta_eta_L[i]->GetRMSError());
    h2_resVsEta_eta_M->SetBinContent(i+1, h_resVsEta_eta_M[i]->GetRMS());
    h2_resVsEta_eta_M->SetBinError(  i+1, h_resVsEta_eta_M[i]->GetRMSError());
    h2_resVsEta_eta_H->SetBinContent(i+1, h_resVsEta_eta_H[i]->GetRMS());
    h2_resVsEta_eta_H->SetBinError(  i+1, h_resVsEta_eta_H[i]->GetRMSError());

    h2_mresVsEta_eta  ->SetBinContent(i+1, h_resVsEta_eta[i]  ->GetMean());
    h2_mresVsEta_eta  ->SetBinError(  i+1, h_resVsEta_eta[i]  ->GetMeanError());
    h2_mresVsEta_eta_L->SetBinContent(i+1, h_resVsEta_eta_L[i]->GetMean());
    h2_mresVsEta_eta_L->SetBinError(  i+1, h_resVsEta_eta_L[i]->GetMeanError());
    h2_mresVsEta_eta_M->SetBinContent(i+1, h_resVsEta_eta_M[i]->GetMean());
    h2_mresVsEta_eta_M->SetBinError(  i+1, h_resVsEta_eta_M[i]->GetMeanError());
    h2_mresVsEta_eta_H->SetBinContent(i+1, h_resVsEta_eta_H[i]->GetMean());
    h2_mresVsEta_eta_H->SetBinError(  i+1, h_resVsEta_eta_H[i]->GetMeanError());

    h2_resVsEta_z0 ->SetBinContent(i+1, h_resVsEta_z0[i] ->GetRMS());
    h2_resVsEta_z0 ->SetBinError(  i+1, h_resVsEta_z0[i] ->GetRMSError());
    h2_resVsEta_z0_L ->SetBinContent(i+1, h_resVsEta_z0_L[i] ->GetRMS());
    h2_resVsEta_z0_L ->SetBinError(  i+1, h_resVsEta_z0_L[i] ->GetRMSError());
    h2_resVsEta_z0_M ->SetBinContent(i+1, h_resVsEta_z0_M[i] ->GetRMS());
    h2_resVsEta_z0_M ->SetBinError(  i+1, h_resVsEta_z0_M[i] ->GetRMSError());
    h2_resVsEta_z0_H ->SetBinContent(i+1, h_resVsEta_z0_H[i] ->GetRMS());
    h2_resVsEta_z0_H ->SetBinError(  i+1, h_resVsEta_z0_H[i] ->GetRMSError());

    h2_resVsEta_phi->SetBinContent(i+1, h_resVsEta_phi[i]->GetRMS());
    h2_resVsEta_phi->SetBinError(  i+1, h_resVsEta_phi[i]->GetRMSError());
    h2_resVsEta_phi_L->SetBinContent(i+1, h_resVsEta_phi_L[i]->GetRMS());
    h2_resVsEta_phi_L->SetBinError(  i+1, h_resVsEta_phi_L[i]->GetRMSError());
    h2_resVsEta_phi_M->SetBinContent(i+1, h_resVsEta_phi_M[i]->GetRMS());
    h2_resVsEta_phi_M->SetBinError(  i+1, h_resVsEta_phi_M[i]->GetRMSError());
    h2_resVsEta_phi_H->SetBinContent(i+1, h_resVsEta_phi_H[i]->GetRMS());
    h2_resVsEta_phi_H->SetBinError(  i+1, h_resVsEta_phi_H[i]->GetRMSError());

    h2_resVsEta_pt   ->SetBinContent(i+1, h_resVsEta_pt[i]   ->GetRMS());
    h2_resVsEta_pt   ->SetBinError(  i+1, h_resVsEta_pt[i]   ->GetRMSError());

    h2_resVsEta_ptRel->SetBinContent(i+1, h_resVsEta_ptRel[i]->GetRMS());
    h2_resVsEta_ptRel->SetBinError(  i+1, h_resVsEta_ptRel[i]->GetRMSError());
    h2_resVsEta_ptRel_L->SetBinContent(i+1, h_resVsEta_ptRel_L[i]->GetRMS());
    h2_resVsEta_ptRel_L->SetBinError(  i+1, h_resVsEta_ptRel_L[i]->GetRMSError());
    h2_resVsEta_ptRel_M->SetBinContent(i+1, h_resVsEta_ptRel_M[i]->GetRMS());
    h2_resVsEta_ptRel_M->SetBinError(  i+1, h_resVsEta_ptRel_M[i]->GetRMSError());
    h2_resVsEta_ptRel_H->SetBinContent(i+1, h_resVsEta_ptRel_H[i]->GetRMS());
    h2_resVsEta_ptRel_H->SetBinError(  i+1, h_resVsEta_ptRel_H[i]->GetRMSError());

    // *** 5-parameter fit versions ***
    // set bin content and error
    h2_res5pVsEta_eta  ->SetBinContent(i+1, h_res5pVsEta_eta[i]  ->GetRMS());
    h2_res5pVsEta_eta  ->SetBinError(  i+1, h_res5pVsEta_eta[i]  ->GetRMSError());
    h2_res5pVsEta_z0 ->SetBinContent(i+1, h_res5pVsEta_z0[i] ->GetRMS());
    h2_res5pVsEta_z0 ->SetBinError(  i+1, h_res5pVsEta_z0[i] ->GetRMSError());
    h2_res5pVsEta_d0 ->SetBinContent(i+1, h_res5pVsEta_d0[i] ->GetRMS());
    h2_res5pVsEta_d0 ->SetBinError(  i+1, h_res5pVsEta_d0[i] ->GetRMSError());
    h2_res5pVsEta_phi->SetBinContent(i+1, h_res5pVsEta_phi[i]->GetRMS());
    h2_res5pVsEta_phi->SetBinError(  i+1, h_res5pVsEta_phi[i]->GetRMSError());
    h2_res5pVsEta_pt   ->SetBinContent(i+1, h_res5pVsEta_pt[i]   ->GetRMS());
    h2_res5pVsEta_pt   ->SetBinError(  i+1, h_res5pVsEta_pt[i]   ->GetRMSError());
    h2_res5pVsEta_ptRel->SetBinContent(i+1, h_res5pVsEta_ptRel[i]->GetRMS());
    h2_res5pVsEta_ptRel->SetBinError(  i+1, h_res5pVsEta_ptRel[i]->GetRMSError());



    // ---------------------------------------------------------------------------------------------------
    // gaussian fit instead
    // ---------------------------------------------------------------------------------------------------
  
    if (doGausFit) {

      TCanvas cfit;
      char text[500];
      
      float sigma = 0;
      float esigma = 0;
      TF1* fit;
      
      float rms = 0;
      float erms = 0;
      
      fit = new TF1("fit", "gaus", -0.01,0.01);
      h_resVsEta_eta_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_eta_L[i]->GetRMS();
      erms = h_resVsEta_eta_L[i]->GetRMSError();
      h3_resVsEta_eta_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_eta_L->SetBinError(i+1, esigma);   
      h_resVsEta_eta_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_eta_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.01,0.01);
      h_resVsEta_eta_M[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_eta_M[i]->GetRMS();
      erms = h_resVsEta_eta_M[i]->GetRMSError();
      h3_resVsEta_eta_M->SetBinContent(i+1, sigma);   
      h3_resVsEta_eta_M->SetBinError(i+1, esigma);   
      h_resVsEta_eta_M[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_eta_M_"+etarange[i]+".png");
      delete fit;

      fit = new TF1("fit", "gaus", -0.01,0.01);
      h_resVsEta_eta_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_eta_H[i]->GetRMS();
      erms = h_resVsEta_eta_H[i]->GetRMSError();
      h3_resVsEta_eta_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_eta_H->SetBinError(i+1, esigma);   
      h_resVsEta_eta_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_eta_H_"+etarange[i]+".png");
      delete fit;
      
      
      if (doPixelTrack) fit = new TF1("fit", "gaus", -0.3,0.3);
      else fit = new TF1("fit", "gaus", -1,1);
      h_resVsEta_z0_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_z0_L[i]->GetRMS();
      erms = h_resVsEta_z0_L[i]->GetRMSError();
      h3_resVsEta_z0_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_z0_L->SetBinError(i+1, esigma);   
      h_resVsEta_z0_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_z0_L_"+etarange[i]+".png");
      delete fit;
      
      if (doPixelTrack) fit = new TF1("fit", "gaus", -0.3,0.3);
      else fit = new TF1("fit", "gaus", -1,1);
      h_resVsEta_z0_M[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_z0_M[i]->GetRMS();
      erms = h_resVsEta_z0_M[i]->GetRMSError();
      h3_resVsEta_z0_M->SetBinContent(i+1, sigma);   
      h3_resVsEta_z0_M->SetBinError(i+1, esigma);   
      h_resVsEta_z0_M[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_z0_M_"+etarange[i]+".png");
      delete fit;
      
      if (doPixelTrack) fit = new TF1("fit", "gaus", -0.3,0.3);
      else fit = new TF1("fit", "gaus", -1,1);
      h_resVsEta_z0_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_z0_H[i]->GetRMS();
      erms = h_resVsEta_z0_H[i]->GetRMSError();
      h3_resVsEta_z0_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_z0_H->SetBinError(i+1, esigma);   
      h_resVsEta_z0_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_z0_H_"+etarange[i]+".png");
      delete fit;
      
      
      fit = new TF1("fit", "gaus", -0.005,0.005);
      h_resVsEta_phi_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_phi_L[i]->GetRMS();
      erms = h_resVsEta_phi_L[i]->GetRMSError();
      h3_resVsEta_phi_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_phi_L->SetBinError(i+1, esigma);   
      h_resVsEta_phi_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_phi_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.005,0.005);
      h_resVsEta_phi_M[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_phi_M[i]->GetRMS();
      erms = h_resVsEta_phi_M[i]->GetRMSError();
      h3_resVsEta_phi_M->SetBinContent(i+1, sigma);   
      h3_resVsEta_phi_M->SetBinError(i+1, esigma);   
      h_resVsEta_phi_M[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_phi_M_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.005,0.005);
      h_resVsEta_phi_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_phi_H[i]->GetRMS();
      erms = h_resVsEta_phi_H[i]->GetRMSError();
      h3_resVsEta_phi_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_phi_H->SetBinError(i+1, esigma);   
      h_resVsEta_phi_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_phi_H_"+etarange[i]+".png");
      delete fit;
      
      
      fit = new TF1("fit", "gaus", -0.5,0.5);
      h_resVsEta_ptRel_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_ptRel_L[i]->GetRMS();
      erms = h_resVsEta_ptRel_L[i]->GetRMSError();
      h3_resVsEta_ptRel_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_ptRel_L->SetBinError(i+1, esigma);   
      h_resVsEta_ptRel_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_ptRel_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.5,0.5);
      h_resVsEta_ptRel_M[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_ptRel_M[i]->GetRMS();
      erms = h_resVsEta_ptRel_M[i]->GetRMSError();
      h3_resVsEta_ptRel_M->SetBinContent(i+1, sigma);   
      h3_resVsEta_ptRel_M->SetBinError(i+1, esigma);   
      h_resVsEta_ptRel_M[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_ptRel_M_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.5,0.5);
      h_resVsEta_ptRel_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_ptRel_H[i]->GetRMS();
      erms = h_resVsEta_ptRel_H[i]->GetRMSError();
      h3_resVsEta_ptRel_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_ptRel_H->SetBinError(i+1, esigma);   
      h_resVsEta_ptRel_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      cfit.SaveAs(fitdir+"resVsEta_ptRel_H_"+etarange[i]+".png");
      delete fit;

    }//end doGausFit

  }


  // set minimum to zero
  h2_resVsPt_pt  ->SetMinimum(0);
  h2_resVsPt_pt_C->SetMinimum(0);
  h2_resVsPt_pt_I->SetMinimum(0);
  h2_resVsPt_pt_F->SetMinimum(0);

  h2_resVsPt_ptRel  ->SetMinimum(0);
  h2_resVsPt_ptRel_C->SetMinimum(0);
  h2_resVsPt_ptRel_I->SetMinimum(0);
  h2_resVsPt_ptRel_F->SetMinimum(0);

  h2_resVsPt_z0  ->SetMinimum(0);
  h2_resVsPt_z0_C->SetMinimum(0);
  h2_resVsPt_z0_I->SetMinimum(0);
  h2_resVsPt_z0_F->SetMinimum(0);

  h2_resVsPt_phi  ->SetMinimum(0);
  h2_resVsPt_phi_C->SetMinimum(0);
  h2_resVsPt_phi_I->SetMinimum(0);
  h2_resVsPt_phi_F->SetMinimum(0);

  h2_resVsPt_eta->SetMinimum(0);

  h2_resVsEta_eta  ->SetMinimum(0);
  h2_resVsEta_eta_L->SetMinimum(0);
  h2_resVsEta_eta_M->SetMinimum(0);
  h2_resVsEta_eta_H->SetMinimum(0);

  h2_resVsEta_z0 ->SetMinimum(0);
  h2_resVsEta_z0_L ->SetMinimum(0);
  h2_resVsEta_z0_M ->SetMinimum(0);
  h2_resVsEta_z0_H ->SetMinimum(0);

  h2_resVsEta_phi->SetMinimum(0);
  h2_resVsEta_phi_L->SetMinimum(0);
  h2_resVsEta_phi_M->SetMinimum(0);
  h2_resVsEta_phi_H->SetMinimum(0);

  h2_resVsEta_pt->SetMinimum(0);

  h2_resVsEta_ptRel->SetMinimum(0);
  h2_resVsEta_ptRel_L->SetMinimum(0);
  h2_resVsEta_ptRel_M->SetMinimum(0);
  h2_resVsEta_ptRel_H->SetMinimum(0);

  h2_res5pVsPt_pt  ->SetMinimum(0);
  h2_res5pVsPt_ptRel  ->SetMinimum(0);
  h2_res5pVsPt_z0  ->SetMinimum(0);
  h2_res5pVsPt_d0  ->SetMinimum(0);
  h2_res5pVsPt_phi  ->SetMinimum(0);
  h2_res5pVsPt_eta->SetMinimum(0);

  h2_res5pVsEta_eta  ->SetMinimum(0);
  h2_res5pVsEta_z0 ->SetMinimum(0);
  h2_res5pVsEta_d0 ->SetMinimum(0);
  h2_res5pVsEta_phi->SetMinimum(0);
  h2_res5pVsEta_pt   ->SetMinimum(0);
  h2_res5pVsEta_ptRel->SetMinimum(0);


  // -------------------------------------------------------------------------------------------
  // output file for histograms
  // -------------------------------------------------------------------------------------------

  if (useTight) type = type+"_tight";

  TString fout_name = "";
  if (doPixelTrack && doHighD0) fout_name = "_pix_highd0";
  else if (doPixelTrack) fout_name = "_pix";
  else if (doHighD0) fout_name = "_highd0";

  TFile* fout = new TFile("output_"+type+fout_name+".root","recreate");


  // -------------------------------------------------------------------------------------------
  // draw and save plots
  // -------------------------------------------------------------------------------------------

  char ctxt[500];
  TCanvas c;

  TString DIR = "TrkPlots/";

  h2_resVsPt_pt->Draw();
  h2_resVsPt_pt->Write();
  c.SaveAs(DIR+type+"_resVsPt_pt.eps");
  c.SaveAs(DIR+type+"_resVsPt_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf("); // keep the pdf file open

  h2_res5pVsPt_pt->Draw();
  h2_res5pVsPt_pt->Write();
  c.SaveAs(DIR+type+"_res5pVsPt_pt.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsPt_pt_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_pt_C.eps");
    c.SaveAs(DIR+type+"_resVsPt_pt_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_pt_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_pt_I.eps");
    c.SaveAs(DIR+type+"_resVsPt_pt_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_pt_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_pt_F.eps");
    c.SaveAs(DIR+type+"_resVsPt_pt_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_resVsPt_ptRel->Draw();
  c.SaveAs(DIR+type+"_resVsPt_ptRel.eps");
  c.SaveAs(DIR+type+"_resVsPt_ptRel.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsPt_ptRel->Draw();
  c.SaveAs(DIR+type+"_res5pVsPt_ptRel.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_ptRel.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsPt_ptRel_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_ptRel_C.eps");
    c.SaveAs(DIR+type+"_resVsPt_ptRel_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_ptRel_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_ptRel_I.eps");
    c.SaveAs(DIR+type+"_resVsPt_ptRel_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_ptRel_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_ptRel_F.eps");
    c.SaveAs(DIR+type+"_resVsPt_ptRel_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_mresVsPt_pt->Draw();
  c.SaveAs(DIR+type+"_mresVsPt_pt.eps");
  c.SaveAs(DIR+type+"_mresVsPt_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_mresVsPt_pt_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsPt_pt_C.eps");
    c.SaveAs(DIR+type+"_mresVsPt_pt_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_mresVsPt_pt_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsPt_pt_I.eps");
    c.SaveAs(DIR+type+"_mresVsPt_pt_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_mresVsPt_pt_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsPt_pt_F.eps");
    c.SaveAs(DIR+type+"_mresVsPt_pt_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    // combined plot for the pt residual bias
    h2_mresVsPt_pt_C->GetYaxis()->SetRangeUser(-1.5,1.5);
    h2_mresVsPt_pt_C->Draw();
    h2_mresVsPt_pt_I->Draw("same");
    h2_mresVsPt_pt_I->SetMarkerColor(kRed);
    h2_mresVsPt_pt_I->SetLineColor(kRed);
    h2_mresVsPt_pt_F->Draw("same");
    h2_mresVsPt_pt_F->SetMarkerColor(kBlue);
    h2_mresVsPt_pt_F->SetLineColor(kBlue);
    
    TLegend* l = new TLegend(0.22,0.22,0.42,0.42);
    l->SetFillStyle(0);
    l->SetBorderSize(0);
    l->SetTextSize(0.04);
    l->AddEntry(h2_mresVsPt_pt_C," |eta| < 0.8","l");
    l->AddEntry(h2_mresVsPt_pt_I," 0.8 < |eta| < 1.6","l");
    l->AddEntry(h2_mresVsPt_pt_F," |eta| > 1.6","l");
    l->SetTextFont(42);
    l->Draw();	

    c.SaveAs(DIR+type+"_mresVsPt_pt_comb.eps");
    c.SaveAs(DIR+type+"_mresVsPt_pt_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
  
  h2_resVsPt_z0->Draw();
  c.SaveAs(DIR+type+"_resVsPt_z0.eps");
  c.SaveAs(DIR+type+"_resVsPt_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsPt_z0->Draw();
  c.SaveAs(DIR+type+"_res5pVsPt_z0.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsPt_d0->Draw();
  c.SaveAs(DIR+type+"_res5pVsPt_d0.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_d0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsPt_z0_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.22,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_z0_C.eps");
    c.SaveAs(DIR+type+"_resVsPt_z0_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_z0_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.22,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_z0_I.eps");
    c.SaveAs(DIR+type+"_resVsPt_z0_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_z0_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.22,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_z0_F.eps");
    c.SaveAs(DIR+type+"_resVsPt_z0_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_resVsPt_phi->Draw();
  c.SaveAs(DIR+type+"_resVsPt_phi.eps");
  c.SaveAs(DIR+type+"_resVsPt_phi.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsPt_phi->Draw();
  c.SaveAs(DIR+type+"_res5pVsPt_phi.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_phi.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsPt_phi_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_phi_C.eps");
    c.SaveAs(DIR+type+"_resVsPt_phi_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_phi_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_phi_I.eps");
    c.SaveAs(DIR+type+"_resVsPt_phi_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsPt_phi_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsPt_phi_F.eps");
    c.SaveAs(DIR+type+"_resVsPt_phi_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
    
  h2_resVsPt_eta->Draw();
  c.SaveAs(DIR+type+"_resVsPt_eta.eps");
  c.SaveAs(DIR+type+"_resVsPt_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsPt_eta->Draw();
  c.SaveAs(DIR+type+"_res5pVsPt_eta.eps");
  c.SaveAs(DIR+type+"_res5pVsPt_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_eta->Draw();
  h2_resVsEta_eta->Write();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_resVsEta_eta.eps");
  c.SaveAs(DIR+type+"_resVsEta_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsEta_eta->Draw();
  h2_res5pVsEta_eta->Write();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5pVsEta_eta.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsEta_eta_L->Draw();
    h2_resVsEta_eta_L->Write();
    sprintf(ctxt,"p_{T} < 5 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_eta_L.eps");
    c.SaveAs(DIR+type+"_resVsEta_eta_L.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsEta_eta_M->Draw();
    h2_resVsEta_eta_M->Write();
    sprintf(ctxt,"5 < p_{T} < 15 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_eta_M.eps");
    c.SaveAs(DIR+type+"_resVsEta_eta_M.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsEta_eta_H->Draw();
    h2_resVsEta_eta_H->Write();
    sprintf(ctxt,"p_{T} > 15 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_eta_H.eps");
    c.SaveAs(DIR+type+"_resVsEta_eta_H.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_mresVsEta_eta->Draw();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_mresVsEta_eta.eps");
  c.SaveAs(DIR+type+"_mresVsEta_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_mresVsEta_eta_L->Draw();
    sprintf(ctxt,"p_{T} < 5 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsEta_eta_L.eps");
    c.SaveAs(DIR+type+"_mresVsEta_eta_L.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_mresVsEta_eta_M->Draw();
    sprintf(ctxt,"5 < p_{T} < 15 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsEta_eta_M.eps");
    c.SaveAs(DIR+type+"_mresVsEta_eta_M.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_mresVsEta_eta_H->Draw();
    sprintf(ctxt,"p_{T} > 15 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_mresVsEta_eta_H.eps");
    c.SaveAs(DIR+type+"_mresVsEta_eta_H.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_resVsEta_z0->Draw();
  h2_resVsEta_z0->Write();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_resVsEta_z0.eps");
  c.SaveAs(DIR+type+"_resVsEta_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h2_resVsEta_z0_L->Draw();
    h2_resVsEta_z0_L->Write();
    sprintf(ctxt,"Full p_{T} range");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_z0_L.eps");
    c.SaveAs(DIR+type+"_resVsEta_z0_L.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsEta_z0_M->Draw();
    h2_resVsEta_z0_M->Write();
    sprintf(ctxt,"Full p_{T} range");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_z0_M.eps");
    c.SaveAs(DIR+type+"_resVsEta_z0_M.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h2_resVsEta_z0_H->Draw();
    h2_resVsEta_z0_H->Write();
    sprintf(ctxt,"Full p_{T} range");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_z0_H.eps");
    c.SaveAs(DIR+type+"_resVsEta_z0_H.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h2_res5pVsEta_z0->Draw();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5pVsEta_z0.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsEta_d0->Draw();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5pVsEta_d0.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_d0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");


  h2_resVsEta_phi->Draw();
  h2_resVsEta_phi->Write();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_resVsEta_phi.eps");
  c.SaveAs(DIR+type+"_resVsEta_phi.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_phi_L->Write();
  h2_resVsEta_phi_M->Write();
  h2_resVsEta_phi_H->Write();

  h2_res5pVsEta_phi->Draw();
  sprintf(ctxt,"Full p_{T} range");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5pVsEta_phi.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_phi.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_pt->Draw();
  c.SaveAs(DIR+type+"_resVsEta_pt.eps");
  c.SaveAs(DIR+type+"_resVsEta_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsEta_pt->Draw();
  c.SaveAs(DIR+type+"_res5pVsEta_pt.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsEta_pt->Draw();
  h2_res5pVsEta_pt->Write();
  c.SaveAs(DIR+type+"_res5pVsEta_pt.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_ptRel->Draw();
  h2_resVsEta_ptRel->Write();
  c.SaveAs(DIR+type+"_resVsEta_ptRel.eps");
  c.SaveAs(DIR+type+"_resVsEta_ptRel.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_ptRel_L->Draw();
  h2_resVsEta_ptRel_L->Write();
  c.SaveAs(DIR+type+"_resVsEta_ptRel_L.eps");
  c.SaveAs(DIR+type+"_resVsEta_ptRel_L.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_ptRel_M->Draw();
  h2_resVsEta_ptRel_M->Write();
  c.SaveAs(DIR+type+"_resVsEta_ptRel_M.eps");
  c.SaveAs(DIR+type+"_resVsEta_ptRel_M.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_resVsEta_ptRel_H->Draw();
  h2_resVsEta_ptRel_H->Write();
  c.SaveAs(DIR+type+"_resVsEta_ptRel_H.eps");
  c.SaveAs(DIR+type+"_resVsEta_ptRel_H.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h2_res5pVsEta_ptRel->Draw();
  c.SaveAs(DIR+type+"_res5pVsEta_ptRel.eps");
  c.SaveAs(DIR+type+"_res5pVsEta_ptRel.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");


  if (doGausFit) {
    h3_resVsEta_eta_L->Write();
    h3_resVsEta_z0_L->Write();
    h3_resVsEta_phi_L->Write();
    h3_resVsEta_ptRel_L->Write();
    
    h3_resVsEta_eta_M->Write();
    h3_resVsEta_z0_M->Write();
    h3_resVsEta_phi_M->Write();
    h3_resVsEta_ptRel_M->Write();
    
    h3_resVsEta_eta_H->Write();
    h3_resVsEta_z0_H->Write();
    h3_resVsEta_phi_H->Write();
    h3_resVsEta_ptRel_H->Write();
  }


  // ----------------------------------------------------------------------------------------------------------------
  // track quality plots
  // ----------------------------------------------------------------------------------------------------------------

  if (doDetailedPlots) {

    // draw and save plots
    h_match_trk_nstub->Draw();
    c.SaveAs(DIR+type+"_match_trk_nstub.png");
    c.SaveAs(DIR+type+"_match_trk_nstub.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_nstub_C->Draw();
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_nstub_C.png");
    c.SaveAs(DIR+type+"_match_trk_nstub_C.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_nstub_I->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_nstub_I.png");
    c.SaveAs(DIR+type+"_match_trk_nstub_I.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_nstub_F->Draw();
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_nstub_F.png");
    c.SaveAs(DIR+type+"_match_trk_nstub_F.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
    
  h_match_trk_chi2->Draw();
  sprintf(ctxt,"|eta| < 2.5; Full p_{T} range");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match_trk_chi2.png");
  c.SaveAs(DIR+type+"_match_trk_chi2.eps");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h_match5p_trk_chi2->Draw();
  sprintf(ctxt,"|eta| < 2.5; Full p_{T} range");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match5p_trk_chi2.png");
  c.SaveAs(DIR+type+"_match5p_trk_chi2.eps");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h_match_trk_chi2_C_L->Draw();
    sprintf(ctxt,"|eta| < 0.8; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_C_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_C_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_I_L->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_I_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_I_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_F_L->Draw();
    sprintf(ctxt,"|eta| > 1.6; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_F_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_F_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_C_M->Draw();
    sprintf(ctxt,"|eta| < 0.8; 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_C_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_C_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_I_M->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6, 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_I_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_I_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_F_M->Draw();
    sprintf(ctxt,"|eta| > 1.6; 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_F_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_F_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_C_H->Draw();
    sprintf(ctxt,"|eta| < 0.8; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_C_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_C_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_I_H->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_I_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_I_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_F_H->Draw();
    sprintf(ctxt,"|eta| > 1.6; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_F_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_F_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
    
  h_match_trk_chi2_dof->Draw();
  sprintf(ctxt,"|eta| < 2.5; Full p_{T} range");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match_trk_chi2_dof.png");
  c.SaveAs(DIR+type+"_match_trk_chi2_dof.eps");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h_match5p_trk_chi2_dof->Draw();
  sprintf(ctxt,"|eta| < 2.5; Full p_{T} range");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match5p_trk_chi2_dof.png");
  c.SaveAs(DIR+type+"_match5p_trk_chi2_dof.eps");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (doDetailedPlots) {
    h_match_trk_chi2_dof_C_L->Draw();
    sprintf(ctxt,"|eta| < 0.8; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_I_L->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_F_L->Draw();
    sprintf(ctxt,"|eta| > 1.6; p_{T} < 5 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_L.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_L.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_C_M->Draw();
    sprintf(ctxt,"|eta| < 0.8; 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_I_M->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6, 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_F_M->Draw();
    sprintf(ctxt,"|eta| > 1.6; 5 < p_{T} < 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_M.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_M.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_C_H->Draw();
    sprintf(ctxt,"|eta| < 0.8; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_I_H->Draw();
    sprintf(ctxt,"0.8 < |eta| < 1.6; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_F_H->Draw();
    sprintf(ctxt,"|eta| > 1.6; p_{T} > 15 GeV");
    mySmallText(0.52,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_H.png");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_H.eps");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
   
 
  // ----------------------------------------------------------------------------------------------------------------
  // efficiency plots  
  // ----------------------------------------------------------------------------------------------------------------

  // rebin pt/phi plots
  h_tp_pt->Rebin(2);
  h_tp_phi->Rebin(2);
  h_match_tp_pt->Rebin(2);
  h_match_tp_phi->Rebin(2);


  // calculate the effeciency
  h_match_tp_pt->Sumw2();
  h_tp_pt->Sumw2();
  TH1F* h_eff_pt = (TH1F*) h_match_tp_pt->Clone();
  h_eff_pt->SetName("eff_pt");
  h_eff_pt->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt->Divide(h_match_tp_pt, h_tp_pt, 1.0, 1.0, "B");

  h_match_tp_pt_L->Sumw2();
  h_tp_pt_L->Sumw2();
  TH1F* h_eff_pt_L = (TH1F*) h_match_tp_pt_L->Clone();
  h_eff_pt_L->SetName("eff_pt_L");
  h_eff_pt_L->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt_L->Divide(h_match_tp_pt_L, h_tp_pt_L, 1.0, 1.0, "B");

  h_match_tp_eta->Sumw2();
  h_tp_eta->Sumw2();
  TH1F* h_eff_eta = (TH1F*) h_match_tp_eta->Clone();
  h_eff_eta->SetName("eff_eta");
  h_eff_eta->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta->Divide(h_match_tp_eta, h_tp_eta, 1.0, 1.0, "B");
  
  h_match_tp_eta_H->Sumw2();
  h_tp_eta_H->Sumw2();
  TH1F* h_eff_eta_H = (TH1F*) h_match_tp_eta_H->Clone();
  h_eff_eta_H->SetName("eff_eta_H");
  h_eff_eta_H->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta_H->Divide(h_match_tp_eta_H, h_tp_eta_H, 1.0, 1.0, "B");

  h_match_tp_phi->Sumw2();
  h_tp_phi->Sumw2();
  TH1F* h_eff_phi = (TH1F*) h_match_tp_phi->Clone();
  h_eff_phi->SetName("eff_phi");
  h_eff_phi->GetYaxis()->SetTitle("Efficiency");
  h_eff_phi->Divide(h_match_tp_phi, h_tp_phi, 1.0, 1.0, "B");

  h_match_tp_z0->Sumw2();
  h_tp_z0->Sumw2();
  TH1F* h_eff_z0 = (TH1F*) h_match_tp_z0->Clone();
  h_eff_z0->SetName("eff_z0");
  h_eff_z0->GetYaxis()->SetTitle("Efficiency");
  h_eff_z0->Divide(h_match_tp_z0, h_tp_z0, 1.0, 1.0, "B");

  h_match_tp_d0->Sumw2();
  h_tp_d0->Sumw2();
  TH1F* h_eff_d0 = (TH1F*) h_match_tp_d0->Clone();
  h_eff_d0->SetName("eff_d0");
  h_eff_d0->GetYaxis()->SetTitle("Efficiency");
  h_eff_d0->Divide(h_match_tp_d0, h_tp_d0, 1.0, 1.0, "B");


  // set the axis range
  h_eff_pt  ->SetAxisRange(0,1.1,"Y");
  h_eff_pt_L->SetAxisRange(0,1.1,"Y");
  h_eff_eta ->SetAxisRange(0,1.1,"Y");
  h_eff_eta_H ->SetAxisRange(0,1.1,"Y");
  h_eff_phi ->SetAxisRange(0,1.1,"Y");
  h_eff_z0  ->SetAxisRange(0,1.1,"Y");
  h_eff_d0  ->SetAxisRange(0,1.1,"Y");

  if (type.Contains("Electron") || type.Contains("Pion")) h_eff_pt->SetAxisRange(0,49,"X");

  gPad->SetGridx();
  gPad->SetGridy();

  // draw and save plots
  h_eff_pt->Draw();
  h_eff_pt->Write();
  c.SaveAs(DIR+type+"_eff_pt.eps");
  c.SaveAs(DIR+type+"_eff_pt.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  if (type.Contains("SingleMu")) {
    h_eff_pt->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
    c.SaveAs(DIR+type+"_eff_pt_zoom.eps");
    c.SaveAs(DIR+type+"_eff_pt_zoom.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h_eff_pt_L->Draw();
  h_eff_pt_L->Write();
  c.SaveAs(DIR+type+"_eff_pt_L.eps");
  c.SaveAs(DIR+type+"_eff_pt_L.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  
  h_eff_eta->Draw();
  h_eff_eta->Write();
  c.SaveAs(DIR+type+"_eff_eta.eps");
  c.SaveAs(DIR+type+"_eff_eta.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  
  if (type.Contains("SingleMu")) {
    h_eff_eta->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
    c.SaveAs(DIR+type+"_eff_eta_zoom.eps");
    c.SaveAs(DIR+type+"_eff_eta_zoom.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h_eff_eta_H->Draw();
  h_eff_eta_H->Write();
  c.SaveAs(DIR+type+"_eff_eta_H.eps");
  c.SaveAs(DIR+type+"_eff_eta_H.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h_eff_phi->Draw();
  h_eff_phi->Write();
  c.SaveAs(DIR+type+"_eff_phi.eps");
  c.SaveAs(DIR+type+"_eff_phi.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  
  if (type.Contains("SingleMu")) {
    h_eff_phi->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
    c.SaveAs(DIR+type+"_eff_phi_zoom.eps");
    c.SaveAs(DIR+type+"_eff_phi_zoom.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }

  h_eff_z0->Draw();
  h_eff_z0->Write();
  c.SaveAs(DIR+type+"_eff_z0.eps");
  c.SaveAs(DIR+type+"_eff_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  h_eff_d0->Draw();
  h_eff_d0->Write();
  c.SaveAs(DIR+type+"_eff_d0.eps");
  c.SaveAs(DIR+type+"_eff_d0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

  gPad->SetGridx(0);
  gPad->SetGridy(0);


  // ----------------------------------------------------------------------------------------------------------------
  // resolution plots
  // ----------------------------------------------------------------------------------------------------------------

  float rms = 0;

  if (doDetailedPlots) {

    // draw and save plots
    h_res_pt->Draw();
    rms = h_res_pt->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_pt.eps");
    c.SaveAs(DIR+type+"_res_pt.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_ptRel->Draw();
    rms = h_res_ptRel->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_ptRel.eps");
    c.SaveAs(DIR+type+"_res_ptRel.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_eta->Draw();
    rms = h_res_eta->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_eta.eps");
    c.SaveAs(DIR+type+"_res_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_phi->Draw();
    rms = h_res_phi->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_phi.eps");
    c.SaveAs(DIR+type+"_res_phi.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0->Draw();
    rms = h_res_z0->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0.eps");
    c.SaveAs(DIR+type+"_res_z0.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_C->Draw();
    rms = h_res_z0_C->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C.eps");
    c.SaveAs(DIR+type+"_res_z0_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_I->Draw();
    rms = h_res_z0_I->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I.eps");
    c.SaveAs(DIR+type+"_res_z0_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_F->Draw();
    rms = h_res_z0_F->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F.eps");
    c.SaveAs(DIR+type+"_res_z0_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

    h_res_z0_C_low->Draw();
    h_res_z0_C_low->Write();
    rms = h_res_z0_C_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C_low.eps");
    c.SaveAs(DIR+type+"_res_z0_C_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_I_low->Draw();
    h_res_z0_I_low->Write();
    rms = h_res_z0_I_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I_low.eps");
    c.SaveAs(DIR+type+"_res_z0_I_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_F_low->Draw();
    h_res_z0_F_low->Write();
    rms = h_res_z0_F_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F_low.eps");
    c.SaveAs(DIR+type+"_res_z0_F_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");

    h_res_z0_C_high->Draw();
    h_res_z0_C_high->Write();
    rms = h_res_z0_C_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C_high.eps");
    c.SaveAs(DIR+type+"_res_z0_C_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_I_high->Draw();
    h_res_z0_I_high->Write();
    rms = h_res_z0_I_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I_high.eps");
    c.SaveAs(DIR+type+"_res_z0_I_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_F_high->Draw();
    h_res_z0_F_high->Write();
    rms = h_res_z0_F_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F_high.eps");
    c.SaveAs(DIR+type+"_res_z0_F_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");




    h_res_z0_L->Draw();
    h_res_z0_L->Write();
    rms = h_res_z0_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} < 5 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_L.eps");
    c.SaveAs(DIR+type+"_res_z0_L.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_M->Draw();
    h_res_z0_M->Write();
    rms = h_res_z0_M->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"5 < p_{T} < 15 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_M.eps");
    c.SaveAs(DIR+type+"_res_z0_M.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res_z0_H->Draw();
    h_res_z0_H->Write();
    rms = h_res_z0_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} > 15 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_H.eps");
    c.SaveAs(DIR+type+"_res_z0_H.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");



    h_res5p_pt->Draw();
    rms = h_res5p_pt->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_pt.eps");
    c.SaveAs(DIR+type+"_res5p_pt.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_ptRel->Draw();
    rms = h_res5p_ptRel->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_ptRel.eps");
    c.SaveAs(DIR+type+"_res5p_ptRel.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_eta->Draw();
    rms = h_res5p_eta->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_eta.eps");
    c.SaveAs(DIR+type+"_res5p_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_phi->Draw();
    rms = h_res5p_phi->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_phi.eps");
    c.SaveAs(DIR+type+"_res5p_phi.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
        
  }

  h_res5p_z0->Draw();
  rms = h_res5p_z0->GetRMS();
  sprintf(ctxt,"RMS = %.4f",rms);	
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5p_z0.eps");
  c.SaveAs(DIR+type+"_res5p_z0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  
  h_res5p_d0->Draw();
  rms = h_res5p_d0->GetRMS();
  sprintf(ctxt,"RMS = %.4f",rms);	
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_res5p_d0.eps");
  c.SaveAs(DIR+type+"_res5p_d0.png");
  if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  


  if (doDetailedPlots) {
    
    h_res5p_d0_C->Draw();
    h_res5p_d0_C->Write();
    rms = h_res5p_d0_C->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_C.eps");
    c.SaveAs(DIR+type+"_res5p_d0_C.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_I->Draw();
    h_res5p_d0_I->Write();
    rms = h_res5p_d0_I->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_I.eps");
    c.SaveAs(DIR+type+"_res5p_d0_I.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_F->Draw();
    h_res5p_d0_F->Write();
    rms = h_res5p_d0_F->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_F.eps");
    c.SaveAs(DIR+type+"_res5p_d0_F.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    
    h_res5p_d0_C_low->Draw();
    h_res5p_d0_C_low->Write();
    rms = h_res5p_d0_C_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_C_low.eps");
    c.SaveAs(DIR+type+"_res5p_d0_C_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_I_low->Draw();
    h_res5p_d0_I_low->Write();
    rms = h_res5p_d0_I_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_I_low.eps");
    c.SaveAs(DIR+type+"_res5p_d0_I_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_F_low->Draw();
    h_res5p_d0_F_low->Write();
    rms = h_res5p_d0_F_low->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_F_low.eps");
    c.SaveAs(DIR+type+"_res5p_d0_F_low.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    
    h_res5p_d0_C_high->Draw();
    h_res5p_d0_C_high->Write();
    rms = h_res5p_d0_C_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_C_high.eps");
    c.SaveAs(DIR+type+"_res5p_d0_C_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_I_high->Draw();
    h_res5p_d0_I_high->Write();
    rms = h_res5p_d0_I_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_I_high.eps");
    c.SaveAs(DIR+type+"_res5p_d0_I_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_F_high->Draw();
    h_res5p_d0_F_high->Write();
    rms = h_res5p_d0_F_high->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_F_high.eps");
    c.SaveAs(DIR+type+"_res5p_d0_F_high.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    
    h_res5p_d0_L->Draw();
    h_res5p_d0_L->Write();
    rms = h_res5p_d0_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} < 5 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_L.eps");
    c.SaveAs(DIR+type+"_res5p_d0_L.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_M->Draw();
    h_res5p_d0_M->Write();
    rms = h_res5p_d0_M->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"5 < p_{T} < 15 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_M.eps");
    c.SaveAs(DIR+type+"_res5p_d0_M.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_res5p_d0_H->Draw();
    h_res5p_d0_H->Write();
    rms = h_res5p_d0_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} > 15 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res5p_d0_H.eps");
    c.SaveAs(DIR+type+"_res5p_d0_H.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
  }
  
 
  // ----------------------------------------------------------------------------------------------------------------
  // 2D histogram plots
  // ----------------------------------------------------------------------------------------------------------------

  if (doDetailedPlots) {

    // draw and save plots
    h_2d_logchi2_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_logchi2_eta.eps");
    c.SaveAs(DIR+type+"_2d_logchi2_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_2d_logchi2_dof_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_logchi2_dof_eta.eps");
    c.SaveAs(DIR+type+"_2d_logchi2_dof_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_2d_dz0_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_dz0_eta.eps");
    c.SaveAs(DIR+type+"_2d_dz0_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_2d_deta_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_deta_eta.eps");
    c.SaveAs(DIR+type+"_2d_deta_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_2d_dphi_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_dphi_eta.eps");
    c.SaveAs(DIR+type+"_2d_dphi_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_2d_dptRel_eta->Draw("colz");
    c.SaveAs(DIR+type+"_2d_dptRel_eta.eps");
    c.SaveAs(DIR+type+"_2d_dptRel_eta.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
  }
    
  // ----------------------------------------------------------------------------------------------------------------
  // logarithmic plots
  // ----------------------------------------------------------------------------------------------------------------
  
  // set y-axis to a log scale
  c.SetLogy();
    
  if (doDetailedPlots) {
    
    // draw and save plots
    h_match_trk_chi2->Draw();
    c.SaveAs(DIR+type+"_match_trk_chi2_log.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_log.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_C_H->Draw();
    h_match_trk_chi2_C_M->Draw("same");
    h_match_trk_chi2_C_M->SetLineColor(kRed);
    h_match_trk_chi2_C_L->Draw("same");
    h_match_trk_chi2_C_L->SetLineColor(kBlue);

    TLegend* lh = new TLegend(0.65,0.7,0.85,0.9);
    lh->SetFillStyle(0);
    lh->SetBorderSize(0);
    lh->SetTextSize(0.04);
    lh->AddEntry(h_match_trk_chi2_C_H," p_{T} > 15 GeV","l");
    lh->AddEntry(h_match_trk_chi2_C_M," 5 < p_{T} < 15 GeV","l");
    lh->AddEntry(h_match_trk_chi2_C_L," p_{T} < 5 GeV","l");
    lh->SetTextFont(42);
    lh->Draw();	
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.66,0.6,1,ctxt);

    c.SaveAs(DIR+type+"_match_trk_chi2_C_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_C_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_I_H->Draw();
    h_match_trk_chi2_I_M->Draw("same");
    h_match_trk_chi2_I_M->SetLineColor(kRed);
    h_match_trk_chi2_I_L->Draw("same");
    h_match_trk_chi2_I_L->SetLineColor(kBlue);
    lh->Draw();	
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.66,0.6,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_I_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_I_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_F_H->Draw();
    h_match_trk_chi2_F_M->Draw("same");
    h_match_trk_chi2_F_M->SetLineColor(kRed);
    h_match_trk_chi2_F_L->Draw("same");
    h_match_trk_chi2_F_L->SetLineColor(kBlue);
    lh->Draw();	
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.66,0.6,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_F_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_F_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof->Draw();
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_log.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_log.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_C_H->Draw();
    h_match_trk_chi2_dof_C_M->Draw("same");
    h_match_trk_chi2_dof_C_M->SetLineColor(kRed);
    h_match_trk_chi2_dof_C_L->Draw("same");
    h_match_trk_chi2_dof_C_L->SetLineColor(kBlue);
    lh->Draw();	
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.66,0.6,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_C_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_I_H->Draw();
    h_match_trk_chi2_dof_I_M->Draw("same");
    h_match_trk_chi2_dof_I_M->SetLineColor(kRed);
    h_match_trk_chi2_dof_I_L->Draw("same");
    h_match_trk_chi2_dof_I_L->SetLineColor(kBlue);
    lh->Draw();	
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.66,0.6,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_I_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf");
    
    h_match_trk_chi2_dof_F_H->Draw();
    h_match_trk_chi2_dof_F_M->Draw("same");
    h_match_trk_chi2_dof_F_M->SetLineColor(kRed);
    h_match_trk_chi2_dof_F_L->Draw("same");
    h_match_trk_chi2_dof_F_L->SetLineColor(kBlue);
    lh->Draw();	
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.66,0.6,1,ctxt);
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_log_comb.eps");
    c.SaveAs(DIR+type+"_match_trk_chi2_dof_F_log_comb.png");
    if (makeCanvas) c.SaveAs(type+"_canvas.pdf)"); // close the pdf file
  }
  else if (makeCanvas) {
    c.Clear();
    c.SaveAs(type+"_canvas.pdf)"); // close the pdf file
  }


  fout->Close();


}


void SetPlotStyle() {

  // from ATLAS plot style macro

  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetHistLineColor(1);

  gStyle->SetPalette(1);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20,26);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.4);
  gStyle->SetTitleYOffset(1.4);

  // use large fonts
  gStyle->SetTextFont(42);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelFont(42,"x");
  gStyle->SetTitleFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetTitleFont(42,"y");
  gStyle->SetLabelFont(42,"z");
  gStyle->SetTitleFont(42,"z");
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.05,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.05,"z");
  gStyle->SetTitleSize(0.05,"z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2.);
  gStyle->SetLineStyleString(2,"[12 12]");

  // get rid of error bar caps
  gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

}


void mySmallText(Double_t x,Double_t y,Color_t color,char *text) {
  Double_t tsize=0.044;
  TLatex l;
  l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);
}


