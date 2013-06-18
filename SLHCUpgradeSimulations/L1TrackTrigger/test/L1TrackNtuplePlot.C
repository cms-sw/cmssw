// ----------------------------------------------------------------------------------------------------------------
// Basic example script for making tracking performance plots using the ntuples produced by L1TrackNtupleMaker.cc
// By Louise Skinnari, July 2013
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
void L1TrackNtuplePlot(TString type) {

  SetPlotStyle();
  

  // ----------------------------------------------------------------------------------------------------------------
  // read ntuples
  if ( !(type=="SingleElectron_BE" || type=="SingleMuon_BE" || type=="SingleMuon_LB" || type=="TTBar_BE") ) {
    cout << "This ROOT file doesn't exist..." << endl;
    return;
  }

  TChain* tree = new TChain("L1TrackNtuple/eventTree");
  tree->Add("/afs/cern.ch/work/s/skinnari/public/CMSSW_6_1_2_SLHC2/"+type+"_TrkPerf.root");



  // ----------------------------------------------------------------------------------------------------------------
  // define leafs & branches

  // basic track properties, plotted for *all* tracks, regardless of origin
  vector<float>* trk_pt;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_z0;
  vector<int>*   trk_chi2;
  vector<int>*   trk_charge;
  vector<int>*   trk_nstub;

  // sim track properties
  vector<float>* simtrk_pt;
  vector<float>* simtrk_eta;
  vector<float>* simtrk_phi;
  vector<float>* simtrk_z0;
  vector<int>*   simtrk_id;   //simtrackID
  vector<int>*   simtrk_type; //PDG ID of the sim track (13==muon, 11==electron, and so on)

  // *sim track* properties, for sim tracks that are matched to an L1 track using simtrackID
  vector<float>* matchID_simtrk_pt;
  vector<float>* matchID_simtrk_eta;
  vector<float>* matchID_simtrk_phi;
  vector<float>* matchID_simtrk_z0;
  vector<int>*   matchID_simtrk_id;
  vector<int>*   matchID_simtrk_type;

  // *L1 track* properties, for sim tracks that are matched to an L1 track using simtrackID
  vector<float>* matchID_trk_pt;
  vector<float>* matchID_trk_eta;
  vector<float>* matchID_trk_phi;
  vector<float>* matchID_trk_z0;
  vector<int>*   matchID_trk_chi2; 
  vector<int>*   matchID_trk_charge;
  vector<int>*   matchID_trk_nstub;
  vector<int>*   matchID_trk_nmatch;

  // *sim track* properties, for sim tracks that are matched to an L1 track using dR
  vector<float>* matchDR_simtrk_pt;
  vector<float>* matchDR_simtrk_eta;
  vector<float>* matchDR_simtrk_phi;
  vector<float>* matchDR_simtrk_z0;
  vector<int>*   matchDR_simtrk_id;
  vector<int>*   matchDR_simtrk_type;

  // *L1 track* properties, for sim tracks that are matched to an L1 track using dR
  vector<float>* matchDR_trk_pt;
  vector<float>* matchDR_trk_eta;
  vector<float>* matchDR_trk_phi;
  vector<float>* matchDR_trk_z0;
  vector<int>*   matchDR_trk_chi2; 
  vector<int>*   matchDR_trk_charge;
  vector<int>*   matchDR_trk_nstub;
  vector<int>*   matchDR_trk_nmatch;

  
  TBranch* b_trk_pt;
  TBranch* b_trk_eta;
  TBranch* b_trk_phi;
  TBranch* b_trk_z0;
  TBranch* b_trk_chi2;
  TBranch* b_trk_charge;
  TBranch* b_trk_nstub;

  TBranch* b_simtrk_pt;
  TBranch* b_simtrk_eta;
  TBranch* b_simtrk_phi;
  TBranch* b_simtrk_z0;
  TBranch* b_simtrk_id;
  TBranch* b_simtrk_type;

  TBranch* b_matchID_simtrk_pt;
  TBranch* b_matchID_simtrk_eta;
  TBranch* b_matchID_simtrk_phi;
  TBranch* b_matchID_simtrk_z0;
  TBranch* b_matchID_simtrk_id;
  TBranch* b_matchID_simtrk_type;

  TBranch* b_matchID_trk_pt;
  TBranch* b_matchID_trk_eta;
  TBranch* b_matchID_trk_phi;
  TBranch* b_matchID_trk_z0;
  TBranch* b_matchID_trk_chi2; 
  TBranch* b_matchID_trk_charge;
  TBranch* b_matchID_trk_nstub;
  TBranch* b_matchID_trk_nmatch;

  TBranch* b_matchDR_simtrk_pt;
  TBranch* b_matchDR_simtrk_eta;
  TBranch* b_matchDR_simtrk_phi;
  TBranch* b_matchDR_simtrk_z0;
  TBranch* b_matchDR_simtrk_id;
  TBranch* b_matchDR_simtrk_type;

  TBranch* b_matchDR_trk_pt;
  TBranch* b_matchDR_trk_eta;
  TBranch* b_matchDR_trk_phi;
  TBranch* b_matchDR_trk_z0;
  TBranch* b_matchDR_trk_chi2; 
  TBranch* b_matchDR_trk_charge;
  TBranch* b_matchDR_trk_nstub;
  TBranch* b_matchDR_trk_nmatch;

  trk_pt  = 0;
  trk_eta = 0;
  trk_phi = 0;
  trk_z0  = 0;
  trk_chi2   = 0;
  trk_charge = 0;
  trk_nstub  = 0;

  simtrk_pt   = 0;
  simtrk_eta  = 0;
  simtrk_phi  = 0;
  simtrk_z0   = 0;
  simtrk_id   = 0;
  simtrk_type = 0;

  matchID_simtrk_pt  = 0;
  matchID_simtrk_eta = 0;
  matchID_simtrk_phi = 0;
  matchID_simtrk_z0  = 0;
  matchID_simtrk_id  = 0;
  matchID_simtrk_type  = 0;

  matchID_trk_pt  = 0;
  matchID_trk_eta = 0;
  matchID_trk_phi = 0;
  matchID_trk_z0  = 0;
  matchID_trk_chi2   = 0; 
  matchID_trk_charge = 0;
  matchID_trk_nstub  = 0;
  matchID_trk_nmatch = 0;
  
  matchDR_simtrk_pt  = 0;
  matchDR_simtrk_eta = 0;
  matchDR_simtrk_phi = 0;
  matchDR_simtrk_z0  = 0;
  matchDR_simtrk_id  = 0;
  matchDR_simtrk_type  = 0;

  matchDR_trk_pt  = 0;
  matchDR_trk_eta = 0;
  matchDR_trk_phi = 0;
  matchDR_trk_z0  = 0;
  matchDR_trk_chi2   = 0; 
  matchDR_trk_charge = 0;
  matchDR_trk_nstub  = 0;
  matchDR_trk_nmatch = 0;
  

  tree->SetBranchAddress("trk_pt",    &trk_pt,    &b_trk_pt);
  tree->SetBranchAddress("trk_eta",   &trk_eta,   &b_trk_eta);
  tree->SetBranchAddress("trk_phi",   &trk_phi,   &b_trk_phi);
  tree->SetBranchAddress("trk_z0",    &trk_z0,    &b_trk_z0);
  tree->SetBranchAddress("trk_chi2",  &trk_chi2,  &b_trk_chi2);
  tree->SetBranchAddress("trk_charge",&trk_charge,&b_trk_charge);
  tree->SetBranchAddress("trk_nstub", &trk_nstub, &b_trk_nstub);

  tree->SetBranchAddress("simtrk_pt",  &simtrk_pt,  &b_simtrk_pt);
  tree->SetBranchAddress("simtrk_eta", &simtrk_eta, &b_simtrk_eta);
  tree->SetBranchAddress("simtrk_phi", &simtrk_phi, &b_simtrk_phi);
  tree->SetBranchAddress("simtrk_z0",  &simtrk_z0,  &b_simtrk_z0);
  tree->SetBranchAddress("simtrk_id",  &simtrk_id,  &b_simtrk_id);
  tree->SetBranchAddress("simtrk_type",&simtrk_type,&b_simtrk_type);

  tree->SetBranchAddress("matchID_simtrk_pt",  &matchID_simtrk_pt,  &b_matchID_simtrk_pt);
  tree->SetBranchAddress("matchID_simtrk_eta", &matchID_simtrk_eta, &b_matchID_simtrk_eta);
  tree->SetBranchAddress("matchID_simtrk_phi", &matchID_simtrk_phi, &b_matchID_simtrk_phi);
  tree->SetBranchAddress("matchID_simtrk_z0",  &matchID_simtrk_z0,  &b_matchID_simtrk_z0);
  tree->SetBranchAddress("matchID_simtrk_id",  &matchID_simtrk_id,  &b_matchID_simtrk_id);
  tree->SetBranchAddress("matchID_simtrk_type",&matchID_simtrk_type,&b_matchID_simtrk_type);

  tree->SetBranchAddress("matchID_trk_pt",    &matchID_trk_pt,    &b_matchID_trk_pt);
  tree->SetBranchAddress("matchID_trk_eta",   &matchID_trk_eta,   &b_matchID_trk_eta);
  tree->SetBranchAddress("matchID_trk_phi",   &matchID_trk_phi,   &b_matchID_trk_phi);
  tree->SetBranchAddress("matchID_trk_z0",    &matchID_trk_z0,    &b_matchID_trk_z0);
  tree->SetBranchAddress("matchID_trk_chi2",  &matchID_trk_chi2,  &b_matchID_trk_chi2);
  tree->SetBranchAddress("matchID_trk_charge",&matchID_trk_charge,&b_matchID_trk_charge);
  tree->SetBranchAddress("matchID_trk_nstub", &matchID_trk_nstub, &b_matchID_trk_nstub);
  tree->SetBranchAddress("matchID_trk_nmatch",&matchID_trk_nmatch,&b_matchID_trk_nmatch);

  tree->SetBranchAddress("matchDR_simtrk_pt",  &matchDR_simtrk_pt,  &b_matchDR_simtrk_pt);
  tree->SetBranchAddress("matchDR_simtrk_eta", &matchDR_simtrk_eta, &b_matchDR_simtrk_eta);
  tree->SetBranchAddress("matchDR_simtrk_phi", &matchDR_simtrk_phi, &b_matchDR_simtrk_phi);
  tree->SetBranchAddress("matchDR_simtrk_z0",  &matchDR_simtrk_z0,  &b_matchDR_simtrk_z0);
  tree->SetBranchAddress("matchDR_simtrk_id",  &matchDR_simtrk_id,  &b_matchDR_simtrk_id);
  tree->SetBranchAddress("matchDR_simtrk_type",&matchDR_simtrk_type,&b_matchDR_simtrk_type);

  tree->SetBranchAddress("matchDR_trk_pt",    &matchDR_trk_pt,    &b_matchDR_trk_pt);
  tree->SetBranchAddress("matchDR_trk_eta",   &matchDR_trk_eta,   &b_matchDR_trk_eta);
  tree->SetBranchAddress("matchDR_trk_phi",   &matchDR_trk_phi,   &b_matchDR_trk_phi);
  tree->SetBranchAddress("matchDR_trk_z0",    &matchDR_trk_z0,    &b_matchDR_trk_z0);
  tree->SetBranchAddress("matchDR_trk_chi2",  &matchDR_trk_chi2,  &b_matchDR_trk_chi2);
  tree->SetBranchAddress("matchDR_trk_charge",&matchDR_trk_charge,&b_matchDR_trk_charge);
  tree->SetBranchAddress("matchDR_trk_nstub", &matchDR_trk_nstub, &b_matchDR_trk_nstub);
  tree->SetBranchAddress("matchDR_trk_nmatch",&matchDR_trk_nmatch,&b_matchDR_trk_nmatch);
  

  // ----------------------------------------------------------------------------------------------------------------
  // histograms
  TH1F* h_simtrk_pt  = new TH1F("simtrk_pt", ";Sim track p_{T} [GeV]; Sim tracks / 1.0 GeV", 100,0,100.0);
  TH1F* h_simtrk_eta = new TH1F("simtrk_eta",";Sim track #eta; Sim tracks / 0.1", 50,-2.5,2.5);
  TH1F* h_simtrk_phi = new TH1F("simtrk_phi",";Sim track #phi; Sim tracks / 0.1", 64,-3.2,3.2);

  TH1F* h_matchID_simtrk_pt  = new TH1F("matchID_simtrk_pt", ";Sim track p_{T} [GeV]; Sim tracks / 1.0 GeV", 100,0,100.0);
  TH1F* h_matchID_simtrk_eta = new TH1F("matchID_simtrk_eta",";Sim track #eta; Sim tracks / 0.1", 50,-2.5,2.5);
  TH1F* h_matchID_simtrk_phi = new TH1F("matchID_simtrk_phi",";Sim track #phi; Sim tracks / 0.1", 64,-3.2,3.2);

  TH1F* h_res_pt    = new TH1F("res_pt",   ";p_{T} resolution (L1 - sim) [GeV]; L1 tracks / 0.05", 200,-5.0,5.0);
  TH1F* h_res_ptRel = new TH1F("res_ptRel",";p_{T} resolution (L1 - sim) / p_{T}; L1 tracks / 0.01", 200,-1.0,1.0);
  TH1F* h_res_eta   = new TH1F("res_eta",  "; #eta resolution (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
  TH1F* h_res_phi   = new TH1F("res_phi",  "; #phi resolution (L1 - sim); L1 tracks / 0.0001", 100,-0.005,0.005);
  TH1F* h_res_z0    = new TH1F("res_z0",   ";z_{0} resolution (L1 - sim) [cm]; L1 tracks / 0.02", 100,-1,1);

  const int nRANGE = 19;
  TString ptrange[nRANGE] = {"5-10", "10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55",
  			     "55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95","95-100"};
  const int nETARANGE = 10;
  TString etarange[nETARANGE] = {"0-0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-1.25","1.25-1.5","1.5-1.75","1.75-2.0","2.0-2.25","2.25-2.5"};
  
  TH1F* h_resVsPt_pt[nRANGE];
  TH1F* h_resVsPt_ptRel[nRANGE];
  TH1F* h_resVsPt_eta[nRANGE];
  TH1F* h_resVsPt_phi[nRANGE];
  TH1F* h_resVsPt_z0[nRANGE];

  TH1F* h_resVsEta_pt[nETARANGE];
  TH1F* h_resVsEta_ptRel[nETARANGE];
  TH1F* h_resVsEta_eta[nETARANGE];
  TH1F* h_resVsEta_phi[nETARANGE];
  TH1F* h_resVsEta_z0[nETARANGE];

  for (int i=0; i<nRANGE; i++) {
    h_resVsPt_pt[i]    = new TH1F("resVsPt_pt_"+ptrange[i],   ";p_{T} resolution (L1 - sim) [GeV]; L1 tracks / 0.1", 100,-5.0,5.0);
    h_resVsPt_ptRel[i] = new TH1F("resVsPt_ptRel_"+ptrange[i],";p_{T} resolution (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-1.0,1.0);
    h_resVsPt_eta[i]   = new TH1F("resVsPt_eta_"+ptrange[i],  ";#eta resolution (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsPt_phi[i]   = new TH1F("resVsPt_phi_"+ptrange[i],  ";#phi resolution (L1 - sim); L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsPt_z0[i]    = new TH1F("resVsPt_z0_"+ptrange[i],   ";z_{0} resolution (L1 - sim) [cm]; L1 tracks / 0.02", 100,-1,1);
  }

  for (int i=0; i<nETARANGE; i++) {
    h_resVsEta_pt[i]    = new TH1F("resVsEta_pt_"+etarange[i],   ";p_{T} resolution (L1 - sim) [GeV]; L1 tracks / 0.1", 100,-5.0,5.0);
    h_resVsEta_ptRel[i] = new TH1F("resVsEta_ptRel_"+etarange[i],";p_{T} resolution (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-1.0,1.0);
    h_resVsEta_eta[i]   = new TH1F("resVsEta_eta_"+etarange[i],  ";#eta resolution (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_phi[i]   = new TH1F("resVsEta_phi_"+etarange[i],  ";#phi resolution (L1 - sim); L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_z0[i]    = new TH1F("resVsEta_z0_"+etarange[i],   ";z_{0} resolution (L1 - sim) [cm]; L1 tracks / 0.01", 100,-1,1);
  }


  
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
    // sim track loop
    for (int it=0; it<(int)simtrk_pt->size(); it++) {

      // for single-gun sample, ensure that only the actual single-gun particle is used
      if (type.Contains("SinglePion") && abs(simtrk_type->at(it)) != 211) continue;
      if (type.Contains("SingleMuon") && abs(simtrk_type->at(it)) != 13 && simtrk_id->at(it) != 1) continue;
      if (type.Contains("SingleElectron") && abs(simtrk_type->at(it)) != 11 && simtrk_id->at(it) != 1) continue;

      // for ttbar sample, look at pion tracks
      if (type.Contains("TTbar") && abs(simtrk_type->at(it)) != 211) continue;


      h_simtrk_pt ->Fill(simtrk_pt->at(it));
      h_simtrk_eta->Fill(simtrk_eta->at(it));
      h_simtrk_phi->Fill(simtrk_phi->at(it));
    }


    // ----------------------------------------------------------------------------------------------------------------
    // loop over matched L1tracks-simtracks
    for (int it=0; it<(int)matchID_simtrk_pt->size(); it++) {
      
      // make selection on chi2 for BarrelEndcap geometry
      if (type.Contains("BE") && matchID_trk_chi2->at(it) > 100.0) continue;

      // for single-gun sample, ensure that only the actual single-gun particle is used
      if (type.Contains("SinglePion") && abs(matchID_simtrk_type->at(it)) != 211) continue;
      if (type.Contains("SingleMuon") && abs(matchID_simtrk_type->at(it)) != 13 && matchID_simtrk_id->at(it) != 1) continue;
      if (type.Contains("SingleElectron") && abs(matchID_simtrk_type->at(it)) != 11 && matchID_simtrk_id->at(it) != 1) continue;

      // for ttbar sample, look at pion tracks
      if (type.Contains("TTbar") && abs(matchID_simtrk_type->at(it)) != 211) continue;

      
      h_matchID_simtrk_pt ->Fill(matchID_simtrk_pt->at(it));
      h_matchID_simtrk_eta->Fill(matchID_simtrk_eta->at(it));
      h_matchID_simtrk_phi->Fill(matchID_simtrk_phi->at(it));
	
      h_res_pt   ->Fill(matchID_trk_pt->at(it)  - matchID_simtrk_pt->at(it));
      h_res_ptRel->Fill((matchID_trk_pt->at(it) - matchID_simtrk_pt->at(it))/matchID_simtrk_pt->at(it));
      h_res_eta  ->Fill(matchID_trk_eta->at(it) - matchID_simtrk_eta->at(it));
      h_res_phi  ->Fill(matchID_trk_phi->at(it) - matchID_simtrk_phi->at(it));
      h_res_z0   ->Fill(matchID_trk_z0->at(it)  - matchID_simtrk_z0->at(it));
      
      for (int im=0; im<nRANGE; im++) {
       if ( (matchID_simtrk_pt->at(it) > (float)im*5.0+5.0) && (matchID_simtrk_pt->at(it) < (float)im*5.0+10.0) ) {
	  h_resVsPt_pt[im]   ->Fill(matchID_trk_pt->at(it)  - matchID_simtrk_pt->at(it));
	  h_resVsPt_ptRel[im]->Fill((matchID_trk_pt->at(it) - matchID_simtrk_pt->at(it))/matchID_simtrk_pt->at(it));
	  h_resVsPt_eta[im]  ->Fill(matchID_trk_eta->at(it) - matchID_simtrk_eta->at(it));
	  h_resVsPt_phi[im]  ->Fill(matchID_trk_phi->at(it) - matchID_simtrk_phi->at(it));
	  h_resVsPt_z0[im]   ->Fill(matchID_trk_z0->at(it)  - matchID_simtrk_z0->at(it));
	}
      }

     for (int im=0; im<nETARANGE; im++) {
       if ( (fabs(matchID_simtrk_eta->at(it)) > (float)im*0.25) && (fabs(matchID_simtrk_eta->at(it)) < (float)im*0.25+0.25) ) {
	  h_resVsEta_pt[im]   ->Fill(matchID_trk_pt->at(it)  - matchID_simtrk_pt->at(it));
	  h_resVsEta_ptRel[im]->Fill((matchID_trk_pt->at(it) - matchID_simtrk_pt->at(it))/matchID_simtrk_pt->at(it));
	  h_resVsEta_eta[im]  ->Fill(matchID_trk_eta->at(it) - matchID_simtrk_eta->at(it));
	  h_resVsEta_phi[im]  ->Fill(matchID_trk_phi->at(it) - matchID_simtrk_phi->at(it));
	  h_resVsEta_z0[im]   ->Fill(matchID_trk_z0->at(it)  - matchID_simtrk_z0->at(it));
	}
      }
      
    }
    
    
  } // end of event loop
  // ----------------------------------------------------------------------------------------------------------------
  

  // ----------------------------------------------------------------------------------------------------------------
  // 2D plots  
  // ----------------------------------------------------------------------------------------------------------------

  TH1F* h2_resVsPt_pt    = new TH1F("resVsPt2_pt",   ";Sim track p_{T} [GeV]; RMS(p_{T} resolution)", 19,0,19);
  TH1F* h2_resVsPt_ptRel = new TH1F("resVsPt2_ptRel",";Sim track p_{T} [GeV]; RMS(p_{T} resolution / p_{T})", 19,0,19);
  TH1F* h2_resVsPt_eta   = new TH1F("resVsPt2_eta",  ";Sim track p_{T} [GeV]; RMS(#eta resolution)", 19,0,19);
  TH1F* h2_resVsPt_phi   = new TH1F("resVsPt2_phi",  ";Sim track p_{T} [GeV]; RMS(#phi resolution)", 19,0,19);
  TH1F* h2_resVsPt_z0    = new TH1F("resVsPt2_z0",   ";Sim track p_{T} [GeV]; RMS(z_{0} resolution)", 19,0,19);

  TH1F* h2_resVsEta_pt    = new TH1F("resVsEta2_pt",   ";Sim track #eta; RMS(p_{T} resolution)", 10,0,10);
  TH1F* h2_resVsEta_ptRel = new TH1F("resVsEta2_ptRel",";Sim track #eta; RMS(p_{T} resolution / p_{T})", 10,0,10);
  TH1F* h2_resVsEta_eta   = new TH1F("resVsEta2_eta",  ";Sim track #eta; RMS(#eta resolution)", 10,0,10);
  TH1F* h2_resVsEta_phi   = new TH1F("resVsEta2_phi",  ";Sim track #eta; RMS(#phi resolution)", 10,0,10);
  TH1F* h2_resVsEta_z0    = new TH1F("resVsEta2_z0",   ";Sim track #eta; RMS(z_{0} resolution)", 10,0,10);

  for (int i=0; i<nRANGE; i++) {
    h2_resVsPt_pt->GetXaxis()->SetBinLabel(i+1,ptrange[i]);
    h2_resVsPt_ptRel->GetXaxis()->SetBinLabel(i+1,ptrange[i]);
    h2_resVsPt_eta->GetXaxis()->SetBinLabel(i+1,ptrange[i]);
    h2_resVsPt_phi->GetXaxis()->SetBinLabel(i+1,ptrange[i]);
    h2_resVsPt_z0->GetXaxis()->SetBinLabel(i+1,ptrange[i]);

    h2_resVsPt_pt->SetBinContent(i+1, h_resVsPt_pt[i]->GetRMS());
    h2_resVsPt_pt->SetBinError(i+1, h_resVsPt_pt[i]->GetRMSError());
    h2_resVsPt_ptRel->SetBinContent(i+1, h_resVsPt_ptRel[i]->GetRMS());
    h2_resVsPt_ptRel->SetBinError(i+1, h_resVsPt_ptRel[i]->GetRMSError());
    h2_resVsPt_eta->SetBinContent(i+1, h_resVsPt_eta[i]->GetRMS());
    h2_resVsPt_eta->SetBinError(i+1, h_resVsPt_eta[i]->GetRMSError());
    h2_resVsPt_phi->SetBinContent(i+1, h_resVsPt_phi[i]->GetRMS());
    h2_resVsPt_phi->SetBinError(i+1, h_resVsPt_phi[i]->GetRMSError());
    h2_resVsPt_z0->SetBinContent(i+1, h_resVsPt_z0[i]->GetRMS());
    h2_resVsPt_z0->SetBinError(i+1, h_resVsPt_z0[i]->GetRMSError());
  }

  for (int i=0; i<nETARANGE; i++) {
    h2_resVsEta_pt->GetXaxis()->SetBinLabel(i+1,etarange[i]);
    h2_resVsEta_ptRel->GetXaxis()->SetBinLabel(i+1,etarange[i]);
    h2_resVsEta_eta->GetXaxis()->SetBinLabel(i+1,etarange[i]);
    h2_resVsEta_phi->GetXaxis()->SetBinLabel(i+1,etarange[i]);
    h2_resVsEta_z0->GetXaxis()->SetBinLabel(i+1,etarange[i]);

    h2_resVsEta_pt->SetBinContent(i+1, h_resVsEta_pt[i]->GetRMS());
    h2_resVsEta_pt->SetBinError(i+1, h_resVsEta_pt[i]->GetRMSError());
    h2_resVsEta_ptRel->SetBinContent(i+1, h_resVsEta_ptRel[i]->GetRMS());
    h2_resVsEta_ptRel->SetBinError(i+1, h_resVsEta_ptRel[i]->GetRMSError());
    h2_resVsEta_eta->SetBinContent(i+1, h_resVsEta_eta[i]->GetRMS());
    h2_resVsEta_eta->SetBinError(i+1, h_resVsEta_eta[i]->GetRMSError());
    h2_resVsEta_phi->SetBinContent(i+1, h_resVsEta_phi[i]->GetRMS());
    h2_resVsEta_phi->SetBinError(i+1, h_resVsEta_phi[i]->GetRMSError());
    h2_resVsEta_z0->SetBinContent(i+1, h_resVsEta_z0[i]->GetRMS());
    h2_resVsEta_z0->SetBinError(i+1, h_resVsEta_z0[i]->GetRMSError());
  }



  h2_resVsPt_pt->SetMinimum(0);
  h2_resVsPt_ptRel->SetMinimum(0);
  h2_resVsPt_eta->SetMinimum(0);
  h2_resVsPt_phi->SetMinimum(0);
  h2_resVsPt_z0->SetMinimum(0);

  h2_resVsEta_pt->SetMinimum(0);
  h2_resVsEta_ptRel->SetMinimum(0);
  h2_resVsEta_eta->SetMinimum(0);
  h2_resVsEta_phi->SetMinimum(0);
  h2_resVsEta_z0->SetMinimum(0);


  TCanvas c2d;
  h2_resVsPt_pt->Draw();
  c2d.SaveAs(type+"_resVsPt_pt.png");
  c2d.SaveAs(type+"_resVsPt_pt.eps");

  h2_resVsPt_ptRel->Draw();
  c2d.SaveAs(type+"_resVsPt_ptRel.png");
  c2d.SaveAs(type+"_resVsPt_ptRel.eps");

  h2_resVsPt_eta->Draw();
  c2d.SaveAs(type+"_resVsPt_eta.png");
  c2d.SaveAs(type+"_resVsPt_eta.eps");

  h2_resVsPt_phi->Draw();
  c2d.SaveAs(type+"_resVsPt_phi.png");
  c2d.SaveAs(type+"_resVsPt_phi.eps");

  h2_resVsPt_z0->Draw();
  c2d.SaveAs(type+"_resVsPt_z0.png");
  c2d.SaveAs(type+"_resVsPt_z0.eps");

  h2_resVsEta_pt->Draw();
  c2d.SaveAs(type+"_resVsEta_pt.png");
  c2d.SaveAs(type+"_resVsEta_pt.eps");

  h2_resVsEta_ptRel->Draw();
  c2d.SaveAs(type+"_resVsEta_ptRel.png");
  c2d.SaveAs(type+"_resVsEta_ptRel.eps");

  h2_resVsEta_eta->Draw();
  c2d.SaveAs(type+"_resVsEta_eta.png");
  c2d.SaveAs(type+"_resVsEta_eta.eps");

  h2_resVsEta_phi->Draw();
  c2d.SaveAs(type+"_resVsEta_phi.png");
  c2d.SaveAs(type+"_resVsEta_phi.eps");

  h2_resVsEta_z0->Draw();
  c2d.SaveAs(type+"_resVsEta_z0.png");
  c2d.SaveAs(type+"_resVsEta_z0.eps");


  // ----------------------------------------------------------------------------------------------------------------
  // efficiency plots  
  // ----------------------------------------------------------------------------------------------------------------
  h_matchID_simtrk_pt->Rebin(2);
  h_matchID_simtrk_phi->Rebin(2);
  h_simtrk_pt->Rebin(2);
  h_simtrk_phi->Rebin(2);

  h_matchID_simtrk_pt->Sumw2();
  h_simtrk_pt->Sumw2();
  TH1F* h_eff_pt = (TH1F*) h_matchID_simtrk_pt->Clone();
  h_eff_pt->SetName("eff_pt");
  h_eff_pt->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt->Divide(h_matchID_simtrk_pt, h_simtrk_pt, 1.0, 1.0, "B");

  h_matchID_simtrk_eta->Sumw2();
  h_simtrk_eta->Sumw2();
  TH1F* h_eff_eta = (TH1F*) h_matchID_simtrk_eta->Clone();
  h_eff_eta->SetName("eff_eta");
  h_eff_eta->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta->Divide(h_matchID_simtrk_eta, h_simtrk_eta, 1.0, 1.0, "B");
  
  h_matchID_simtrk_phi->Sumw2();
  h_simtrk_phi->Sumw2();
  TH1F* h_eff_phi = (TH1F*) h_matchID_simtrk_phi->Clone();
  h_eff_phi->SetName("eff_phi");
  h_eff_phi->GetYaxis()->SetTitle("Efficiency");
  h_eff_phi->Divide(h_matchID_simtrk_phi, h_simtrk_phi, 1.0, 1.0, "B");


  h_eff_pt->SetAxisRange(0,1.1,"Y");
  h_eff_eta->SetAxisRange(0,1.1,"Y");
  h_eff_phi->SetAxisRange(0,1.1,"Y");

  TCanvas c;

  h_eff_pt->Draw();
  c.SaveAs(type+"_eff_pt.eps");
  c.SaveAs(type+"_eff_pt.png");
  h_eff_eta->Draw();
  c.SaveAs(type+"_eff_eta.eps");
  c.SaveAs(type+"_eff_eta.png");
  h_eff_phi->Draw();
  c.SaveAs(type+"_eff_phi.eps");
  c.SaveAs(type+"_eff_phi.png");


  // ----------------------------------------------------------------------------------------------------------------
  // resolution plots
  // ----------------------------------------------------------------------------------------------------------------

  char stxt[500];
  float rms = 0;

  h_res_pt->Draw();
  rms  = h_res_pt->GetRMS();
  sprintf(stxt,"RMS  = %.4f",rms);	
  mySmallText(0.22,0.82,1,stxt);
  c.SaveAs(type+"_res_pt.eps");
  c.SaveAs(type+"_res_pt.png");

  h_res_ptRel->Draw();
  rms  = h_res_ptRel->GetRMS();
  sprintf(stxt,"RMS  = %.4f",rms);	
  mySmallText(0.22,0.82,1,stxt);
  c.SaveAs(type+"_res_ptRel.eps");
  c.SaveAs(type+"_res_ptRel.png");

  h_res_eta->Draw();
  rms  = h_res_eta->GetRMS();
  sprintf(stxt,"RMS  = %.3e",rms);	
  mySmallText(0.22,0.82,1,stxt);
  c.SaveAs(type+"_res_eta.eps");
  c.SaveAs(type+"_res_eta.png");

  h_res_phi->Draw();
  rms  = h_res_phi->GetRMS();
  sprintf(stxt,"RMS  = %.3e",rms);	
  mySmallText(0.22,0.82,1,stxt);
  c.SaveAs(type+"_res_phi.eps");
  c.SaveAs(type+"_res_phi.png");

  h_res_z0->Draw();
  rms  = h_res_z0->GetRMS();
  sprintf(stxt,"RMS  = %.4f",rms);	
  mySmallText(0.22,0.82,1,stxt);
  c.SaveAs(type+"_res_z0.eps");
  c.SaveAs(type+"_res_z0.png");

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


