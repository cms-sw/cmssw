
#include <iostream>
#include <string>
#include "TFile.h"
#include "THashList.h"
#include "TH1.h"
#include "TKey.h"
#include "TClass.h"
#include "TSystem.h"

int makeCompositeEff( TString fname, TString assocName )
{
  int a = makeValidTrackEff(fname,assocName);
  return a;
}

int makeValidTrackEff( TString fname, TString assocName)
{
  
  TFile* source = TFile::Open( fname , "UPDATE");
  if( source==0 ){
    return 1;
  }

  bool glbDir = source->cd("DQMData/Track/cutsGLB_"+assocName);
  if(!glbDir) return -1;
  
  TH1F *glbSimRec_eta, *glbSimRec_pt, *glbreco2sim_eta, *glbreco_eta;
  TH1F *glbeffic, *glbeffic_pt, *glbfakerate;
  gDirectory->GetObject("num_assoc(simToReco)_eta",glbSimRec_eta);
  gDirectory->GetObject("num_assoc(simToReco)_pT",glbSimRec_pt);
  //gDirectory->GetObject("num_assoc(recoToSim)_eta",glbreco2sim_eta);
  //gDirectory->GetObject("num_reco_eta",glbreco_eta);
  //glbeffic = (TH1*)gDirectory->Get("effic")->Clone(); glbeffic->Reset();
  //glbeffic_pt = (TH1*)gDirectory->Get("efficPt")->Clone(); glbeffic_pt->Reset();
  //glbfakerate = (TH1*)gDirectory->Get("fakerate")->Clone(); glbfakerate->Reset();  

  bool staDir = source->cd("DQMData/Track/cutsSTA_"+assocName);
  if(!staDir) return -1;


  TH1F *staSimRec_eta, *staSimRec_pt, *stareco2sim_eta, *stareco_eta;
  TH1F *staeffic, *staeffic_pt, *stafakerate;
  gDirectory->GetObject("num_assoc(simToReco)_eta",staSimRec_eta);
  gDirectory->GetObject("num_assoc(simToReco)_pT",staSimRec_pt);
  //gDirectory->GetObject("num_assoc(recoToSim)_eta",stareco2sim_eta);
  //gDirectory->GetObject("num_reco_eta",stareco_eta);
  //staeffic = (TH1*)gDirectory->Get("effic")->Clone(); staeffic->Reset();
  //staeffic_pt = (TH1*)gDirectory->Get("efficPt")->Clone(); staeffic_pt->Reset();
  //stafakerate = (TH1*)gDirectory->Get("fakerate")->Clone(); stafakerate->Reset();  

  bool seedDir = source->cd("DQMData/Track/muonSeedTrack_"+assocName);
  if(!seedDir) return -1;

  TH1F *seedSimRec_eta, *seedSimRec_pt, *seedreco2sim_eta, *seedreco_eta;
  TH1F *seedeffic, *seedeffic_pt, *seedfakerate;
  gDirectory->GetObject("num_assoc(simToReco)_eta",seedSimRec_eta);
  gDirectory->GetObject("num_assoc(simToReco)_pT",seedSimRec_pt);
  //gDirectory->GetObject("num_assoc(recoToSim)_eta",seedreco2sim_eta);
  //gDirectory->GetObject("num_reco_eta",seedreco_eta);
  //seedeffic = (TH1*)gDirectory->Get("effic")->Clone(); seedeffic->Reset();
  //seedeffic_pt = (TH1*)gDirectory->Get("efficPt")->Clone(); seedeffic_pt->Reset();
  //seedfakerate = (TH1*)gDirectory->Get("fakerate")->Clone(); seedfakerate->Reset();  

  source->cd("DQMData/Track");

  TString effPath("eff_"+assocName);

  TDirectory* effDir = gDirectory->mkdir(effPath.Data());

  effDir->cd();

  TGraphAsymmErrors * glbStaEff = new TGraphAsymmErrors(glbSimRec_eta,staSimRec_eta);
  TGraphAsymmErrors * glbSeedEff = new TGraphAsymmErrors(glbSimRec_eta,seedSimRec_eta);
  TGraphAsymmErrors * staSeedEff = new TGraphAsymmErrors(staSimRec_eta,seedSimRec_eta);

  TGraphAsymmErrors * glbStaEff_pt = new TGraphAsymmErrors(glbSimRec_pt,staSimRec_pt);
  TGraphAsymmErrors * glbSeedEff_pt = new TGraphAsymmErrors(glbSimRec_pt,seedSimRec_pt);
  TGraphAsymmErrors * staSeedEff_pt = new TGraphAsymmErrors(staSimRec_pt,seedSimRec_pt);

  glbStaEff->SetNameTitle("glbStaEff_eta","Global-Standalone Efficiency vs #eta");
  glbSeedEff->SetNameTitle("glbSeedEff_eta","Global-Seed Efficiency vs #eta");
  staSeedEff->SetNameTitle("staSeedEff_eta","Standalone-Seed Efficiency vs #eta");
  glbStaEff_pt->SetNameTitle("glbStaEff_pt","Global-Standalone Efficiency vs #eta");
  glbSeedEff_pt->SetNameTitle("glbSeedEff_pt","Global-Seed Efficiency vs #eta");
  staSeedEff_pt->SetNameTitle("staSeedEff_pt","Standalone-Seed Efficiency vs #eta");

  glbStaEff->Write("",TObject::kOverwrite);
  glbSeedEff->Write("",TObject::kOverwrite);
  staSeedEff->Write("",TObject::kOverwrite);
  glbStaEff_pt->Write("",TObject::kOverwrite);
  glbSeedEff_pt->Write("",TObject::kOverwrite);
  staSeedEff_pt->Write("",TObject::kOverwrite);

  //TGraphAsymmErrors * glbStaFake = new TGraphAsymmErrors(glbSimRec_eta,staSimRec_eta);
  //TGraphAsymmErrors * glbSeedFake = new TGraphAsymmErrors(glbSimRec_eta,seedSimRec_eta);
  //TGraphAsymmErrors * staSeedFake = new TGraphAsymmErrors(staSimRec_eta,seedSimRec_eta);

  return 0;
}
