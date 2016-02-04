
#include <iostream>
#include <string>
#include "TFile.h"
#include "THashList.h"
#include "TH1.h"
#include "TKey.h"
#include "TClass.h"
#include "TSystem.h"

int remakeComposites( TString fname )
{
  int a = remakeValidTrack(fname);
  int b = remakeValidMuon(fname);
  return a+b;
}

int remakeValidTrack( TString fname)
{
  
  TFile* source = TFile::Open( fname , "UPDATE");
  if( source==0 ){
    return 1;
  }
  bool multiTrack = source->cd("DQMData/Track");
  if(!multiTrack) return -1;

  TString path( (char*)strstr( gDirectory->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TDirectory *current_sourcedir = source->GetDirectory(path);
  if (!current_sourcedir) {
    continue;
  }
  
  // loop over all keys in this directory
  TChain *globChain = 0;
  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  //gain time, do not add the objects in the list in memory
  //TH1::AddDirectory(kFALSE);
  THashList allNames;
  while ( (key = (TKey*)nextkey())) {
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;
    
    // read object from first source file
    current_sourcedir->cd();
    TObject *obj = key->ReadObj();
    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      allNames.Add(new TObjString(key->GetName()));      
      cout << "Found subdirectory " << obj->GetName() << endl;
      ((TDirectory*)obj)->cd();

      TH1F * sim2reco_eta, *reco2sim_eta, *reco_eta, *sim_eta;
      TH1F * sim2reco_pt, *reco2sim_pt, *reco_pt, *sim_pt;
      TH1F *effic, *effic_pt, *fakerate;

      gDirectory->GetObject("num_assoc(simToReco)_eta",sim2reco_eta);
      gDirectory->GetObject("num_assoc(recoToSim)_eta",reco2sim_eta);
      gDirectory->GetObject("num_assoc(simToReco)_pT",sim2reco_pt);
      //gDirectory->GetObject("num_assoc(recoToSim)_pT",reco2sim_pt);
      gDirectory->GetObject("num_reco_eta",reco_eta);
      gDirectory->GetObject("num_simul_eta",sim_eta);
      //gDirectory->GetObject("num_reco_pT",reco_pt);
      gDirectory->GetObject("num_simul_pT",sim_pt);

      gDirectory->GetObject("effic",effic); effic->Reset();
      gDirectory->GetObject("efficPt",effic_pt); effic_pt->Reset();
      gDirectory->GetObject("fakerate",fakerate); fakerate->Reset();

      //refill efficiency plot vs eta

      makeEffTH1(sim2reco_eta,sim_eta,effic); 
      makeEffTH1(sim2reco_pt,sim_pt,effic_pt); 
      makeFakeTH1(reco2sim_eta,reco_eta,fakerate); 

      effic->Write("",TObject::kOverwrite);
      effic_pt->Write("",TObject::kOverwrite);
      fakerate->Write("",TObject::kOverwrite);

      //refill slices
      computeSlice("dxyres_vs_eta","sigmadxy");
      computeSlice("dxyres_vs_pt","sigmadxyPt");
      computeSlice("ptres_vs_eta","sigmapt");
      computeSlice("ptres_vs_pt","sigmaptPt");
      computeSlice("dzres_vs_eta","sigmadz");
      computeSlice("dzres_vs_pt","sigmadzPt");
      computeSlice("phires_vs_eta","sigmaphi");
      computeSlice("phires_vs_pt","sigmaphiPt");
      computeSlice("cotThetares_vs_eta","sigmacotTheta");
      computeSlice("cotThetares_vs_pt","sigmacotThetaPt");
      //TH1F* ph1 = (TH1F*) chi2_vs_eta->ProfileX();
      //TH1F* ph2 = (TH1F*) nhits_vs_eta->ProfileX();
      computeSlice("dxypull_vs_eta","h_dxypulleta");
      computeSlice("ptpull_vs_eta","h_ptpulleta","h_ptshifteta");
      computeSlice("dzpull_vs_eta","h_dzpulleta");
      computeSlice("phipull_vs_eta","h_phipulleta");
      computeSlice("thetapull_vs_eta","h_thetapulleta");

    } 
  }

  source->Close();  

  return 0;
}

int remakeValidMuon( TString fname)
{

  
  TFile* source = TFile::Open( fname , "UPDATE");
  if( source==0 ){
    return 1;
  }
  bool singleMuon = source->cd("DQMData/RecoMuonTask");  
  if(!singleMuon) return -1;


  TString path( (char*)strstr( gDirectory->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TDirectory *current_sourcedir = source->GetDirectory(path);
  if (!current_sourcedir) {
    continue;
  }
  
  // loop over all keys in this directory
  TChain *globChain = 0;
  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  //gain time, do not add the objects in the list in memory
  //TH1::AddDirectory(kFALSE);
  THashList allNames;
  while ( (key = (TKey*)nextkey())) {
    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;
    
    // read object from first source file
    current_sourcedir->cd();
    TObject *obj = key->ReadObj();
    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      allNames.Add(new TObjString(key->GetName()));      
      cout << "Found subdirectory " << obj->GetName() << endl;
      ((TDirectory*)obj)->cd();

      computeMuonEff("Glb","Sim");
      computeMuonEff("Glb","Seed");
      computeMuonEff("Glb","Sta");
      computeMuonEff("Glb","Tk");
      
      computeMuonEff("Sta","Sim");
      computeMuonEff("Sta","Seed");
      
      computeMuonEff("Seed","Sim");
      
      computeSlice("GlbEtaVsErrQPt","GlbPtResSigma","GlbPtResMean");
      computeSlice("StaEtaVsErrQPt","StaPtResSigma","StaPtResMean");
      computeSlice("SeedEtaVsErrQPt","SeedPtResSigma","SeedPtResMean");

      computeSlice("GlbEtaVsPullPt","GlbPullPtSigma","GlbPullPtMean");
      computeSlice("StaEtaVsPullPt","StaPullPtSigma","StaPullPtMean");
      computeSlice("SeedEtaVsPullPt","SeedPullPtSigma","SeedPullPtMean");
      
    } 
  }
  
  source->Close();  
  
  return 0;
}

void computeSlice(TString name1, TString name2="", TString name3="")
{
  TH2F *theTh2;
  cout << "computeSlice " << name1.Data() << endl;
  gDirectory->GetObject(name1,theTh2);
  
  
  theTh2->FitSlicesY();
  TH1* h1=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_1")->Clone();
  TH1* h2=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_2")->Clone();
  TH1* h3=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_chi2")->Clone();
  
  h1->SetTitle(TString(theTh2->GetTitle())+" Gaussian Mean");
  h2->SetTitle(TString(theTh2->GetTitle())+" Gaussian Width");
  h3->SetTitle(TString(theTh2->GetTitle())+" Gaussian fit #chi^{2}");
  
  TH1* h1a = h1->Clone();
  TH1* h2a = h2->Clone();
  h1a->Write(name3,TObject::kOverwrite);
  h2a->Write(name2,TObject::kOverwrite);

  h1->Write("",TObject::kOverwrite);
  h2->Write("",TObject::kOverwrite);
  h3->Write("",TObject::kOverwrite);
}


void computeSlice(TH2F* theTh2) 
{
  theTh2->FitSlicesY();
  TH1* h1=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_1")->Clone();
  TH1* h2=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_2")->Clone();
  TH1* h3=(TH1*)gDirectory->Get(TString(theTh2->GetName())+"_chi2")->Clone();

  h1->SetTitle(TString(theTh2->GetTitle())+" Gaussian Mean");
  h2->SetTitle(TString(theTh2->GetTitle())+" Gaussian Width");
  h3->SetTitle(TString(theTh2->GetTitle())+" Gaussian fit #chi^{2}");

  h1->Write("",TObject::kOverwrite);
  h2->Write("",TObject::kOverwrite);
  h3->Write("",TObject::kOverwrite);
}

void computeMuonEff(TString first, TString second)
{

  TH2F *numTH2, *denTH2;
  TH1D *num, *den;
  TH1F *effic;

  gDirectory->GetObject(first+"EtaVsPhi",numTH2);
  gDirectory->GetObject(second+"EtaVsPhi",denTH2);
  gDirectory->GetObject(first+second+"_effEta",effic);
  num  = numTH2 ->ProjectionX();
  den  = denTH2 ->ProjectionX();

  makeEffTH1(num,den,effic);
  effic->Write("",TObject::kOverwrite);
}

void makeEffTH1(TH1* num, TH1* den, TH1* effic)
{

     //fill efficiency plot vs eta
      double eff,err;
      int nBins = num->GetNbinsX();
      for(int bin = 0; bin <= nBins + 1 ; bin++) {
        if (den->GetBinContent(bin) != 0 ){
          eff = ((double) num->GetBinContent(bin))/((double) den->GetBinContent(bin));
	  err = sqrt(eff*(1-eff)/((double) den->GetBinContent(bin)));
          effic->SetBinContent(bin, eff);
          effic->SetBinError(bin,err);
        }
        else {
          effic->SetBinContent(bin, 0);
        }
      }
}

void makeFakeTH1(TH1F* num, TH1F* den, TH1F* fakerate)
{
      //fill fakerate plot
      double frate,ferr;
      int nBins = num->GetNbinsX();
      for (int bin=0; bin <= nBins + 1; bin++){
        if (den->GetBinContent(bin) != 0 ){
          frate = 1-((double) num->GetBinContent(bin))/((double) den->GetBinContent(bin));
	  ferr = sqrt( frate*(1-frate)/(double) den->GetBinContent(bin) );
          fakerate->SetBinContent(bin, frate);
	  fakerate->SetBinError(bin,ferr);
        }
        else {
          fakerate->SetBinContent(bin, 0.);
        }
      }
}
