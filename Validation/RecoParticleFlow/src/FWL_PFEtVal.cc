#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include <iostream>
#include "../interface/FWL_PFEtVal.h"
#include <TFile.h>
#include <TBranch.h>
#include <TTree.h>
#include <vector>
using namespace std;
using namespace reco;

FWL_PFEtVal::FWL_PFEtVal()
{
  file_=TFile::Open("tau2hadron2.root");
  RECO_caloJet_=new reco::CaloJetCollection();
  PF_Jet_=new reco::PFJetCollection();
  GEN_caloJet_=new reco::GenJetCollection();
}

FWL_PFEtVal::~FWL_PFEtVal()
{

}

void FWL_PFEtVal::readData()
{

  cout<<" starting readData() "<<endl;

 tree_=(TTree*)file_->Get("Events");
  if(!tree_)
    {
      cout<<"Node \"Events\" not found in root-file!"<<endl;
      return;
    }
  
  RECO_caloJetBranch_=tree_->GetBranch("recoCaloJets_iterativeCone5CaloJets__PROD.obj");
  if(!RECO_caloJetBranch_)
    {
      cout<<"Could not find branch!"<<endl;
      return;
    }
  PF_JetBranch_=tree_->GetBranch("recoPFJets_iterativeCone5PFJets__PROD.obj");
  if(!PF_JetBranch_)
    {
      cout<<"Could not find branch!"<<endl;
      return;
    }
  GEN_caloJetBranch_=tree_->GetBranch("recoGenJets_iterativeCone5GenJets__PROD.obj");
  if(!GEN_caloJetBranch_)
    {
      cout<<"Could not find branch!"<<endl;
      return;
    }

  RECO_caloJetBranch_->SetAddress(RECO_caloJet_);
  GEN_caloJetBranch_->SetAddress(GEN_caloJet_);
  PF_JetBranch_->SetAddress(PF_Jet_);
  
  cout<<"entries in tree: "<<tree_->GetEntries()<<endl;
  cout<<"entries in branch: "<<RECO_caloJetBranch_->GetEntries()<<endl;
  

  for(int i=0;i<5;i++)
    {
      RECO_caloJetBranch_->GetEntry(i);
      PF_JetBranch_->GetEntry(i);
      for(unsigned int a=0;a<RECO_caloJet_->size();a++)
	cout<<(*RECO_caloJet_)[a].et()<<endl;
      
      
      
      
      
   }
}


