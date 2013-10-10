// -*- C++ -*-
//
// Package:    L1T
// Class:      L1Validator
// 
/**\class L1T L1Validator.cc Validation/L1T/plugins/L1Validator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Scott Wilbur
//         Created:  Wed, 28 Aug 2013 09:42:55 GMT
// $Id$
//
//

#include <string>

#include <Validation/L1T/interface/L1Validator.h>

#include "DataFormats/Math/interface/deltaR.h"

#include "TFile.h"

//defining as a macro instead of a function because inheritance doesn't work:
#define FINDRECOPART(TYPE, COLLECTION1, COLLECTION2) \
const TYPE *RecoPart=NULL; \
double BestDist=999.; \
for(uint i=0; i < COLLECTION1->size(); i++){ \
  const TYPE *ThisPart = &COLLECTION1->at(i); \
  double ThisDist = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi()); \
  if(ThisDist < 1.0 && ThisDist < BestDist){ \
    BestDist = ThisDist; \
    RecoPart = ThisPart; \
  } \
} \
if(COLLECTION1.product() != COLLECTION2.product()){ \
  for(uint i=0; i < COLLECTION2->size(); i++){ \
    const TYPE *ThisPart = &COLLECTION2->at(i); \
    double ThisDist = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi()); \
    if(ThisDist < 1.0 && ThisDist < BestDist){ \
      BestDist = ThisDist; \
      RecoPart = ThisPart; \
    } \
  } \
}

L1Validator::L1Validator(const edm::ParameterSet& iConfig){
  //_dbe = edm::Service<DQMStore>().operator->();
  _dirName = iConfig.getParameter<std::string>("dirName");
  _fileName = iConfig.getParameter<std::string>("fileName");

  _Hists = new L1ValidatorHists(/*_dbe*/);
}


L1Validator::~L1Validator(){
}

void L1Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  using namespace std;
  using namespace l1extra;
  using namespace reco;

  Handle<GenParticleCollection> GenParticles;
  Handle<L1EmParticleCollection> IsoEGs;
  Handle<L1EmParticleCollection> NonIsoEGs;
  Handle<L1JetParticleCollection> CenJets;
  Handle<L1JetParticleCollection> ForJets;
  Handle<L1JetParticleCollection> Taus;
  Handle<L1MuonParticleCollection> Muons;
  //Handle<L1EtMissParticleCollection> METs;

  iEvent.getByLabel("genParticles", GenParticles);
  iEvent.getByLabel("l1extraParticles", "Isolated", IsoEGs);
  iEvent.getByLabel("l1extraParticles", "NonIsolated", NonIsoEGs);
  iEvent.getByLabel("l1extraParticles", "Central", CenJets);
  iEvent.getByLabel("l1extraParticles", "Forward", ForJets);
  iEvent.getByLabel("l1extraParticles", "Tau", Taus);
  iEvent.getByLabel("l1extraParticles", Muons);
  //iEvent.getByLabel("l1extraParticles", METs);


  _Hists->NEvents++;

  _Hists->FillNumber(L1ValidatorHists::Type::IsoEG, IsoEGs->size());
  _Hists->FillNumber(L1ValidatorHists::Type::NonIsoEG, NonIsoEGs->size());
  _Hists->FillNumber(L1ValidatorHists::Type::CenJet, CenJets->size());
  _Hists->FillNumber(L1ValidatorHists::Type::ForJet, ForJets->size());
  _Hists->FillNumber(L1ValidatorHists::Type::TauJet, Taus->size());
  _Hists->FillNumber(L1ValidatorHists::Type::Muon, Muons->size());

  for(uint i=0; i < GenParticles->size(); i++){
    const GenParticle *GenPart = &GenParticles->at(i);

    int pdg = GenPart->pdgId(), status = GenPart->status();

    if(status==1 && (abs(pdg)==11 || pdg==22)){
      FINDRECOPART(L1EmParticle, IsoEGs, NonIsoEGs)

      if(RecoPart==NULL){
 	_Hists->Fill(L1ValidatorHists::Type::IsoEG, GenPart, NULL);
 	_Hists->Fill(L1ValidatorHists::Type::NonIsoEG, GenPart, NULL);
      }else if(RecoPart->type() == L1EmParticle::EmType::kIsolated){
 	_Hists->Fill(L1ValidatorHists::Type::IsoEG, GenPart, RecoPart);
 	_Hists->Fill(L1ValidatorHists::Type::NonIsoEG, GenPart, NULL);
      }else if(RecoPart->type() == L1EmParticle::EmType::kNonIsolated){
 	_Hists->Fill(L1ValidatorHists::Type::IsoEG, GenPart, NULL);
 	_Hists->Fill(L1ValidatorHists::Type::NonIsoEG, GenPart, RecoPart);
      }
    }else if(status==1 && abs(pdg)==13){
      FINDRECOPART(L1MuonParticle, Muons, Muons)

      _Hists->Fill(L1ValidatorHists::Type::Muon, GenPart, RecoPart);
    }else if(status==3 && abs(pdg)==15){
      FINDRECOPART(L1JetParticle, Taus, Taus)

      _Hists->Fill(L1ValidatorHists::Type::TauJet, GenPart, RecoPart);
    }else if(status==3 && (abs(pdg)<=5 || pdg==21)){
      FINDRECOPART(L1JetParticle, CenJets, ForJets)

      if(RecoPart==NULL){
 	_Hists->Fill(L1ValidatorHists::Type::CenJet, GenPart, NULL);
 	_Hists->Fill(L1ValidatorHists::Type::ForJet, GenPart, NULL);
      }else if(RecoPart->type() == L1JetParticle::JetType::kCentral){
 	_Hists->Fill(L1ValidatorHists::Type::CenJet, GenPart, RecoPart);
 	_Hists->Fill(L1ValidatorHists::Type::ForJet, GenPart, NULL);
      }else if(RecoPart->type() == L1JetParticle::JetType::kForward){
 	_Hists->Fill(L1ValidatorHists::Type::CenJet, GenPart, NULL);
 	_Hists->Fill(L1ValidatorHists::Type::ForJet, GenPart, RecoPart);
      }
    }else continue;

    //cout << GenPart->pt() << '\t' << GenPart->eta() << '\t' << GenPart->phi() << '\t' << GenPart->pdgId() << endl;
  }
}


// ------------ method called once each job just before starting event loop  ------------
/*
void L1Validator::beginJob(){
}
*/

// ------------ method called once each job just after ending the event loop  ------------
void L1Validator::endJob(){
  _Hists->Normalize();

  TFile OutFile(_fileName.c_str(), "recreate");
  _Hists->Write();
  OutFile.Close();
}

// ------------ method called when starting to processes a run  ------------
/*
void L1Validator::beginRun(edm::Run const&, edm::EventSetup const&){
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void L1Validator::endRun(edm::Run const&, edm::EventSetup const&){
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void L1Validator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void L1Validator::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/


//The next three are exactly the same, but apparently inheritance doesn't work like I thought it did.
const reco::LeafCandidate *L1Validator::FindBest(const reco::GenParticle *GenPart, const std::vector<l1extra::L1EmParticle> *Collection1, const std::vector<l1extra::L1EmParticle> *Collection2=NULL){
  const reco::LeafCandidate *BestPart=NULL;
  double BestDR=999.;

  for(uint i=0; i < Collection1->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection1->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  if(Collection2==NULL) return BestPart;

  for(uint i=0; i < Collection2->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection2->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  return BestPart;
}

const reco::LeafCandidate *L1Validator::FindBest(const reco::GenParticle *GenPart, const std::vector<l1extra::L1JetParticle> *Collection1, const std::vector<l1extra::L1JetParticle> *Collection2=NULL){
  const reco::LeafCandidate *BestPart=NULL;
  double BestDR=999.;

  for(uint i=0; i < Collection1->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection1->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  if(Collection2==NULL) return BestPart;

  for(uint i=0; i < Collection2->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection2->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  return BestPart;
}

const reco::LeafCandidate *L1Validator::FindBest(const reco::GenParticle *GenPart, const std::vector<l1extra::L1MuonParticle> *Collection1){
  const reco::LeafCandidate *BestPart=NULL;
  double BestDR=999.;

  for(uint i=0; i < Collection1->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection1->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  return BestPart;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1Validator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1Validator);
