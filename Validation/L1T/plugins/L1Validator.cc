// -*- C++ -*-
//
// Package:    L1T
// Class:      L1Validator
// 
/**
 * \class L1T L1Validator.cc Validation/L1T/plugins/L1Validator.cc
 *
 * Description: [one line class summary]
 * 
 * Implementation:
 *    [Notes on implementation]
 */
//
// Original Author:  Scott Wilbur
//         Created:  Wed, 28 Aug 2013 09:42:55 GMT
// $Id$
//
//

#include <string>

#include <Validation/L1T/interface/L1Validator.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

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
  _dirName = iConfig.getParameter<std::string>("dirName");
  _GenSource = consumes<reco::GenParticleCollection> (iConfig.getParameter<edm::InputTag>("GenSource"));

  _L1MuonBXSource = consumes<l1t::MuonBxCollection> (iConfig.getParameter<edm::InputTag>("L1MuonBXSource"));
  _L1EGammaBXSource = consumes<l1t::EGammaBxCollection> (iConfig.getParameter<edm::InputTag>("L1EGammaBXSource"));
  _L1TauBXSource = consumes<l1t::TauBxCollection> (iConfig.getParameter<edm::InputTag>("L1TauBXSource"));
  _L1JetBXSource = consumes<l1t::JetBxCollection> (iConfig.getParameter<edm::InputTag>("L1JetBXSource"));
  _srcToken = mayConsume<GenEventInfoProduct>( iConfig.getParameter<edm::InputTag>("srcToken") );
  _L1GenJetSource = consumes<reco::GenJetCollection>( iConfig.getParameter<edm::InputTag>("L1GenJetSource"));

  //_fileName = iConfig.getParameter<std::string>("fileName");
}


L1Validator::~L1Validator(){
}

void L1Validator::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  iBooker.setCurrentFolder(_dirName);
  _Hists.Book(iBooker);
};

void L1Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  using namespace std;
  using namespace l1extra;
  using namespace reco;

  Handle<GenParticleCollection> GenParticles;
  Handle<l1t::MuonBxCollection> MuonsBX;
  Handle<l1t::EGammaBxCollection> EGammasBX;
  Handle<l1t::TauBxCollection> TausBX;
  Handle<l1t::JetBxCollection> JetsBX;
  Handle<GenEventInfoProduct> genEvtInfoProduct;
  Handle<reco::GenJetCollection> GenJets;

  bool GotEverything=true;

  if(!iEvent.getByToken(_GenSource, GenParticles)) GotEverything=false;
  if(!iEvent.getByToken(_L1MuonBXSource, MuonsBX)) GotEverything=false;
  if(!iEvent.getByToken(_L1EGammaBXSource, EGammasBX)) GotEverything=false;
  if(!iEvent.getByToken(_L1TauBXSource, TausBX)) GotEverything=false;
  if(!iEvent.getByToken(_L1JetBXSource, JetsBX)) GotEverything=false;
  if(!iEvent.getByToken(_srcToken, genEvtInfoProduct)) GotEverything=false;
  if(!iEvent.getByToken(_L1GenJetSource, GenJets)) GotEverything=false;  

  if(!GotEverything) return;

  /*
  std::string moduleName = "";
  if( genEvtInfoProduct.isValid() ) {
	  const edm::Provenance& prov = iEvent.getProvenance(genEvtInfoProduct.id());
	  moduleName = edm::moduleName(prov);
	  //cout<<" generator name: "<<moduleName<<endl;
  }
  */ 

  _Hists.NEvents++;

  int nL1Muons = 0, nL1EGammas = 0, nL1Taus = 0, nL1Jets = 0;
  if(MuonsBX->getFirstBX()>=0) nL1Muons = MuonsBX->size(0);
  if(EGammasBX->getFirstBX()>=0) nL1EGammas = EGammasBX->size(0);
  if(TausBX->getFirstBX()>=0) nL1Taus = TausBX->size(0);
  if(JetsBX->getFirstBX()>=0) nL1Jets = JetsBX->size(0);

  _Hists.FillNumber(L1ValidatorHists::Type::Muon, nL1Muons);
  _Hists.FillNumber(L1ValidatorHists::Type::Egamma, nL1EGammas);
  _Hists.FillNumber(L1ValidatorHists::Type::Tau, nL1Taus);
  _Hists.FillNumber(L1ValidatorHists::Type::Jet, nL1Jets);

  //For gen jet

  for(auto &Genjet : *GenJets ){

     // eta within calorimeter acceptance 4.7
     if(fabs((&Genjet)->eta())>4.7) continue;

     // only consider the gen jet with pt greater than 10 GeV
     if((&Genjet)->pt()<10.0) continue;

     double minDR = 999.0;

     // match L1T object
     const l1t::Jet        *L1Part=nullptr;
     for(int iBx = JetsBX->getFirstBX();  iBx<=JetsBX->getLastBX(); ++iBx){
          if(iBx>0) continue;
          for(std::vector<l1t::Jet>::const_iterator jet = JetsBX->begin(iBx); jet != JetsBX->end(iBx); ++jet){
                double idR = reco::deltaR((&Genjet)->eta(), (&Genjet)->phi(), jet->eta(), jet->phi());
                if( idR < minDR ){
                         minDR = idR;
                         L1Part = &(*jet);
                }
          }
     }
     _Hists.Fill(L1ValidatorHists::Type::Jet, &Genjet, L1Part);
  }


  for(uint i=0; i < GenParticles->size(); i++){
    const GenParticle *GenPart = &GenParticles->at(i);

    int pdg = GenPart->pdgId(), status = GenPart->status();

    double minDR = 999.0;

    // only consider the gen particle with pt greater than 10 GeV
    if(GenPart->pt()<10.0) continue;

    /// select the final state (i.e status==1) muons (pdg==+/-13)
    if(status==1 && abs(pdg)==13){  //Muon

       // eta within tracker acceptance 2.4
       if(fabs(GenPart->eta())>2.4) continue;

       // match L1T object
       const l1t::Muon 	*L1Part=nullptr;
       for(int iBx = MuonsBX->getFirstBX();  iBx<=MuonsBX->getLastBX(); ++iBx){
	  if(iBx>0) continue;
          for(std::vector<l1t::Muon>::const_iterator mu = MuonsBX->begin(iBx); mu != MuonsBX->end(iBx); ++mu){
	  	double idR = reco::deltaR(GenPart->eta(), GenPart->phi(), mu->eta(), mu->phi());  
		if(idR < minDR ){
			 minDR = idR;
			 L1Part = &(*mu);
		}
	 		
	  }
          _Hists.Fill(L1ValidatorHists::Type::Muon, GenPart, L1Part);
       } 


    /// select the final state (i.e status==1) electrons (pdg==+/-11) and photons (pdg==22)
    }  else if(status==1 && (abs(pdg)==11 || pdg==22)){  //Egamma

       // eta within EM calorimeter acceptance 2.5
       if(fabs(GenPart->eta())>2.5) continue;

       // exclude the calorimeter barrel and endcap overlap region 
       if(fabs(GenPart->eta())>1.4442 && fabs(GenPart->eta())<1.5660) continue;

       // match L1T object
       const l1t::EGamma 	*L1Part=nullptr;
       for(int iBx = EGammasBX->getFirstBX();  iBx<=EGammasBX->getLastBX(); ++iBx){
	  if(iBx>0) continue;
          for(std::vector<l1t::EGamma>::const_iterator eg = EGammasBX->begin(iBx); eg != EGammasBX->end(iBx); ++eg){
	  	double idR = reco::deltaR(GenPart->eta(), GenPart->phi(), eg->eta(), eg->phi());  
		if(idR < minDR ){
			 minDR = idR;
			 L1Part = &(*eg);
		}
	  }
       }
       _Hists.Fill(L1ValidatorHists::Type::Egamma, GenPart, L1Part);


    /// select the matrix element (i.e status==2) taus (pdg==+/-15) before decay
     } else if(status==2 && abs(pdg)==15){  //Tau

       // eta within tracker acceptance 2.4
       if(fabs(GenPart->eta())>2.4) continue;

       // match L1T object
       const l1t::Tau 	*L1Part=nullptr;
       for(int iBx = TausBX->getFirstBX();  iBx<=TausBX->getLastBX(); ++iBx){
	  if(iBx>0) continue;
          for(std::vector<l1t::Tau>::const_iterator tau = TausBX->begin(iBx); tau != TausBX->end(iBx); ++tau){
	  	double idR = reco::deltaR(GenPart->eta(), GenPart->phi(), tau->eta(), tau->phi());  
		if(idR < minDR ){
			 minDR = idR;
			 L1Part = &(*tau);
		}
	  }
       }
       _Hists.Fill(L1ValidatorHists::Type::Tau, GenPart, L1Part);

    }

  }

}

//The next three are exactly the same, but apparently inheritance doesn't work like I thought it did.
const reco::LeafCandidate *L1Validator::FindBest(const reco::GenParticle *GenPart, const std::vector<l1extra::L1EmParticle> *Collection1, const std::vector<l1extra::L1EmParticle> *Collection2=nullptr){
  const reco::LeafCandidate *BestPart=nullptr;
  double BestDR=999.;

  for(uint i=0; i < Collection1->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection1->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  if(Collection2==nullptr) return BestPart;

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

const reco::LeafCandidate *L1Validator::FindBest(const reco::GenParticle *GenPart, const std::vector<l1extra::L1JetParticle> *Collection1, const std::vector<l1extra::L1JetParticle> *Collection2=nullptr){
  const reco::LeafCandidate *BestPart=nullptr;
  double BestDR=999.;

  for(uint i=0; i < Collection1->size(); i++){
    const reco::LeafCandidate *ThisPart = &Collection1->at(i);
    double ThisDR = reco::deltaR(GenPart->eta(), GenPart->phi(), ThisPart->eta(), ThisPart->phi());
    if(ThisDR < BestDR){
      BestDR = ThisDR;
      BestPart = ThisPart;
    }
  }

  if(Collection2==nullptr) return BestPart;

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
  const reco::LeafCandidate *BestPart=nullptr;
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
