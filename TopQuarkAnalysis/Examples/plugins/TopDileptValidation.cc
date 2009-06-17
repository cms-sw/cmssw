// -*- C++ -*-
//
// Package:    TopDileptValidation
// Class:      TopDileptValidation
// 
/**\class TopDileptValidation TopDileptValidation.cc TopPagDataVal/TopDileptValidation/src/TopDileptValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremy Andrea
//         Created:  Fri May 22 21:51:29 CEST 2009
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <list>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


//--------------------PAT includes

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"



#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"



//--------------------Gen info includes
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//#include <DataFormats/HepMCCandidate/interface/GenParticleCandidate.h> 
#include <DataFormats/Candidate/interface/Candidate.h>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


#include "TH1F.h"
#include "TLorentzVector.h"

//
// class decleration
//

using namespace std;
using namespace edm;
using namespace reco;

class TopDileptValidation : public edm::EDAnalyzer {
public:
  explicit TopDileptValidation(const edm::ParameterSet&);
  ~TopDileptValidation();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  
  
  void fillDileptonAnalysis(int tmeme, const pat::Jet & thejet, int theApproxChannel );
  void fillDileptonAnalysis(int tmeme, int leptonOrigin, const pat::Electron & theElectron);
  void fillDileptonAnalysis(int tmeme, int leptonOrigin, const pat::Muon     & theMuon    );

  void fillMuonJetsAnalysis();
  void fillElectronJetsAnalysis();
  void fillSingleTopAnalysis();
  
  int Tmeme;
  int setTmeme(edm::Handle<reco::GenParticleCollection> genParticles);
  
  bool lookAtDilepton_;
  bool lookAtMuonJets_;
  bool lookAtElectronJets_;
  bool lookAtSingleTop_;
  edm::Service<TFileService> fs;
  bool isTTBarEvents_;
  
  
  int getElectronOrigin(edm::Handle<reco::GenParticleCollection> genParticles, const pat::Electron * thePatElectron);
  int getMuonOrigin(edm::Handle<reco::GenParticleCollection> genParticles, const pat::Muon * thePatMuon);
  
  // ----------member data ---------------------------
  
  
  InputTag      electronProducer_;
  InputTag      muonProducer_;
  InputTag      jetProducer_;
  InputTag      metProducer_;
  InputTag      metProducer_uncorr_;
  
  
  
  
  
  //--------------------------------------------------
  //all channels
  TH1F * h_com_channels,    *h_com_branchinRatio;
  TH1F * h_com_elecCombIso, *h_com_elecCaloIso, *h_com_elecTckIso;
  TH1F * h_com_muCombIso  , *h_com_muCaloIso  , *h_com_muTckIso;
  TH1F * h_com_nJets,       *h_com_jetPt,       *h_com_jetEta;
  TH1F * h_com_nSelMuon,    *h_com_nSelElectron;
  
  
  
  
  //--------------------------------------------------
  //di-lepton (e-e, e-mu, mu-mu) channels
 
  TH1F * h_ee_BTag, *h_ee_MET, *h_ee_MET_nocorr ;
  TH1F * h_em_BTag, *h_em_MET, *h_em_MET_nocorr ;
  TH1F * h_mm_BTag, *h_mm_MET, *h_mm_MET_nocorr ;
  
  
  TH1F * h_sel_ee_lept_Pt, *h_sel_ee_lept_Eta, *h_sel_ee_lept_Phi ;
  TH1F * h_sel_em_lept_Pt, *h_sel_em_lept_Eta, *h_sel_em_lept_Phi ;
  TH1F * h_sel_mm_lept_Pt, *h_sel_mm_lept_Eta, *h_sel_mm_lept_Phi ;
  
  
  TH1F * h_sel_ee_Jet_Pt, *h_sel_ee_Jet_Eta, *h_sel_ee_Jet_Phi, *h_sel_ee_Jet_NJet ;
  TH1F * h_sel_em_Jet_Pt, *h_sel_em_Jet_Eta, *h_sel_em_Jet_Phi, *h_sel_em_Jet_NJet ;
  TH1F * h_sel_mm_Jet_Pt, *h_sel_mm_Jet_Eta, *h_sel_mm_Jet_Phi, *h_sel_mm_Jet_NJet ;
  
  
  TH1F * h_sel_ee_SelJet_Phi, *h_sel_ee_SelJet_NJet ;
  TH1F * h_sel_em_SelJet_Phi, *h_sel_em_SelJet_NJet ;
  TH1F * h_sel_mm_SelJet_Phi, *h_sel_mm_SelJet_NJet ;
  
  TH1F * h_sel_ee_BTag, *h_sel_ee_MET, *h_sel_ee_MET_nocorr ;
  TH1F * h_sel_em_BTag, *h_sel_em_MET, *h_sel_em_MET_nocorr ;
  TH1F * h_sel_mm_BTag, *h_sel_mm_MET, *h_sel_mm_MET_nocorr ;
  
  
  TH1F * h_ee_Minv,* h_sel_ee_Minv ;
  TH1F * h_mm_Minv,* h_sel_mm_Minv ;
  
  
  
  //--------------------------------------------------
  // muon + jets  channels
  // to be filled
  
  
  
  
  //--------------------------------------------------
  // electron + jets  channels
  // to be filled
  
  
  //--------------------------------------------------
  // single top
  // to be filled
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TopDileptValidation::TopDileptValidation(const edm::ParameterSet& iConfig)
  
{
  //now do what ever initialization is needed
  lookAtDilepton_     = iConfig.getParameter<bool>("lookAtDilepton");
  lookAtMuonJets_     = iConfig.getParameter<bool>("lookAtMuonJets");
  lookAtElectronJets_ = iConfig.getParameter<bool>("lookAtElectronJets");
  lookAtSingleTop_    = iConfig.getParameter<bool>("lookAtSingleTop");
  isTTBarEvents_      = iConfig.getParameter<bool>("isTTBarEvents");
  
  
  
  electronProducer_    = iConfig.getParameter<InputTag>("electronProducer");
  muonProducer_        = iConfig.getParameter<InputTag>("muonProducer");
  jetProducer_         = iConfig.getParameter<InputTag>("jetProducer");
  metProducer_         = iConfig.getParameter<InputTag>("metProducer");
  
  
  

  
  Tmeme = 0;
  
  TFileDirectory dirMuonJets     = fs->mkdir( "muonjets" );
  TFileDirectory dirElectronJets = fs->mkdir( "electronjets" );
  TFileDirectory dirSingTop      = fs->mkdir( "singletop" );
  
  TFileDirectory dirCommon       = fs->mkdir( "common" );
  
  //h_test = subDirDilepton.make<TH1F>( "test"  , "test", 100,  0., 100. );
  

  h_com_channels       = dirCommon.make<TH1F>("h_com_channels"       ,"channels"       , 10, 0.5, 10.5);
  h_com_elecCombIso    = dirCommon.make<TH1F>("h_com_elecCombIso"    ,"elecCombIso"    , 100, 0., 1.);
  h_com_elecCaloIso    = dirCommon.make<TH1F>("h_com_elecCaloIso"    ,"elecCaloIso"    , 100, 0., 1.);
  h_com_elecTckIso     = dirCommon.make<TH1F>("h_com_elecTckIso"     ,"elecTckIso"     , 100, 0., 1.);
  h_com_muCombIso      = dirCommon.make<TH1F>("h_com_muCombIso"      ,"muCombIso"      , 100, 0., 1.);
  h_com_muCaloIso      = dirCommon.make<TH1F>("h_com_muCaloIso"      ,"muCaloIso"      , 100, 0., 1.);
  h_com_muTckIso       = dirCommon.make<TH1F>("h_com_muTckIso"       ,"muTckIso"       , 100, 0., 1.);
  h_com_nSelMuon       = dirCommon.make<TH1F>("h_com_nSelMuon"       ,"nSelMuon"       , 10, 0., 10.);    
  h_com_nSelElectron   = dirCommon.make<TH1F>("h_com_nSelElectron"   ,"nSelElectron"   , 10, 0., 10.);
  h_com_nJets          = dirCommon.make<TH1F>("h_com_nJets"          ,"nJets"          , 50, -0.5, 49.5);
  h_com_jetPt          = dirCommon.make<TH1F>("h_com_jetPt"          ,"jetPt"          , 100, 0., 100.);
  h_com_jetEta         = dirCommon.make<TH1F>("h_com_jetEta"         ,"jetEta"         , 50, 0., 3.);
  
  TFileDirectory dirDilepton     = fs->mkdir( "dilepton" );
  TFileDirectory subDirDilepton  = dirDilepton.mkdir( "dilepton_sel" );
  
  
  
  
  //-------------------------------------------------
  //Histograms for the di-lepton  selection
  //-------------------------------------------------
  
  
  if(lookAtDilepton_) {
  
    
    h_ee_BTag       = dirDilepton.make<TH1F>("h_ee_BTag"       ,"h_ee_BTag"      , 100, -50., 50.) ;
    h_ee_MET        = dirDilepton.make<TH1F>("h_ee_MET"        ,"h_ee_MET"       , 100, 0., 200.) ;
    h_ee_MET_nocorr = dirDilepton.make<TH1F>("h_ee_MET_nocorr" ,"h_ee_MET_nocorr", 100, 0., 200.) ;
    h_em_BTag       = dirDilepton.make<TH1F>("h_em_BTag"       ,"h_em_BTag"      , 100, -50., 50.) ;
    h_em_MET        = dirDilepton.make<TH1F>("h_em_MET"        ,"h_em_MET"       , 100, 0., 200.) ;
    h_em_MET_nocorr = dirDilepton.make<TH1F>("h_em_MET_nocorr" ,"h_em_MET_nocorr", 100, 0., 200.) ;
    h_mm_BTag       = dirDilepton.make<TH1F>("h_mm_BTag"       ,"h_mm_BTag"      , 100, -50., 50.) ;
    h_mm_MET        = dirDilepton.make<TH1F>("h_mm_MET"        ,"h_mm_MET"       , 100, 0., 200.) ;
    h_mm_MET_nocorr = dirDilepton.make<TH1F>("h_mm_MET_nocorr" ,"h_mm_MET_nocorr", 100, 0., 200.) ;
    
    
    h_sel_ee_lept_Pt  = dirDilepton.make<TH1F>("h_sel_ee_lept_Pt"  ,"h_sel_ee_lept_Pt"  , 100, 0., 100.) ;
    h_sel_ee_lept_Eta = dirDilepton.make<TH1F>("h_sel_ee_lept_Eta" ,"h_sel_ee_lept_Eta" , 50, 0., 3.) ;
    h_sel_ee_lept_Phi = dirDilepton.make<TH1F>("h_sel_ee_lept_Phi" ,"h_sel_ee_lept_Phi" , 50, 0., 3.2) ;
    h_sel_em_lept_Pt  = dirDilepton.make<TH1F>("h_sel_em_lept_Pt"  ,"h_sel_em_lept_Pt"  , 100, 0., 100.) ;
    h_sel_em_lept_Eta = dirDilepton.make<TH1F>("h_sel_em_lept_Eta" ,"h_sel_em_lept_Eta" , 50, 0., 3.) ;
    h_sel_em_lept_Phi = dirDilepton.make<TH1F>("h_sel_em_lept_Phi" ,"h_sel_em_lept_Phi" , 50, 0., 3.2) ;
    h_sel_mm_lept_Pt  = dirDilepton.make<TH1F>("h_sel_mm_lept_Pt"  ,"h_sel_mm_lept_Pt"  , 100, 0., 100.) ;
    h_sel_mm_lept_Eta = dirDilepton.make<TH1F>("h_sel_mm_lept_Eta" ,"h_sel_mm_lept_Eta" , 50, 0., 3.) ;
    h_sel_mm_lept_Phi = dirDilepton.make<TH1F>("h_sel_mm_lept_Phi" ,"h_sel_mm_lept_Phi" , 50, 0., 3.2) ;
    
    
    h_sel_ee_Jet_Pt   = dirDilepton.make<TH1F>("h_sel_ee_Jet_Pt"   ,"h_sel_ee_Jet_Pt"  , 100, 0., 100.) ;
    h_sel_ee_Jet_Eta  = dirDilepton.make<TH1F>("h_sel_ee_Jet_Eta"  ,"h_sel_ee_Jet_Eta" , 50, 0., 100.) ;
    h_sel_ee_Jet_Phi  = dirDilepton.make<TH1F>("h_sel_ee_Jet_Phi"  ,"h_sel_ee_Jet_Phi" , 50, 0., 100.) ;
    h_sel_ee_Jet_NJet = dirDilepton.make<TH1F>("h_sel_ee_Jet_NJet" ,"h_sel_ee_Jet_NJet", 50, -0.5, 49.5) ;
    h_sel_em_Jet_Pt   = dirDilepton.make<TH1F>("h_sel_em_Jet_Pt"   ,"h_sel_em_Jet_Pt"  , 100, 0., 100.) ;
    h_sel_em_Jet_Eta  = dirDilepton.make<TH1F>("h_sel_em_Jet_Eta"  ,"h_sel_em_Jet_Eta" , 50, 0., 100.) ;
    h_sel_em_Jet_Phi  = dirDilepton.make<TH1F>("h_sel_em_Jet_Phi"  ,"h_sel_em_Jet_Phi" , 50, 0., 100.) ;
    h_sel_em_Jet_NJet = dirDilepton.make<TH1F>("h_sel_em_Jet_NJet" ,"h_sel_em_Jet_NJet", 50, -0.5, 49.5) ;
    h_sel_mm_Jet_Pt   = dirDilepton.make<TH1F>("h_sel_mm_Jet_Pt"   ,"h_sel_mm_Jet_Pt"  , 100, 0., 100.) ;
    h_sel_mm_Jet_Eta  = dirDilepton.make<TH1F>("h_sel_mm_Jet_Eta"  ,"h_sel_mm_Jet_Eta" , 50, 0., 3.) ;
    h_sel_mm_Jet_Phi  = dirDilepton.make<TH1F>("h_sel_mm_Jet_Phi"  ,"h_sel_mm_Jet_Phi" , 50, 0., 3.2) ;
    h_sel_mm_Jet_NJet = dirDilepton.make<TH1F>("h_sel_mm_Jet_NJet" ,"h_sel_mm_Jet_NJet", 50, -0.5, 49.5) ;
    
    
    h_ee_Minv      = dirDilepton.make<TH1F>("h_ee_Minv", "h_ee_Minv", 100, 0, 200);
    h_mm_Minv      = dirDilepton.make<TH1F>("h_mm_Minv", "h_mm_Minv", 100, 0, 200);
    h_sel_ee_Minv  = dirDilepton.make<TH1F>("h_sel_ee_Minv", "h_sel_ee_Minv", 100, 0, 200);
    h_sel_mm_Minv  = dirDilepton.make<TH1F>("h_sel_mm_Minv", "h_sel_mm_Minv", 100, 0, 200);
    
    
    
  }
  
  
  
  
  //-------------------------------------------------
  //Histograms for the muon+jets  selection
  //-------------------------------------------------
  
  if(lookAtMuonJets_){
    
    //to be filled 
    
  }
 
  
  
  
  //-------------------------------------------------
  //Histograms for the electron+jets  selection
  //-------------------------------------------------
  
  if(lookAtElectronJets_){
    
    //to be filled 
    
  }
  
  
  //-------------------------------------------------
  //Histograms for the single top  selection
  //-------------------------------------------------
  
  
  
  if(lookAtSingleTop_){
    
    //to be filled 
    
  }
  
  
}


TopDileptValidation::~TopDileptValidation()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TopDileptValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  

  //-------------------------------------------------
  //get gen particules
  //-------------------------------------------------  
  
  
  Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles",genParticles);
  if(isTTBarEvents_){
    Tmeme = setTmeme( genParticles);
    //full hadronic
    if(Tmeme == 0) h_com_channels->Fill( 0 );
    //electron + jets
    if(Tmeme == 1) h_com_channels->Fill( 1 );
    //muon + jets
    if(Tmeme == 10) h_com_channels->Fill( 2 );
    //tau + jets
    if(Tmeme == 10000 || Tmeme == 10001 ||   Tmeme == 10010) h_com_channels->Fill( 3 );
    //di electrons
    if(Tmeme == 2) h_com_channels->Fill( 4 );
    //di muons
    if(Tmeme == 20) h_com_channels->Fill( 5 );
    // electron-muon
    if(Tmeme == 11) h_com_channels->Fill( 6 );
    // di tau
    if(Tmeme == 20000 || Tmeme == 20100 || Tmeme == 21000 || 
    	Tmeme == 22000 || Tmeme == 20200) h_com_channels->Fill( 7 );
    // electron-tau 
    if(Tmeme == 10001 || Tmeme == 10101 || Tmeme == 11001 ) h_com_channels->Fill( 8 );
    // muon-tau 
    if(Tmeme == 10010 || Tmeme == 11010 || Tmeme == 10110 ) h_com_channels->Fill( 9 );
    
  }
  
  //-------------------------------------------------
  //triggers selection : need to be filled
  //-------------------------------------------------
  
  TriggerResults tr;
  Handle<TriggerResults> h_trigRes;
  iEvent.getByLabel(InputTag("TriggerResults::HLT"), h_trigRes);
  
  vector<string> triggerList;
  Service<service::TriggerNamesService> tns;
  bool foundNames = tns->getTrigPaths(tr,triggerList);
  if (!foundNames) std::cout << "Could not get trigger names!\n";
  if (tr.size()!=triggerList.size()) std::cout << "ERROR: length of names and paths not the same: " << triggerList.size() << "," << tr.size() << endl;
  
  
  // dump trigger list at first event
  //cout << "List of triggers: \n";
  /*for(unsigned int i=0; i< tr.size(); i++) {
    cout.width(3); cout << i;
    cout << " - " << triggerList[i] << endl;
  }*/
  
  
  //-------------------------------------------------
  //electron selection
  //-------------------------------------------------
  
  
  Handle<vector<pat::Electron> > elHa;
  iEvent.getByLabel(electronProducer_, elHa);
  const vector<pat::Electron>* ele = elHa.product();
  int nSelElec   = 0;
  int nSelElec_p = 0;
  int nSelElec_m = 0;
  for(vector<pat::Electron>::const_iterator it = ele->begin();  it!=ele->end(); it++){
    
    const pat::Electron* elec = &*it;
    
    if(elec->pt()<20 && fabs(elec->eta()) > 2.4 && elec->electronID("eidLoose") !=1 ) continue;
    
    int leptOrigin  = getElectronOrigin(genParticles, elec);
    double trackIso = elec->trackIso();
    double caloIso  = elec->caloIso();
    
    double combIso  = elec->pt()/(elec->pt()+ caloIso+trackIso );
    
    trackIso = (elec->pt())/(elec->pt()+ trackIso);
    caloIso  = (elec->pt())/(elec->pt()+ caloIso);
    
    if(trackIso<0.9 && caloIso<0.8) continue ;
    nSelElec++;
    if(elec->charge() ==  1) nSelElec_p++;
    if(elec->charge() == -1) nSelElec_m++;
    if(leptOrigin == 1 && isTTBarEvents_ ){
      h_com_elecCombIso->Fill(combIso);
      h_com_elecCaloIso->Fill(caloIso) ;
      h_com_elecTckIso->Fill(trackIso) ;
      
      fillDileptonAnalysis(Tmeme,leptOrigin, *elec);
      
      
      
    }
    if(!isTTBarEvents_){
      h_com_elecCombIso->Fill(combIso);
      h_com_elecCaloIso->Fill(caloIso) ;
      h_com_elecTckIso->Fill(trackIso) ;
    }
    
    
    
  }
 
  h_com_nSelElectron->Fill(nSelElec);
  
  
  
  //-------------------------------------------------
  //muon selection
  //-------------------------------------------------

  int nSelMuon=0;
  int nSelMuon_p=0;
  int nSelMuon_m=0;
  
  Handle<vector<pat::Muon> > muHa;
  iEvent.getByLabel(muonProducer_, muHa);
  const vector<pat::Muon>* mus = muHa.product();
  
  
  for(vector<pat::Muon>::const_iterator it = mus->begin(); it != mus->end(); it++){
    
    const pat::Muon* mu = &*it;
    int leptOrigin  = getMuonOrigin(genParticles, mu);
    double trackIso = mu->trackIso();
    double caloIso  = mu->caloIso();
    if(mu->pt()<20 && fabs(mu->eta()) > 2.4 && mu->isGlobalMuon()!=1 ) continue;
    double combIso  = mu->pt()/(mu->pt()+ caloIso+trackIso );
    
    trackIso = (mu->pt())/(mu->pt()+ trackIso);
    caloIso  = (mu->pt())/(mu->pt()+ caloIso);
    
    if(trackIso<0.9 && caloIso<0.9) continue ;
    
    nSelMuon++;
    if(mu->charge() ==  1) nSelMuon_p++;
    if(mu->charge() == -1) nSelMuon_m++;
    fillDileptonAnalysis(Tmeme,leptOrigin, *mu);

    if(leptOrigin == 1 && isTTBarEvents_ ){
      h_com_muCombIso->Fill(combIso);
      h_com_muCaloIso->Fill(caloIso) ;
      h_com_muTckIso->Fill(trackIso) ;
    }
    if(!isTTBarEvents_){
      h_com_muCombIso->Fill(combIso);
      h_com_muCaloIso->Fill(caloIso) ;
      h_com_muTckIso->Fill(trackIso) ;
    }
    
  }
  
  h_com_nSelMuon->Fill(nSelMuon);
  
  int theApproxChannel = 0;
  //--------------------------------------------------------------
  //select d-lepton channel
  if(  nSelMuon_p == 1  &&  nSelMuon_m ==1                                            ) theApproxChannel += 1;
  if( (nSelMuon_p == 1  &&  nSelElec_m == 1) ||  (nSelMuon_m == 1 && nSelElec_p == 1) ) theApproxChannel += 10;  
  if(  nSelElec_p == 1  &&  nSelElec_m ==1                                            ) theApproxChannel += 100;
    
    
  
    
  //-------------------------------------------------
  //jets selection
  //-------------------------------------------------
  
 
  int nSelJets = 0;
  int nSelJets_ee = 0;
  int nSelJets_mm = 0;
  int nSelJets_em = 0;
  
  Handle<vector<pat::Jet> > jetHa;
  iEvent.getByLabel(jetProducer_, jetHa);
  const vector<pat::Jet>* jets = jetHa.product();
  
  pat::Jet theHighestPtJet ;
  pat::Jet theSecHighestPtJet;
  int jetSize = jetHa->size();
  
  
  if(jetSize > 0 ) theHighestPtJet     = (*jetHa)[0];
  if(jetSize > 1 ) theSecHighestPtJet  = (*jetHa)[1];
  
  
  for(vector<pat::Jet>::const_iterator it = jets->begin(); it != jets->end(); it++){
    
    if(it->pt()<30 && fabs(it->eta()>2.4) ) continue;
    
    
    
    nSelJets++;
    h_com_jetPt->Fill(it->pt());
    h_com_jetEta->Fill(fabs(it->eta()));
    fillDileptonAnalysis(Tmeme, *it, theApproxChannel);
    
    if(theApproxChannel == 1 &&  (Tmeme == 20 || Tmeme == 11010 || Tmeme == 20020 ) ){
      h_mm_BTag->Fill(it->bDiscriminator("trackCountingHighEffBJetTags")) ;
      nSelJets_mm++;
    }
    
    if(theApproxChannel == 10 && (Tmeme == 11 || Tmeme == 10110 || Tmeme == 11001 || Tmeme == 20011 ) ){
      h_em_BTag->Fill(it->bDiscriminator("trackCountingHighEffBJetTags")) ;
      nSelJets_em++;
      
    }
    
    if(theApproxChannel == 100 &&(Tmeme == 2 || Tmeme == 10101 || Tmeme == 20002  ) ){
      h_ee_BTag->Fill(it->bDiscriminator("trackCountingHighEffBJetTags")) ;
      nSelJets_ee++;
    }
       
  }
  h_com_nJets->Fill(nSelJets);
  
  

  
    
  h_sel_ee_Jet_NJet->Fill(nSelJets_ee) ;
  h_sel_em_Jet_NJet->Fill(nSelJets_em) ;
  h_sel_mm_Jet_NJet->Fill(nSelJets_mm) ;
  
  
  
  
  //-------------------------------------------------
  //MET selection
  //-------------------------------------------------
  
  TLorentzVector v1 ;
  TLorentzVector v2 ;
  TLorentzVector v3 ; 
  
  
  edm::Handle<edm::View<pat::MET> > metHandle;
  iEvent.getByLabel(metProducer_,metHandle);
  edm::View<pat::MET> mets = *metHandle;
  if (metHandle->size() != 0){  
    const pat::MET met = mets.front(); 
    
    if(nSelJets >=2){
      
      if(theApproxChannel == 1 &&  (Tmeme == 20 || Tmeme == 11010 || Tmeme == 20020 ) ){
	h_mm_MET->Fill(met.pt()) ;
	h_mm_MET_nocorr->Fill(met.uncorrectedPt()) ;
	v1.SetPtEtaPhiE((*mus)[0].pt(), (*mus)[0].eta(),(*mus)[0].phi(),(*mus)[0].energy());
        v2.SetPtEtaPhiE((*mus)[1].pt(), (*mus)[1].eta(),(*mus)[1].phi(),(*mus)[1].energy());
        v3 = v1 + v2; 
        double M_inv = v3.M();
	h_mm_Minv->Fill(M_inv);
	if(met.pt()>30) h_sel_mm_Minv->Fill(M_inv);
 
      }
      
      if(theApproxChannel == 10 && (Tmeme == 11 || Tmeme == 10110 || Tmeme == 11001 || Tmeme == 20011 ) ){
	h_em_MET->Fill(met.pt()) ;
	h_em_MET_nocorr->Fill(met.uncorrectedPt()) ;      
      }
      
      if(theApproxChannel == 100 &&(Tmeme == 2 || Tmeme == 10101 || Tmeme == 20002  ) ){
	h_ee_MET->Fill(met.pt()) ;
	h_ee_MET_nocorr->Fill(met.uncorrectedPt()) ;
	v1.SetPtEtaPhiE((*elHa)[0].pt(), (*elHa)[0].eta(),(*elHa)[0].phi(),(*elHa)[0].energy());
        v2.SetPtEtaPhiE((*elHa)[1].pt(), (*elHa)[1].eta(),(*elHa)[1].phi(),(*elHa)[1].energy());
        v3 = v1 + v2; 
        double M_inv = v3.M();
	h_ee_Minv->Fill(M_inv);
	if(met.pt()>30) h_sel_ee_Minv->Fill(M_inv);
      }
    }
  }
  
  
 
  
 
  
  
  
  //InputTag      vertexProducer;
  
  
  
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
TopDileptValidation::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TopDileptValidation::endJob() {
}

  
 // ------------ fill histograms  ------------    
       
void 
TopDileptValidation::fillDileptonAnalysis(int tmeme, int leptonOrigin, const pat::Electron & theElectron){
  
  
  if(leptonOrigin == 1){
    if( Tmeme == 2 || Tmeme == 10101 || Tmeme == 20002 ){
      h_sel_ee_lept_Pt->Fill(theElectron.pt())  ;
      h_sel_ee_lept_Eta->Fill(fabs(theElectron.eta())) ;
      h_sel_ee_lept_Phi->Fill(fabs(theElectron.phi())) ;
    }
    
    if(Tmeme == 11 || Tmeme == 10110 || Tmeme == 11001 || Tmeme == 20011 ){
      h_sel_em_lept_Pt->Fill(theElectron.pt()) ;
      h_sel_em_lept_Eta->Fill(fabs(theElectron.eta())) ;
      h_sel_em_lept_Phi->Fill(fabs(theElectron.phi())) ;
    }
  }else{
    h_sel_ee_lept_Pt->Fill(theElectron.pt())  ;
    h_sel_ee_lept_Eta->Fill(fabs(theElectron.eta())) ;
    h_sel_ee_lept_Phi->Fill(fabs(theElectron.phi())) ;
  }
    
}


void 
TopDileptValidation::fillDileptonAnalysis(int tmeme, int leptonOrigin, const pat::Muon & theMuon){
       
  if(leptonOrigin == 1){
    if(Tmeme == 20 || Tmeme == 11010 || Tmeme == 20020 ){
      h_sel_mm_lept_Pt->Fill(theMuon.pt())  ;
      h_sel_mm_lept_Eta->Fill(fabs(theMuon.eta())) ;
      h_sel_mm_lept_Phi->Fill(fabs(theMuon.phi())) ;
    }
    
    if(Tmeme == 11 || Tmeme == 10110 || Tmeme == 11001 || Tmeme == 20011){
      h_sel_em_lept_Pt->Fill(theMuon.pt()) ;
      h_sel_em_lept_Eta->Fill(fabs(theMuon.eta())) ;
      h_sel_em_lept_Phi->Fill(fabs(theMuon.phi())) ;
    }
  }else{
    h_sel_em_lept_Pt->Fill(theMuon.pt()) ;
    h_sel_em_lept_Eta->Fill(fabs(theMuon.eta())) ;
    h_sel_em_lept_Phi->Fill(fabs(theMuon.phi())) ;
  }
  
}


void 
TopDileptValidation::fillDileptonAnalysis(int tmeme, const pat::Jet & thejet, int theApproxChannel ){
  
  if( (Tmeme == 2 || Tmeme == 10101 || Tmeme == 20002) && theApproxChannel == 100 ){
    h_sel_ee_Jet_Pt->Fill(thejet.pt())    ;
    h_sel_ee_Jet_Eta->Fill(fabs(thejet.eta()))  ;
    h_sel_ee_Jet_Phi->Fill(fabs(thejet.phi()))   ;
  }
  if( (Tmeme == 11 || Tmeme == 10110 || Tmeme == 11001 || Tmeme == 20011 ) &&  theApproxChannel == 10 ){
    h_sel_em_Jet_Pt->Fill(thejet.pt())     ;
    h_sel_em_Jet_Eta->Fill(fabs(thejet.eta()))   ;
    h_sel_em_Jet_Phi->Fill(fabs(thejet.phi()))   ;
  }
  if( (Tmeme == 20 || Tmeme == 11010 || Tmeme == 20020) &&  theApproxChannel == 1 ){
    h_sel_mm_Jet_Pt->Fill(thejet.pt())     ;
    h_sel_mm_Jet_Eta->Fill(fabs(thejet.eta()))   ;
    h_sel_mm_Jet_Phi->Fill(fabs(thejet.phi()))   ;
  }
  
  if(Tmeme == 0){
    
    if( theApproxChannel == 100 ){
      h_sel_ee_Jet_Pt->Fill(thejet.pt())     ;
      h_sel_ee_Jet_Eta->Fill(fabs(thejet.eta()))   ;
      h_sel_ee_Jet_Phi->Fill(fabs(thejet.phi()))   ;
    }
    if( theApproxChannel == 10 ){
      h_sel_em_Jet_Pt->Fill(thejet.pt())     ;
      h_sel_em_Jet_Eta->Fill(fabs(thejet.eta()))   ;
      h_sel_em_Jet_Phi->Fill(fabs(thejet.phi()))   ;
    }
    if( theApproxChannel == 1 ){
      h_sel_mm_Jet_Pt->Fill(thejet.pt())     ;
      h_sel_mm_Jet_Eta->Fill(fabs(thejet.eta()))   ;
      h_sel_mm_Jet_Phi->Fill(fabs(thejet.phi()))   ;
    }
    
    
    
  }
  
  
  
}


void 
TopDileptValidation::fillMuonJetsAnalysis(){

}


void
TopDileptValidation::fillElectronJetsAnalysis(){

}


void 
TopDileptValidation::fillSingleTopAnalysis(){

}
  








int 
TopDileptValidation::setTmeme(edm::Handle<reco::GenParticleCollection> genParticles){

  double Qlep1=0;
  double Qlep2=0;
  const Candidate* lep1=0;
  const Candidate* lep2=0;

  const Candidate* W1=0;
  const Candidate* W2=0;
  
  const Candidate* Nu1=0;
  const Candidate* Nu2=0;

  const Candidate* Top1=0;
  const Candidate* Top2=0;
    
  const Candidate* B1=0;
  const Candidate* B2=0;

  const Candidate* B1Rad=0;
  const Candidate* B2Rad=0;
  int tmeme = 0;
  
  for(size_t i = 0; i < genParticles->size(); ++ i) {
    const GenParticle & paIt = (*genParticles)[i];
    const Candidate * W    = &(*genParticles)[i];
    
    
    //    // finding top and antitop 
    //    //------------------------
    if( abs(paIt.pdgId()) == 6 && paIt.status() == 3 ) {
      if ( paIt.pdgId()==6 ){
	Top1 = &(*genParticles)[i];
      }
      if ( paIt.pdgId()==-6 ){
	Top2 = &(*genParticles)[i];
      }
    }   
    
    //       // finding B1 & B2:
    //       //------------------------          
    if( abs(paIt.pdgId()) == 6 && paIt.status() == 3 ) {
      int FoundWB = 0;
      int FoundB1 = -1;
      int FoundB2 = -1;
      for (unsigned int j=0; j<paIt.numberOfDaughters(); j++){
	if( abs(paIt.daughter(j)->pdgId()) == 24 || abs(paIt.daughter(j)->pdgId()) == 5  ){
	  FoundWB++;
	  if (paIt.daughter(j)->pdgId()==5)  FoundB1=j;
	  if (paIt.daughter(j)->pdgId()==-5) FoundB2=j;
	}
      }
      if (FoundWB>=2) {
	if (FoundB1>=0) B1 = paIt.daughter(FoundB1); 
	if (FoundB2>=0) B2 = paIt.daughter(FoundB2);
      }	          	 
    }
    
    if ( B1 ) {
      for (unsigned int j=0; j<B1->numberOfDaughters(); j++){
	if( abs(B1->daughter(j)->pdgId()) == 5 && B1->daughter(j)->status()==2 ) B1Rad = B1->daughter(j);
      } 	  
    }
    
    if ( B2 ) {
      for (unsigned int j=0; j<B2->numberOfDaughters(); j++){
	if( abs(B2->daughter(j)->pdgId()) == 5 && B2->daughter(j)->status()==2 )  B2Rad = B2->daughter(j);
      }	  
    }
    
    if( abs(paIt.pdgId()) != 24  ) continue;
    if(paIt.status() != 3) continue;
    
    for (unsigned int j=0; j<paIt.numberOfDaughters(); j++){
      if( abs(paIt.daughter(j)->pdgId()) == 12 || abs(paIt.daughter(j)->pdgId()) == 14 || abs(paIt.daughter(j)->pdgId()) == 16 ){
	//     // finding nue/antinue ( t->W->nue )
	//     //--------------------------------
	if( abs(paIt.daughter(j)->pdgId())==12 ) { 
	  if ( paIt.daughter(j)->pdgId()==-12 ) {
	    Nu1   = paIt.daughter(j);
	  }
	  if ( paIt.daughter(j)->pdgId()==12 ) {
	    Nu2   = paIt.daughter(j);
	  }
	} 
	//     // finding numu/antinumu ( t->W->numu )
	//     //------------------------------------
	if( abs(paIt.daughter(j)->pdgId())==14 ) { 
	  if ( paIt.daughter(j)->pdgId()==-14 ) {
	    Nu1   = paIt.daughter(j);
	  }
	  if ( paIt.daughter(j)->pdgId()==14 ) {
	    Nu2   = paIt.daughter(j);
	  }
	}   
	//     // finding nutau/antinutau ( t->W->nutau )
	//     //---------------------------------------
	if( abs(paIt.daughter(j)->pdgId())==16 ) { 
	  if ( paIt.daughter(j)->pdgId()==-16 ) {
	    Nu1   = paIt.daughter(j);
	  }
	  if ( paIt.daughter(j)->pdgId()==16 ) {
	    Nu2   = paIt.daughter(j);
	  }
	}   
	
      }
      
      if( abs(paIt.daughter(j)->pdgId()) == 11 || abs(paIt.daughter(j)->pdgId()) == 13 || abs(paIt.daughter(j)->pdgId()) == 15 ){
	
	//     // finding electron/positron ( t->W->e )
	//     //--------------------------------
	if( abs(paIt.daughter(j)->pdgId())==11 ) { 
	  tmeme +=1;
	  if ( paIt.daughter(j)->pdgId()==11 ) {
	    W1     = &(*genParticles)[i];
	    lep1   = paIt.daughter(j);
	    Qlep1  = ( paIt.daughter(j)->pdgId() == 11 ) ? -1 : 1;
	  }
	  if ( paIt.daughter(j)->pdgId()==-11 ) {
	    W2     = &(*genParticles)[i];
	    lep2   = paIt.daughter(j);
	    Qlep2  = ( paIt.daughter(j)->pdgId() == 11 ) ? -1 : 1;
	  }
	}   
	//     // finding muon/antimuon ( t->W->muon )
	//     //-------------------------------------
	if( abs(paIt.daughter(j)->pdgId())==13 ) { 
	  tmeme +=10;
	  if ( paIt.daughter(j)->pdgId()==13 ) {
	    W1     = &(*genParticles)[i];
	    lep1   = paIt.daughter(j);
	    Qlep1  = ( paIt.daughter(j)->pdgId() == 13 ) ? -1 : 1;
	  }
	  if ( paIt.daughter(j)->pdgId()==-13 ) {
	    W2     = &(*genParticles)[i];
	    lep2   = paIt.daughter(j);
	    Qlep2  = ( paIt.daughter(j)->pdgId() == 13 ) ? -1 : 1;
	  }
	}
	//     // finding tau/antitau ( t->W->tau )
	//     //-------------------------------------
	if( abs(paIt.daughter(j)->pdgId())==15 ) { 
	  tmeme +=10000;
	  if ( paIt.daughter(j)->pdgId()==15 ) {
	    W1     = &(*genParticles)[i];
	    lep1   = paIt.daughter(j);
	    Qlep1  = ( paIt.daughter(j)->pdgId() == 15 ) ? -1 : 1;
	  }
	  if ( paIt.daughter(j)->pdgId()==-15 ) {
	    W2     = &(*genParticles)[i];
	    lep2   = paIt.daughter(j);
	    Qlep2  = ( paIt.daughter(j)->pdgId() == 15 ) ? -1 : 1;
	  }
	}
	
	
	int indxtau = -1;
	for (unsigned int k=0; k<W->numberOfDaughters(); k++){
	  if ( abs(W->daughter(k)->pdgId()) == 15 ) indxtau = k;
	}
	while( indxtau>=0 ){
	  //	    std::cout<<"ok1,abs(W->pdgId()) abs(W->daughter(indxtau)->pdgId()) "<<abs(W->pdgId())<<std::endl;
	  if ( !W ) std::cout<<"NULL "<<std::endl;
	  bool FoundTau = false;
	  for (unsigned int k=0; k<W->numberOfDaughters(); k++){
	    if ( abs(W->daughter(k)->pdgId())==24 ) continue;
	    //	    std::cout<<"ok2 "<<k<<" "<< abs(W->pdgId())<<" "<<W->numberOfDaughters()<<" "<<abs(W->daughter(k)->pdgId())<<std::endl;
	    if ( abs(W->daughter(k)->pdgId()) == 11 || abs(W->daughter(k)->pdgId()) == 13 ) {
	      if ( abs(W->daughter(k)->pdgId()) == 11 ) tmeme +=100; 
	      if ( abs(W->daughter(k)->pdgId()) == 13 ) tmeme +=1000; 
	      indxtau = -1;
	    }
	    if ( abs(W->daughter(k)->pdgId()) == 15) {  indxtau = k; FoundTau = true;} 
	    //               std::cout<<"fin boucle "<<FoundTau<<" "<<indxtau<<std::endl;
	  } //k
	  if (FoundTau) { W = W->daughter(indxtau);} else {indxtau = -1;}
	  //               std::cout<<"fin  "<<FoundTau<<" "<<indxtau<<std::endl;
	}
         }
    }
  }
  
  return tmeme;
}





int TopDileptValidation::getElectronOrigin(edm::Handle<reco::GenParticleCollection> genParticles, const pat::Electron * thePatElectron){
  
  int electronOrigin = 0;
  reco::Candidate * theGenElectron;
  bool matchedGenLepton = false;
  
  for (reco::GenParticleCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++p){
    reco::Candidate * aGenElectron = (dynamic_cast<reco::Candidate *>(const_cast<reco::GenParticle *>(&*p)));
    
    if (abs(p->pdgId()) == 11 && p->status() == 1){
      if ((thePatElectron->genLepton() != NULL) && abs(thePatElectron->genLepton()->pt()-aGenElectron->pt()) < 0.00001){
	theGenElectron = aGenElectron;
	matchedGenLepton = true;
      }
    }
  } 
  
  if (matchedGenLepton){
    bool isFromBDecay = false;
    bool isFromCDecay = false;
    if(theGenElectron->mother() !=0 && abs(theGenElectron->pdgId()) == 11 ){
      const reco::Candidate * aMotherGenElectron1 = theGenElectron->mother();
      const reco::Candidate * aMotherGenElectron2 = theGenElectron->mother();
      while(aMotherGenElectron2->mother() !=0){
	aMotherGenElectron2 = aMotherGenElectron2->mother();
	if( abs(aMotherGenElectron2->pdgId()) == 24 && abs(aMotherGenElectron1->pdgId()) == 11) electronOrigin = electronOrigin+1;
	if( abs(aMotherGenElectron2->pdgId()) == 24 && abs(aMotherGenElectron1->pdgId()) == 15) electronOrigin = electronOrigin+1;
	if( abs(aMotherGenElectron2->pdgId()) == 23 && abs(aMotherGenElectron1->pdgId()) == 11) electronOrigin = electronOrigin+10;
	if( abs(aMotherGenElectron1->pdgId()) == 4 || (abs(aMotherGenElectron1->pdgId()) > 39 && abs(aMotherGenElectron1->pdgId()) < 50) || (abs(aMotherGenElectron1->pdgId()) > 390 && abs(aMotherGenElectron1->pdgId()) < 500) || (abs(aMotherGenElectron1->pdgId()) > 3900 && abs(aMotherGenElectron1->pdgId()) < 5000)) isFromCDecay = true;
	if( abs(aMotherGenElectron1->pdgId()) == 5 || (abs(aMotherGenElectron1->pdgId()) > 49 && abs(aMotherGenElectron1->pdgId()) < 60) || (abs(aMotherGenElectron1->pdgId()) > 490 && abs(aMotherGenElectron1->pdgId()) < 600) || (abs(aMotherGenElectron1->pdgId()) > 4900 && abs(aMotherGenElectron1->pdgId()) < 6000)) isFromBDecay = true;
      }
      aMotherGenElectron1 = aMotherGenElectron2;
    }
    
    if(isFromCDecay) electronOrigin = electronOrigin+1000;
    if(isFromBDecay) electronOrigin = electronOrigin+100;
  }
  return electronOrigin;
}





int TopDileptValidation::getMuonOrigin(edm::Handle<reco::GenParticleCollection> genParticles, const pat::Muon * thePatMuon){
  
  int muonOrigin = 0;
  const reco::Candidate * theGenMuon;
  bool matchedGenLepton = false;
  
  for (reco::GenParticleCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++p){
    const reco::Candidate * aGenMuon = (dynamic_cast<reco::Candidate *>(const_cast<reco::GenParticle *>(&*p)));
    
    if (abs(p->pdgId()) == 13 && p->status() == 1){
      if ((thePatMuon->genLepton() != NULL) && abs(thePatMuon->genLepton()->pt()-aGenMuon->pt()) < 0.00001){
	theGenMuon = aGenMuon;
	matchedGenLepton = true; 
      }
    }
  } 
  
  if (matchedGenLepton){
    bool isFromBDecay = false;
    bool isFromCDecay = false;
    if(theGenMuon->mother() !=0 && abs(theGenMuon->pdgId()) == 13 ){
      const reco::Candidate * aMotherGenMuon1 = theGenMuon->mother();
      const reco::Candidate * aMotherGenMuon2 = theGenMuon->mother();
      while(aMotherGenMuon2->mother() !=0){
	aMotherGenMuon2 = aMotherGenMuon2->mother();
	if( abs(aMotherGenMuon2->pdgId()) == 24 && abs(aMotherGenMuon1->pdgId()) == 13) muonOrigin = muonOrigin+1;	// muon from W
	if( abs(aMotherGenMuon2->pdgId()) == 24 && abs(aMotherGenMuon1->pdgId()) == 15) muonOrigin = muonOrigin+1;	// muon from W->tau
	if( abs(aMotherGenMuon2->pdgId()) == 23 && abs(aMotherGenMuon1->pdgId()) == 13) muonOrigin = muonOrigin+10;
	if( abs(aMotherGenMuon1->pdgId()) == 4 || (abs(aMotherGenMuon1->pdgId()) > 39 && abs(aMotherGenMuon1->pdgId()) < 50) || (abs(aMotherGenMuon1->pdgId()) > 390 && abs(aMotherGenMuon1->pdgId()) < 500) || (abs(aMotherGenMuon1->pdgId()) > 3900 && abs(aMotherGenMuon1->pdgId()) < 5000)) isFromCDecay = true;
	if( abs(aMotherGenMuon1->pdgId()) == 5 || (abs(aMotherGenMuon1->pdgId()) > 49 && abs(aMotherGenMuon1->pdgId()) < 60) || (abs(aMotherGenMuon1->pdgId()) > 490 && abs(aMotherGenMuon1->pdgId()) < 600) || (abs(aMotherGenMuon1->pdgId()) > 4900 && abs(aMotherGenMuon1->pdgId()) < 6000)) isFromBDecay = true;
      }
      aMotherGenMuon1 = aMotherGenMuon2;
    }
    if(isFromCDecay) muonOrigin = muonOrigin+1000;
    if(isFromBDecay) muonOrigin = muonOrigin+100;
  }
  return muonOrigin;  //REMARK : cbZW format!
}








//define this as a plug-in
DEFINE_FWK_MODULE(TopDileptValidation);
