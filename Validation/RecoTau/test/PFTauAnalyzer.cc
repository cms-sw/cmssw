/* class PFTauAnalyzer
 *  EDAnalyzer of the PFTau objects, 
 *  created: Sep 15 2007,
 *  revised: Dec 28 2007
 *  contributors : Ludovic Houchu, Anne-Catherine Le Bihan.
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include <memory>
#include <string>
#include <iostream>
#include <limits>

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>

using namespace std; 
using namespace edm;
using namespace reco;
using namespace math;

class PFTauAnalyzer : public EDAnalyzer {
public:
  explicit PFTauAnalyzer(const ParameterSet&);
  ~PFTauAnalyzer() {;}
  virtual void analyze(const Event& iEvent, const EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
private:
  string GenJetProd_;
  string HepMCProductProd_;
  string PFSimParticleProd_;
  string PFTauProd_;
  string PFTauDiscriminatorByIsolationProd_;
  string PVProd_;
  bool UseCHCandLeadCHCand_tksDZconstraint_;
  double CHCandLeadCHCand_tksmaxDZ_;
  double test_matchingcone_size_;
  double test_leadCHCand_minpt_;
  double test_Cand_minpt_;
  double test_trackersignalcone_size_;
  double test_trackerisolcone_size_;
  double test_ECALsignalcone_size_;
  double test_ECALisolcone_size_;
  double test_HCALsignalcone_size_;
  double test_HCALisolcone_size_;
  string output_filename_;
  TFile* thefile;
  TTree* theEventTree; 
  TTree* thePFTauTree; 
  int gets_recPV;
  static const int MaxGenTausnumber=100;
  int GenTausnumber;
  int GenTau_pid[MaxGenTausnumber];
  float GenTau_e[MaxGenTausnumber]; 
  float GenTau_et[MaxGenTausnumber];
  float GenTau_theta[MaxGenTausnumber];
  float GenTau_phi[MaxGenTausnumber];
  float GenTau_visproducts_e[MaxGenTausnumber];
  float GenTau_visproducts_et[MaxGenTausnumber];
  float GenTau_visproducts_theta[MaxGenTausnumber];
  float GenTau_visproducts_phi[MaxGenTausnumber];
  int   GenTau_decaytype[MaxGenTausnumber];  
  int   GenTau_pftau[MaxGenTausnumber];  
  int   GenTau_signalCHCands_number[MaxGenTausnumber];  
  int   GenTau_isolCHCands_number[MaxGenTausnumber];  
  int   GenTau_NHCands_number[MaxGenTausnumber];  
  int   GenTau_GCands_number[MaxGenTausnumber];    
  static const int MaxSimTausnumber=100;
  int SimTausnumber;
  int SimTau_pid[MaxSimTausnumber];
  float SimTau_visproducts_e[MaxSimTausnumber];
  float SimTau_visproducts_p[MaxSimTausnumber];
  float SimTau_visproducts_theta[MaxSimTausnumber];
  float SimTau_visproducts_phi[MaxSimTausnumber];
  int SimTau_decaytype[MaxSimTausnumber];
  static const int MaxGenJet05snumber=200;
  int GenJet05snumber;
  float GenJet05_e[MaxGenJet05snumber];
  float GenJet05_et[MaxGenJet05snumber];
  float GenJet05_eta[MaxGenJet05snumber];
  float GenJet05_phi[MaxGenJet05snumber];
  int PFTau_eventgets_recPV;
  float PFTau_e;
  float PFTau_et;
  float PFTau_eta;
  float PFTau_phi;
  float PFTau_invmass;
  float PFTau_mode;
  float PFTau_modet;
  float PFTau_modeta;
  float PFTau_modphi;
  float PFTau_modinvmass;
  float PFTau_discriminator;
  int PFTau_passed_tracksel;
  int PFTau_passed_trackisolsel;
  int PFTau_passed_ECALisolsel;
  float PFTau_leadCHCand_pt;
  static const int PFTau_CHCands_numbermax=100;
  int PFTau_CHCands_number;
  float PFTau_CHCandDR[PFTau_CHCands_numbermax];
  int PFTau_signalCHCands_number;
  int PFTau_isolCHCands_number;
  static const int PFTau_NHCands_numbermax=100;
  int PFTau_NHCands_number;
  float PFTau_NHCandDR[PFTau_NHCands_numbermax];
  float PFTau_NHCandEt[PFTau_NHCands_numbermax];
  static const int PFTau_GCands_numbermax=100;
  int PFTau_GCands_number;
  float PFTau_GCandDR[PFTau_GCands_numbermax];
  float PFTau_GCandEt[PFTau_GCands_numbermax];
  float PFTau_GenTau_visproducts_e;
  float PFTau_GenTau_visproducts_et;
  float PFTau_GenTau_visproducts_eta;
  float PFTau_GenTau_visproducts_phi;
  int PFTau_GenTau_visproducts_type;
  float PFTau_SimTau_visproducts_e;
  float PFTau_SimTau_visproducts_et;
  float PFTau_SimTau_visproducts_eta;
  float PFTau_SimTau_visproducts_phi;
  int PFTau_SimTau_visproducts_type;
  float PFTau_GenJet05_e;
  float PFTau_GenJet05_et;
  float PFTau_GenJet05_eta;
  float PFTau_GenJet05_phi;
  class BydecreasingEt {
  public:
    bool operator()(math::XYZTLorentzVector a,math::XYZTLorentzVector b) {
      return (double)a.Et()>(double)b.Et();      
    }
  };
};

PFTauAnalyzer::PFTauAnalyzer(const edm::ParameterSet& iConfig){  
  GenJetProd_                           = iConfig.getParameter<string>("GenJetProd");
  HepMCProductProd_                     = iConfig.getParameter<string>("HepMCProductProd");
  PFSimParticleProd_                    = iConfig.getParameter<string>("PFSimParticleProd");
  PFTauProd_                            = iConfig.getParameter<string>("PFTauProd");
  PFTauDiscriminatorByIsolationProd_    = iConfig.getParameter<string>("PFTauDiscriminatorByIsolationProd");
  PVProd_                               = iConfig.getParameter<string>("PVProd");
  UseCHCandLeadCHCand_tksDZconstraint_  = iConfig.getParameter<bool>("UseCHCandLeadCHCand_tksDZconstraint");
  CHCandLeadCHCand_tksmaxDZ_            = iConfig.getParameter<double>("CHCandLeadCHCand_tksmaxDZ");
  test_matchingcone_size_               = iConfig.getParameter<double>("test_matchingcone_size");
  test_leadCHCand_minpt_                = iConfig.getParameter<double>("test_leadCHCand_minpt");
  test_Cand_minpt_                      = iConfig.getParameter<double>("test_Cand_minpt");
  test_trackersignalcone_size_          = iConfig.getParameter<double>("test_trackersignalcone_size");
  test_trackerisolcone_size_            = iConfig.getParameter<double>("test_trackerisolcone_size");
  test_ECALsignalcone_size_             = iConfig.getParameter<double>("test_ECALsignalcone_size");
  test_ECALisolcone_size_               = iConfig.getParameter<double>("test_ECALisolcone_size");
  test_HCALsignalcone_size_             = iConfig.getParameter<double>("test_HCALsignalcone_size");
  test_HCALisolcone_size_               = iConfig.getParameter<double>("test_HCALisolcone_size");
  output_filename_                      = iConfig.getParameter<string>("output_filename");
  thefile=TFile::Open(output_filename_.c_str(),"recreate");  
  thefile->cd();
  theEventTree = new TTree("theEventTree", "theEventTree");
  theEventTree->Branch("gets_recPV",&gets_recPV,"gets_recPV/I");
  theEventTree->Branch("SimTausnumber",&SimTausnumber,"SimTausnumber/I");
  theEventTree->Branch("SimTau_pid",SimTau_pid,"SimTau_pid[SimTausnumber]/I");
  theEventTree->Branch("SimTau_visproducts_e",SimTau_visproducts_e,"SimTau_visproducts_e[SimTausnumber]/F");
  theEventTree->Branch("SimTau_visproducts_p",SimTau_visproducts_p,"SimTau_visproducts_p[SimTausnumber]/F");
  theEventTree->Branch("SimTau_visproducts_theta",SimTau_visproducts_theta,"SimTau_visproducts_theta[SimTausnumber]/F");
  theEventTree->Branch("SimTau_visproducts_phi",SimTau_visproducts_phi,"SimTau_visproducts_phi[SimTausnumber]/F");
  theEventTree->Branch("SimTau_decaytype",SimTau_decaytype,"SimTau_decaytype[SimTausnumber]/I");
  theEventTree->Branch("GenTausnumber",&GenTausnumber,"GenTausnumber/I");
  theEventTree->Branch("GenTau_pid",GenTau_pid,"GenTau_pid[GenTausnumber]/I");
  theEventTree->Branch("GenTau_e",GenTau_e,"GenTau_e[GenTausnumber]/F");
  theEventTree->Branch("GenTau_et",GenTau_et,"GenTau_et[GenTausnumber]/F");
  theEventTree->Branch("GenTau_theta",GenTau_theta,"GenTau_theta[GenTausnumber]/F");
  theEventTree->Branch("GenTau_phi",GenTau_phi,"GenTau_phi[GenTausnumber]/F");
  theEventTree->Branch("GenTau_visproducts_e",GenTau_visproducts_e,"GenTau_visproducts_e[GenTausnumber]/F");
  theEventTree->Branch("GenTau_visproducts_et",GenTau_visproducts_et,"GenTau_visproducts_et[GenTausnumber]/F");
  theEventTree->Branch("GenTau_visproducts_theta",GenTau_visproducts_theta,"GenTau_visproducts_theta[GenTausnumber]/F");
  theEventTree->Branch("GenTau_visproducts_phi",GenTau_visproducts_phi,"GenTau_visproducts_phi[GenTausnumber]/F");
  theEventTree->Branch("GenTau_decaytype",GenTau_decaytype,"GenTau_decaytype[GenTausnumber]/I");  
  theEventTree->Branch("GenTau_pftau",GenTau_pftau,"GenTau_pftau[GenTausnumber]/I");
  theEventTree->Branch("GenTau_signalCHCands_number",GenTau_signalCHCands_number,"GenTau_signalCHCands_number[GenTausnumber]/I");
  theEventTree->Branch("GenTau_isolCHCands_number",GenTau_isolCHCands_number,"GenTau_isolCHCands_number[GenTausnumber]/I");
  theEventTree->Branch("GenTau_NHCands_number",GenTau_NHCands_number,"GenTau_NHCands_number[GenTausnumber]/I");
  theEventTree->Branch("GenTau_GCands_number",GenTau_GCands_number,"GenTau_GCands_number[GenTausnumber]/I");       
  theEventTree->Branch("GenJet05snumber",&GenJet05snumber,"GenJet05snumber/I");
  theEventTree->Branch("GenJet05_e",GenJet05_e,"GenJet05_e[GenJet05snumber]/F");
  theEventTree->Branch("GenJet05_et",GenJet05_et,"GenJet05_et[GenJet05snumber]/F");
  theEventTree->Branch("GenJet05_eta",GenJet05_eta,"GenJet05_eta[GenJet05snumber]/F");
  theEventTree->Branch("GenJet05_phi",GenJet05_phi,"GenJet05_phi[GenJet05snumber]/F");
  thePFTauTree = new TTree("thePFTauTree", "thePFTauTree"); 
  thePFTauTree->Branch("PFTau_eventgets_recPV",&PFTau_eventgets_recPV,"PFTau_eventgets_recPV/I");
  thePFTauTree->Branch("PFTau_e",&PFTau_e,"PFTau_e/F");
  thePFTauTree->Branch("PFTau_et",&PFTau_et,"PFTau_et/F");
  thePFTauTree->Branch("PFTau_eta",&PFTau_eta,"PFTau_eta/F");
  thePFTauTree->Branch("PFTau_phi",&PFTau_phi,"PFTau_phi/F");
  thePFTauTree->Branch("PFTau_invmass",&PFTau_invmass,"PFTau_invmass/F");
  thePFTauTree->Branch("PFTau_mode",&PFTau_mode,"PFTau_mode/F");
  thePFTauTree->Branch("PFTau_modet",&PFTau_modet,"PFTau_modet/F");
  thePFTauTree->Branch("PFTau_modeta",&PFTau_modeta,"PFTau_modeta/F");
  thePFTauTree->Branch("PFTau_modphi",&PFTau_modphi,"PFTau_modphi/F");
  thePFTauTree->Branch("PFTau_modinvmass",&PFTau_modinvmass,"PFTau_modinvmass/F");
  thePFTauTree->Branch("PFTau_discriminator",&PFTau_discriminator,"PFTau_discriminator/F");
  thePFTauTree->Branch("PFTau_passed_tracksel",&PFTau_passed_tracksel,"PFTau_passed_tracksel/I");
  thePFTauTree->Branch("PFTau_passed_trackisolsel",&PFTau_passed_trackisolsel,"PFTau_passed_trackisolsel/I");
  thePFTauTree->Branch("PFTau_passed_ECALisolsel",&PFTau_passed_ECALisolsel,"PFTau_passed_ECALisolsel/I");
  thePFTauTree->Branch("PFTau_leadCHCand_pt",&PFTau_leadCHCand_pt,"PFTau_leadCHCand_pt/F");
  thePFTauTree->Branch("PFTau_CHCands_number",&PFTau_CHCands_number,"PFTau_CHCands_number/I");
  thePFTauTree->Branch("PFTau_CHCandDR",PFTau_CHCandDR,"PFTau_CHCandDR[PFTau_CHCands_number]/F");
  thePFTauTree->Branch("PFTau_signalCHCands_number",&PFTau_signalCHCands_number,"PFTau_signalCHCands_number/I");
  thePFTauTree->Branch("PFTau_isolCHCands_number",&PFTau_isolCHCands_number,"PFTau_isolCHCands_number/I");
  thePFTauTree->Branch("PFTau_NHCands_number",&PFTau_NHCands_number,"PFTau_NHCands_number/I");
  thePFTauTree->Branch("PFTau_NHCandDR",PFTau_NHCandDR,"PFTau_NHCandDR[PFTau_NHCands_number]/F");
  thePFTauTree->Branch("PFTau_NHCandEt",PFTau_NHCandEt,"PFTau_NHCandEt[PFTau_NHCands_number]/F");
  thePFTauTree->Branch("PFTau_GCands_number",&PFTau_GCands_number,"PFTau_GCands_number/I");
  thePFTauTree->Branch("PFTau_GCandDR",PFTau_GCandDR,"PFTau_GCandDR[PFTau_GCands_number]/F");
  thePFTauTree->Branch("PFTau_GCandEt",PFTau_GCandEt,"PFTau_GCandEt[PFTau_GCands_number]/F");
  thePFTauTree->Branch("PFTau_GenTau_visproducts_e",&PFTau_GenTau_visproducts_e,"PFTau_GenTau_visproducts_e/F");
  thePFTauTree->Branch("PFTau_GenTau_visproducts_et",&PFTau_GenTau_visproducts_et,"PFTau_GenTau_visproducts_et/F");
  thePFTauTree->Branch("PFTau_GenTau_visproducts_eta",&PFTau_GenTau_visproducts_eta,"PFTau_GenTau_visproducts_eta/F");
  thePFTauTree->Branch("PFTau_GenTau_visproducts_phi",&PFTau_GenTau_visproducts_phi,"PFTau_GenTau_visproducts_phi/F");
  thePFTauTree->Branch("PFTau_GenTau_visproducts_type",&PFTau_GenTau_visproducts_type,"PFTau_GenTau_visproducts_type/I");
  thePFTauTree->Branch("PFTau_SimTau_visproducts_e",&PFTau_SimTau_visproducts_e,"PFTau_SimTau_visproducts_e/F");
  thePFTauTree->Branch("PFTau_SimTau_visproducts_et",&PFTau_SimTau_visproducts_et,"PFTau_SimTau_visproducts_et/F");
  thePFTauTree->Branch("PFTau_SimTau_visproducts_eta",&PFTau_SimTau_visproducts_eta,"PFTau_SimTau_visproducts_eta/F");
  thePFTauTree->Branch("PFTau_SimTau_visproducts_phi",&PFTau_SimTau_visproducts_phi,"PFTau_SimTau_visproducts_phi/F");
  thePFTauTree->Branch("PFTau_SimTau_visproducts_type",&PFTau_SimTau_visproducts_type,"PFTau_SimTau_visproducts_type/I");
  thePFTauTree->Branch("PFTau_GenJet05_e",&PFTau_GenJet05_e,"PFTau_GenJet05_e/F");
  thePFTauTree->Branch("PFTau_GenJet05_et",&PFTau_GenJet05_et,"PFTau_GenJet05_et/F");
  thePFTauTree->Branch("PFTau_GenJet05_eta",&PFTau_GenJet05_eta,"PFTau_GenJet05_eta/F");
  thePFTauTree->Branch("PFTau_GenJet05_phi",&PFTau_GenJet05_phi,"PFTau_GenJet05_phi/F");
}
void PFTauAnalyzer::beginJob(){}
void PFTauAnalyzer::endJob(){
  thefile->cd();
  theEventTree->Write();
  thePFTauTree->Write();
  thefile->Write();
  thefile->Close();
}
void PFTauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  vector<pair<math::XYZTLorentzVector,int> > GenTau_pair;
  GenTau_pair.clear();
  GenTausnumber=0;
  vector<pair<math::XYZTLorentzVector,int> > SimTau_pair;
  SimTau_pair.clear();
  SimTausnumber=0;
  vector<math::XYZTLorentzVector> GenJet05_LorentzVect;
  GenJet05_LorentzVect.clear();
  GenJet05snumber=0;
  /* *******************************************************************
     generation step
     ******************************************************************* */
  
  Handle<HepMCProduct> evt;
  iEvent.getByLabel(HepMCProductProd_,evt);
 
  // select susy processes
  /*
  if ((*(evt->GetEvent())).signal_process_id()<200 ||  (*(evt->GetEvent())).signal_process_id()>300){
    return;
  }
  */
   
  // select gamma*/Z0 processes
  /*
  if ((*(evt->GetEvent())).signal_process_id()!=1){
    return;
  }
  */
  
  // select QCD-dijet processes
  /*   
  if ((*(evt->GetEvent())).signal_process_id()!=11 
      && (*(evt->GetEvent())).signal_process_id()!=12 
      && (*(evt->GetEvent())).signal_process_id()!=13 
      && (*(evt->GetEvent())).signal_process_id()!=68 
      && (*(evt->GetEvent())).signal_process_id()!=28 
      && (*(evt->GetEvent())).signal_process_id()!=53){
    return;
  }
  */
  int iGenTau = 0; 
  for (HepMC::GenEvent::particle_const_iterator iter=(*(evt->GetEvent())).particles_begin();iter!=(*(evt->GetEvent())).particles_end();iter++) {
     if ((**iter).status()==2 && (abs((**iter).pdg_id())==15)){
      HepMC::GenParticle* TheParticle=(*iter);
      math::XYZTLorentzVector TheParticle_LorentzVect(TheParticle->momentum().px(),TheParticle->momentum().py(),TheParticle->momentum().pz(),TheParticle->momentum().e());
      math::XYZTLorentzVector TheTauJet_LorentzVect=TheParticle_LorentzVect;
      HepMC::GenParticle* TheTauneutrino=0;
      HepMC::GenParticle* TheTauneutrinobar=0;
      int tau_children_n=TheParticle->end_vertex()->particles_out_size();
      for (HepMC::GenVertex::particles_out_const_iterator i_taudaughter=TheParticle->end_vertex()->particles_out_const_begin();i_taudaughter!=TheParticle->end_vertex()->particles_out_const_end();i_taudaughter++){
	if ((**i_taudaughter).status()==1 && abs((int)(**i_taudaughter).pdg_id())==16){
	  TheTauneutrino=*i_taudaughter;
	  math::XYZTLorentzVector TheTauneutrino_LorentzVect(TheTauneutrino->momentum().px(),TheTauneutrino->momentum().py(),TheTauneutrino->momentum().pz(),TheTauneutrino->momentum().e());
	  TheTauJet_LorentzVect-=TheTauneutrino_LorentzVect;
	}
      }
      bool tau_decayingtoenunubar = false;
      bool tau_decayingtomununubar = false;
      bool tau_decayingto1prongnu = false;
      bool tau_decayingtopi0chargedpinu  = false;
      bool tau_decayingto2pi0chargedpinu = false;
      bool tau_decayingto3pi0chargedpinu = false;
      bool tau_decayingto4pi0chargedpinu = false;  
      bool tau_decayingtopi0chargedpinugam  = false;
      bool tau_decayingto3prongsnu          = false;
      bool tau_decayingto3prongs1pi0nu      = false;
      bool tau_decayingto3prongs2pi0nu      = false;
      
      if (tau_children_n==2) {
	HepMC::GenVertex::particles_out_const_iterator i_1sttaudaughter=TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_2ndtaudaughter=++TheParticle->end_vertex()->particles_out_const_begin();
	if (abs((**i_1sttaudaughter).pdg_id())==211 || abs((**i_2ndtaudaughter).pdg_id())==211) tau_decayingto1prongnu = true;
      }
      if (tau_children_n==2) {
	HepMC::GenVertex::particles_out_const_iterator i_1sttaudaughter=TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_2ndtaudaughter=++TheParticle->end_vertex()->particles_out_const_begin();
	if (abs((**i_1sttaudaughter).pdg_id())==213 || abs((**i_2ndtaudaughter).pdg_id())==213) tau_decayingtopi0chargedpinu=true;
      }
      if (tau_children_n==3) {
	HepMC::GenVertex::particles_out_const_iterator i_1sttaudaughter=TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_2ndtaudaughter=++TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_3rdtaudaughter=++(++TheParticle->end_vertex()->particles_out_const_begin());
	if ((abs((**i_1sttaudaughter).pdg_id())==16 && abs((**i_2ndtaudaughter).pdg_id())==211 && (int)(**i_3rdtaudaughter).pdg_id()==111)
	    || (abs((**i_1sttaudaughter).pdg_id())==16 && abs((**i_3rdtaudaughter).pdg_id())==211 && (int)(**i_2ndtaudaughter).pdg_id()==111)
	    || (abs((**i_2ndtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && (int)(**i_3rdtaudaughter).pdg_id()==111)
	    || (abs((**i_2ndtaudaughter).pdg_id())==16 && abs((**i_3rdtaudaughter).pdg_id())==211 && (int)(**i_1sttaudaughter).pdg_id()==111)
	    || (abs((**i_3rdtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && (int)(**i_2ndtaudaughter).pdg_id()==111)
	    || (abs((**i_3rdtaudaughter).pdg_id())==16 && abs((**i_2ndtaudaughter).pdg_id())==211 && (int)(**i_1sttaudaughter).pdg_id()==111)
	    ) tau_decayingtopi0chargedpinu = true;
	if ((abs((**i_1sttaudaughter).pdg_id())==16 && abs((**i_2ndtaudaughter).pdg_id())==211 && (int)(**i_3rdtaudaughter).pdg_id()==113)
	    || (abs((**i_1sttaudaughter).pdg_id())==16 && abs((**i_3rdtaudaughter).pdg_id())==211 && (int)(**i_2ndtaudaughter).pdg_id()==113)
	    || (abs((**i_2ndtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && (int)(**i_3rdtaudaughter).pdg_id()==113)
	    || (abs((**i_2ndtaudaughter).pdg_id())==16 && abs((**i_3rdtaudaughter).pdg_id())==211 && (int)(**i_1sttaudaughter).pdg_id()==113)
	    || (abs((**i_3rdtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && (int)(**i_2ndtaudaughter).pdg_id()==113)
	    || (abs((**i_3rdtaudaughter).pdg_id())==16 && abs((**i_2ndtaudaughter).pdg_id())==211 && (int)(**i_1sttaudaughter).pdg_id()==113)
	    ) tau_decayingto3prongsnu = true;
      }
      if (tau_children_n==4) {
	HepMC::GenVertex::particles_out_const_iterator i_1sttaudaughter=TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_2ndtaudaughter=++TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_3rdtaudaughter=++(++TheParticle->end_vertex()->particles_out_const_begin());
	HepMC::GenVertex::particles_out_const_iterator i_4thtaudaughter=++(++(++TheParticle->end_vertex()->particles_out_const_begin()));
	if ((abs((**i_1sttaudaughter).pdg_id())==16 && abs((**i_2ndtaudaughter).pdg_id())==211 && abs((**i_3rdtaudaughter).pdg_id())==211 && abs((**i_4thtaudaughter).pdg_id())==211)
	    || (abs((**i_2ndtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && abs((**i_3rdtaudaughter).pdg_id())==211 && abs((**i_4thtaudaughter).pdg_id())==211)
	    || (abs((**i_3rdtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && abs((**i_2ndtaudaughter).pdg_id())==211 && abs((**i_4thtaudaughter).pdg_id())==211)
	    || (abs((**i_4thtaudaughter).pdg_id())==16 && abs((**i_1sttaudaughter).pdg_id())==211 && abs((**i_2ndtaudaughter).pdg_id())==211 && abs((**i_3rdtaudaughter).pdg_id())==211)
	    )  tau_decayingto3prongsnu = true;
      }
      if (tau_children_n==3) {
	HepMC::GenVertex::particles_out_const_iterator i_1sttaudaughter=TheParticle->end_vertex()->particles_out_const_begin();
	HepMC::GenVertex::particles_out_const_iterator i_2ndtaudaughter=++TheParticle->end_vertex()->particles_out_const_begin();	
	HepMC::GenVertex::particles_out_const_iterator i_3rdtaudaughter=++(++TheParticle->end_vertex()->particles_out_const_begin());	
	if (abs((**i_1sttaudaughter).pdg_id())==12 || abs((**i_2ndtaudaughter).pdg_id())==12 || abs((**i_3rdtaudaughter).pdg_id())==12) {
	  if (abs((**i_1sttaudaughter).pdg_id())==12) TheTauneutrinobar=(*i_1sttaudaughter);
	  if (abs((**i_2ndtaudaughter).pdg_id())==12) TheTauneutrinobar=(*i_2ndtaudaughter);
	  if (abs((**i_3rdtaudaughter).pdg_id())==12) TheTauneutrinobar=(*i_3rdtaudaughter);
	  math::XYZTLorentzVector TheTauneutrinobar_LorentzVect(TheTauneutrinobar->momentum().px(),TheTauneutrinobar->momentum().py(),TheTauneutrinobar->momentum().pz(),TheTauneutrinobar->momentum().e());
	  TheTauJet_LorentzVect = TheTauJet_LorentzVect-TheTauneutrinobar_LorentzVect;
	  tau_decayingtoenunubar = true;
	}
	if (abs((**i_1sttaudaughter).pdg_id())==14 || abs((**i_2ndtaudaughter).pdg_id())==14 || abs((**i_3rdtaudaughter).pdg_id())==14) {
	  if (abs((**i_1sttaudaughter).pdg_id())==14) TheTauneutrinobar=(*i_1sttaudaughter);
	  if (abs((**i_2ndtaudaughter).pdg_id())==14) TheTauneutrinobar=(*i_2ndtaudaughter);
	  if (abs((**i_3rdtaudaughter).pdg_id())==14) TheTauneutrinobar=(*i_3rdtaudaughter);
	  math::XYZTLorentzVector TheTauneutrinobar_LorentzVect(TheTauneutrinobar->momentum().px(),TheTauneutrinobar->momentum().py(),TheTauneutrinobar->momentum().pz(),TheTauneutrinobar->momentum().e());
	  TheTauJet_LorentzVect = TheTauJet_LorentzVect-TheTauneutrinobar_LorentzVect;
	  tau_decayingtomununubar = true;
	}	
      }
      GenTau_pid[iGenTau] = (int)(**iter).pdg_id();
      GenTau_e[iGenTau] = (float)(**iter).momentum().e();
      GenTau_et[iGenTau] = (float)((**iter).momentum().e()*fabs(sin((**iter).momentum().theta())));
      GenTau_theta[iGenTau] = (float)(**iter).momentum().theta();
      GenTau_phi[iGenTau] = (float)(**iter).momentum().phi();
      GenTau_visproducts_e[iGenTau] = (float)TheTauJet_LorentzVect.e();
      GenTau_visproducts_et[iGenTau] = (float)TheTauJet_LorentzVect.Et();
      GenTau_visproducts_theta[iGenTau] = (float)TheTauJet_LorentzVect.theta();
      GenTau_visproducts_phi[iGenTau] = (float)TheTauJet_LorentzVect.phi();
      if (tau_decayingtoenunubar) {
	GenTau_decaytype[iGenTau] = 1;
      }
      if (tau_decayingtomununubar) {
	GenTau_decaytype[iGenTau] = 2;
      }
      if (tau_decayingto1prongnu) {
	GenTau_decaytype[iGenTau] = 3;
      }
      if (tau_decayingtopi0chargedpinu) {
	GenTau_decaytype[iGenTau] = 4;
      }
      if (tau_decayingto3prongsnu) {
	GenTau_decaytype[iGenTau] = 9;
      }
      if (!tau_decayingtoenunubar && !tau_decayingtomununubar && !tau_decayingto1prongnu && !tau_decayingtopi0chargedpinu && !tau_decayingto3prongsnu) {
	int npi  = 0;
	int npi0 = 0;
	int ngam = 0;	
	if(tau_children_n>0){
	HepMC::GenVertex::particles_out_const_iterator isttaudaughter = TheParticle->end_vertex()->particles_out_const_begin();
	for (int j=0; j< tau_children_n; j++)
	{ 
	  if  (abs((**isttaudaughter).pdg_id()) == 211) npi++ ;
	  if  (abs((**isttaudaughter).pdg_id()) == 111) npi0++; 
	  if  (abs((**isttaudaughter).pdg_id()) == 22)  ngam++;
	  
	  if ( abs((**isttaudaughter).pdg_id()) == 223 || abs((**isttaudaughter).pdg_id()) == 221 || abs((**isttaudaughter).pdg_id()) == 213 || abs((**isttaudaughter).pdg_id()) == 113)
	  { 
	    int daughter_children_n = (**isttaudaughter).end_vertex()->particles_out_size();
	    if (daughter_children_n>0) 
	    { 
	      HepMC::GenVertex::particles_out_const_iterator istbaby = (**isttaudaughter).end_vertex()->particles_out_const_begin();
	      for (int y=0; y<daughter_children_n; y++)
	      { 
	      
	       if  (abs((**istbaby).pdg_id()) == 211) npi++ ;
	       if  (abs((**istbaby).pdg_id()) == 111) npi0++;  
	       if  (abs((**istbaby).pdg_id()) == 22)  ngam++;
	       istbaby++;
	       }
	     }
	   }
	  isttaudaughter++;
	}
       }
       if      (npi == 1 && npi0 == 2)  	   {tau_decayingto2pi0chargedpinu   = true; GenTau_decaytype[iGenTau] = 5; }
       else if (npi == 1 && npi0 == 3)  	   {tau_decayingto3pi0chargedpinu   = true; GenTau_decaytype[iGenTau] = 6; }
       else if (npi == 1 && npi0 == 4)  	   {tau_decayingto4pi0chargedpinu   = true; GenTau_decaytype[iGenTau] = 7; }  
       else if (npi == 1 && npi0 == 1 && ngam >0)  {tau_decayingtopi0chargedpinugam = true; GenTau_decaytype[iGenTau] = 8; } 
       else if (npi == 3 && npi0 == 1)  	   {tau_decayingto3prongs1pi0nu     = true; GenTau_decaytype[iGenTau] = 10;}
       else if (npi == 3 && npi0 == 2)  	   {tau_decayingto3prongs2pi0nu     = true; GenTau_decaytype[iGenTau] = 11;}
       else                             	   {GenTau_decaytype[iGenTau] = 12;}
      }
      
      pair<math::XYZTLorentzVector,int> TheTauJet_pair(TheTauJet_LorentzVect,GenTau_decaytype[iGenTau]);
      GenTau_pair.push_back(TheTauJet_pair); 
           
      float minDeltaR = 99.;
      PFTauCollection::size_type minDeltaR_iPFTau=0;
      
      Handle<PFTauCollection> thePFTauHandle;
      iEvent.getByLabel(PFTauProd_,thePFTauHandle);
      
      if (thePFTauHandle->size()>0){
	for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) { 
	  PFTauRef thePFTau(thePFTauHandle,iPFTau);
	  math::XYZTLorentzVector ThePFTauJet_LorentzVect=(*thePFTau).p4();
	  if (ROOT::Math::VectorUtil::DeltaR(TheTauJet_LorentzVect,ThePFTauJet_LorentzVect)< minDeltaR ) {
	    minDeltaR_iPFTau = iPFTau;
	    minDeltaR=ROOT::Math::VectorUtil::DeltaR(TheTauJet_LorentzVect,ThePFTauJet_LorentzVect);
	  }      
        }
      }      
      if (minDeltaR <0.5){
	PFTauRef thePFTau(thePFTauHandle,minDeltaR_iPFTau);
	GenTau_pftau[iGenTau] = 1;
	GenTau_signalCHCands_number[iGenTau]=(int)(*thePFTau).signalPFChargedHadrCands().size();
	GenTau_isolCHCands_number[iGenTau]  =(int)(*thePFTau).isolationPFChargedHadrCands().size();
	GenTau_NHCands_number[iGenTau]	  =(int)(*thePFTau).pfTauTagInfoRef()->PFNeutrHadrCands().size();
	GenTau_GCands_number[iGenTau]	  =(int)(*thePFTau).pfTauTagInfoRef()->PFGammaCands().size();
      }
      else {
	GenTau_pftau[iGenTau] = -1;
	GenTau_signalCHCands_number[iGenTau]= 0;
	GenTau_isolCHCands_number[iGenTau]  = 0;
	GenTau_NHCands_number[iGenTau]      = 0;
	GenTau_GCands_number[iGenTau]	  = 0; 
      }          
      ++iGenTau;
    }		 
  }  
  GenTausnumber=iGenTau;
  
  int iSimTau = 0;
  /*
  Handle<PFSimParticleCollection> thePFSimParticleCollectionHandle;
  iEvent.getByLabel(PFSimParticleProd_,thePFSimParticleCollectionHandle);
  const PFSimParticleCollection& thePFSimParticleCollection=*(thePFSimParticleCollectionHandle.product());
  for (PFSimParticleCollection::const_iterator iPFSimParticle=thePFSimParticleCollection.begin();iPFSimParticle!=thePFSimParticleCollection.end();++iPFSimParticle) {
    const PFSimParticle& thePFSimParticle=(*iPFSimParticle);
  
    if (abs(thePFSimParticle.pdgCode())==15) { 
      const vector<int>& thePFSimParticledaughters = thePFSimParticle.daughterIds();
      math::XYZTLorentzVector TheTauJet_LorentzVect(0.,0.,0.,0.);
      bool tau_decayingtoenunubar=false;
      bool tau_decayingtomununubar=false;
      bool tau_decayingto1prongnu = false;
      bool tau_decayingtopi0chargedpinu = false;
      bool tau_decayingto3prongsnu = false;
      if ((int)thePFSimParticledaughters.size()==1){
	const PFSimParticle& theTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[0]];
	const PFTrajectoryPoint& theTauDaughter_1stTP=theTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector theTauDaughter_LorentzVect(theTauDaughter_1stTP.momentum().Px(),theTauDaughter_1stTP.momentum().Py(),theTauDaughter_1stTP.momentum().Pz(),theTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect=theTauDaughter_LorentzVect;
	unsigned pdgdaugter=theTauDaughter.pdgCode();
	if ((int)pdgdaugter==11) tau_decayingtoenunubar=true;
	if ((int)pdgdaugter==13) tau_decayingtomununubar=true;
	if (abs((int)pdgdaugter)==211) tau_decayingto1prongnu=true;
      }
      if ((int)thePFSimParticledaughters.size()==2){
	const PFSimParticle& the1stTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[0]];
	const PFTrajectoryPoint& the1stTauDaughter_1stTP=the1stTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector the1stTauDaughter_LorentzVect(the1stTauDaughter_1stTP.momentum().Px(),the1stTauDaughter_1stTP.momentum().Py(),the1stTauDaughter_1stTP.momentum().Pz(),the1stTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect=the1stTauDaughter_LorentzVect;
	unsigned pdg1stdaugter=the1stTauDaughter.pdgCode();
	const PFSimParticle& the2ndTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[1]];
	const PFTrajectoryPoint& the2ndTauDaughter_1stTP=the2ndTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector the2ndTauDaughter_LorentzVect(the2ndTauDaughter_1stTP.momentum().Px(),the2ndTauDaughter_1stTP.momentum().Py(),the2ndTauDaughter_1stTP.momentum().Pz(),the2ndTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect+=the2ndTauDaughter_LorentzVect;
	unsigned pdg2nddaugter=the2ndTauDaughter.pdgCode();
	if ((abs((int)pdg1stdaugter)==211 && (int)pdg2nddaugter==111) || (abs((int)pdg2nddaugter)==211 && (int)pdg1stdaugter==111)) tau_decayingtopi0chargedpinu=true;
      }
      if ((int)thePFSimParticledaughters.size()==3){
	const PFSimParticle& the1stTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[0]];
	const PFTrajectoryPoint& the1stTauDaughter_1stTP=the1stTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector the1stTauDaughter_LorentzVect(the1stTauDaughter_1stTP.momentum().Px(),the1stTauDaughter_1stTP.momentum().Py(),the1stTauDaughter_1stTP.momentum().Pz(),the1stTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect=the1stTauDaughter_LorentzVect;
	unsigned pdg1stdaugter=the1stTauDaughter.pdgCode();
	const PFSimParticle& the2ndTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[1]];
	const PFTrajectoryPoint& the2ndTauDaughter_1stTP=the2ndTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector the2ndTauDaughter_LorentzVect(the2ndTauDaughter_1stTP.momentum().Px(),the2ndTauDaughter_1stTP.momentum().Py(),the2ndTauDaughter_1stTP.momentum().Pz(),the2ndTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect+=the2ndTauDaughter_LorentzVect;
	unsigned pdg2nddaugter=the2ndTauDaughter.pdgCode();
	const PFSimParticle& the3rdTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[2]];
	const PFTrajectoryPoint& the3rdTauDaughter_1stTP=the3rdTauDaughter.trajectoryPoint(0);
	math::XYZTLorentzVector the3rdTauDaughter_LorentzVect(the3rdTauDaughter_1stTP.momentum().Px(),the3rdTauDaughter_1stTP.momentum().Py(),the3rdTauDaughter_1stTP.momentum().Pz(),the3rdTauDaughter_1stTP.momentum().E());
	TheTauJet_LorentzVect+=the3rdTauDaughter_LorentzVect;
	unsigned pdg3rddaugter=the3rdTauDaughter.pdgCode();
	if (abs((int)pdg1stdaugter)==211 && abs((int)pdg2nddaugter)==211 && abs((int)pdg3rddaugter)==211) tau_decayingto3prongsnu=true;
      }
      if ((int)thePFSimParticledaughters.size()>3){
	for (unsigned int idaughter=0;idaughter<thePFSimParticledaughters.size();++idaughter){
	  const PFSimParticle& theTauDaughter=thePFSimParticleCollection[thePFSimParticledaughters[idaughter]];
	  const PFTrajectoryPoint& theTauDaughter_1stTP=theTauDaughter.trajectoryPoint(0);
	  math::XYZTLorentzVector theTauDaughter_LorentzVect(theTauDaughter_1stTP.momentum().Px(),theTauDaughter_1stTP.momentum().Py(),theTauDaughter_1stTP.momentum().Pz(),theTauDaughter_1stTP.momentum().E());
	  TheTauJet_LorentzVect+=theTauDaughter_LorentzVect;
	}
      }
      
      if (tau_decayingtoenunubar) SimTau_decaytype[iSimTau] = 1;
      if (tau_decayingtomununubar) SimTau_decaytype[iSimTau] = 2;
      if (tau_decayingto1prongnu) SimTau_decaytype[iSimTau] = 3;
      if (tau_decayingtopi0chargedpinu) SimTau_decaytype[iSimTau] = 4;
      if (tau_decayingto3prongsnu) SimTau_decaytype[iSimTau] = 5;
      if (!tau_decayingtoenunubar && !tau_decayingtomununubar && !tau_decayingto1prongnu && !tau_decayingtopi0chargedpinu && !tau_decayingto3prongsnu) SimTau_decaytype[iSimTau] = 6;
      
       
      SimTau_pid[iSimTau] = thePFSimParticle.pdgCode();
      SimTau_visproducts_e[iSimTau] = (float)TheTauJet_LorentzVect.E();
      SimTau_visproducts_p[iSimTau] = (float)TheTauJet_LorentzVect.Rho();
      SimTau_visproducts_theta[iSimTau] = (float)TheTauJet_LorentzVect.Theta();
      SimTau_visproducts_phi[iSimTau] = (float)TheTauJet_LorentzVect.Phi();      
      pair<math::XYZTLorentzVector,int> TheTauJet_pair(TheTauJet_LorentzVect,SimTau_decaytype[iSimTau]);
      SimTau_pair.push_back(TheTauJet_pair);
      ++iSimTau;
    }
  }
  */
  SimTausnumber=iSimTau;
  
  int iGenJet05 = 0; 
  /*
  Handle<GenJetCollection> genIter05Jets;
  iEvent.getByLabel(GenJetProd_,genIter05Jets);   
  for(GenJetCollection::const_iterator i_genIter05Jet=genIter05Jets->begin();i_genIter05Jet!=genIter05Jets->end();++i_genIter05Jet) {
    math::XYZTLorentzVector myGenJet05_LorentzVect(i_genIter05Jet->px(),i_genIter05Jet->py(),i_genIter05Jet->pz(),i_genIter05Jet->energy());
    GenJet05_e[iGenJet05] = myGenJet05_LorentzVect.E();
    GenJet05_et[iGenJet05] = myGenJet05_LorentzVect.Et();
    GenJet05_eta[iGenJet05] = myGenJet05_LorentzVect.Eta();
    GenJet05_phi[iGenJet05] = myGenJet05_LorentzVect.Phi();
    GenJet05_LorentzVect.push_back(myGenJet05_LorentzVect);
    ++iGenJet05;
  }  
  */
  GenJet05snumber=iGenJet05;
  stable_sort(GenJet05_LorentzVect.begin(),GenJet05_LorentzVect.end(),BydecreasingEt());
  
  Handle<VertexCollection> vertices;
  iEvent.getByLabel(PVProd_,vertices);
  const VertexCollection vertCollection = *(vertices.product());
  Vertex myPV;
  if(!vertCollection.size()) {
     gets_recPV=0;
     myPV = vertCollection[0];
  }
  else gets_recPV=1;
  
  Handle<PFTauCollection> thePFTauHandle;
  iEvent.getByLabel(PFTauProd_,thePFTauHandle);
  
  Handle<PFTauDiscriminatorByIsolation> thePFTauDiscriminatorByIsolation;
  iEvent.getByLabel(PFTauDiscriminatorByIsolationProd_,thePFTauDiscriminatorByIsolation);
  
  for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) {
    PFTauRef thePFTau(thePFTauHandle,iPFTau);
    math::XYZTLorentzVector ThePFTauJet_LorentzVect=(*thePFTau).p4();
    PFTau_eventgets_recPV=gets_recPV;
    PFTau_e=ThePFTauJet_LorentzVect.E();
    PFTau_et=ThePFTauJet_LorentzVect.Et();
    PFTau_eta=ThePFTauJet_LorentzVect.Eta();
    PFTau_phi=ThePFTauJet_LorentzVect.Phi();
    PFTau_invmass=ThePFTauJet_LorentzVect.M(); 
    PFTau_mode=(*thePFTau).alternatLorentzVect().E();
    PFTau_modet=(*thePFTau).alternatLorentzVect().Et();
    PFTau_modeta=(*thePFTau).alternatLorentzVect().Eta();
    PFTau_modphi=(*thePFTau).alternatLorentzVect().Phi();
    PFTau_modinvmass=(*thePFTau).alternatLorentzVect().M(); 
    if ((*thePFTau).leadPFChargedHadrCand().isNonnull())PFTau_leadCHCand_pt=(*thePFTau).leadPFChargedHadrCand()->momentum().Rho();  
    else PFTau_leadCHCand_pt=-100.;
    PFTau_signalCHCands_number=(int)(*thePFTau).signalPFChargedHadrCands().size();
    PFTau_isolCHCands_number=(int)(*thePFTau).isolationPFChargedHadrCands().size();
    PFTau_NHCands_number=(int)(*thePFTau).pfTauTagInfoRef()->PFNeutrHadrCands().size();
    PFTau_GCands_number=(int)(*thePFTau).pfTauTagInfoRef()->PFGammaCands().size();
    //
    PFTau_discriminator=(*thePFTauDiscriminatorByIsolation)[thePFTau];    
    // Example of recomputing leading charged hadron PFCandidate, lists of signal cone and isolation annulus PFCandidates
    // BEGIN ***
    PFTau* thePFTauClone=(*thePFTau).clone();
    PFTauElementsOperators thePFTauElementsOperators(*thePFTauClone);
    PFCandidateRef theleadPFChargedHadrCand=thePFTauElementsOperators.leadPFChargedHadrCand("DR",test_matchingcone_size_,test_leadCHCand_minpt_);
    bool theleadPFChargedHadrCand_rectkavailable=false;
    double theleadPFChargedHadrCand_rectkDZ=0.;
    double thePFTau_refInnerPosition_x=0.;
    double thePFTau_refInnerPosition_y=0.;
    double thePFTau_refInnerPosition_z=0.;
    int ChargedHadrCands_n=0;
    if(theleadPFChargedHadrCand.isNonnull()){    
      // search for the Track which is the main constituent of a charged hadron PFCandidate ...
      TrackRef theleadPFChargedHadrCand_rectk=(*theleadPFChargedHadrCand).trackRef();
      if(theleadPFChargedHadrCand_rectk.isNonnull()){
	theleadPFChargedHadrCand_rectkavailable=true;
	theleadPFChargedHadrCand_rectkDZ=(*theleadPFChargedHadrCand_rectk).dz();
	if((*theleadPFChargedHadrCand_rectk).innerOk()){
	  thePFTau_refInnerPosition_x=(*theleadPFChargedHadrCand_rectk).innerPosition().x(); 
	  thePFTau_refInnerPosition_y=(*theleadPFChargedHadrCand_rectk).innerPosition().y(); 
	  thePFTau_refInnerPosition_z=(*theleadPFChargedHadrCand_rectk).innerPosition().z(); 
	}
      }    	      
        
      int NeutrHadrCands_n=0;
      int GammaCands_n=0;
      
      for (PFCandidateRefVector::const_iterator iChargedHadrCand=(*thePFTau).pfTauTagInfoRef()->PFChargedHadrCands().begin();iChargedHadrCand!=(*thePFTau).pfTauTagInfoRef()->PFChargedHadrCands().end();++iChargedHadrCand){
	if (UseCHCandLeadCHCand_tksDZconstraint_){
	  if (!theleadPFChargedHadrCand_rectkavailable) continue; // ch. hadr. cand. w/o track ???
	  bool iChargedHadrCand_rectkavailable=false;
	  double iChargedHadrCand_rectkDZ=0.;
	  TrackRef iChargedHadrCand_rectk=(**iChargedHadrCand).trackRef();
	  if(iChargedHadrCand_rectk.isNonnull()){
	    iChargedHadrCand_rectkavailable=true;
	    iChargedHadrCand_rectkDZ=(*iChargedHadrCand_rectk).dz();
	  }
	  if (!iChargedHadrCand_rectkavailable || fabs(iChargedHadrCand_rectkDZ-theleadPFChargedHadrCand_rectkDZ)>CHCandLeadCHCand_tksmaxDZ_) continue;
	}
	PFTau_CHCandDR[ChargedHadrCands_n]=-100.;
	if ((*iChargedHadrCand)!=theleadPFChargedHadrCand) PFTau_CHCandDR[ChargedHadrCands_n]=ROOT::Math::VectorUtil::DeltaR((**iChargedHadrCand).p4(),(*theleadPFChargedHadrCand).p4());
	++ChargedHadrCands_n;
      }
      
      for (PFCandidateRefVector::const_iterator iNeutrHadrCand=(*thePFTau).pfTauTagInfoRef()->PFNeutrHadrCands().begin();iNeutrHadrCand!=(*thePFTau).pfTauTagInfoRef()->PFNeutrHadrCands().end();++iNeutrHadrCand){
	PFTau_NHCandDR[NeutrHadrCands_n]=ROOT::Math::VectorUtil::DeltaR((**iNeutrHadrCand).p4(),(*theleadPFChargedHadrCand).p4());
	PFTau_NHCandEt[NeutrHadrCands_n]=(**iNeutrHadrCand).et();
	++NeutrHadrCands_n;
      }
      
      for (PFCandidateRefVector::const_iterator iGammaCand=(*thePFTau).pfTauTagInfoRef()->PFGammaCands().begin();iGammaCand!=(*thePFTau).pfTauTagInfoRef()->PFGammaCands().end();++iGammaCand){
	PFTau_GCandDR[GammaCands_n]=ROOT::Math::VectorUtil::DeltaR((**iGammaCand).p4(),(*theleadPFChargedHadrCand).p4());
	PFTau_GCandEt[GammaCands_n]=(**iGammaCand).et();
	++GammaCands_n;
      }
      
      PFCandidateRefVector theSignalPFChargedHadrCands,theSignalPFNeutrHadrCands,theSignalPFGammaCands,theSignalPFCands;
      if (UseCHCandLeadCHCand_tksDZconstraint_ && theleadPFChargedHadrCand_rectkavailable) 
         theSignalPFChargedHadrCands=thePFTauElementsOperators.PFChargedHadrCandsInCone((*theleadPFChargedHadrCand).momentum(),"DR",test_trackersignalcone_size_,test_Cand_minpt_,CHCandLeadCHCand_tksmaxDZ_,theleadPFChargedHadrCand_rectkDZ, myPV);
      else theSignalPFChargedHadrCands=thePFTauElementsOperators.PFChargedHadrCandsInCone((*theleadPFChargedHadrCand).momentum(),"DR",test_trackersignalcone_size_,test_Cand_minpt_);
      theSignalPFNeutrHadrCands=thePFTauElementsOperators.PFNeutrHadrCandsInCone((*theleadPFChargedHadrCand).momentum(),"DR",test_HCALsignalcone_size_,test_Cand_minpt_);
      theSignalPFGammaCands=thePFTauElementsOperators.PFGammaCandsInCone((*theleadPFChargedHadrCand).momentum(),"DR",test_ECALsignalcone_size_,test_Cand_minpt_);
      
      PFCandidateRefVector theIsolPFChargedHadrCands,theIsolPFNeutrHadrCands,theIsolPFGammaCands,theIsolPFCands;
      if (UseCHCandLeadCHCand_tksDZconstraint_ && theleadPFChargedHadrCand_rectkavailable) theIsolPFChargedHadrCands=thePFTauElementsOperators.PFChargedHadrCandsInAnnulus((*theleadPFChargedHadrCand).momentum(),"DR",test_trackersignalcone_size_,"DR",test_trackerisolcone_size_,test_Cand_minpt_,CHCandLeadCHCand_tksmaxDZ_,theleadPFChargedHadrCand_rectkDZ, myPV);
      else theIsolPFChargedHadrCands=thePFTauElementsOperators.PFChargedHadrCandsInAnnulus((*theleadPFChargedHadrCand).momentum(),"DR",test_trackersignalcone_size_,"DR",test_trackerisolcone_size_,test_Cand_minpt_);
      theIsolPFNeutrHadrCands=thePFTauElementsOperators.PFNeutrHadrCandsInAnnulus((*theleadPFChargedHadrCand).momentum(),"DR",test_HCALsignalcone_size_,"DR",test_HCALisolcone_size_,test_Cand_minpt_);
      theIsolPFGammaCands=thePFTauElementsOperators.PFGammaCandsInAnnulus((*theleadPFChargedHadrCand).momentum(),"DR",test_ECALsignalcone_size_,"DR",test_ECALisolcone_size_,test_Cand_minpt_);  
      //
      if ((int)theIsolPFChargedHadrCands.size()==0){
	PFTau_passed_trackisolsel=1;
	if ((int)theSignalPFChargedHadrCands.size()==1 || (int)theSignalPFChargedHadrCands.size()==3)PFTau_passed_tracksel=1;
	else PFTau_passed_tracksel=0;
      }else{
	PFTau_passed_tracksel=0;
	PFTau_passed_trackisolsel=0;
      }
      if ((int)theIsolPFGammaCands.size()==0) PFTau_passed_ECALisolsel=1;
      else PFTau_passed_ECALisolsel=0;
    }else{
      PFTau_passed_tracksel=0;
      PFTau_passed_trackisolsel=0;
      PFTau_passed_ECALisolsel=0;
    }
    PFTau_CHCands_number=ChargedHadrCands_n;
    // END ***
    
    PFTau_GenTau_visproducts_e=-100.;
    PFTau_GenTau_visproducts_et=-100.;
    PFTau_GenTau_visproducts_eta=-100.;
    PFTau_GenTau_visproducts_phi=-100.;
    PFTau_GenTau_visproducts_type=-100;
    PFTau_SimTau_visproducts_e=-100.;
    PFTau_SimTau_visproducts_et=-100.;
    PFTau_SimTau_visproducts_eta=-100.;
    PFTau_SimTau_visproducts_phi=-100.;
    PFTau_SimTau_visproducts_type=-100;
    PFTau_GenJet05_e=-100.;
    PFTau_GenJet05_et=-100.;
    PFTau_GenJet05_eta=-100.;    
    PFTau_GenJet05_phi=-100.;    
    double min1stdeltaR_GenTau_deltaR=100.;
    pair<math::XYZTLorentzVector,int> min1stdeltaR_GenTau_pair;
    double min1stdeltaR_SimTau_deltaR=100.;
    pair<math::XYZTLorentzVector,int> min1stdeltaR_SimTau_pair;
    double min1stdeltaR_GenJet05_deltaR=100.;
    math::XYZTLorentzVector min1stdeltaR_GenJet05_LorentzVect(0.,0.,0.,0.);
    
    if (GenTausnumber>0){
      for (vector<pair<math::XYZTLorentzVector,int> >::iterator iGenTau_pair=GenTau_pair.begin();iGenTau_pair!=GenTau_pair.end();iGenTau_pair++) {
	if (ROOT::Math::VectorUtil::DeltaR((*iGenTau_pair).first,ThePFTauJet_LorentzVect)<min1stdeltaR_GenTau_deltaR) {
	  min1stdeltaR_GenTau_pair=(*iGenTau_pair);
	  min1stdeltaR_GenTau_deltaR=ROOT::Math::VectorUtil::DeltaR((*iGenTau_pair).first,ThePFTauJet_LorentzVect);
	}      
      }
    }
    
    if (SimTausnumber>0){
      for (vector<pair<math::XYZTLorentzVector,int> >::iterator iSimTau_pair=SimTau_pair.begin();iSimTau_pair!=SimTau_pair.end();iSimTau_pair++) {
	if (ROOT::Math::VectorUtil::DeltaR((*iSimTau_pair).first,ThePFTauJet_LorentzVect)<min1stdeltaR_SimTau_deltaR) {
	  min1stdeltaR_SimTau_pair=(*iSimTau_pair);
	  min1stdeltaR_SimTau_deltaR=ROOT::Math::VectorUtil::DeltaR((*iSimTau_pair).first,ThePFTauJet_LorentzVect);
	}      
      }
    }
    
    if (GenJet05snumber>0){
      iGenJet05 = 0;
      for (vector<math::XYZTLorentzVector>::iterator iGenJet05_LorentzVect=GenJet05_LorentzVect.begin();iGenJet05_LorentzVect!=GenJet05_LorentzVect.end();iGenJet05_LorentzVect++) {
	if (iGenJet05>1)break;
	if (ROOT::Math::VectorUtil::DeltaR((*iGenJet05_LorentzVect),ThePFTauJet_LorentzVect)<min1stdeltaR_GenJet05_deltaR) {
	  min1stdeltaR_GenJet05_LorentzVect=(*iGenJet05_LorentzVect);
	  min1stdeltaR_GenJet05_deltaR=ROOT::Math::VectorUtil::DeltaR((*iGenJet05_LorentzVect),ThePFTauJet_LorentzVect);
	} 
	++iGenJet05;     
      }
    }
    
    if (min1stdeltaR_GenTau_deltaR<0.15){
      PFTau_GenTau_visproducts_e=(min1stdeltaR_GenTau_pair).first.E();
      PFTau_GenTau_visproducts_et=(min1stdeltaR_GenTau_pair).first.Et();
      PFTau_GenTau_visproducts_eta=(min1stdeltaR_GenTau_pair).first.Eta();
      PFTau_GenTau_visproducts_phi=(min1stdeltaR_GenTau_pair).first.Phi();
      PFTau_GenTau_visproducts_type=min1stdeltaR_GenTau_pair.second;          
    }
    if (min1stdeltaR_SimTau_deltaR<0.15){
      PFTau_SimTau_visproducts_e=(min1stdeltaR_SimTau_pair).first.E();
      PFTau_SimTau_visproducts_et=(min1stdeltaR_SimTau_pair).first.Et();
      PFTau_SimTau_visproducts_eta=(min1stdeltaR_SimTau_pair).first.Eta();
      PFTau_SimTau_visproducts_phi=(min1stdeltaR_SimTau_pair).first.Phi();
      PFTau_SimTau_visproducts_type=min1stdeltaR_SimTau_pair.second;          
    }
    if (min1stdeltaR_GenJet05_deltaR<.25){
      PFTau_GenJet05_e=min1stdeltaR_GenJet05_LorentzVect.E();
      PFTau_GenJet05_et=min1stdeltaR_GenJet05_LorentzVect.Et();
      PFTau_GenJet05_eta=min1stdeltaR_GenJet05_LorentzVect.Eta(); 
      PFTau_GenJet05_phi=min1stdeltaR_GenJet05_LorentzVect.Phi(); 
    }    
    thePFTauTree->Fill();        
  }
  theEventTree->Fill(); 
}

DEFINE_FWK_MODULE(PFTauAnalyzer);
