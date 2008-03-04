#ifndef PFTauTagVal_h
#define PFTauTagVal_h

// -*- C++ -*-
//
// Package:    PFTauTagVal
// Class:      PFTauTagVal
// 
/**\class PFTauTagVal PFTauTagVal.cc 

 Description: EDAnalyzer to validate the Collections from the ConeIsolation Producer
 It is supposed to be used for Offline Tau Reconstrction, so PrimaryVertex should be used.
 Implementation:

*/
// Original Author:  Simone Gennai
// Modified by Ricardo Vasquez Sierra On August 29, 2007
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HepMC/GenParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/JetReco/interface/PFJet.h"
//#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include "TH1.h"
#include "TH1F.h"
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"


#include "DQMServices/Core/interface/MonitorElement.h"


// class declaration    
class PFTauTagVal : public edm::EDAnalyzer {

public:
  explicit PFTauTagVal(const edm::ParameterSet&);
  ~PFTauTagVal() {}

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();

  //  Function to get the Daughters of a Generated Particle 

private:
  //------------HELPER FUNCTIONS---------------------------

  std::vector<TLorentzVector> getVectorOfVisibleTauJets(HepMC::GenEvent *theEvent);
  //  std::vector<TLorentzVector> getVectorOfGenJets(edm::Handle< reco::GenJetCollection >& genJets );
  std::vector<HepMC::GenParticle*> getGenStableDecayProducts(const HepMC::GenParticle* particle);
std::vector<TLorentzVector> getVectorOfGenJets(edm::Handle< reco::GenJetCollection >& genJets );
  // ----------- MEMBER DATA--------------------------------
  enum tauDecayModes {kElectron, kMuon, 
		      kOneProng0pi0, kOneProng1pi0, kOneProng2pi0,
		      kThreeProng0pi0, kThreeProng1pi0,
		      kOther, kUndefined};

  edm::InputTag ExtensionName_, genJetSrc_;
  std::string outPutFile_;
  std::string dataType_;
  std::string outputhistograms_;
  std::string PFTauProducer_;
  std::string PFTauDiscriminatorByIsolationProducer_;
  std::string tversion;

  // MonteCarlo Taus -- to see what kind of Taus do we originally have!
  MonitorElement* ptTauMC_;
  MonitorElement* etaTauMC_;
  MonitorElement* phiTauMC_;
  MonitorElement* energyTauMC_;
  MonitorElement* hGenTauDecay_DecayModes_;
  MonitorElement* hGenTauDecay_DecayModesChosen_;
  MonitorElement* nMCTaus_ptTauJet_;
  MonitorElement* nMCTaus_etaTauJet_;
  MonitorElement* nMCTaus_phiTauJet_;
  MonitorElement* nMCTaus_energyTauJet_;
 
    // Number of PFTau Candidates matched to MC Taus
  MonitorElement* nPFTauCand_ptTauJet_;      
  MonitorElement* nPFTauCand_etaTauJet_;     
  MonitorElement* nPFTauCand_phiTauJet_;    
  MonitorElement* nPFTauCand_energyTauJet_;
  MonitorElement* nPFTauCand_ChargedHadronsSignal_;	
  MonitorElement* nPFTauCand_ChargedHadronsIsolAnnulus_;
  MonitorElement* nPFTauCand_GammasSignal_;		
  MonitorElement* nPFTauCand_GammasIsolAnnulus_;  	
  MonitorElement* nPFTauCand_NeutralHadronsSignal_;	
  MonitorElement* nPFTauCand_NeutralHadronsIsolAnnulus_;
  
  // Number of PFTau Candidates with a Leading charged hadron in it (within a cone of 0.1 avound the jet axis and a minimum pt of 6 GeV)
  MonitorElement* nPFTau_LeadingChargedHadron_ptTauJet_;   
  MonitorElement* nPFTau_LeadingChargedHadron_etaTauJet_;  
  MonitorElement* nPFTau_LeadingChargedHadron_phiTauJet_;  
  MonitorElement* nPFTau_LeadingChargedHadron_energyTauJet_; 
  MonitorElement* nPFTau_LeadingChargedHadron_ChargedHadronsSignal_;	  
  MonitorElement* nPFTau_LeadingChargedHadron_ChargedHadronsIsolAnnulus_; 
  MonitorElement* nPFTau_LeadingChargedHadron_GammasSignal_;		  
  MonitorElement* nPFTau_LeadingChargedHadron_GammasIsolAnnulus_;
  MonitorElement* nPFTau_LeadingChargedHadron_NeutralHadronsSignal_;	 
  MonitorElement* nPFTau_LeadingChargedHadron_NeutralHadronsIsolAnnulus_;
    
  // Isolated PFTau with a Leading charged hadron with no Charged Hadrons inside the isolation annulus
  MonitorElement* nIsolated_NoChargedHadrons_ptTauJet_;      
  MonitorElement* nIsolated_NoChargedHadrons_etaTauJet_;     
  MonitorElement* nIsolated_NoChargedHadrons_phiTauJet_;     
  MonitorElement* nIsolated_NoChargedHadrons_energyTauJet_;
  MonitorElement* nIsolated_NoChargedHadrons_ChargedHadronsSignal_;	  

  MonitorElement* nIsolated_NoChargedHadrons_GammasSignal_;		  
  MonitorElement* nIsolated_NoChargedHadrons_GammasIsolAnnulus_;         
  MonitorElement* nIsolated_NoChargedHadrons_NeutralHadronsSignal_;	
  MonitorElement* nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_;

  // Isolated PFTau with a Leading charge hadron with no Charged Hadron inside the isolation annulus with no Ecal/Gamma candidates in the isolation annulus
  MonitorElement* nIsolated_NoChargedNoGammas_ptTauJet_;     
  MonitorElement* nIsolated_NoChargedNoGammas_etaTauJet_;     
  MonitorElement* nIsolated_NoChargedNoGammas_phiTauJet_;     
  MonitorElement* nIsolated_NoChargedNoGammas_energyTauJet_;
  MonitorElement* nIsolated_NoChargedNoGammas_ChargedHadronsSignal_;    

  MonitorElement* nIsolated_NoChargedNoGammas_GammasSignal_;
         
  MonitorElement* nIsolated_NoChargedNoGammas_NeutralHadronsSignal_;	 
  MonitorElement* nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_;


  MonitorElement* nChargedHadronsSignalCone_isolated_;
  MonitorElement* nGammasSignalCone_isolated_;
  MonitorElement* nNeutralHadronsSignalCone_isolated_;
  MonitorElement* N_1_ChargedHadronsSignal_;	  
  MonitorElement* N_1_ChargedHadronsIsolAnnulus_; 
  MonitorElement* N_1_GammasSignal_;		  
  MonitorElement* N_1_GammasIsolAnnulus_;          
  MonitorElement* N_1_NeutralHadronsSignal_;	 
  MonitorElement* N_1_NeutralHadronsIsolAnnulus_;
 
  // book-keeping variables

  int numEvents_;

};

#endif











