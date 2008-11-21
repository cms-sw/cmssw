#ifndef CaloTauTagVal_h
#define CaloTauTagVal_h

// -*- C++ -*-
//
// Package:    CaloTauTagVal
// Class:      CaloTauTagVal
// 
/**\class CaloTauTagVal CaloTauTagVal.cc 

 Description: EDAnalyzer to validate the Collections from the ConeIsolation Producer
 It is supposed to be used for Offline Tau Reconstrction, so PrimaryVertex should be used.
 Implementation:

*/
// Original Author: Ricardo Vasquez Sierra  on Feb 28, 2007
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
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
//#include "DataFormats/TauReco/interface/CaloTauDiscriminatorAgainstElectron.h"
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
class CaloTauTagVal : public edm::EDAnalyzer {

public:
  explicit CaloTauTagVal(const edm::ParameterSet&);
  ~CaloTauTagVal() {}

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
  std::string CaloTauProducer_;
  std::string CaloTauDiscriminatorByIsolationProducer_;
  std::string CaloTauDiscriminatorAgainstElectronProducer_;
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
 
    // Number of CaloJets matched to MC Taus
  MonitorElement* nCaloJet_ptTauJet_;      
  MonitorElement* nCaloJet_etaTauJet_;     
  MonitorElement* nCaloJet_phiTauJet_;    
  MonitorElement* nCaloJet_energyTauJet_;

  MonitorElement* nCaloJet_Tracks_;
  MonitorElement* nCaloJet_isolationECALhitsEtSum_;
  
  // Number of CaloJets with a Leading Track (within a cone of 0.1 around the jet axis and a minimum pt of 5. GeV)
  MonitorElement* nCaloJet_LeadingTrack_ptTauJet_;   
  MonitorElement* nCaloJet_LeadingTrack_etaTauJet_;  
  MonitorElement* nCaloJet_LeadingTrack_phiTauJet_;  
  MonitorElement* nCaloJet_LeadingTrack_energyTauJet_; 

  MonitorElement* nCaloJet_LeadingTrack_signalTracksInvariantMass_;	  
  MonitorElement* nCaloJet_LeadingTrack_signalTracks_;
  MonitorElement* nCaloJet_LeadingTrack_isolationTracks_;
  MonitorElement* nCaloJet_LeadingTrack_isolationECALhitsEtSum_;              
    
  // Track Isolated CaloTau with a Leading Track
  MonitorElement* nTrackIsolated_ptTauJet_;      
  MonitorElement* nTrackIsolated_etaTauJet_;     
  MonitorElement* nTrackIsolated_phiTauJet_;     
  MonitorElement* nTrackIsolated_energyTauJet_;

  MonitorElement* nTrackIsolated_isolationECALhitsEtSum_; 
  MonitorElement* nTrackIsolated_signalTracksInvariantMass_;
  MonitorElement* nTrackIsolated_signalTracks_;             	          

  // EM Isolated CaloTau with a Leading with no tracks in the Isolation Annulus
  MonitorElement* nEMIsolated_ptTauJet_;     
  MonitorElement* nEMIsolated_etaTauJet_;     
  MonitorElement* nEMIsolated_phiTauJet_;     
  MonitorElement* nEMIsolated_energyTauJet_;

  MonitorElement* nEMIsolated_signalTracksInvariantMass_;    
  MonitorElement* nEMIsolated_signalTracks_;                          
         
  // book-keeping variables

  int numEvents_;

};

#endif
