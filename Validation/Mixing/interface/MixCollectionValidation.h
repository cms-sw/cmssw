#ifndef MixCollectionValidation_H
#define MixCollectionValidation_H

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include <string>

//
// class declaration
//

class MixCollectionValidation : public edm::EDAnalyzer {
public:
  explicit MixCollectionValidation(const edm::ParameterSet&);
  ~MixCollectionValidation();

  void beginJob(edm::EventSetup const&iSetup);
  void endJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  template<class T1, class T2> void fillMultiplicity(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillGenParticleMulti(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillSimHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillCaloHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  std::string outputFile_;
  int minbunch_;
  int maxbunch_;

  bool verbose_;

  MonitorElement * nrHepMCProductH_;
  MonitorElement * nrSimTrackH_;
  MonitorElement * nrSimVertexH_;

  MonitorElement * nrSimHitBSCHitsH_;
  MonitorElement * nrSimHitFP420SIH_;
  MonitorElement * nrSimHitMuonCSCHitsH_; 
  MonitorElement * nrSimHitMuonDTHitsH_; 
  MonitorElement * nrSimHitMuonRPCHitsH_; 
  MonitorElement * nrSimHitTotemHitsRPH_; 
  MonitorElement * nrSimHitTotemHitsT1H_; 
  MonitorElement * nrSimHitTotemHitsT2GemH_; 
  MonitorElement * nrSimHitTrackerHitsPixelBarrelHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsPixelBarrelLowTofH_; 
  MonitorElement * nrSimHitTrackerHitsPixelEndcapHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsPixelEndcapLowTofH_; 
  MonitorElement * nrSimHitTrackerHitsTECHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsTECLowTofH_; 
  MonitorElement * nrSimHitTrackerHitsTIBHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsTIBLowTofH_; 
  MonitorElement * nrSimHitTrackerHitsTIDHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsTIDLowTofH_; 
  MonitorElement * nrSimHitTrackerHitsTOBHighTofH_; 
  MonitorElement * nrSimHitTrackerHitsTOBLowTofH_;

  std::map<std::string,MonitorElement *> SimHitNrmap_;

  MonitorElement * timeSimHitBSCHitsH_;
  MonitorElement * timeSimHitFP420SIH_;
  MonitorElement * timeSimHitMuonCSCHitsH_; 
  MonitorElement * timeSimHitMuonDTHitsH_; 
  MonitorElement * timeSimHitMuonRPCHitsH_; 
  MonitorElement * timeSimHitTotemHitsRPH_; 
  MonitorElement * timeSimHitTotemHitsT1H_; 
  MonitorElement * timeSimHitTotemHitsT2GemH_; 
  MonitorElement * timeSimHitTrackerHitsPixelBarrelHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsPixelBarrelLowTofH_; 
  MonitorElement * timeSimHitTrackerHitsPixelEndcapHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsPixelEndcapLowTofH_; 
  MonitorElement * timeSimHitTrackerHitsTECHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsTECLowTofH_; 
  MonitorElement * timeSimHitTrackerHitsTIBHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsTIBLowTofH_; 
  MonitorElement * timeSimHitTrackerHitsTIDHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsTIDLowTofH_; 
  MonitorElement * timeSimHitTrackerHitsTOBHighTofH_; 
  MonitorElement * timeSimHitTrackerHitsTOBLowTofH_;

  std::map<std::string,MonitorElement *> SimHitTimemap_;

  MonitorElement * nrCaloHitCaloHitsTkH_;
  MonitorElement * nrCaloHitCastorBUH_; 
  MonitorElement * nrCaloHitCastorFIH_;  
  MonitorElement * nrCaloHitCastorPLH_;  
  MonitorElement * nrCaloHitCastorTUH_;  
  MonitorElement * nrCaloHitEcalHitsEBH_;  
  MonitorElement * nrCaloHitEcalHitsEEH_;  
  MonitorElement * nrCaloHitEcalHitsESH_;  
  MonitorElement * nrCaloHitEcalTBH4BeamHitsH_;  
  MonitorElement * nrCaloHitHcalHitsH_;  
  MonitorElement * nrCaloHitHcalTB06BeamHitsH_;  
  MonitorElement * nrCaloHitZDCHITSH_; 

  std::map<std::string,MonitorElement *> CaloHitNrmap_;

  MonitorElement * timeCaloHitCaloHitsTkH_;
  MonitorElement * timeCaloHitCastorBUH_; 
  MonitorElement * timeCaloHitCastorFIH_;  
  MonitorElement * timeCaloHitCastorPLH_;  
  MonitorElement * timeCaloHitCastorTUH_;  
  MonitorElement * timeCaloHitEcalHitsEBH_;  
  MonitorElement * timeCaloHitEcalHitsEEH_;  
  MonitorElement * timeCaloHitEcalHitsESH_;  
  MonitorElement * timeCaloHitEcalTBH4BeamHitsH_;  
  MonitorElement * timeCaloHitHcalHitsH_;  
  MonitorElement * timeCaloHitHcalTB06BeamHitsH_;  
  MonitorElement * timeCaloHitZDCHITSH_; 

  std::map<std::string,MonitorElement *> CaloHitTimemap_;

  DQMStore* dbe_;
  
  std::vector<std::string> names_;
  std::vector<edm::InputTag> HepMCProductTags_;
  std::vector<edm::InputTag> SimTrackTags_;
  std::vector<edm::InputTag> SimVertexTags_;
  std::vector<edm::InputTag> PSimHitTags_;
  std::vector<edm::InputTag> PCaloHitTags_;

  int nbin_;
  
};

#endif
