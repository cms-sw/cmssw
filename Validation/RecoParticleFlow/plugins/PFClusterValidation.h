#ifndef PFClusterValidation_h
#define PFClusterValidation_h
 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// including PFCluster 
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

// end include PFCluster
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class PFClusterValidation : public DQMEDAnalyzer {
 public:
  PFClusterValidation(edm::ParameterSet const& conf);
  ~PFClusterValidation() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  double dR(double eta1, double phi1, double eta2, double phi2);
  double sumEnergy(edm::Handle<reco::PFClusterCollection> pfCluster1);
  std::string outputFile_;
  //  std::string hcalselector_;
  std::string mc_;
  bool        useAllHistos_;

  typedef math::RhoEtaPhiVector Vector;

  //  edm::EDGetTokenT<std::vector<reco::PFCluster> > PFClusterTok_;
  edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterECALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHCALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHOTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHFTok_;

  int isub;
  int nevent;

  int imc;
  
  // test function scope
  double partR, eta_MC, phi_MC;

  // eta limits to calcualte MET, SET (not to include HF if not needed)
  double etaMax[3];
  double etaMin[3];


  //************Modules

  
  // ieta scan
  MonitorElement*  emean_vs_eta_E;
  MonitorElement*  emean_vs_eta_H;
  MonitorElement*  emean_vs_eta_EH;

  MonitorElement*  emean_vs_eta_HF;
  MonitorElement*  emean_vs_eta_HO;
  MonitorElement*  emean_vs_eta_EHF;
  MonitorElement*  emean_vs_eta_EHFO;
  // MonitorElement*  emean_vs_eta_H1;
  //MonitorElement*  emean_vs_eta_EH1;
  


};

#endif
