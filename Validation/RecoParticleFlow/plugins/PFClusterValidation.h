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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class PFClusterValidation : public DQMEDAnalyzer {
public:
  PFClusterValidation(edm::ParameterSet const& conf);
  ~PFClusterValidation() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  double dR(double eta1, double phi1, double eta2, double phi2);
  double sumEnergy(edm::Handle<reco::PFClusterCollection> pfCluster1);
  std::string outputFile_;
  //std::string mc_;
  bool mc_;

  typedef math::RhoEtaPhiVector Vector;

  edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterECALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHCALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHOTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHFTok_;

  int imc;

  // this acts throught the entire class
  // no reinitialization required
  const double partR = 0.3;  // dr cutoff
  double eta_MC, phi_MC, energy_MC = 9999.;

  //************Modules

  MonitorElement* emean_vs_eta_E;
  MonitorElement* emean_vs_eta_H;
  MonitorElement* emean_vs_eta_EH;

  MonitorElement* emean_vs_eta_HF;
  MonitorElement* emean_vs_eta_HO;
  MonitorElement* emean_vs_eta_EHF;
  MonitorElement* emean_vs_eta_EHFO;

  MonitorElement* Ratio_Esummed_ECAL_0;
  MonitorElement* Ratio_Esummed_HCAL_0;
  MonitorElement* Ratio_Esummed_HO_0;

  MonitorElement* Ratio_Esummed_ECAL_1;
  MonitorElement* Ratio_Esummed_HCAL_1;
  MonitorElement* Ratio_Esummed_HO_1;

  MonitorElement* Ratio_Esummed_ECAL_2;
  MonitorElement* Ratio_Esummed_HCAL_2;
  MonitorElement* Ratio_Esummed_HO_2;

  MonitorElement* Ratio_Esummed_ECAL_3;
  MonitorElement* Ratio_Esummed_HCAL_3;
  MonitorElement* Ratio_Esummed_HO_3;

  MonitorElement* Ratio_Esummed_ECAL_4;
  MonitorElement* Ratio_Esummed_HCAL_4;
  MonitorElement* Ratio_Esummed_HO_4;

  MonitorElement* Ratio_Esummed_HF_5;
  MonitorElement* Ratio_Esummed_HF_6;

  MonitorElement* Ratio_Esummed_ECAL_HCAL_0;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_HO_0;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_1;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_HO_1;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_2;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_HO_2;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_3;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_HO_3;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_4;
  MonitorElement* Ratio_Esummed_ECAL_HCAL_HO_4;

  MonitorElement* Egen_MC;
};

#endif
