#ifndef Validation_RecoParticleFlow_plugins_PFClusterValidation_h
#define Validation_RecoParticleFlow_plugins_PFClusterValidation_h

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class PFClusterValidation : public DQMEDAnalyzer {
public:
  PFClusterValidation(edm::ParameterSet const& conf);
  ~PFClusterValidation() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  static constexpr double partR2 = 0.3 * 0.3;  // dr cutoff (squared)
  static double sumEnergy(edm::Handle<reco::PFClusterCollection> const& pfClusters, double eta, double phi);

  edm::EDGetTokenT<edm::HepMCProduct> hepMCTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterECALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterHCALTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterHOTok_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterHFTok_;

  MonitorElement* emean_vs_eta_E_;
  MonitorElement* emean_vs_eta_H_;
  MonitorElement* emean_vs_eta_EH_;

  MonitorElement* emean_vs_eta_HF_;
  MonitorElement* emean_vs_eta_HO_;
  MonitorElement* emean_vs_eta_EHF_;
  MonitorElement* emean_vs_eta_EHFO_;

  MonitorElement* ratio_Esummed_ECAL_0_;
  MonitorElement* ratio_Esummed_HCAL_0_;
  MonitorElement* ratio_Esummed_HO_0_;

  MonitorElement* ratio_Esummed_ECAL_1_;
  MonitorElement* ratio_Esummed_HCAL_1_;
  MonitorElement* ratio_Esummed_HO_1_;

  MonitorElement* ratio_Esummed_ECAL_2_;
  MonitorElement* ratio_Esummed_HCAL_2_;
  MonitorElement* ratio_Esummed_HO_2_;

  MonitorElement* ratio_Esummed_ECAL_3_;
  MonitorElement* ratio_Esummed_HCAL_3_;
  MonitorElement* ratio_Esummed_HO_3_;

  MonitorElement* ratio_Esummed_ECAL_4_;
  MonitorElement* ratio_Esummed_HCAL_4_;
  MonitorElement* ratio_Esummed_HO_4_;

  MonitorElement* ratio_Esummed_HF_5_;
  MonitorElement* ratio_Esummed_HF_6_;

  MonitorElement* ratio_Esummed_ECAL_HCAL_0_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_HO_0_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_1_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_HO_1_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_2_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_HO_2_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_3_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_HO_3_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_4_;
  MonitorElement* ratio_Esummed_ECAL_HCAL_HO_4_;

  MonitorElement* egen_MC_;
};

#endif  // Validation_RecoParticleFlow_plugins_PFClusterValidation_h
