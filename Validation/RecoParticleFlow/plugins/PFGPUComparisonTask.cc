#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/deltaR.h"
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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

class PFGPUComparisonTask : public DQMEDAnalyzer {
public:
  PFGPUComparisonTask(edm::ParameterSet const& conf);
  ~PFGPUComparisonTask() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterHBHETok_ref_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterHBHETok_target_;

  MonitorElement* energy_GPUvsCPU_;
  //MonitorElement* energy_CPU_;

  // Need MonitorElement* for rechit multiplicity?

};

PFGPUComparisonTask::PFGPUComparisonTask(const edm::ParameterSet& conf) {
  pfClusterHBHETok_ref_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("particleFlowClusterHBHE"));
  pfClusterHBHETok_target_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("particleFlowClusterHBHEonGPU"));
}

PFGPUComparisonTask::~PFGPUComparisonTask() {}

void PFGPUComparisonTask::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& irun,
                                         edm::EventSetup const& isetup) {
  constexpr auto size = 100;
  char histo[size];
  //char histo2[size];

  ibooker.setCurrentFolder("ParticleFlow/PFClusterV");

  strncpy(histo, "energy_GPUvsCPU_", size);
  //strncpy(histo2, "energy_CPU_", size);
  energy_GPUvsCPU_ = ibooker.book2D(histo, histo, 500, 0, 10000, 500, 0, 10000);
}
void PFGPUComparisonTask::analyze(edm::Event const& event, edm::EventSetup const& c) {

  edm::Handle<reco::PFClusterCollection> pfClusterHBHE_ref;
  event.getByToken(pfClusterHBHETok_ref_, pfClusterHBHE_ref);

  edm::Handle<reco::PFClusterCollection> pfClusterHBHE_target;
  event.getByToken(pfClusterHBHETok_ref_, pfClusterHBHE_target);

  double energy_CPU = 0;
  for (auto pf = pfClusterHBHE_ref->begin(); pf != pfClusterHBHE_ref->end(); ++pf) {
      energy_CPU = pf->energy();
  }

  double energy_GPU = 0;
  for (auto pf = pfClusterHBHE_target->begin(); pf != pfClusterHBHE_target->end(); ++pf) {
      energy_GPU = pf->energy();
  }

  energy_GPUvsCPU_->Fill(energy_GPU, energy_CPU);
}
  
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFGPUComparisonTask);
