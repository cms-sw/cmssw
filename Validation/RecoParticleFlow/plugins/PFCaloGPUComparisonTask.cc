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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#else
#define LOGVERB(x) LogTrace(x)
#endif

class PFCaloGPUComparisonTask : public DQMEDAnalyzer {
public:
  PFCaloGPUComparisonTask(edm::ParameterSet const& conf);
  ~PFCaloGPUComparisonTask() override = default;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_ref_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_target_;

  MonitorElement* pfCluster_Multiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Energy_GPUvsCPU_;
  MonitorElement* pfCluster_RecHitMultiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Layer_GPUvsCPU_;
  MonitorElement* pfCluster_Depth_GPUvsCPU_;
  MonitorElement* pfCluster_Eta_GPUvsCPU_;
  MonitorElement* pfCluster_Phi_GPUvsCPU_;
  MonitorElement* pfCluster_DuplicateMatches_GPUvsCPU_;

  std::string pfCaloGPUCompDir_;
};

PFCaloGPUComparisonTask::PFCaloGPUComparisonTask(const edm::ParameterSet& conf)
    : pfClusterTok_ref_{consumes<reco::PFClusterCollection>(
          conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_ref"))},
      pfClusterTok_target_{
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_target"))},
      pfCaloGPUCompDir_{conf.getUntrackedParameter<std::string>("pfCaloGPUCompDir")} {}

void PFCaloGPUComparisonTask::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& irun,
                                             edm::EventSetup const& isetup) {
  const char* histo;

  ibooker.setCurrentFolder("ParticleFlow/" + pfCaloGPUCompDir_);

  histo = "pfCluster_Multiplicity_GPUvsCPU";
  pfCluster_Multiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);

  histo = "pfCluster_Energy_GPUvsCPU";
  pfCluster_Energy_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 500, 100, 0, 500);

  histo = "pfCluster_RecHitMultiplicity_GPUvsCPU";
  pfCluster_RecHitMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Layer_GPUvsCPU";
  pfCluster_Layer_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Depth_GPUvsCPU";
  pfCluster_Depth_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Eta_GPUvsCPU";
  pfCluster_Eta_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Phi_GPUvsCPU";
  pfCluster_Phi_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_DuplicateMatches_GPUvsCPU";
  pfCluster_DuplicateMatches_GPUvsCPU_ = ibooker.book1D(histo, histo, 100, 0., 1000);
}
void PFCaloGPUComparisonTask::analyze(edm::Event const& event, edm::EventSetup const& c) {
  edm::Handle<reco::PFClusterCollection> pfClusters_ref;
  event.getByToken(pfClusterTok_ref_, pfClusters_ref);

  edm::Handle<reco::PFClusterCollection> pfClusters_target;
  event.getByToken(pfClusterTok_target_, pfClusters_target);

  //
  // Compare per-event PF cluster multiplicity

  if (pfClusters_ref->size() != pfClusters_target->size())
    LOGVERB("PFCaloGPUComparisonTask") << " PFCluster multiplicity " << pfClusters_ref->size() << " "
                                       << pfClusters_target->size();
  pfCluster_Multiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->size(), (float)pfClusters_target->size());

  //
  // Find matching PF cluster pairs
  std::vector<int> matched_idx;
  matched_idx.reserve(pfClusters_ref->size());
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
      if (pfClusters_ref->at(i).seed() == pfClusters_target->at(j).seed()) {
        if (!matched) {
          matched = true;
          matched_idx.push_back((int)j);
        } else {
          edm::LogWarning("PFCaloGPUComparisonTask") << "Found duplicate match";
          pfCluster_DuplicateMatches_GPUvsCPU_->Fill((int)j);
        }
      }
    }
    if (!matched)
      matched_idx.push_back(-1);  // if you don't find a match, put a dummy number
  }

  //
  // Plot matching PF cluster variables
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    if (matched_idx[i] >= 0) {
      unsigned int j = matched_idx[i];
      int ref_energy_bin = pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClusters_ref->at(i).energy());
      int target_energy_bin =
          pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClusters_target->at(j).energy());
      if (ref_energy_bin != target_energy_bin)
        edm::LogPrint("PFCaloGPUComparisonTask")
            << "Off-diagonal energy bin entries: " << pfClusters_ref->at(i).energy() << " "
            << pfClusters_ref->at(i).eta() << " " << pfClusters_ref->at(i).phi() << " "
            << pfClusters_target->at(j).energy() << " " << pfClusters_target->at(j).eta() << " "
            << pfClusters_target->at(j).phi() << std::endl;
      pfCluster_Energy_GPUvsCPU_->Fill(pfClusters_ref->at(i).energy(), pfClusters_target->at(j).energy());
      pfCluster_Layer_GPUvsCPU_->Fill(pfClusters_ref->at(i).layer(), pfClusters_target->at(j).layer());
      pfCluster_Eta_GPUvsCPU_->Fill(pfClusters_ref->at(i).eta(), pfClusters_target->at(j).eta());
      pfCluster_Phi_GPUvsCPU_->Fill(pfClusters_ref->at(i).phi(), pfClusters_target->at(j).phi());
      pfCluster_Depth_GPUvsCPU_->Fill(pfClusters_ref->at(i).depth(), pfClusters_target->at(j).depth());
      pfCluster_RecHitMultiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->at(i).recHitFractions().size(),
                                                   (float)pfClusters_target->at(j).recHitFractions().size());
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCaloGPUComparisonTask);
