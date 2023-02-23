#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/deltaR.h"
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

class PFRecHitCreatorGPUComparisonTask : public DQMEDAnalyzer {
public:
  PFRecHitCreatorGPUComparisonTask(edm::ParameterSet const& conf);
  ~PFRecHitCreatorGPUComparisonTask() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
  edm::EDGetTokenT<reco::PFRecHitCollection> pfRecHitsTokenCPU, pfRecHitsTokenGPU;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTokenCPU, pfClusterTokenGPU;

  MonitorElement* pfCluster_Multiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Energy_GPUvsCPU_;
  MonitorElement* pfCluster_RecHitMultiplicity_GPUvsCPU_;

  std::string pfCaloGPUCompDir;
};

PFRecHitCreatorGPUComparisonTask::PFRecHitCreatorGPUComparisonTask(const edm::ParameterSet& conf)
    : recHitsToken(
          consumes<edm::SortedCollection<HBHERecHit>>(conf.getUntrackedParameter<edm::InputTag>("recHitsSourceCPU"))),
      pfRecHitsTokenCPU(
          consumes<reco::PFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceCPU"))),
      pfRecHitsTokenGPU(
          consumes<reco::PFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceGPU"))),
      pfClusterTokenCPU(
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterSourceCPU"))),
      pfClusterTokenGPU(
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterSourceGPU"))),
      pfCaloGPUCompDir(conf.getUntrackedParameter<std::string>("pfCaloGPUCompDir")) {}

PFRecHitCreatorGPUComparisonTask::~PFRecHitCreatorGPUComparisonTask() {}

void PFRecHitCreatorGPUComparisonTask::bookHistograms(DQMStore::IBooker& ibooker,
                                                      edm::Run const& irun,
                                                      edm::EventSetup const& isetup) {
  constexpr auto size = 100;
  char histo[size];
  ibooker.setCurrentFolder("ParticleFlow/" + pfCaloGPUCompDir);

  strncpy(histo, "pfCluster_Multiplicity_GPUvsCPU_", size);
  pfCluster_Multiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);

  strncpy(histo, "pfCluster_Energy_GPUvsCPU_", size);
  pfCluster_Energy_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 500, 100, 0, 500);

  strncpy(histo, "pfCluster_RecHitMultiplicity_GPUvsCPU_", size);
  pfCluster_RecHitMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);
}

void PFRecHitCreatorGPUComparisonTask::analyze(edm::Event const& event, edm::EventSetup const& c) {
  // Rec Hits
  edm::Handle<edm::SortedCollection<HBHERecHit>> recHits;
  event.getByToken(recHitsToken, recHits);
  printf("Found %zd recHits\n", recHits->size());
  fprintf(stderr, "Found %zd recHits\n", recHits->size());
  for (size_t i = 0; i < recHits->size(); i++)
    printf("recHit %4lu %u\n", i, recHits->operator[](i).id().rawId());

  // PF Rec Hits
  // paste <(grep "^CPU" validation.log | sort -nk3) <(grep "^GPU" validation.log | sort -nk3) | awk '$3!=$13 || $4!=$14 || $5!=$15 || $6!=$16 || $9!=$19 {print}' | head
  edm::Handle<reco::PFRecHitCollection> pfRecHitsCPU, pfRecHitsGPU;
  event.getByToken(pfRecHitsTokenCPU, pfRecHitsCPU);
  event.getByToken(pfRecHitsTokenGPU, pfRecHitsGPU);
  printf("Found %zd/%zd pfRecHits on CPU/GPU\n", pfRecHitsCPU->size(), pfRecHitsGPU->size());
  fprintf(stderr, "Found %zd/%zd pfRecHits on CPU/GPU\n", pfRecHitsCPU->size(), pfRecHitsGPU->size());
  for (size_t i = 0; i < pfRecHitsCPU->size(); i++)
    printf("CPU %4lu %u %d %d %u : %f %f (%f,%f,%f)\n",
           i,
           pfRecHitsCPU->at(i).detId(),
           pfRecHitsCPU->at(i).depth(),
           pfRecHitsCPU->at(i).layer(),
           pfRecHitsCPU->at(i).neighbours().size(),
           pfRecHitsCPU->at(i).time(),
           pfRecHitsCPU->at(i).energy(),
           0.,  //pfRecHitsCPU->at(i).position().x(),
           0.,  //pfRecHitsCPU->at(i).position().y(),
           0.   //pfRecHitsCPU->at(i).position().z()
    );
  for (size_t i = 0; i < pfRecHitsGPU->size(); i++)
    printf("GPU %4lu %u %d %d %u : %f %f (%f,%f,%f)\n",
           i,
           pfRecHitsGPU->at(i).detId(),
           pfRecHitsGPU->at(i).depth(),
           pfRecHitsGPU->at(i).layer(),
           pfRecHitsGPU->at(i).neighbours().size(),
           pfRecHitsGPU->at(i).time(),
           pfRecHitsGPU->at(i).energy(),
           0.,  //pfRecHitsGPU->at(i).position().x(),
           0.,  //pfRecHitsGPU->at(i).position().y(),
           0.   //pfRecHitsGPU->at(i).position().z()
    );

  static int cnt = 0;
  if (++cnt >= 1)
    exit(1);

  // PF Clusters
  edm::Handle<reco::PFClusterCollection> pfClustersCPU, pfClustersGPU;
  event.getByToken(pfClusterTokenCPU, pfClustersCPU);
  event.getByToken(pfClusterTokenGPU, pfClustersGPU);

  // Compare per-event PF cluster multiplicity
  if (pfClustersCPU->size() != pfClustersGPU->size())
    LOGVERB("PFRecHitCreatorGPUComparisonTask")
        << " PFCluster multiplicity " << pfClustersCPU->size() << " " << pfClustersGPU->size();
  pfCluster_Multiplicity_GPUvsCPU_->Fill((float)pfClustersCPU->size(), (float)pfClustersGPU->size());

  // Find matching PF cluster pairs
  std::vector<int> matched_idx;
  for (unsigned i = 0; i < pfClustersCPU->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < pfClustersGPU->size(); ++j) {
      if (pfClustersCPU->at(i).seed() == pfClustersGPU->at(j).seed()) {
        if (!matched) {
          matched = true;
          matched_idx.push_back((int)j);
        } else {
          LOGWARN("PFRecHitCreatorGPUComparisonTask") << " another matching? ";
        }
      }
    }
    if (!matched)
      matched_idx.push_back(-1);  // if you don't find a match, put a dummy number
  }

  // Check matches
  std::vector<int> tmp = matched_idx;
  sort(tmp.begin(), tmp.end());
  const bool hasDuplicates = std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end();
  if (hasDuplicates)
    LOGWARN("PFRecHitCreatorGPUComparisonTask") << "find duplicated matched";

  // Plot matching PF cluster variables
  for (unsigned i = 0; i < pfClustersCPU->size(); ++i) {
    if (matched_idx[i] >= 0) {
      unsigned int j = matched_idx[i];
      int ref_energy_bin = pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClustersCPU->at(i).energy());
      int target_energy_bin = pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClustersGPU->at(j).energy());
      if (ref_energy_bin != target_energy_bin)
        std::cout << "Off-diagonal energy bin entries: " << pfClustersCPU->at(i).energy() << " "
                  << pfClustersCPU->at(i).eta() << " " << pfClustersCPU->at(i).phi() << " "
                  << pfClustersGPU->at(j).energy() << " " << pfClustersGPU->at(j).eta() << " "
                  << pfClustersGPU->at(j).phi() << std::endl;
      pfCluster_Energy_GPUvsCPU_->Fill(pfClustersCPU->at(i).energy(), pfClustersGPU->at(j).energy());
      pfCluster_RecHitMultiplicity_GPUvsCPU_->Fill((float)pfClustersCPU->at(i).recHitFractions().size(),
                                                   (float)pfClustersGPU->at(j).recHitFractions().size());
    }
  }
}

// void PFRecHitCreatorGPUComparisonTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   edm::ParameterSetDescription desc;
//   //desc.setUnknown();
//   desc.add<edm::InputTag>("pfClusterToken_ref", edm::InputTag("particleFlowClusterHBHE"));
//   desc.add<edm::InputTag>("pfClusterToken_target", edm::InputTag("particleFlowClusterHBHEonGPU"));
//   desc.addUntracked<std::string>("pfCaloGPUCompDir", "pfClusterHBHEGPUv");
//   descriptions.addDefault(desc);
// }

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitCreatorGPUComparisonTask);
