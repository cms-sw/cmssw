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
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

class PFCaloGPUComparisonTask : public DQMEDAnalyzer {
public:
  PFCaloGPUComparisonTask(edm::ParameterSet const& conf);
  ~PFCaloGPUComparisonTask() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_ref_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_target_;

  MonitorElement* pfCluster_Multiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Energy_GPUvsCPU_;
  MonitorElement* pfCluster_RecHitMultiplicity_GPUvsCPU_;

  std::string pfCaloGPUCompDir;

};

PFCaloGPUComparisonTask::PFCaloGPUComparisonTask(const edm::ParameterSet& conf) {
  pfClusterTok_ref_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_ref"));
  pfClusterTok_target_ =
      consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_target"));
  pfCaloGPUCompDir = conf.getUntrackedParameter<std::string>("pfCaloGPUCompDir");
}

PFCaloGPUComparisonTask::~PFCaloGPUComparisonTask() {}

void PFCaloGPUComparisonTask::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& irun,
                                         edm::EventSetup const& isetup) {
  constexpr auto size = 100;
  char histo[size];
  ibooker.setCurrentFolder("ParticleFlow/"+pfCaloGPUCompDir);

  strncpy(histo, "pfCluster_Multiplicity_GPUvsCPU_", size);
  pfCluster_Multiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);

  strncpy(histo, "pfCluster_Energy_GPUvsCPU_", size);
  pfCluster_Energy_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 500, 100, 0, 500);

  strncpy(histo, "pfCluster_RecHitMultiplicity_GPUvsCPU_", size);
  pfCluster_RecHitMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

}
void PFCaloGPUComparisonTask::analyze(edm::Event const& event, edm::EventSetup const& c) {

  edm::Handle<reco::PFClusterCollection> pfClusters_ref;
  event.getByToken(pfClusterTok_ref_, pfClusters_ref);

  edm::Handle<reco::PFClusterCollection> pfClusters_target;
  event.getByToken(pfClusterTok_target_, pfClusters_target);

  //
  // Compare per-event PF cluster multiplicity
  
  if (pfClusters_ref->size()!=pfClusters_target->size()) 
    LOGVERB("PFCaloGPUComparisonTask") << " PFCluster multiplicity " <<  pfClusters_ref->size() << " " <<  pfClusters_target->size();
  pfCluster_Multiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->size(),(float)pfClusters_target->size());

  //
  // Find matching PF cluster pairs
  std::vector<int> matched_idx;
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    bool matched=false;
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
      if (pfClusters_ref->at(i).seed() == pfClusters_target->at(j).seed()){
	if (!matched){
	  matched=true;
	  matched_idx.push_back((int)j);
	} else {
	  LOGWARN("PFCaloGPUComparisonTask") << " another matching? ";
	}
      }
    }
    if (!matched) matched_idx.push_back(-1); // if you don't find a match, put a dummy number
  }

  //
  // Check matches
  std::vector<int> tmp = matched_idx;
  sort(tmp.begin(), tmp.end());
  const bool hasDuplicates = std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end();  
  if (hasDuplicates) LOGWARN("PFCaloGPUComparisonTask") << "find duplicated matched";
  
  // 
  // Plot matching PF cluster variables
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    if (matched_idx[i]>=0){
      unsigned int j = matched_idx[i];
      /*
      if (pfClusters_ref->at(i).energy() != pfClusters_target->at(j).energy())
	std::cout << pfClusters_ref->at(i).energy() << " " << pfClusters_ref->at(i).eta() << " " << pfClusters_ref->at(i).phi() << " "
		  << pfClusters_target->at(j).energy() << " " << pfClusters_target->at(j).eta() << " " << pfClusters_target->at(j).phi() << std::endl;
      */
      pfCluster_Energy_GPUvsCPU_->Fill(pfClusters_ref->at(i).energy(),
				       pfClusters_target->at(j).energy());
      pfCluster_RecHitMultiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->at(i).recHitFractions().size(),
						   (float)pfClusters_target->at(j).recHitFractions().size());
    }
  }

}
void PFCaloGPUComparisonTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //desc.setUnknown();
  desc.add<edm::InputTag>("pfClusterToken_ref", edm::InputTag("particleFlowClusterHBHE"));
  desc.add<edm::InputTag>("pfClusterToken_target", edm::InputTag("particleFlowClusterHBHEonGPU"));
  desc.addUntracked<std::string>("", "pfClusterHBHEGPUv");
  descriptions.addDefault(desc);
}
  
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCaloGPUComparisonTask);
