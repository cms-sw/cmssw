#ifndef BarrelValidator_h
#define BarrelValidator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "Validation/HGCalValidation/interface/TICLCandidateValidator.h"
#include "Validation/HGCalValidation/interface/BarrelVHistoProducerAlgo.h"
#include "Validation/HGCalValidation/interface/CaloParticleSelector.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Common/interface/MultiSpan.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfoFwd.h"

struct BarrelValidatorHistograms {
  BarrelVHistoProducerAlgoHistograms histoProducerAlgo;
  std::vector<dqm::reco::MonitorElement*> h_layerclusters_coll;
};

class BarrelValidator : public DQMGlobalEDAnalyzer<BarrelValidatorHistograms> {
public:
  using Histograms = BarrelValidatorHistograms;
  using TracksterToTracksterMap =
      ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>;
  using SimClusterToCaloParticleMap =
      ticl::AssociationMap<ticl::oneToOneMapWithFraction, std::vector<SimCluster>, std::vector<CaloParticle>>;

  /// Constructor
  BarrelValidator(const edm::ParameterSet& pset);

  /// Destructor
  ~BarrelValidator() override;

  /// Method called once per event
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const Histograms&) const override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;

  void cpParametersAndSelection(const Histograms& histograms,
                                std::vector<CaloParticle> const& cPeff,
                                std::vector<SimVertex> const& simVertices,
                                std::vector<size_t>& selected_cPeff,
                                unsigned int layers,
                                std::unordered_map<DetId, const unsigned int> const&,
                                edm::MultiSpan<reco::PFRecHit> const& barrelHits) const;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  edm::InputTag label_lcl;
  std::vector<edm::InputTag> associator_;
  std::vector<edm::InputTag> associatorSim_;
  const bool SaveGeneralInfo_;
  const bool doCaloParticlePlots_;
  const bool doCaloParticleSelection_;
  const bool doSimClustersPlots_;
  std::string label_SimClustersPlots_, label_SimClustersLevel_;
  const bool doLayerClustersPlots_;
  std::string label_layerClustersPlots_, label_LCToCPLinking_;
  std::vector<edm::InputTag> label_clustersmask;

  std::vector<edm::EDGetTokenT<reco::CaloClusterCollection>> labelToken;
  edm::EDGetTokenT<std::vector<SimCluster>> simClusters_;
  edm::EDGetTokenT<reco::CaloClusterCollection> layerclusters_;
  edm::EDGetTokenT<std::vector<CaloParticle>> label_cp_effic;
  edm::EDGetTokenT<std::vector<CaloParticle>> label_cp_fake;
  edm::EDGetTokenT<std::vector<SimVertex>> simVertices_;
  std::vector<edm::EDGetTokenT<std::vector<float>>> clustersMaskTokens_;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> barrelHitMap_;
  std::vector<edm::EDGetTokenT<ticl::RecoToSimCollectionT<reco::CaloClusterCollection>>> associatorMapRtS;
  std::vector<edm::EDGetTokenT<ticl::SimToRecoCollectionT<reco::CaloClusterCollection>>> associatorMapStR;
  std::vector<edm::EDGetTokenT<ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>>>
      associatorMapSimtR;
  std::vector<edm::EDGetTokenT<ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>>>
      associatorMapRtSim;
  std::unique_ptr<BarrelVHistoProducerAlgo> histoProducerAlgo_;
  edm::EDGetTokenT<edm::RefProdVector<reco::PFRecHitCollection>> hitsToken_;
  edm::EDGetTokenT<SimClusterToCaloParticleMap> scToCpMapToken_;

private:
  CaloParticleSelector cpSelector;
  std::shared_ptr<hgcal::RecHitTools> tools_;
  std::vector<int> particles_to_monitor_;
  unsigned totallayers_to_monitor_;
  std::string dirName_;
};

#endif
