#ifndef HGCalValidator_h
#define HGCalValidator_h

/** \class HGCalValidator
 *  Class that produces histograms to validate HGCal Reconstruction performances
 *
 *  \author HGCal
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "Validation/HGCalValidation/interface/CaloParticleSelector.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

class PileupSummaryInfo;

struct HGCalValidatorHistograms {
  HGVHistoProducerAlgoHistograms histoProducerAlgo;
  std::vector<dqm::reco::MonitorElement*> h_layerclusters_coll;
};

class HGCalValidator : public DQMGlobalEDAnalyzer<HGCalValidatorHistograms> {
public:
  using Histograms = HGCalValidatorHistograms;

  /// Constructor
  HGCalValidator(const edm::ParameterSet& pset);

  /// Destructor
  ~HGCalValidator() override;

  /// Method called once per event
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const Histograms&) const override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;

  void cpParametersAndSelection(const Histograms& histograms,
                                std::vector<CaloParticle> const& cPeff,
                                std::vector<SimVertex> const& simVertices,
                                std::vector<size_t>& selected_cPeff,
                                unsigned int layers,
                                std::unordered_map<DetId, const HGCRecHit*> const&) const;

protected:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  edm::InputTag label_lcl;
  std::vector<edm::InputTag> label_tst;
  edm::InputTag label_simTS, label_simTSFromCP;
  edm::InputTag associator_;
  edm::InputTag associatorSim_;
  const bool SaveGeneralInfo_;
  const bool doCaloParticlePlots_;
  const bool doCaloParticleSelection_;
  const bool doSimClustersPlots_;
  edm::InputTag label_SimClustersPlots_, label_SimClustersLevel_;
  const bool doLayerClustersPlots_;
  edm::InputTag label_layerClustersPlots_, label_LCToCPLinking_;
  const bool doTrackstersPlots_;
  std::string label_TS_, label_TSToCPLinking_, label_TSToSTSPR_;
  std::vector<edm::InputTag> label_clustersmask;
  const edm::FileInPath cummatbudinxo_;

  std::vector<edm::EDGetTokenT<reco::CaloClusterCollection>> labelToken;
  edm::EDGetTokenT<std::vector<SimCluster>> simClusters_;
  edm::EDGetTokenT<reco::CaloClusterCollection> layerclusters_;
  std::vector<edm::EDGetTokenT<ticl::TracksterCollection>> label_tstTokens;
  edm::EDGetTokenT<ticl::TracksterCollection> simTracksters_;
  edm::EDGetTokenT<ticl::TracksterCollection> simTracksters_fromCPs_;
  edm::EDGetTokenT<std::map<uint, std::vector<uint>>> simTrackstersMap_;
  edm::EDGetTokenT<std::vector<CaloParticle>> label_cp_effic;
  edm::EDGetTokenT<std::vector<CaloParticle>> label_cp_fake;
  edm::EDGetTokenT<std::vector<SimVertex>> simVertices_;
  std::vector<edm::EDGetTokenT<std::vector<float>>> clustersMaskTokens_;
  edm::EDGetTokenT<std::unordered_map<DetId, const HGCRecHit*>> hitMap_;
  edm::EDGetTokenT<hgcal::RecoToSimCollection> associatorMapRtS;
  edm::EDGetTokenT<hgcal::SimToRecoCollection> associatorMapStR;
  edm::EDGetTokenT<hgcal::SimToRecoCollectionWithSimClusters> associatorMapSimtR;
  edm::EDGetTokenT<hgcal::RecoToSimCollectionWithSimClusters> associatorMapRtSim;
  std::unique_ptr<HGVHistoProducerAlgo> histoProducerAlgo_;

private:
  CaloParticleSelector cpSelector;
  std::shared_ptr<hgcal::RecHitTools> tools_;
  std::map<double, double> cummatbudg;
  std::vector<int> particles_to_monitor_;
  unsigned totallayers_to_monitor_;
  std::vector<int> thicknesses_to_monitor_;
  std::string dirName_;
};

#endif
