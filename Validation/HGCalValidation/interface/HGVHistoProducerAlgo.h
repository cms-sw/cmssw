#ifndef Validation_HGCalValidation_HGVHistoProducerAlgo_h
#define Validation_HGCalValidation_HGVHistoProducerAlgo_h

/* \author HGCal
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

#include "DQMServices/Core/interface/DQMStore.h"

struct HGVHistoProducerAlgoHistograms {
  //Info
  //To be able to spot any issues both in -z and +z a layer id was introduced
  //that spans from 0 to 103 for hgcal_v9 geometry. The mapping for hgcal_v9 is:
  //-z: 0->51
  //+z: 52->103
  //We will pick the numbers below from RecHitTools just to avoid future problems
  dqm::reco::MonitorElement* lastLayerEEzm;  // last layer of EE -z
  dqm::reco::MonitorElement* lastLayerFHzm;  // last layer of FH -z
  dqm::reco::MonitorElement* maxlayerzm;     // last layer of BH -z
  dqm::reco::MonitorElement* lastLayerEEzp;  // last layer of EE +z
  dqm::reco::MonitorElement* lastLayerFHzp;  // last layer of FH +z
  dqm::reco::MonitorElement* maxlayerzp;     // last layer of BH +z

  //1D
  std::vector<dqm::reco::MonitorElement*> h_cluster_eta;
  std::vector<dqm::reco::MonitorElement*> h_mixedhitscluster_zminus;
  std::vector<dqm::reco::MonitorElement*> h_mixedhitscluster_zplus;
  std::vector<dqm::reco::MonitorElement*> h_energyclustered_zminus;
  std::vector<dqm::reco::MonitorElement*> h_energyclustered_zplus;
  std::vector<dqm::reco::MonitorElement*> h_longdepthbarycentre_zminus;
  std::vector<dqm::reco::MonitorElement*> h_longdepthbarycentre_zplus;

  std::unordered_map<int, dqm::reco::MonitorElement*> h_clusternum_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_energyclustered_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_score_layercl2caloparticle_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_score_caloparticle2layercl_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_energy_vs_score_caloparticle2layercl_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_energy_vs_score_layercl2caloparticle_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2layercl_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2layercl_vs_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2layercl_vs_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_layercl2caloparticle_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_layercl2caloparticle_vs_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_sharedenergy_layercl2caloparticle_vs_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_num_caloparticle_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_numDup_caloparticle_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_denom_caloparticle_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_num_caloparticle_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_numDup_caloparticle_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_denom_caloparticle_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_num_layercl_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_numMerge_layercl_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_denom_layercl_eta_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_num_layercl_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_numMerge_layercl_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_denom_layercl_phi_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_cellAssociation_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_clusternum_perthick;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_cellsenedens_perthick;

  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_cellsnum_perthickperlayer;
  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_distancetoseedcell_perthickperlayer;
  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_distancetoseedcell_perthickperlayer_eneweighted;
  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_distancetomaxcell_perthickperlayer;
  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_distancetomaxcell_perthickperlayer_eneweighted;
  std::unordered_map<std::string, dqm::reco::MonitorElement*> h_distancebetseedandmaxcell_perthickperlayer;
  std::unordered_map<std::string, dqm::reco::MonitorElement*>
      h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer;

  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_eta;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_eta_Zorigin;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_energy;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_selfenergy;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_energyDifference;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_pt;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_phi;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nSimClusters;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nHitsInSimClusters;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_firstlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_lastlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_layersnum;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nHitsInSimClusters_matchedtoRecHit;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nHits_matched_energy;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nHits_matched_energy_layer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_nHits_matched_energy_layer_1SimCl;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_sum_energy_layer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_firstlayer_matchedtoRecHit;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_lastlayer_matchedtoRecHit;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_caloparticle_layersnum_matchedtoRecHit;

  //For SimClusters
  std::unordered_map<int, dqm::reco::MonitorElement*> h_simclusternum_perlayer;
  std::unordered_map<int, dqm::reco::MonitorElement*> h_simclusternum_perthick;
  dqm::reco::MonitorElement* h_mixedhitssimcluster_zminus;
  dqm::reco::MonitorElement* h_mixedhitssimcluster_zplus;

  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_denom_layercl_in_simcl_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_denom_layercl_in_simcl_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_score_layercl2simcluster_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_layercl2simcluster_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_energy_vs_score_layercl2simcluster_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_num_layercl_in_simcl_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_num_layercl_in_simcl_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_numMerge_layercl_in_simcl_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_numMerge_layercl_in_simcl_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_layercl2simcluster_vs_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_layercl2simcluster_vs_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_denom_simcluster_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_denom_simcluster_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_score_simcluster2layercl_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_simcluster2layercl_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_energy_vs_score_simcluster2layercl_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_num_simcluster_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_num_simcluster_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_numDup_simcluster_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_numDup_simcluster_phi_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_simcluster2layercl_vs_eta_perlayer;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_sharedenergy_simcluster2layercl_vs_phi_perlayer;

  //For Tracksters
  std::vector<dqm::reco::MonitorElement*> h_score_trackster2caloparticle;
  std::vector<dqm::reco::MonitorElement*> h_score_caloparticle2trackster;
  std::vector<dqm::reco::MonitorElement*> h_energy_vs_score_trackster2caloparticle;
  std::vector<dqm::reco::MonitorElement*> h_energy_vs_score_caloparticle2trackster;
  std::vector<dqm::reco::MonitorElement*> h_num_trackster_eta;
  std::vector<dqm::reco::MonitorElement*> h_num_trackster_phi;
  std::vector<dqm::reco::MonitorElement*> h_numMerge_trackster_eta;
  std::vector<dqm::reco::MonitorElement*> h_numMerge_trackster_phi;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_trackster2caloparticle;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2trackster;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2trackster_assoc;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_trackster2caloparticle_vs_eta;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_trackster2caloparticle_vs_phi;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2trackster_vs_eta;
  std::vector<dqm::reco::MonitorElement*> h_sharedenergy_caloparticle2trackster_vs_phi;
  std::vector<dqm::reco::MonitorElement*> h_denom_trackster_eta;
  std::vector<dqm::reco::MonitorElement*> h_denom_trackster_phi;
  std::vector<dqm::reco::MonitorElement*> h_numEff_caloparticle_eta;
  std::vector<dqm::reco::MonitorElement*> h_numEff_caloparticle_phi;
  std::vector<dqm::reco::MonitorElement*> h_num_caloparticle_eta;
  std::vector<dqm::reco::MonitorElement*> h_num_caloparticle_phi;
  std::vector<dqm::reco::MonitorElement*> h_numDup_trackster_eta;
  std::vector<dqm::reco::MonitorElement*> h_numDup_trackster_phi;
  std::vector<dqm::reco::MonitorElement*> h_denom_caloparticle_eta;
  std::vector<dqm::reco::MonitorElement*> h_denom_caloparticle_phi;
  std::vector<dqm::reco::MonitorElement*> h_tracksternum;
  std::vector<dqm::reco::MonitorElement*> h_conttracksternum;
  std::vector<dqm::reco::MonitorElement*> h_nonconttracksternum;
  std::vector<dqm::reco::MonitorElement*> h_clusternum_in_trackster;
  std::vector<std::unordered_map<int, dqm::reco::MonitorElement*>> h_clusternum_in_trackster_perlayer;
  std::vector<dqm::reco::MonitorElement*> h_multiplicityOfLCinTST;
  std::vector<dqm::reco::MonitorElement*> h_multiplicity_numberOfEventsHistogram;
  std::vector<dqm::reco::MonitorElement*> h_multiplicity_zminus_numberOfEventsHistogram;
  std::vector<dqm::reco::MonitorElement*> h_multiplicity_zplus_numberOfEventsHistogram;
  std::vector<dqm::reco::MonitorElement*> h_multiplicityOfLCinTST_vs_layercluster;
  std::vector<dqm::reco::MonitorElement*> h_multiplicityOfLCinTST_vs_layercluster_zminus;
  std::vector<dqm::reco::MonitorElement*> h_multiplicityOfLCinTST_vs_layercluster_zplus;
  std::vector<dqm::reco::MonitorElement*> h_multiplicityOfLCinTST_vs_layerclusterenergy;
  std::vector<dqm::reco::MonitorElement*> h_clusternum_in_trackster_vs_layer;
  std::vector<dqm::reco::MonitorElement*> h_trackster_pt;
  std::vector<dqm::reco::MonitorElement*> h_trackster_eta;
  std::vector<dqm::reco::MonitorElement*> h_trackster_phi;
  std::vector<dqm::reco::MonitorElement*> h_trackster_energy;
  std::vector<dqm::reco::MonitorElement*> h_trackster_x;
  std::vector<dqm::reco::MonitorElement*> h_trackster_y;
  std::vector<dqm::reco::MonitorElement*> h_trackster_z;
  std::vector<dqm::reco::MonitorElement*> h_trackster_firstlayer;
  std::vector<dqm::reco::MonitorElement*> h_trackster_lastlayer;
  std::vector<dqm::reco::MonitorElement*> h_trackster_layersnum;
};

using Density = hgcal_clustering::Density;

class HGVHistoProducerAlgo {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  HGVHistoProducerAlgo(const edm::ParameterSet& pset);
  ~HGVHistoProducerAlgo();

  using Histograms = HGVHistoProducerAlgoHistograms;

  void bookInfo(DQMStore::IBooker& ibook, Histograms& histograms);
  void bookCaloParticleHistos(DQMStore::IBooker& ibook, Histograms& histograms, int pdgid, unsigned int layers);

  void bookSimClusterHistos(DQMStore::IBooker& ibook,
                            Histograms& histograms,
                            unsigned int layers,
                            std::vector<int> thicknesses);

  void bookSimClusterAssociationHistos(DQMStore::IBooker& ibook,
                                       Histograms& histograms,
                                       unsigned int layers,
                                       std::vector<int> thicknesses);

  void bookClusterHistos_ClusterLevel(DQMStore::IBooker& ibook,
                                      Histograms& histograms,
                                      unsigned int layers,
                                      std::vector<int> thicknesses,
                                      std::string pathtomatbudfile);

  void bookClusterHistos_LCtoCP_association(DQMStore::IBooker& ibook, Histograms& histograms, unsigned int layers);

  void bookClusterHistos_CellLevel(DQMStore::IBooker& ibook,
                                   Histograms& histograms,
                                   unsigned int layers,
                                   std::vector<int> thicknesses);

  void bookTracksterHistos(DQMStore::IBooker& ibook, Histograms& histograms, unsigned int layers);

  void layerClusters_to_CaloParticles(const Histograms& histograms,
                                      edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                      const reco::CaloClusterCollection& clusters,
                                      edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
                                      std::vector<CaloParticle> const& cP,
                                      std::vector<size_t> const& cPIndices,
                                      std::vector<size_t> const& cPSelectedIndices,
                                      std::unordered_map<DetId, const HGCRecHit*> const&,
                                      unsigned int layers,
                                      const hgcal::RecoToSimCollection& recSimColl,
                                      const hgcal::SimToRecoCollection& simRecColl) const;
  void layerClusters_to_SimClusters(const Histograms& histograms,
                                    int count,
                                    edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                    const reco::CaloClusterCollection& clusters,
                                    edm::Handle<std::vector<SimCluster>> simClusterHandle,
                                    std::vector<SimCluster> const& simClusters,
                                    std::vector<size_t> const& sCIndices,
                                    const std::vector<float>& mask,
                                    std::unordered_map<DetId, const HGCRecHit*> const&,
                                    unsigned int layers,
                                    const hgcal::RecoToSimCollectionWithSimClusters& recSimColl,
                                    const hgcal::SimToRecoCollectionWithSimClusters& simRecColl) const;
  void tracksters_to_CaloParticles(const Histograms& histograms,
                                   int count,
                                   const ticl::TracksterCollection& Tracksters,
                                   const reco::CaloClusterCollection& layerClusters,
                                   std::vector<CaloParticle> const& cP,
                                   std::vector<size_t> const& cPIndices,
                                   std::vector<size_t> const& cPSelectedIndices,
                                   std::unordered_map<DetId, const HGCRecHit*> const&,
                                   unsigned int layers) const;
  void fill_info_histos(const Histograms& histograms, unsigned int layers) const;
  void fill_caloparticle_histos(const Histograms& histograms,
                                int pdgid,
                                const CaloParticle& caloparticle,
                                std::vector<SimVertex> const& simVertices,
                                unsigned int layers,
                                std::unordered_map<DetId, const HGCRecHit*> const&) const;
  void fill_generic_cluster_histos(const Histograms& histograms,
                                   int count,
                                   edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                   const reco::CaloClusterCollection& clusters,
                                   const Density& densities,
                                   edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
                                   std::vector<CaloParticle> const& cP,
                                   std::vector<size_t> const& cPIndices,
                                   std::vector<size_t> const& cPSelectedIndices,
                                   std::unordered_map<DetId, const HGCRecHit*> const&,
                                   std::map<double, double> cummatbudg,
                                   unsigned int layers,
                                   std::vector<int> thicknesses,
                                   const hgcal::RecoToSimCollection& recSimColl,
                                   const hgcal::SimToRecoCollection& simRecColl) const;
  void fill_simCluster_histos(const Histograms& histograms,
                              std::vector<SimCluster> const& simClusters,
                              unsigned int layers,
                              std::vector<int> thicknesses) const;
  void fill_simClusterAssociation_histos(const Histograms& histograms,
                                         int count,
                                         edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                         const reco::CaloClusterCollection& clusters,
                                         edm::Handle<std::vector<SimCluster>> simClusterHandle,
                                         std::vector<SimCluster> const& simClusters,
                                         std::vector<size_t> const& sCIndices,
                                         const std::vector<float>& mask,
                                         std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
                                         unsigned int layers,
                                         const hgcal::RecoToSimCollectionWithSimClusters& recSimColl,
                                         const hgcal::SimToRecoCollectionWithSimClusters& simRecColl) const;
  void fill_cluster_histos(const Histograms& histograms, int count, const reco::CaloCluster& cluster) const;
  void fill_trackster_histos(const Histograms& histograms,
                             int count,
                             const ticl::TracksterCollection& Tracksters,
                             const reco::CaloClusterCollection& layerClusters,
                             std::vector<CaloParticle> const& cP,
                             std::vector<size_t> const& cPIndices,
                             std::vector<size_t> const& cPSelectedIndices,
                             std::unordered_map<DetId, const HGCRecHit*> const&,
                             unsigned int layers) const;
  double distance2(const double x1, const double y1, const double x2, const double y2) const;
  double distance(const double x1, const double y1, const double x2, const double y2) const;

  void setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools);

  DetId findmaxhit(const reco::CaloCluster& cluster, std::unordered_map<DetId, const HGCRecHit*> const&) const;

  struct detIdInfoInCluster {
    bool operator==(const detIdInfoInCluster& o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
  };

  struct detIdInfoInTrackster {
    bool operator==(const detIdInfoInTrackster& o) const { return tracksterId == o.tracksterId; };
    unsigned int tracksterId;
    long unsigned int clusterId;
    float fraction;
  };

  struct caloParticleOnLayer {
    unsigned int caloParticleId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

private:
  double getEta(double eta) const;

  std::shared_ptr<hgcal::RecHitTools> recHitTools_;

  //private data members
  double minEta_, maxEta_;
  int nintEta_;
  bool useFabsEta_;
  double minEne_, maxEne_;
  int nintEne_;
  double minPt_, maxPt_;
  int nintPt_;
  double minPhi_, maxPhi_;
  int nintPhi_;
  double minMixedHitsSimCluster_, maxMixedHitsSimCluster_;
  int nintMixedHitsSimCluster_;
  double minMixedHitsCluster_, maxMixedHitsCluster_;
  int nintMixedHitsCluster_;
  double minEneCl_, maxEneCl_;
  int nintEneCl_;
  double minLongDepBary_, maxLongDepBary_;
  int nintLongDepBary_;
  double minZpos_, maxZpos_;
  int nintZpos_;
  double minTotNsimClsperlay_, maxTotNsimClsperlay_;
  int nintTotNsimClsperlay_;
  double minTotNClsperlay_, maxTotNClsperlay_;
  int nintTotNClsperlay_;
  double minEneClperlay_, maxEneClperlay_;
  int nintEneClperlay_;
  double minScore_, maxScore_;
  int nintScore_;
  double minSharedEneFrac_, maxSharedEneFrac_;
  int nintSharedEneFrac_;
  double minTSTSharedEneFracEfficiency_;
  double minTSTSharedEneFrac_, maxTSTSharedEneFrac_;
  int nintTSTSharedEneFrac_;
  double minTotNsimClsperthick_, maxTotNsimClsperthick_;
  int nintTotNsimClsperthick_;
  double minTotNClsperthick_, maxTotNClsperthick_;
  int nintTotNClsperthick_;
  double minTotNcellsperthickperlayer_, maxTotNcellsperthickperlayer_;
  int nintTotNcellsperthickperlayer_;
  double minDisToSeedperthickperlayer_, maxDisToSeedperthickperlayer_;
  int nintDisToSeedperthickperlayer_;
  double minDisToSeedperthickperlayerenewei_, maxDisToSeedperthickperlayerenewei_;
  int nintDisToSeedperthickperlayerenewei_;
  double minDisToMaxperthickperlayer_, maxDisToMaxperthickperlayer_;
  int nintDisToMaxperthickperlayer_;
  double minDisToMaxperthickperlayerenewei_, maxDisToMaxperthickperlayerenewei_;
  int nintDisToMaxperthickperlayerenewei_;
  double minDisSeedToMaxperthickperlayer_, maxDisSeedToMaxperthickperlayer_;
  int nintDisSeedToMaxperthickperlayer_;
  double minClEneperthickperlayer_, maxClEneperthickperlayer_;
  int nintClEneperthickperlayer_;
  double minCellsEneDensperthick_, maxCellsEneDensperthick_;
  int nintCellsEneDensperthick_;
  double minTotNTSTs_, maxTotNTSTs_;
  int nintTotNTSTs_;
  double minTotNClsinTSTs_, maxTotNClsinTSTs_;
  int nintTotNClsinTSTs_;
  double minTotNClsinTSTsperlayer_, maxTotNClsinTSTsperlayer_;
  int nintTotNClsinTSTsperlayer_;
  double minMplofLCs_, maxMplofLCs_;
  int nintMplofLCs_;
  double minSizeCLsinTSTs_, maxSizeCLsinTSTs_;
  int nintSizeCLsinTSTs_;
  double minClEnepermultiplicity_, maxClEnepermultiplicity_;
  int nintClEnepermultiplicity_;
  double minX_, maxX_;
  int nintX_;
  double minY_, maxY_;
  int nintY_;
  double minZ_, maxZ_;
  int nintZ_;
};

#endif
