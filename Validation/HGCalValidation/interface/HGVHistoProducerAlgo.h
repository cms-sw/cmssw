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

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "DQMServices/Core/interface/DQMStore.h"

struct HGVHistoProducerAlgoHistograms {
  //Info
  //To be able to spot any issues both in -z and +z a layer id was introduced 
  //that spans from 0 to 103 for hgcal_v9 geometry. The mapping for hgcal_v9 is: 
  //-z: 0->51
  //+z: 52->103
  //We will pick the numbers below from RecHitTools just to avoid future problems
  ConcurrentMonitorElement lastLayerEEzm; // last layer of EE -z
  ConcurrentMonitorElement lastLayerFHzm; // last layer of FH -z
  ConcurrentMonitorElement maxlayerzm; // last layer of BH -z
  ConcurrentMonitorElement lastLayerEEzp; // last layer of EE +z
  ConcurrentMonitorElement lastLayerFHzp; // last layer of FH +z
  ConcurrentMonitorElement maxlayerzp; // last layer of BH +z

  //1D
  std::vector<ConcurrentMonitorElement>  h_cluster_eta;
  std::vector<ConcurrentMonitorElement>  h_mixedhitscluster_zminus;
  std::vector<ConcurrentMonitorElement>  h_mixedhitscluster_zplus;
  std::vector<ConcurrentMonitorElement>  h_energyclustered_zminus;
  std::vector<ConcurrentMonitorElement>  h_energyclustered_zplus;
  std::vector<ConcurrentMonitorElement>  h_longdepthbarycentre_zminus;
  std::vector<ConcurrentMonitorElement>  h_longdepthbarycentre_zplus;

  std::unordered_map<int, ConcurrentMonitorElement > h_clusternum_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_energyclustered_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_score_layercl2caloparticle_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_score_caloparticle2layercl_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_energy_vs_score_caloparticle2layercl_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_energy_vs_score_layercl2caloparticle_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_caloparticle2layercl_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_caloparticle2layercl_vs_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_caloparticle2layercl_vs_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_layercl2caloparticle_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_layercl2caloparticle_vs_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_sharedenergy_layercl2caloparticle_vs_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_num_caloparticle_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_numDup_caloparticle_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_denom_caloparticle_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_num_caloparticle_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_numDup_caloparticle_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_denom_caloparticle_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_num_layercl_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_numMerge_layercl_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_denom_layercl_eta_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_num_layercl_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_numMerge_layercl_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_denom_layercl_phi_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_cellAssociation_perlayer;
  std::unordered_map<int, ConcurrentMonitorElement > h_clusternum_perthick;
  std::unordered_map<int, ConcurrentMonitorElement > h_cellsenedens_perthick;

  std::unordered_map< std::string, ConcurrentMonitorElement > h_cellsnum_perthickperlayer;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancetoseedcell_perthickperlayer;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancetoseedcell_perthickperlayer_eneweighted;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancetomaxcell_perthickperlayer;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancetomaxcell_perthickperlayer_eneweighted;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancebetseedandmaxcell_perthickperlayer;
  std::unordered_map< std::string, ConcurrentMonitorElement > h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer;

  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_eta;
  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_eta_Zorigin;
  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_energy;
  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_pt;
  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_phi;

};

using Density = hgcal_clustering::Density;

class HGVHistoProducerAlgo {
 public:
  HGVHistoProducerAlgo(const edm::ParameterSet& pset) ;
  ~HGVHistoProducerAlgo();

  using Histograms = HGVHistoProducerAlgoHistograms;

  void bookInfo(DQMStore::ConcurrentBooker& ibook, Histograms& histograms);
  void bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid);

  void bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,unsigned layers,
			 std::vector<int> thicknesses, std::string pathtomatbudfile);
  void layerClusters_to_CaloParticles(const Histograms& histograms,
      const reco::CaloClusterCollection &clusters,
      std::vector<CaloParticle> const & cP,
      std::map<DetId, const HGCRecHit*> const &,
      unsigned layers) const ;
  void fill_info_histos(const Histograms& histograms, unsigned layers) const;
  void fill_caloparticle_histos(const Histograms& histograms,
				int pdgid,
				const CaloParticle & caloparticle,
				std::vector<SimVertex> const & simVertices) const ;

  void fill_cluster_histos(const Histograms& histograms,
			   int count,
			   const reco::CaloCluster & cluster) const;

  void fill_generic_cluster_histos(const Histograms& histograms,
				   int count,
				   const reco::CaloClusterCollection &clusters,
				   const Density &densities,
				   std::vector<CaloParticle> const & cP,
                                   std::map<DetId, const HGCRecHit*> const &,
				   std::map<double, double> cummatbudg,
				   unsigned layers,
				   std::vector<int> thicknesses) const ;

  double distance2(const double x1, const double y1, const double x2, const double y2) const;
  double distance(const double x1, const double y1, const double x2, const double y2) const;

  void setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools );

  DetId findmaxhit(const reco::CaloCluster & cluster,
		   std::map<DetId, const HGCRecHit*> const &) const;

  struct detIdInfoInCluster
  {
    bool operator==(const detIdInfoInCluster& o) const { return clusterId == o.clusterId;};
    unsigned int clusterId;
    float fraction;
  };

  struct caloParticleOnLayer
  {
    unsigned int caloParticleId;
    float energy=0;
    std::vector<std::pair<DetId, float> > hits_and_fractions;
    std::unordered_map<int, std::pair<float,float>> layerClusterIdToEnergyAndScore;
  };
 private:

  double getEta(double eta) const;

  std::shared_ptr<hgcal::RecHitTools> recHitTools_;

  //private data members
  double minEta_, maxEta_;  int nintEta_; bool useFabsEta_;
  double minEne_, maxEne_;  int nintEne_;
  double minPt_, maxPt_;  int nintPt_;
  double minPhi_, maxPhi_;  int nintPhi_;
  double minMixedHitsCluster_, maxMixedHitsCluster_;  int nintMixedHitsCluster_;
  double minEneCl_, maxEneCl_;  int nintEneCl_;
  double minLongDepBary_, maxLongDepBary_;  int nintLongDepBary_;
  double minZpos_, maxZpos_; int nintZpos_;
  double minTotNClsperlay_, maxTotNClsperlay_; int nintTotNClsperlay_;
  double minEneClperlay_, maxEneClperlay_; int nintEneClperlay_;
  double minScore_, maxScore_; int nintScore_;
  double minSharedEneFrac_,maxSharedEneFrac_; int nintSharedEneFrac_;
  double minTotNClsperthick_, maxTotNClsperthick_; int nintTotNClsperthick_;
  double minTotNcellsperthickperlayer_, maxTotNcellsperthickperlayer_; int nintTotNcellsperthickperlayer_;
  double minDisToSeedperthickperlayer_, maxDisToSeedperthickperlayer_; int nintDisToSeedperthickperlayer_;
  double minDisToSeedperthickperlayerenewei_, maxDisToSeedperthickperlayerenewei_; int nintDisToSeedperthickperlayerenewei_;
  double minDisToMaxperthickperlayer_, maxDisToMaxperthickperlayer_; int nintDisToMaxperthickperlayer_;
  double minDisToMaxperthickperlayerenewei_, maxDisToMaxperthickperlayerenewei_; int nintDisToMaxperthickperlayerenewei_;
  double minDisSeedToMaxperthickperlayer_, maxDisSeedToMaxperthickperlayer_; int nintDisSeedToMaxperthickperlayer_;
  double minClEneperthickperlayer_, maxClEneperthickperlayer_; int nintClEneperthickperlayer_;
  double minCellsEneDensperthick_, maxCellsEneDensperthick_; int nintCellsEneDensperthick_;

};

#endif
