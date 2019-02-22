#ifndef Validation_HGCalValidation_HGVHistoProducerAlgo_h
#define Validation_HGCalValidation_HGVHistoProducerAlgo_h

/* \author HGCal
 */

#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include <unordered_map>

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

struct HGVHistoProducerAlgoHistograms {
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

class HGVHistoProducerAlgo {
 public:
  HGVHistoProducerAlgo(const edm::ParameterSet& pset) ;
  ~HGVHistoProducerAlgo();

  using Histograms = HGVHistoProducerAlgoHistograms;

  void bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid);

  void bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,unsigned layers, std::vector<int> thicknesses);
  void layerClusters_to_CaloParticles(const Histograms& histograms,
      const reco::CaloClusterCollection &clusters,
      std::vector<CaloParticle> const & cP,
      std::map<DetId, const HGCRecHit*> const &,
      unsigned layers) const ;
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

//  void fillHitMap(const HGCRecHitCollection & HGCEERecHits,
//		  const HGCRecHitCollection & HGCFHRecHits,
//		  const HGCRecHitCollection & HGCBHRecHits);

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

//  std::map<DetId, const HGCRecHit *> * hitMap_;

  //private data members
  double minEta, maxEta;  int nintEta; bool useFabsEta;
  double minCaloEne, maxCaloEne;  int nintCaloEne;
  double minCaloPt, maxCaloPt;  int nintCaloPt;
  double minCaloPhi, maxCaloPhi;  int nintCaloPhi;
  double minMixedHitsCluster, maxMixedHitsCluster;  int nintMixedHitsCluster;
  double minEneCl, maxEneCl;  int nintEneCl;
  double minLongDepBary, maxLongDepBary;  int nintLongDepBary;
  double minZpos, maxZpos; int nintZpos;
  double minTotNClsperlay, maxTotNClsperlay; int nintTotNClsperlay;
  double minEneClperlay, maxEneClperlay; int nintEneClperlay;
  double minTotNClsperthick, maxTotNClsperthick; int nintTotNClsperthick;
  double minTotNcellsperthickperlayer, maxTotNcellsperthickperlayer; int nintTotNcellsperthickperlayer;
  double minDisToSeedperthickperlayer, maxDisToSeedperthickperlayer; int nintDisToSeedperthickperlayer;
  double minDisToSeedperthickperlayerenewei, maxDisToSeedperthickperlayerenewei; int nintDisToSeedperthickperlayerenewei;
  double minDisToMaxperthickperlayer, maxDisToMaxperthickperlayer; int nintDisToMaxperthickperlayer;
  double minDisToMaxperthickperlayerenewei, maxDisToMaxperthickperlayerenewei; int nintDisToMaxperthickperlayerenewei;
  double minDisSeedToMaxperthickperlayer, maxDisSeedToMaxperthickperlayer; int nintDisSeedToMaxperthickperlayer;
  double minClEneperthickperlayer, maxClEneperthickperlayer; int nintClEneperthickperlayer;
  double minCellsEneDensperthick, maxCellsEneDensperthick; int nintCellsEneDensperthick;
};

#endif
