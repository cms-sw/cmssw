#ifndef Validation_HGCalValidation_HGVHistoProducerAlgo_h
#define Validation_HGCalValidation_HGVHistoProducerAlgo_h

/* \author HGCal
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include <unordered_map>

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

struct HGVHistoProducerAlgoHistograms {
  //1D
  std::vector<ConcurrentMonitorElement>  h_cluster_eta;

  std::unordered_map<int, ConcurrentMonitorElement > h_caloparticle_eta;

};

class HGVHistoProducerAlgo {
 public:
  HGVHistoProducerAlgo(const edm::ParameterSet& pset) ;
  ~HGVHistoProducerAlgo();

  using Histograms = HGVHistoProducerAlgoHistograms;

  void bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid);

  void bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms);

  void fill_caloparticle_histos(const Histograms& histograms,
				int pdgid,
				const CaloParticle & caloparticle) const ;
    
  void fill_cluster_histos(const Histograms& histograms,
			   int count,
			   const reco::CaloCluster & cluster) const;
  
 private:

  double getEta(double eta) const;

  //private data members
  double minEta, maxEta;  int nintEta; bool useFabsEta;

};

#endif
