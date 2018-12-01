#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset) {
  //parameters for _vs_eta plots
  minEta  = pset.getParameter<double>("minEta");
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid) {

  histograms.h_caloparticle_eta[pdgid] = ibook.book1D("num_caloparticle_eta","N of caloparticle vs eta",nintEta,minEta,maxEta);

}


void HGVHistoProducerAlgo::bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms) {

  histograms.h_cluster_eta.push_back( ibook.book1D("num_reco_cluster_eta","N of reco clusters vs eta",nintEta,minEta,maxEta) );

}

void HGVHistoProducerAlgo::fill_caloparticle_histos(const Histograms& histograms,
						    int pdgid,
						    const CaloParticle & caloparticle) const {

  const auto eta = getEta(caloparticle.eta());
  if (histograms.h_caloparticle_eta.count(pdgid)){
    histograms.h_caloparticle_eta.at(pdgid).fill(eta);
  }

}

void HGVHistoProducerAlgo::fill_cluster_histos(const Histograms& histograms,
					       int count,
					       const reco::CaloCluster & cluster) const {

  const auto eta = getEta(cluster.eta());
  histograms.h_cluster_eta[count].fill(eta);

}

double HGVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta) return fabs(eta);
  else return eta;
}

