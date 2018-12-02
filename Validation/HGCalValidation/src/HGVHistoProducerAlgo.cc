#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset) {
  //parameters for eta plots
  minEta  = pset.getParameter<double>("minEta");
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for z positionof vertex plots 
  minZpos  = pset.getParameter<double>("minZpos");
  maxZpos  = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");



}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid) {

  histograms.h_caloparticle_eta[pdgid] = ibook.book1D("num_caloparticle_eta","N of caloparticle vs eta",nintEta,minEta,maxEta);
  histograms.h_caloparticle_eta_Zorigin[pdgid] = ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", nintEta, minEta, maxEta, nintZpos, minZpos, maxZpos);
  
}


void HGVHistoProducerAlgo::bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms) {

  histograms.h_cluster_eta.push_back( ibook.book1D("num_reco_cluster_eta","N of reco clusters vs eta",nintEta,minEta,maxEta) );

}

void HGVHistoProducerAlgo::fill_caloparticle_histos(const Histograms& histograms,
						    int pdgid,
						    const CaloParticle & caloparticle,
						    std::vector<SimVertex> const & simVertices) const {

  const auto eta = getEta(caloparticle.eta());
  if (histograms.h_caloparticle_eta.count(pdgid)){ histograms.h_caloparticle_eta.at(pdgid).fill(eta); }
  if (histograms.h_caloparticle_eta_Zorigin.count(pdgid)){ histograms.h_caloparticle_eta_Zorigin.at(pdgid).fill( simVertices.at(caloparticle.g4Tracks()[0].vertIndex()).position().z(), eta ); }

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

