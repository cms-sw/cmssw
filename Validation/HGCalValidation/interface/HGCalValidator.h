#ifndef HGCalValidator_h
#define HGCalValidator_h

/** \class HGCalValidator
 *  Class that produces histograms to validate HGCal Reconstruction performances
 *
 *  \author HGCal
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "Validation/HGCalValidation/interface/CaloParticleSelector.h"

class PileupSummaryInfo;

struct HGCalValidatorHistograms {
  HGVHistoProducerAlgoHistograms histoProducerAlgo;
  std::vector<ConcurrentMonitorElement> h_layerclusters_coll;
};

class HGCalValidator : public DQMGlobalEDAnalyzer<HGCalValidatorHistograms> {
 public:
  using Histograms = HGCalValidatorHistograms;

  /// Constructor
  HGCalValidator(const edm::ParameterSet& pset);
  
  /// Destructor
  ~HGCalValidator() override;


  /// Method called once per event
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const Histograms& ) const override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::ConcurrentBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  
  void cpParametersAndSelection(const Histograms& histograms, std::vector<CaloParticle> const & cPeff, std::vector<SimVertex> const & simVertices, std::vector<size_t>& selected_cPeff) const; 


 protected:

  std::vector<edm::InputTag> label;
  const bool doCaloParticlePlots_;
  const bool dolayerclustersPlots_;

  std::vector<edm::EDGetTokenT<reco::CaloClusterCollection> > labelToken;
  edm::EDGetTokenT<std::vector<CaloParticle> > label_cp_effic;
  edm::EDGetTokenT<std::vector<CaloParticle> > label_cp_fake;
  edm::EDGetTokenT<std::vector<SimVertex> > simVertices_;
  std::unique_ptr<HGVHistoProducerAlgo> histoProducerAlgo_;



 private:

  CaloParticleSelector cpSelector; 
   std::vector<int> particles_to_monitor_;
  std::string dirName_;

};


#endif
