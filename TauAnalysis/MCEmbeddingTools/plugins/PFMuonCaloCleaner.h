#ifndef TauAnalysis_MCEmbeddingTools_PFMuonCaloCleaner_h
#define TauAnalysis_MCEmbeddingTools_PFMuonCaloCleaner_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/View.h"

#include <string>
#include <vector>

class PFMuonCaloCleaner : public edm::EDProducer
{
 public:
  explicit PFMuonCaloCleaner(const edm::ParameterSet&);
  ~PFMuonCaloCleaner() {}
    
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef std::map<uint32_t, float> detIdToFloatMap;
  void fillEnergyDepositMap(const reco::Muon*, const edm::View<reco::PFCandidate>&, detIdToFloatMap&);

  edm::InputTag srcSelectedMuons_;  
  edm::InputTag srcPFCandidates_;  
  double dRmatch_;

  int verbosity_;
};

#endif   
