#ifndef TtSemiLepJetCombWMassMaxSumPt_h
#define TtSemiLepJetCombWMassMaxSumPt_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class TtSemiLepJetCombWMassMaxSumPt : public edm::EDProducer {

 public:

  explicit TtSemiLepJetCombWMassMaxSumPt(const edm::ParameterSet&);
  ~TtSemiLepJetCombWMassMaxSumPt() override;

 private:

  void beginJob() override {};
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;
  void endJob() override {};

  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };

  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT< edm::View<reco::RecoCandidate> > lepsToken_;
  int maxNJets_;
  double wMass_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
};

#endif
