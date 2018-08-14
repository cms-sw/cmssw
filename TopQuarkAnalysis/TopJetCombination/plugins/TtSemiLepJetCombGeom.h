#ifndef TtSemiLepJetCombGeom_h
#define TtSemiLepJetCombGeom_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class TtSemiLepJetCombGeom : public edm::EDProducer {

 public:

  explicit TtSemiLepJetCombGeom(const edm::ParameterSet&);
  ~TtSemiLepJetCombGeom() override;

 private:

  void beginJob() override {};
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;
  void endJob() override {};

  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);

  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT< edm::View<reco::RecoCandidate> > lepsToken_;
  int maxNJets_;
  bool useDeltaR_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
};

#endif
