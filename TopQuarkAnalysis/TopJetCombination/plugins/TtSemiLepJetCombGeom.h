#ifndef TtSemiLepJetCombGeom_h
#define TtSemiLepJetCombGeom_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class TtSemiLepJetCombGeom : public edm::EDProducer {

 public:
  
  explicit TtSemiLepJetCombGeom(const edm::ParameterSet&);
  ~TtSemiLepJetCombGeom();
  
 private:

  virtual void beginJob() {};
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob() {};

  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);

  edm::InputTag jets_;
  edm::InputTag leps_;
  int maxNJets_;
  bool useDeltaR_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
};

#endif
