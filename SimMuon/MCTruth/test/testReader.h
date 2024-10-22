#ifndef testReader_h
#define testReader_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include <memory>

class testReader : public edm::one::EDAnalyzer<> {
public:
  testReader(const edm::ParameterSet &);
  ~testReader() override = default;
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  const edm::InputTag tracksTag_;
  const edm::InputTag tpTag_;
  const edm::InputTag assoMapsTag_;
  const edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  const edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimToken_;
  const edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoToken_;
};

#endif
