#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrackingParticleRefMuonProducer : public edm::stream::EDProducer<> {
public:
  TrackingParticleRefMuonProducer(const edm::ParameterSet &iConfig);

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  std::string skim_;
  double ptmin_;
  double pmin_;
};

TrackingParticleRefMuonProducer::TrackingParticleRefMuonProducer(const edm::ParameterSet &iConfig)
    : tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      skim_(iConfig.getParameter<std::string>("skim")),
      ptmin_(iConfig.getParameter<double>("ptmin")),
      pmin_(iConfig.getParameter<double>("pmin")) {
  edm::LogVerbatim("TrackingParticleRefMuonProducer")
      << "\n constructing TrackingParticleRefMuonProducer: skim = " << skim_;
  if (skim_ == "mu")
    edm::LogVerbatim("TrackingParticleRefMuonProducer") << "\t ptmin = " << ptmin_ << ", pmin = " << pmin_ << "\n";
  else if (skim_ == "track")
    edm::LogVerbatim("TrackingParticleRefMuonProducer") << "\t ptmin = " << ptmin_ << "\n";
  else if (skim_ == "pf")
    edm::LogVerbatim("TrackingParticleRefMuonProducer") << "\t ptmin = " << ptmin_ << ", pmin = " << pmin_ << "\n";
  else
    edm::LogError("TrackingParticleRefMuonProducer") << "\t undefined skim = " << skim_ << "\n";

  produces<TrackingParticleRefVector>();
}

void TrackingParticleRefMuonProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<TrackingParticleCollection> tpH;
  iEvent.getByToken(tpToken_, tpH);

  auto tpskim = std::make_unique<TrackingParticleRefVector>();

  if (skim_ == "mu") {
    for (size_t i = 0, end = tpH->size(); i < end; ++i) {
      auto tp = TrackingParticleRef(tpH, i);

      // test if the TP is a muon with pt and p above minimum thresholds
      bool isMu = (std::abs(tp->pdgId()) == 13);
      bool ptpOk = (tp->pt() > ptmin_) && (tp->p() > pmin_);
      if (isMu && ptpOk)
        tpskim->push_back(tp);
      else {
        // test if the TP has muon hits
        int n_muon_hits = tp->numberOfHits() - tp->numberOfTrackerHits();
        if (n_muon_hits > 0)
          tpskim->push_back(tp);
      }
    }
  } else if (skim_ == "track") {
    for (size_t i = 0, end = tpH->size(); i < end; ++i) {
      auto tp = TrackingParticleRef(tpH, i);

      // test if the pt is above a minimum cut
      if (tp->pt() > ptmin_)
        tpskim->push_back(tp);
    }
  } else if (skim_ == "pf") {
    for (size_t i = 0, end = tpH->size(); i < end; ++i) {
      auto tp = TrackingParticleRef(tpH, i);

      // test if p and pt are above minimum cuts
      if ((tp->pt() > ptmin_) && (tp->p() > pmin_))
        tpskim->push_back(tp);
    }
  }

  iEvent.put(std::move(tpskim));
}

DEFINE_FWK_MODULE(TrackingParticleRefMuonProducer);
