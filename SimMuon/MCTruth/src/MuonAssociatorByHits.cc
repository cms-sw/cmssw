#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"
#include <sstream>

using namespace reco;
using namespace std;
using namespace muonAssociatorByHitsDiagnostics;

namespace muonAssociatorByHitsDiagnostics {
  using TrackHitsCollection = MuonAssociatorByHitsHelper::TrackHitsCollection;

  class InputDumper {
  public:
    InputDumper(const edm::ParameterSet &conf)
        : simtracksTag(conf.getParameter<edm::InputTag>("simtracksTag")),
          simtracksXFTag(conf.getParameter<edm::InputTag>("simtracksXFTag")),
          crossingframe(conf.getParameter<bool>("crossingframe")) {}

    InputDumper(const edm::ParameterSet &conf, edm::ConsumesCollector &&iC) : InputDumper(conf) {
      if (crossingframe) {
        iC.consumes<CrossingFrame<SimTrack>>(simtracksXFTag);
        iC.consumes<CrossingFrame<SimVertex>>(simtracksXFTag);
      } else {
        iC.consumes<edm::SimTrackContainer>(simtracksTag);
        iC.consumes<edm::SimVertexContainer>(simtracksTag);
      }
    }

    void dump(const TrackHitsCollection &, const TrackingParticleCollection &, const edm::Event &) const;

  private:
    edm::InputTag const simtracksTag;
    edm::InputTag const simtracksXFTag;
    bool const crossingframe;
  };

  void InputDumper::dump(const TrackHitsCollection &tC,
                         const TrackingParticleCollection &tPC,
                         const edm::Event &event) const {
    // reco::Track collection
    edm::LogVerbatim("MuonAssociatorByHits") << "\n"
                                             << "reco::Track collection --- size = " << tC.size();

    // TrackingParticle collection
    edm::LogVerbatim("MuonAssociatorByHits") << "\n"
                                             << "TrackingParticle collection --- size = " << tPC.size();
    int j = 0;
    for (TrackingParticleCollection::const_iterator ITER = tPC.begin(); ITER != tPC.end(); ITER++, j++) {
      edm::LogVerbatim("MuonAssociatorByHits")
          << "TrackingParticle " << j << ", q = " << ITER->charge() << ", p = " << ITER->p() << ", pT = " << ITER->pt()
          << ", eta = " << ITER->eta() << ", phi = " << ITER->phi();

      edm::LogVerbatim("MuonAssociatorByHits")
          << "\t pdg code = " << ITER->pdgId() << ", made of " << ITER->numberOfHits() << " PSimHit"
          << " (in " << ITER->numberOfTrackerLayers() << " layers)"
          << " from " << ITER->g4Tracks().size() << " SimTrack:";
      for (TrackingParticle::g4t_iterator g4T = ITER->g4Track_begin(); g4T != ITER->g4Track_end(); g4T++) {
        edm::LogVerbatim("MuonAssociatorByHits") << "\t\t Id:" << g4T->trackId() << "/Evt:(" << g4T->eventId().event()
                                                 << "," << g4T->eventId().bunchCrossing() << ")";
      }
    }

    // SimTrack collection
    edm::Handle<CrossingFrame<SimTrack>> cf_simtracks;
    edm::Handle<edm::SimTrackContainer> simTrackCollection;

    // SimVertex collection
    edm::Handle<CrossingFrame<SimVertex>> cf_simvertices;
    edm::Handle<edm::SimVertexContainer> simVertexCollection;

    if (crossingframe) {
      event.getByLabel(simtracksXFTag, cf_simtracks);
      unique_ptr<MixCollection<SimTrack>> SimTk(new MixCollection<SimTrack>(cf_simtracks.product()));
      edm::LogVerbatim("MuonAssociatorByHits")
          << "\n"
          << "CrossingFrame<SimTrack> collection with InputTag = " << simtracksXFTag << " has size = " << SimTk->size();
      int k = 0;
      for (MixCollection<SimTrack>::MixItr ITER = SimTk->begin(); ITER != SimTk->end(); ITER++, k++) {
        edm::LogVerbatim("MuonAssociatorByHits")
            << "SimTrack " << k << " - Id:" << ITER->trackId() << "/Evt:(" << ITER->eventId().event() << ","
            << ITER->eventId().bunchCrossing() << ")"
            << " pdgId = " << ITER->type() << ", q = " << ITER->charge() << ", p = " << ITER->momentum().P()
            << ", pT = " << ITER->momentum().Pt() << ", eta = " << ITER->momentum().Eta()
            << ", phi = " << ITER->momentum().Phi() << "\n * " << *ITER << endl;
      }
      event.getByLabel(simtracksXFTag, cf_simvertices);
      unique_ptr<MixCollection<SimVertex>> SimVtx(new MixCollection<SimVertex>(cf_simvertices.product()));
      edm::LogVerbatim("MuonAssociatorByHits")
          << "\n"
          << "CrossingFrame<SimVertex> collection with InputTag = " << simtracksXFTag
          << " has size = " << SimVtx->size();
      int kv = 0;
      for (MixCollection<SimVertex>::MixItr VITER = SimVtx->begin(); VITER != SimVtx->end(); VITER++, kv++) {
        edm::LogVerbatim("MuonAssociatorByHits") << "SimVertex " << kv << " : " << *VITER << endl;
      }
    } else {
      event.getByLabel(simtracksTag, simTrackCollection);
      const edm::SimTrackContainer simTC = *(simTrackCollection.product());
      edm::LogVerbatim("MuonAssociatorByHits")
          << "\n"
          << "SimTrack collection with InputTag = " << simtracksTag << " has size = " << simTC.size() << endl;
      int k = 0;
      for (edm::SimTrackContainer::const_iterator ITER = simTC.begin(); ITER != simTC.end(); ITER++, k++) {
        edm::LogVerbatim("MuonAssociatorByHits")
            << "SimTrack " << k << " - Id:" << ITER->trackId() << "/Evt:(" << ITER->eventId().event() << ","
            << ITER->eventId().bunchCrossing() << ")"
            << " pdgId = " << ITER->type() << ", q = " << ITER->charge() << ", p = " << ITER->momentum().P()
            << ", pT = " << ITER->momentum().Pt() << ", eta = " << ITER->momentum().Eta()
            << ", phi = " << ITER->momentum().Phi() << "\n * " << *ITER << endl;
      }
      event.getByLabel(simtracksTag, simVertexCollection);
      const edm::SimVertexContainer simVC = *(simVertexCollection.product());
      edm::LogVerbatim("MuonAssociatorByHits") << "\n"
                                               << "SimVertex collection with InputTag = "
                                               << "g4SimHits"
                                               << " has size = " << simVC.size() << endl;
      int kv = 0;
      for (edm::SimVertexContainer::const_iterator VITER = simVC.begin(); VITER != simVC.end(); VITER++, kv++) {
        edm::LogVerbatim("MuonAssociatorByHits") << "SimVertex " << kv << " : " << *VITER << endl;
      }
    }
  }

}  // namespace muonAssociatorByHitsDiagnostics

MuonAssociatorByHits::MuonAssociatorByHits(const edm::ParameterSet &conf, edm::ConsumesCollector &&iC)
    : helper_(conf), conf_(conf), trackerHitAssociatorConfig_(conf, std::move(iC)) {
  // hack for consumes
  RPCHitAssociator rpctruth(conf, std::move(iC));
  GEMHitAssociator gemtruth(conf, std::move(iC));
  DTHitAssociator dttruth(conf, std::move(iC));
  CSCHitAssociator muonTruth(conf, std::move(iC));
  if (conf.getUntrackedParameter<bool>("dumpInputCollections")) {
    diagnostics_.reset(new InputDumper(conf, std::move(iC)));
  }
}

MuonAssociatorByHits::~MuonAssociatorByHits() {}

RecoToSimCollection MuonAssociatorByHits::associateRecoToSim(
    const edm::RefToBaseVector<reco::Track> &tC,
    const edm::RefVector<TrackingParticleCollection> &TPCollectionH,
    const edm::Event *e,
    const edm::EventSetup *setup) const {
  RecoToSimCollection outputCollection(&e->productGetter());

  MuonAssociatorByHitsHelper::TrackHitsCollection tH;
  for (auto it = tC.begin(), ed = tC.end(); it != ed; ++it) {
    tH.push_back(std::make_pair((*it)->recHitsBegin(), (*it)->recHitsEnd()));
  }

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup->get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo = tTopoHand.product();

  // Tracker hit association
  TrackerHitAssociator trackertruth(*e, trackerHitAssociatorConfig_);
  // CSC hit association
  CSCHitAssociator csctruth(*e, *setup, conf_);
  // DT hit association
  bool printRtS(true);
  DTHitAssociator dttruth(*e, *setup, conf_, printRtS);
  // RPC hit association
  RPCHitAssociator rpctruth(*e, *setup, conf_);
  // GEM hit association
  GEMHitAssociator gemtruth(*e, *setup, conf_);

  MuonAssociatorByHitsHelper::Resources resources = {
      tTopo, &trackertruth, &csctruth, &dttruth, &rpctruth, &gemtruth, {}};

  if (diagnostics_) {
    resources.diagnostics_ = [this, e](const TrackHitsCollection &hC, const TrackingParticleCollection &pC) {
      diagnostics_->dump(hC, pC, *e);
    };
  }

  auto bareAssoc = helper_.associateRecoToSimIndices(tH, TPCollectionH, resources);
  for (auto it = bareAssoc.begin(), ed = bareAssoc.end(); it != ed; ++it) {
    for (auto itma = it->second.begin(), edma = it->second.end(); itma != edma; ++itma) {
      outputCollection.insert(tC[it->first], std::make_pair(TPCollectionH[itma->idx], itma->quality));
    }
  }

  outputCollection.post_insert();  // perhaps not even necessary
  return outputCollection;
}

SimToRecoCollection MuonAssociatorByHits::associateSimToReco(
    const edm::RefToBaseVector<reco::Track> &tC,
    const edm::RefVector<TrackingParticleCollection> &TPCollectionH,
    const edm::Event *e,
    const edm::EventSetup *setup) const {
  SimToRecoCollection outputCollection(&e->productGetter());
  MuonAssociatorByHitsHelper::TrackHitsCollection tH;
  for (auto it = tC.begin(), ed = tC.end(); it != ed; ++it) {
    tH.push_back(std::make_pair((*it)->recHitsBegin(), (*it)->recHitsEnd()));
  }

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup->get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo = tTopoHand.product();

  // Tracker hit association
  TrackerHitAssociator trackertruth(*e, trackerHitAssociatorConfig_);
  // CSC hit association
  CSCHitAssociator csctruth(*e, *setup, conf_);
  // DT hit association
  bool printRtS = false;
  DTHitAssociator dttruth(*e, *setup, conf_, printRtS);
  // RPC hit association
  RPCHitAssociator rpctruth(*e, *setup, conf_);
  // GEM hit association
  GEMHitAssociator gemtruth(*e, *setup, conf_);

  MuonAssociatorByHitsHelper::Resources resources = {
      tTopo, &trackertruth, &csctruth, &dttruth, &rpctruth, &gemtruth, {}};

  auto bareAssoc = helper_.associateSimToRecoIndices(tH, TPCollectionH, resources);
  for (auto it = bareAssoc.begin(), ed = bareAssoc.end(); it != ed; ++it) {
    for (auto itma = it->second.begin(), edma = it->second.end(); itma != edma; ++itma) {
      outputCollection.insert(TPCollectionH[it->first], std::make_pair(tC[itma->idx], itma->quality));
    }
  }

  outputCollection.post_insert();  // perhaps not even necessary
  return outputCollection;
}
