// -*- C++ -*-
//
// Package:    SimMuon/MCTruth
// Class:      MuonToTrackingParticleAssociatorEDProducer
//
/**\class MuonToTrackingParticleAssociatorEDProducer
 MuonToTrackingParticleAssociatorEDProducer.cc
 SimMuon/MCTruth/plugins/MuonToTrackingParticleAssociatorEDProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:30:14 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MuonToTrackingParticleAssociatorByHitsImpl.h"
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"

//
// class declaration
//
namespace {
  using TrackHitsCollection = MuonAssociatorByHitsHelper::TrackHitsCollection;

  class InputDumper {
  public:
    InputDumper(const edm::ParameterSet &conf, edm::ConsumesCollector &&iC)
        : simtracksTag(conf.getParameter<edm::InputTag>("simtracksTag")),
          simtracksXFTag(conf.getParameter<edm::InputTag>("simtracksXFTag")),
          crossingframe(conf.getParameter<bool>("crossingframe")) {
      if (crossingframe) {
        simtracksXFToken_ = iC.consumes<CrossingFrame<SimTrack>>(simtracksXFTag);
        simvertsXFToken_ = iC.consumes<CrossingFrame<SimVertex>>(simtracksXFTag);
      } else {
        simtracksToken_ = iC.consumes<edm::SimTrackContainer>(simtracksTag);
        simvertsToken_ = iC.consumes<edm::SimVertexContainer>(simtracksTag);
      }
    }

    void read(const edm::Event &);
    void dump(const TrackHitsCollection &, const TrackingParticleCollection &) const;

  private:
    edm::InputTag simtracksTag;
    edm::InputTag simtracksXFTag;
    edm::EDGetTokenT<CrossingFrame<SimTrack>> simtracksXFToken_;
    edm::EDGetTokenT<CrossingFrame<SimVertex>> simvertsXFToken_;
    edm::EDGetTokenT<edm::SimTrackContainer> simtracksToken_;
    edm::EDGetTokenT<edm::SimVertexContainer> simvertsToken_;

    edm::Handle<CrossingFrame<SimTrack>> simtracksXF_;
    edm::Handle<CrossingFrame<SimVertex>> simvertsXF_;
    edm::Handle<edm::SimTrackContainer> simtracks_;
    edm::Handle<edm::SimVertexContainer> simverts_;
    bool const crossingframe;
  };

  void InputDumper::read(const edm::Event &iEvent) {
    if (crossingframe) {
      iEvent.getByToken(simtracksXFToken_, simtracksXF_);
      iEvent.getByToken(simvertsXFToken_, simvertsXF_);
    } else {
      iEvent.getByToken(simtracksToken_, simtracks_);
      iEvent.getByToken(simvertsToken_, simverts_);
    }
  }

  void InputDumper::dump(const TrackHitsCollection &tC, const TrackingParticleCollection &tPC) const {
    using namespace std;
    // reco::Track collection
    edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer") << "\n"
                                                                   << "reco::Track collection --- size = " << tC.size();

    // TrackingParticle collection
    edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
        << "\n"
        << "TrackingParticle collection --- size = " << tPC.size();
    int j = 0;
    for (TrackingParticleCollection::const_iterator ITER = tPC.begin(); ITER != tPC.end(); ITER++, j++) {
      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
          << "TrackingParticle " << j << ", q = " << ITER->charge() << ", p = " << ITER->p() << ", pT = " << ITER->pt()
          << ", eta = " << ITER->eta() << ", phi = " << ITER->phi();

      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
          << "\t pdg code = " << ITER->pdgId() << ", made of " << ITER->numberOfHits() << " PSimHit"
          << " (in " << ITER->numberOfTrackerLayers() << " layers)"
          << " from " << ITER->g4Tracks().size() << " SimTrack:";
      for (TrackingParticle::g4t_iterator g4T = ITER->g4Track_begin(); g4T != ITER->g4Track_end(); g4T++) {
        edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
            << "\t\t Id:" << g4T->trackId() << "/Evt:(" << g4T->eventId().event() << ","
            << g4T->eventId().bunchCrossing() << ")";
      }
    }

    if (crossingframe) {
      std::unique_ptr<MixCollection<SimTrack>> SimTk(new MixCollection<SimTrack>(simtracksXF_.product()));
      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
          << "\n"
          << "CrossingFrame<SimTrack> collection with InputTag = " << simtracksXFTag << " has size = " << SimTk->size();
      int k = 0;
      for (MixCollection<SimTrack>::MixItr ITER = SimTk->begin(); ITER != SimTk->end(); ITER++, k++) {
        edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
            << "SimTrack " << k << " - Id:" << ITER->trackId() << "/Evt:(" << ITER->eventId().event() << ","
            << ITER->eventId().bunchCrossing() << ")"
            << ", q = " << ITER->charge() << ", p = " << ITER->momentum().P() << ", pT = " << ITER->momentum().Pt()
            << ", eta = " << ITER->momentum().Eta() << ", phi = " << ITER->momentum().Phi()
            << "\n\t pdgId = " << ITER->type() << ", Vertex index = " << ITER->vertIndex()
            << ", Gen Particle index = " << (ITER->genpartIndex() > 0 ? ITER->genpartIndex() - 1 : ITER->genpartIndex())
            << endl;
      }

      std::unique_ptr<MixCollection<SimVertex>> SimVtx(new MixCollection<SimVertex>(simvertsXF_.product()));
      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
          << "\n"
          << "CrossingFrame<SimVertex> collection with InputTag = " << simtracksXFTag
          << " has size = " << SimVtx->size();
      int kv = 0;
      for (MixCollection<SimVertex>::MixItr VITER = SimVtx->begin(); VITER != SimVtx->end(); VITER++, kv++) {
        edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
            << "SimVertex " << kv << " - Id:" << VITER->vertexId() << ", position = " << VITER->position()
            << ", parent SimTrack Id = " << VITER->parentIndex() << ", processType = " << VITER->processType();
      }
    } else {
      const edm::SimTrackContainer simTC = *(simtracks_.product());
      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
          << "\n"
          << "SimTrack collection with InputTag = " << simtracksTag << " has size = " << simTC.size() << endl;
      int k = 0;
      for (edm::SimTrackContainer::const_iterator ITER = simTC.begin(); ITER != simTC.end(); ITER++, k++) {
        edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
            << "SimTrack " << k << " - Id:" << ITER->trackId() << "/Evt:(" << ITER->eventId().event() << ","
            << ITER->eventId().bunchCrossing() << ")"
            << ", q = " << ITER->charge() << ", p = " << ITER->momentum().P() << ", pT = " << ITER->momentum().Pt()
            << ", eta = " << ITER->momentum().Eta() << ", phi = " << ITER->momentum().Phi()
            << "\n\t pdgId = " << ITER->type() << ", Vertex index = " << ITER->vertIndex()
            << ", Gen Particle index = " << (ITER->genpartIndex() > 0 ? ITER->genpartIndex() - 1 : ITER->genpartIndex())
            << endl;
      }
      const edm::SimVertexContainer simVC = *(simverts_.product());
      edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer") << "\n"
                                                                     << "SimVertex collection with InputTag = "
                                                                     << "g4SimHits"
                                                                     << " has size = " << simVC.size() << endl;
      int kv = 0;
      for (edm::SimVertexContainer::const_iterator VITER = simVC.begin(); VITER != simVC.end(); VITER++, kv++) {
        edm::LogVerbatim("MuonToTrackingParticleAssociatorEDProducer")
            << "SimVertex " << kv << " - Id:" << VITER->vertexId() << ", position = " << VITER->position()
            << ", parent SimTrack Id = " << VITER->parentIndex() << ", processType = " << VITER->processType();
      }
    }
  }

}  // namespace

class MuonToTrackingParticleAssociatorEDProducer : public edm::stream::EDProducer<> {
public:
  explicit MuonToTrackingParticleAssociatorEDProducer(const edm::ParameterSet &);
  ~MuonToTrackingParticleAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  edm::ParameterSet const config_;
  MuonAssociatorByHitsHelper helper_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  TrackerMuonHitExtractor hitExtractor_;

  std::unique_ptr<RPCHitAssociator> rpctruth_;
  std::unique_ptr<GEMHitAssociator> gemtruth_;
  std::unique_ptr<DTHitAssociator> dttruth_;
  std::unique_ptr<CSCHitAssociator> csctruth_;
  std::unique_ptr<TrackerHitAssociator> trackertruth_;
  std::unique_ptr<InputDumper> diagnostics_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonToTrackingParticleAssociatorEDProducer::MuonToTrackingParticleAssociatorEDProducer(const edm::ParameterSet &iConfig)
    : config_(iConfig),
      helper_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      hitExtractor_(iConfig, consumesCollector()) {
  // register your products
  produces<reco::MuonToTrackingParticleAssociator>();

  // hack for consumes
  RPCHitAssociator rpctruth(iConfig, consumesCollector());
  GEMHitAssociator gemtruth(iConfig, consumesCollector());
  DTHitAssociator dttruth(iConfig, consumesCollector());
  CSCHitAssociator cscruth(iConfig, consumesCollector());

  if (iConfig.getUntrackedParameter<bool>("dumpInputCollections")) {
    diagnostics_.reset(new InputDumper(iConfig, consumesCollector()));
  }
}

MuonToTrackingParticleAssociatorEDProducer::~MuonToTrackingParticleAssociatorEDProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MuonToTrackingParticleAssociatorEDProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  hitExtractor_.init(iEvent, iSetup);

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo = tTopoHand.product();

  bool printRtS = true;

  // NOTE: This assumes that produce will not be called until the edm::Event
  // used in the previous call
  // has been deleted. This is true for now. In the future, we may have to have
  // the resources own the memory.

  // Tracker hit association
  trackertruth_.reset(new TrackerHitAssociator(iEvent, trackerHitAssociatorConfig_));
  // CSC hit association
  csctruth_.reset(new CSCHitAssociator(iEvent, iSetup, config_));
  // DT hit association
  printRtS = false;
  dttruth_.reset(new DTHitAssociator(iEvent, iSetup, config_, printRtS));
  // RPC hit association
  rpctruth_.reset(new RPCHitAssociator(iEvent, iSetup, config_));
  // GEM hit association
  gemtruth_.reset(new GEMHitAssociator(iEvent, iSetup, config_));

  MuonAssociatorByHitsHelper::Resources resources = {
      tTopo, trackertruth_.get(), csctruth_.get(), dttruth_.get(), rpctruth_.get(), gemtruth_.get(), {}};

  if (diagnostics_) {
    diagnostics_->read(iEvent);
    resources.diagnostics_ = [this](const TrackHitsCollection &hC, const TrackingParticleCollection &pC) {
      diagnostics_->dump(hC, pC);
    };
  }

  std::unique_ptr<reco::MuonToTrackingParticleAssociatorBaseImpl> impl{
      new MuonToTrackingParticleAssociatorByHitsImpl(hitExtractor_, resources, &helper_)};
  std::unique_ptr<reco::MuonToTrackingParticleAssociator> toPut(
      new reco::MuonToTrackingParticleAssociator(std::move(impl)));
  iEvent.put(std::move(toPut));
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void MuonToTrackingParticleAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonToTrackingParticleAssociatorEDProducer);
