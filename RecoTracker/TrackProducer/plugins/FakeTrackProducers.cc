
//
//

/**
  \class    pat::FakeTrackProducer FakeTrackProducer.h "MuonAnalysis/MuonAssociators/interface/FakeTrackProducer.h"
  \brief    Matcher of reconstructed objects to other reconstructed objects using the tracks inside them 
            
  \author   Giovanni Petrucciani
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

template <class T>
class FakeTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit FakeTrackProducer(const edm::ParameterSet &iConfig);
  ~FakeTrackProducer() override {}

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  /// Labels for input collections
  edm::EDGetTokenT<std::vector<T>> src_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;

  /// Muon selection
  //StringCutObjectSelector<T> selector_;

  const PTrajectoryStateOnDet &getState(const TrajectorySeed &seed) const { return seed.startingState(); }
  const PTrajectoryStateOnDet &getState(const TrackCandidate &seed) const { return seed.trajectoryStateOnDet(); }
};

template <typename T>
FakeTrackProducer<T>::FakeTrackProducer(const edm::ParameterSet &iConfig)
    : src_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("src"))),
      geometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      magFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      trackerTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>())
//,selector_(iConfig.existsAs<std::string>("cut") ? iConfig.getParameter<std::string>("cut") : "", true)
{
  produces<std::vector<reco::Track>>();
  produces<std::vector<reco::TrackExtra>>();
  produces<edm::OwnVector<TrackingRecHit>>();
}

template <typename T>
void FakeTrackProducer<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;

  edm::ESHandle<TrackerGeometry> theGeometry = iSetup.getHandle(geometryToken_);
  edm::ESHandle<MagneticField> theMagField = iSetup.getHandle(magFieldToken_);
  edm::ESHandle<TrackerTopology> httopo = iSetup.getHandle(trackerTopoToken_);
  const TrackerTopology &ttopo = *httopo;

  Handle<vector<T>> src;
  iEvent.getByToken(src_, src);

  unique_ptr<vector<reco::Track>> out(new vector<reco::Track>());
  out->reserve(src->size());
  unique_ptr<vector<reco::TrackExtra>> outEx(new vector<reco::TrackExtra>());
  outEx->reserve(src->size());
  unique_ptr<OwnVector<TrackingRecHit>> outHits(new OwnVector<TrackingRecHit>());

  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  for (typename vector<T>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
    const T &mu = *it;
    //if (!selector_(mu)) continue;
    const PTrajectoryStateOnDet &pstate = getState(mu);
    const GeomDet *det = theGeometry->idToDet(DetId(pstate.detId()));
    if (det == nullptr) {
      std::cerr << "ERROR:  bogus detid " << pstate.detId() << std::endl;
      continue;
    }
    TrajectoryStateOnSurface state = trajectoryStateTransform::transientState(pstate, &det->surface(), &*theMagField);
    GlobalPoint gx = state.globalPosition();
    GlobalVector gp = state.globalMomentum();
    reco::Track::Point x(gx.x(), gx.y(), gx.z());
    reco::Track::Vector p(gp.x(), gp.y(), gp.z());
    int charge = state.localParameters().charge();
    out->push_back(reco::Track(1.0, 1.0, x, p, charge, reco::Track::CovarianceMatrix()));
    auto hits = mu.recHits();
    out->back().appendHits(hits.begin(), hits.end(), ttopo);
    // Now Track Extra
    const TrackingRecHit *hit0 = &*hits.begin();
    const TrackingRecHit *hit1 = &*(hits.end() - 1);
    const GeomDet *det0 = theGeometry->idToDet(hit0->geographicalId());
    const GeomDet *det1 = theGeometry->idToDet(hit1->geographicalId());
    if (det0 == nullptr || det1 == nullptr) {
      std::cerr << "ERROR:  bogus detids at beginning or end of range" << std::endl;
      continue;
    }
    GlobalPoint gx0 = det0->toGlobal(hit0->localPosition());
    GlobalPoint gx1 = det1->toGlobal(hit1->localPosition());
    reco::Track::Point x0(gx0.x(), gx0.y(), gx0.z());
    reco::Track::Point x1(gx1.x(), gx1.y(), gx1.z());
    if (x0.R() > x1.R())
      std::swap(x0, x1);
    outEx->push_back(reco::TrackExtra(x1,
                                      p,
                                      true,
                                      x0,
                                      p,
                                      true,
                                      reco::Track::CovarianceMatrix(),
                                      hit0->geographicalId().rawId(),
                                      reco::Track::CovarianceMatrix(),
                                      hit1->geographicalId().rawId(),
                                      alongMomentum));
    out->back().setExtra(reco::TrackExtraRef(rTrackExtras, outEx->size() - 1));
    reco::TrackExtra &ex = outEx->back();
    auto const firstHitIndex = outHits->size();
    for (auto const &it2 : hits) {
      outHits->push_back(it2);
    }
    ex.setHits(rHits, firstHitIndex, outHits->size() - firstHitIndex);
  }

  iEvent.put(std::move(out));
  iEvent.put(std::move(outEx));
  iEvent.put(std::move(outHits));
}

typedef FakeTrackProducer<TrajectorySeed> FakeTrackProducerFromSeed;
typedef FakeTrackProducer<TrackCandidate> FakeTrackProducerFromCandidate;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FakeTrackProducerFromSeed);
DEFINE_FWK_MODULE(FakeTrackProducerFromCandidate);
