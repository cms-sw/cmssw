/** \class SeedToTrackProducerBase
 *  
 *  See header file for a description of the class
 * 
 *  \author  Hugues Brun 
 */

#include "SimMuon/MCTruth/plugins/SeedToTrackProducerBase.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"

template class SeedToTrackProducerBase<std::vector<TrajectorySeed>>;
template class SeedToTrackProducerBase<std::vector<L2MuonTrajectorySeed>>;

//
// constructors and destructor
//
template <typename SeedCollection>
SeedToTrackProducerBase<SeedCollection>::SeedToTrackProducerBase(const edm::ParameterSet &iConfig)
    : theMGFieldToken(esConsumes()), theTrackingGeometryToken(esConsumes()), theTopoToken(esConsumes()) {
  L2seedsTagT_ = consumes<SeedCollection>(iConfig.getParameter<edm::InputTag>("L2seedsCollection"));
  L2seedsTagS_ = consumes<edm::View<TrajectorySeed>>(iConfig.getParameter<edm::InputTag>("L2seedsCollection"));

  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename SeedCollection>
void SeedToTrackProducerBase<SeedCollection>::produce(edm::StreamID,
                                                      edm::Event &iEvent,
                                                      const edm::EventSetup &iSetup) const {
  using namespace edm;
  using namespace std;

  std::unique_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> selectedTrackExtras(new reco::TrackExtraCollection());
  std::unique_ptr<TrackingRecHitCollection> selectedTrackHits(new TrackingRecHitCollection());

  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;

  // magnetic fied and detector geometry
  auto const &mgField = iSetup.getData(theMGFieldToken);
  auto const &trackingGeometry = iSetup.getData(theTrackingGeometryToken);

  const TrackerTopology &ttopo = iSetup.getData(theTopoToken);

  // now read the L2 seeds collection :
  edm::Handle<SeedCollection> L2seedsCollection;
  iEvent.getByToken(L2seedsTagT_, L2seedsCollection);
  const std::vector<SeedType> *L2seeds = nullptr;
  if (L2seedsCollection.isValid())
    L2seeds = L2seedsCollection.product();
  else {
    edm::LogError("SeedToTrackProducerBase") << "L2 seeds collection not found !! " << endl;
    return;
  }

  edm::Handle<edm::View<TrajectorySeed>> seedHandle;
  iEvent.getByToken(L2seedsTagS_, seedHandle);

  // now  loop on the seeds :
  for (unsigned int i = 0; i < L2seeds->size(); i++) {
    // get the kinematic extrapolation from the seed
    TrajectoryStateOnSurface theTrajectory = seedTransientState(L2seeds->at(i), mgField, trackingGeometry);
    float seedEta = theTrajectory.globalMomentum().eta();
    float seedPhi = theTrajectory.globalMomentum().phi();
    float seedPt = theTrajectory.globalMomentum().perp();
    CovarianceMatrix matrixSeedErr = theTrajectory.curvilinearError().matrix();
    edm::LogVerbatim("SeedToTrackProducerBase")
        << "seedPt=" << seedPt << " seedEta=" << seedEta << " seedPhi=" << seedPhi << endl;
    /*AlgebraicSymMatrix66 errors = theTrajectory.cartesianError().matrix();
    double partialPterror =
    errors(3,3)*pow(theTrajectory.globalMomentum().x(),2) +
    errors(4,4)*pow(theTrajectory.globalMomentum().y(),2);
    edm::LogVerbatim("SeedToTrackProducerBase") <<  "seedPtError=" <<
    sqrt(partialPterror)/theTrajectory.globalMomentum().perp() <<
    "seedPhiError=" << theTrajectory.curvilinearError().matrix()(2,2) << endl;*/
    // fill the track in a way that its pt, phi and eta will be the same as the
    // seed
    math::XYZPoint initPoint(0, 0, 0);
    math::XYZVector initMom(seedPt * cos(seedPhi), seedPt * sin(seedPhi), seedPt * sinh(seedEta));
    reco::Track theTrack(1,
                         1,  // dummy Chi2 and ndof
                         initPoint,
                         initMom,
                         1,
                         matrixSeedErr,
                         reco::TrackBase::TrackAlgorithm::globalMuon,
                         reco::TrackBase::TrackQuality::tight);

    // fill the extra track with dummy information
    math::XYZPoint dummyFinalPoint(1, 1, 1);
    math::XYZVector dummyFinalMom(0, 0, 10);
    edm::RefToBase<TrajectorySeed> seed(seedHandle, i);
    CovarianceMatrix matrixExtra = ROOT::Math::SMatrixIdentity();
    reco::TrackExtra theTrackExtra(dummyFinalPoint,
                                   dummyFinalMom,
                                   true,
                                   initPoint,
                                   initMom,
                                   true,
                                   matrixSeedErr,
                                   1,
                                   matrixExtra,
                                   2,
                                   (L2seeds->at(i)).direction(),
                                   seed);
    theTrack.setExtra(reco::TrackExtraRef(rTrackExtras, idx++));
    edm::LogVerbatim("SeedToTrackProducerBase")
        << "trackPt=" << theTrack.pt() << " trackEta=" << theTrack.eta() << " trackPhi=" << theTrack.phi() << endl;
    edm::LogVerbatim("SeedToTrackProducerBase")
        << "trackPtError=" << theTrack.ptError() << "trackPhiError=" << theTrack.phiError() << endl;

    // fill the seed segments in the track
    unsigned int nHitsAdded = 0;
    for (auto const &recHit : L2seeds->at(i).recHits()) {
      TrackingRecHit *hit = recHit.clone();
      theTrack.appendHitPattern(*hit, ttopo);
      selectedTrackHits->push_back(hit);
      nHitsAdded++;
    }
    theTrackExtra.setHits(rHits, hidx, nHitsAdded);
    hidx += nHitsAdded;
    selectedTracks->push_back(theTrack);
    selectedTrackExtras->push_back(theTrackExtra);
  }
  iEvent.put(std::move(selectedTracks));
  iEvent.put(std::move(selectedTrackExtras));
  iEvent.put(std::move(selectedTrackHits));
}

template <typename SeedCollection>
TrajectoryStateOnSurface SeedToTrackProducerBase<SeedCollection>::seedTransientState(
    const SeedType &tmpSeed, const MagneticField &mgField, const GlobalTrackingGeometry &trackingGeometry) const {
  PTrajectoryStateOnDet tmpTSOD = tmpSeed.startingState();
  DetId tmpDetId(tmpTSOD.detId());
  const GeomDet *tmpGeomDet = trackingGeometry.idToDet(tmpDetId);
  TrajectoryStateOnSurface tmpTSOS =
      trajectoryStateTransform::transientState(tmpTSOD, &(tmpGeomDet->surface()), &mgField);
  return tmpTSOS;
}
