/**
 *  Class: MuonSeedTrack
 *
 * 
 *
 *  Authors :
 *  \author Adam Everett - Purdue University
 *
 */

#include "Validation/RecoMuon/src/MuonSeedTrack.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

using namespace reco;
using namespace edm;
using namespace std;

typedef TrajectoryStateOnSurface TSOS;

//
// constructors
//
MuonSeedTrack::MuonSeedTrack(const edm::ParameterSet& pset) {
  // service parameters
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  ParameterSet updatorPar = pset.getParameter<ParameterSet>("MuonUpdatorAtVertexParameters");
  //theSeedPropagatorName = updatorPar.getParameter<string>("Propagator");
  theSeedsLabel = pset.getParameter<InputTag>("MuonSeed");
  theSeedsToken = consumes<TrajectorySeedCollection>(theSeedsLabel);
  theUpdatorAtVtx = new MuonUpdatorAtVertex(updatorPar, theService);

  theAllowNoVtxFlag = pset.getUntrackedParameter<bool>("AllowNoVertex", false);

  //register products
  setAlias(pset.getParameter<std::string>("@module_label"));
  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");
}

//
// destructor
//
MuonSeedTrack::~MuonSeedTrack() {
  if (theService)
    delete theService;
  if (theUpdatorAtVtx)
    delete theUpdatorAtVtx;
}

//
// member functions
//

/*!
 * For each seed, make a (fake) reco::Track
 */
void MuonSeedTrack::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  using namespace edm;

  // Update the services
  theService->update(eventSetup);

  // the track collectios; they will be loaded in the event
  unique_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection());
  // ... and its reference into the event
  reco::TrackRefProd trackCollectionRefProd = event.getRefBeforePut<reco::TrackCollection>();

  Handle<TrajectorySeedCollection> seeds;
  event.getByToken(theSeedsToken, seeds);

  for (TrajectorySeedCollection::const_iterator iSeed = seeds->begin(); iSeed != seeds->end(); iSeed++) {
    pair<bool, reco::Track> resultOfTrackExtrapAtPCA = buildTrackAtPCA(*iSeed);
    if (!resultOfTrackExtrapAtPCA.first)
      continue;
    // take the "bare" track at PCA
    reco::Track& track = resultOfTrackExtrapAtPCA.second;
    // fill the TrackCollection
    trackCollection->push_back(track);
  }

  event.put(std::move(trackCollection));
}

/*!
 * empty method
 */
void MuonSeedTrack::beginJob() {}

/*!
 * empty method
 */
void MuonSeedTrack::endJob() {}

/*!
 * Get the TrajectoryStateOnSurface from the TrajectorySeed
 */
TrajectoryStateOnSurface MuonSeedTrack::getSeedTSOS(const TrajectorySeed& seed) const {
  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface

  DetId seedDetId(pTSOD.detId());

  const GeomDet* gdet = theService->trackingGeometry()->idToDet(seedDetId);

  TrajectoryStateOnSurface initialState =
      trajectoryStateTransform::transientState(pTSOD, &(gdet->surface()), &*theService->magneticField());

  /*
  // Get the layer on which the seed relies
  const DetLayer *initialLayer = theService->detLayerGeometry()->idToLayer( seedDetId );

  PropagationDirection detLayerOrder = oppositeToMomentum;

  // ask for compatible layers
  vector<const DetLayer*> detLayers;
  detLayers = initialLayer->compatibleLayers( *initialState.freeState(),detLayerOrder);
  
  TrajectoryStateOnSurface result = initialState;
  if(detLayers.size()){
    const DetLayer* finalLayer = detLayers.back();
    const TrajectoryStateOnSurface propagatedState = theService->propagator(theSeedPropagatorName)->propagate(initialState, finalLayer->surface());
    if(propagatedState.isValid())
      result = propagatedState;
  }
  
  return result;
  */

  return initialState;
}

/*!
 * First calls getSeedTSOS, then propagates to the vertex.  Creates a
 * reco::Track from the propagated initial FreeTrajectoryState.
 */
pair<bool, reco::Track> MuonSeedTrack::buildTrackAtPCA(const TrajectorySeed& seed) const {
  const string metname = "MuonSeedTrack";

  MuonPatternRecoDumper debug;

  TSOS seedTSOS = getSeedTSOS(seed);
  // This is needed to extrapolate the tsos at vertex
  LogTrace(metname) << "Propagate to PCA...";
  pair<bool, FreeTrajectoryState> extrapolationResult = theUpdatorAtVtx->propagateToNominalLine(seedTSOS);
  FreeTrajectoryState ftsAtVtx;

  if (extrapolationResult.first) {
    ftsAtVtx = extrapolationResult.second;
  } else {
    if (TrackerBounds::isInside(seedTSOS.globalPosition())) {
      LogWarning(metname) << "Track in the Tracker: taking the innermost state instead of the state at PCA";
      ftsAtVtx = *seedTSOS.freeState();
    } else {
      if (theAllowNoVtxFlag) {
        LogWarning(metname) << "Propagation to PCA failed, taking the innermost state instead of the state at PCA";
        ftsAtVtx = *seedTSOS.freeState();
      } else {
        LogWarning(metname) << "Stand Alone track: this track will be rejected";
        return pair<bool, reco::Track>(false, reco::Track());
      }
    }
  }

  LogTrace(metname) << "TSOS after the extrapolation at vtx";
  LogTrace(metname) << debug.dumpFTS(ftsAtVtx);

  GlobalPoint pca = ftsAtVtx.position();
  math::XYZPoint persistentPCA(pca.x(), pca.y(), pca.z());
  GlobalVector p = ftsAtVtx.momentum();
  math::XYZVector persistentMomentum(p.x(), p.y(), p.z());

  double dummyNDOF = 1.0;
  double dummyChi2 = 1.0;

  reco::Track track(
      dummyChi2, dummyNDOF, persistentPCA, persistentMomentum, ftsAtVtx.charge(), ftsAtVtx.curvilinearError());

  return pair<bool, reco::Track>(true, track);
}
