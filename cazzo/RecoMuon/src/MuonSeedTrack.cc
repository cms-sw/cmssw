/**
 *  Class: MuonSeedTrack
 *
 * 
 *  $Date: 2011/12/22 20:44:37 $
 *  $Revision: 1.4 $
 *
 *  Authors :
 *  \author Adam Everett - Purdue University
 *
 */

#include "Validation/RecoMuon/src/MuonSeedTrack.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

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
MuonSeedTrack::MuonSeedTrack(const edm::ParameterSet& pset)
{
  // service parameters
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);
  
  ParameterSet updatorPar = pset.getParameter<ParameterSet>("MuonUpdatorAtVertexParameters");
  //theSeedPropagatorName = updatorPar.getParameter<string>("Propagator");
  theSeedsLabel = pset.getParameter<InputTag>("MuonSeed");
  theUpdatorAtVtx = new MuonUpdatorAtVertex(updatorPar,theService);

  theAllowNoVtxFlag = pset.getUntrackedParameter<bool>("AllowNoVertex",false);

  //register products
  setAlias(pset.getParameter<std::string>("@module_label"));
  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");

}

//
// destructor
//
MuonSeedTrack::~MuonSeedTrack()
{
  if (theService) delete theService;
  if (theUpdatorAtVtx) delete theUpdatorAtVtx;
}

//
// member functions
//

/*!
 * For each seed, make a (fake) reco::Track
 */
void
MuonSeedTrack::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
   using namespace edm;

  // Update the services
  theService->update(eventSetup);

  // the track collectios; they will be loaded in the event  
  auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection());
  // ... and its reference into the event
  reco::TrackRefProd trackCollectionRefProd = event.getRefBeforePut<reco::TrackCollection>();


  Handle<TrajectorySeedCollection> seeds;
  event.getByLabel(theSeedsLabel, seeds);

  for ( TrajectorySeedCollection::const_iterator iSeed = seeds->begin();
        iSeed != seeds->end(); iSeed++ ) {
    pair<bool,reco::Track> resultOfTrackExtrapAtPCA = buildTrackAtPCA(*iSeed);
    if(!resultOfTrackExtrapAtPCA.first) continue;
    // take the "bare" track at PCA
    reco::Track &track = resultOfTrackExtrapAtPCA.second;
    // fill the TrackCollection
    trackCollection->push_back(track);
  }
  
  event.put(trackCollection);

}

/*!
 * empty method
 */
void
MuonSeedTrack::beginJob()
{
}


/*!
 * empty method
 */
void
MuonSeedTrack::endJob() {
}


/*!
 * Get the TrajectoryStateOnSurface from the TrajectorySeed
 */
TrajectoryStateOnSurface MuonSeedTrack::getSeedTSOS(const TrajectorySeed& seed) const{

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface
  

  DetId seedDetId(pTSOD.detId());

  const GeomDet* gdet = theService->trackingGeometry()->idToDet( seedDetId );

  TrajectoryStateOnSurface initialState = trajectoryStateTransform::transientState(pTSOD, &(gdet->surface()), &*theService->magneticField());

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
pair<bool,reco::Track> MuonSeedTrack::buildTrackAtPCA(const TrajectorySeed& seed) const {

  const string metname = "MuonSeedTrack";

  MuonPatternRecoDumper debug;

  TSOS seedTSOS = getSeedTSOS(seed);
  // This is needed to extrapolate the tsos at vertex
  LogTrace(metname) << "Propagate to PCA...";
  pair<bool,FreeTrajectoryState> 
    extrapolationResult = theUpdatorAtVtx->propagateToNominalLine(seedTSOS);
  FreeTrajectoryState ftsAtVtx;
  
  if(extrapolationResult.first) {
    ftsAtVtx = extrapolationResult.second;
  }  else {    
    if(TrackerBounds::isInside(seedTSOS.globalPosition())){
      LogWarning(metname) << "Track in the Tracker: taking the innermost state instead of the state at PCA";
      ftsAtVtx = *seedTSOS.freeState();
    }
    else{
      if ( theAllowNoVtxFlag ) {
        LogWarning(metname) << "Propagation to PCA failed, taking the innermost state instead of the state at PCA";
        ftsAtVtx = *seedTSOS.freeState();
      } else {
        LogWarning(metname) << "Stand Alone track: this track will be rejected";
        return pair<bool,reco::Track>(false,reco::Track());
      }
    }
  }
    
  LogTrace(metname) << "TSOS after the extrapolation at vtx";
  LogTrace(metname) << debug.dumpFTS(ftsAtVtx);
  
  GlobalPoint pca = ftsAtVtx.position();
  math::XYZPoint persistentPCA(pca.x(),pca.y(),pca.z());
  GlobalVector p = ftsAtVtx.momentum();
  math::XYZVector persistentMomentum(p.x(),p.y(),p.z());
  
  double dummyNDOF = 1.0;
  //double ndof = computeNDOF(seed);
  double dummyChi2 = 1.0;
  
  reco::Track track(dummyChi2, 
		    dummyNDOF,
		    persistentPCA,
		    persistentMomentum,
		    ftsAtVtx.charge(),
		    ftsAtVtx.curvilinearError());
  
  return pair<bool,reco::Track>(true,track);
}

/*!
 * Calculates number of degrees of freedom for the TrajectorySeed
 */
double MuonSeedTrack::computeNDOF(const TrajectorySeed& trajectory) const {
  const string metname = "MuonSeedTrack";

  BasicTrajectorySeed::const_iterator recHits1 = (trajectory.recHits().first);
  BasicTrajectorySeed::const_iterator recHits2 = (trajectory.recHits().second);
  
  double ndof = 0.;

  if ((*recHits1).isValid()) ndof += (*recHits1).dimension();
  if ((*recHits2).isValid()) ndof += (*recHits2).dimension();

  //const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  //for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
  //  rechit != transRecHits.end(); ++rechit)
  //if ((*rechit)->isValid()) ndof += (*rechit)->dimension();
  
  // FIXME! in case of Boff is dof - 4
  return max(ndof - 5., 0.);
}

