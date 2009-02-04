#include "TrackingTools/TrackRefitter/plugins/TracksToTrajectories.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;
using namespace edm;

/// Constructor
TracksToTrajectories::TracksToTrajectories(const ParameterSet& parameterSet){

  theTracksLabel = parameterSet.getParameter<InputTag>("Tracks");

  theTrackTransformer = new TrackTransformer(parameterSet.getParameter<ParameterSet>("TrackTransformer"));
  
  produces<vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
}

/// Destructor
TracksToTrajectories::~TracksToTrajectories(){
  if(theTrackTransformer) delete theTrackTransformer;
}

/// Convert Tracks into Trajectories
void TracksToTrajectories::produce(Event& event, const EventSetup& setup){

  const string metname = "Reco|TrackingTools|TracksToTrajectories";

  theTrackTransformer->setServices(setup);
  
  // Collection of Trajectory
  auto_ptr<vector<Trajectory> > trajectoryCollection(new vector<Trajectory>);
  
  // Get the reference
  RefProd<vector<Trajectory> > trajectoryCollectionRefProd 
    = event.getRefBeforePut<vector<Trajectory> >();
  
  // Association map between Trajectory and Track
  auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection);
 
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByLabel(theTracksLabel,tracks);
  
  Ref<vector<Trajectory> >::key_type trajectoryIndex = 0;
  reco::TrackRef::key_type trackIndex = 0;

  // Loop over the Rec tracks
  for (reco::TrackCollection::const_iterator newTrack = tracks->begin(); 
       newTrack != tracks->end(); ++newTrack) {
    
    vector<Trajectory> trajectoriesSM = theTrackTransformer->transform(*newTrack);
    
    if(!trajectoriesSM.empty()){
      // Load the trajectory in the Trajectory Container
      trajectoryCollection->push_back(trajectoriesSM.front());

      // Make the association between the Trajectory and the original Track
      trajTrackMap->insert(Ref<vector<Trajectory> >(trajectoryCollectionRefProd,trajectoryIndex++),
			   reco::TrackRef(tracks,trackIndex++));
    }
    else
      LogError(metname) << "Error in the Track refitting. This must not happen!";
  }
  LogDebug(metname)<<"Load the Trajectory Collection";
  event.put(trajectoryCollection);
  event.put(trajTrackMap);
}
