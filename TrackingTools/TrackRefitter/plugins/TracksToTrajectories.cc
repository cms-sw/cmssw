#include "TrackingTools/TrackRefitter/plugins/TracksToTrajectories.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerForGlobalCosmicMuons.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"

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
TracksToTrajectories::TracksToTrajectories(const ParameterSet& parameterSet):theTrackTransformer(0),
									     theNTracks(0),theNFailures(0){

  theTracksLabel = parameterSet.getParameter<InputTag>("Tracks");

  ParameterSet trackTransformerParam = parameterSet.getParameter<ParameterSet>("TrackTransformer");

  string type = parameterSet.getParameter<string>("Type");

  if(type == "Default") theTrackTransformer = new TrackTransformer(trackTransformerParam);
  else if(type == "GlobalCosmicMuonsForAlignment") theTrackTransformer = new TrackTransformerForGlobalCosmicMuons(trackTransformerParam);
  else if(type == "CosmicMuonsForAlignment") theTrackTransformer = new TrackTransformerForCosmicMuons(trackTransformerParam);
  else{
    throw cms::Exception("TracksToTrajectories") 
      <<"The selected algorithm does not exist"
      << "\n"
      << "Possible choices are:"
      << "\n"
      << "Type = [Default, GlobalCosmicMuonsForAlignment, CosmicMuonsForAlignment]";
  }

  produces<vector<Trajectory> >("Refitted");
  produces<TrajTrackAssociationCollection>("Refitted");
}
 

/// Destructor
TracksToTrajectories::~TracksToTrajectories(){
  if(theTrackTransformer) delete theTrackTransformer;
}

void TracksToTrajectories::endJob(){
  const string metname = "Reco|TrackingTools|TracksToTrajectories";
  
  if(theNFailures!=0)
    LogWarning(metname) << "During the refit there were " 
			<< theNFailures << " out of " << theNTracks << " tracks, i.e. failure rate is: " << double(theNFailures)/theNTracks;
  else{
    LogTrace(metname) << "Refit of the tracks done without any failure";
  }
}


/// Convert Tracks into Trajectories
void TracksToTrajectories::produce(Event& event, const EventSetup& setup){

  const string metname = "Reco|TrackingTools|TracksToTrajectories";

  theTrackTransformer->setServices(setup);
  
  // Collection of Trajectory
  auto_ptr<vector<Trajectory> > trajectoryCollection(new vector<Trajectory>);
  
  // Get the reference
  RefProd<vector<Trajectory> > trajectoryCollectionRefProd 
    = event.getRefBeforePut<vector<Trajectory> >("Refitted");
  
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
    
    ++theNTracks;

    vector<Trajectory> trajectoriesSM = theTrackTransformer->transform(*newTrack);
    
    if(!trajectoriesSM.empty()){
      // Load the trajectory in the Trajectory Container
      trajectoryCollection->push_back(trajectoriesSM.front());

      // Make the association between the Trajectory and the original Track
      trajTrackMap->insert(Ref<vector<Trajectory> >(trajectoryCollectionRefProd,trajectoryIndex++),
			   reco::TrackRef(tracks,trackIndex++));
    }
    else{
      LogTrace(metname) << "Error in the Track refitting. This should not happen";
      ++theNFailures;
    }
  }
  LogTrace(metname)<<"Load the Trajectory Collection";
  event.put(trajectoryCollection,"Refitted");
  event.put(trajTrackMap,"Refitted");
}
