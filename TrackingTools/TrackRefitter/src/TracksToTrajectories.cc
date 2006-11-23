#include "TrackingTools/TrackRefitter/src/TracksToTrajectories.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace std;
using namespace edm;

/// Constructor
TracksToTrajectories::TracksToTrajectories(const ParameterSet& parameterSet){

  theTracksLabel = parameterSet.getParameter<InputTag>("Tracks");

  theTrackTransformer = new TrackTransformer(parameterSet.getParameter<ParameterSet>("TrackTransformer"));
  
  produces<vector<Trajectory> >();
}

/// Destructor
TracksToTrajectories::~TracksToTrajectories(){
  if(theTrackTransformer) delete theTrackTransformer;
}

/// Convert Tracks into Trajectories
void TracksToTrajectories::produce(Event& event, const EventSetup& setup){

  const string metname = "Reco|TrackingTools|TracksToTrajectories";
  
  auto_ptr<vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);
  
  theTrackTransformer->setServices(setup);
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByLabel(theTracksLabel,tracks);
  
  // Loop over the Rec tracks
  for (reco::TrackCollection::const_iterator newTrack = tracks->begin(); 
       newTrack != tracks->end(); ++newTrack) {
    
    vector<Trajectory> trajectoriesSM = theTrackTransformer->transform(*newTrack);
    
    if(trajectoriesSM.size())
      trajectoryCollection->push_back(trajectoriesSM.front());
    
  }
  LogDebug(metname)<<"Load the Trajectory Collection";
  event.put(trajectoryCollection);
}
