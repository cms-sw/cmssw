#include "TrackingTools/TrackRefitter/test/TrackReader.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include "TH1F.h"


using namespace std;
using namespace edm;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackReader);

/// Constructor
TrackReader::TrackReader(const ParameterSet& parameterSet){
  
  theInputLabel = parameterSet.getParameter<InputTag>("InputLabel");

  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");
}

/// Destructor
TrackReader::~TrackReader(){}


// Operations
void TrackReader::beginJob(){}

void TrackReader::endJob(){}
 


void TrackReader::analyze(const Event & event, const EventSetup& setup){
  
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
  
  setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
  setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  
  const std::string metname = "Reco|TrackingTools|TrackReader";
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByLabel(theInputLabel,tracks);
  
  for (reco::TrackCollection::const_iterator track = tracks->begin(); 
       track != tracks->end(); ++track){
    for (trackingRecHit_iterator hit = track->recHitsBegin(); hit != track->recHitsEnd(); ++hit) {

      if((*hit)->isValid()) {
	if ( (*hit)->geographicalId().det() == DetId::Tracker ){
	  LogDebug("TrackReader") << "Tracker hit"; 
	  TransientTrackingRecHit::RecHitPointer tthit = theTrackerRecHitBuilder->build(&**hit);
	  //	  TransientTrackingRecHit::RecHitPointer preciseHit = tthit.clone(predTsos); 
	  LogTrace("TrackReader") << "Position: " << tthit->globalPosition();
	} 
	else if ( (*hit)->geographicalId().det() == DetId::Muon ){
	  LogDebug("TrackReader") << "Muon hit"; 
	  TransientTrackingRecHit::RecHitPointer tthit = theMuonRecHitBuilder->build(&**hit);
	  LogTrace("TrackReader") << "Position: " << tthit->globalPosition();
	}
      }
    }
  }
}
