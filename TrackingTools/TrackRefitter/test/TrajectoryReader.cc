#include "TrackingTools/TrackRefitter/test/TrajectoryReader.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include "TH1F.h"


using namespace std;
using namespace edm;

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrajectoryReader);

/// Constructor
TrajectoryReader::TrajectoryReader(const ParameterSet& pset){
  
  theInputLabel = pset.getParameter<InputTag>("InputLabel");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
}

/// Destructor
TrajectoryReader::~TrajectoryReader(){}


// Operations
void TrajectoryReader::beginJob(const EventSetup& eventSetup){

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hDPtIn = new TH1F("DeltaPtIn","P_{t}^{Track}-P_{t}^{Traj} inner state",10000,-20,20);
  hDPtOut = new TH1F("DeltaPtOut","P_{t}^{Track}-P_{t}^{Traj} outer state",10000,-20,20);

}

void TrajectoryReader::endJob(){
  theFile->cd();

  // Write the histos to file
  hDPtIn->Write();
  hDPtOut->Write();
  
  theFile->Close();
}
 


void TrajectoryReader::analyze(const Event & event, const EventSetup& eventSetup){
  
  // Global Tracking Geometry
  ESHandle<GlobalTrackingGeometry> trackingGeometry; 
  eventSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry); 
  
  // Magfield Field
  ESHandle<MagneticField> magField;    
  eventSetup.get<IdealMagneticFieldRecord>().get(magField);

  const std::string metname = "Reco|TrackingTools|TrajectoryReader";
  
  // Get the Trajectory collection from the event
  Handle<Trajectories> trajectories;
  event.getByLabel(theInputLabel.label(),trajectories);
  
  for(Trajectories::const_iterator trajectory = trajectories->begin(); 
      trajectory != trajectories->end(); ++trajectory){
    
    TransientTrackingRecHit::ConstRecHitContainer rechits = trajectory->recHits();
    LogDebug(metname) << "Size of the RecHit container: " << rechits.size();
    
    int i = 0;
    for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator recHit = rechits.begin(); 
	recHit != rechits.end(); ++recHit)
      if((*recHit)->isValid()){
	const GeomDet* geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
	double r = geomDet->surface().position().perp();
	double z = geomDet->toGlobal((*recHit)->localPosition()).z();
	LogDebug(metname) <<  i++ <<" r: "<< r <<" z: "<<z <<" "<<geomDet->toGlobal((*recHit)->localPosition())
			  <<endl;
      }
  }
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByLabel(theInputLabel.label(),tracks);

  for (reco::TrackCollection::const_iterator tr = tracks->begin(); 
       tr != tracks->end(); ++tr) {
    
    reco::TransientTrack track(*tr,&*magField,trackingGeometry); 
    LogDebug(metname) << "Valid RecHits: "<<track.found() << " invalid RecHits: " << track.lost();

    int i = 0;
    for(trackingRecHit_iterator recHit = track.recHitsBegin(); recHit != track.recHitsEnd(); ++recHit)
      if((*recHit)->isValid()){
	const GeomDet* geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
	double r = geomDet->surface().position().perp();
	double z = geomDet->toGlobal((*recHit)->localPosition()).z();
	LogDebug(metname) << i++ <<" r: "<< r <<" z: "<<z <<" "<<geomDet->toGlobal((*recHit)->localPosition())
	    <<endl;
      }
  }

  // Check the difference in Pt

  int traj_size = trajectories->size();
  int track_size = tracks->size();

  if(traj_size != track_size){
    LogDebug(metname)
      <<"Mismatch between the # of Tracks ("<<track_size<<") and the # of Trajectories! ("
      <<traj_size<<")";
  }
  else{
    unsigned int position = 0;
    
    for(Trajectories::const_iterator trajectory = trajectories->begin(); 
	trajectory != trajectories->end(); ++trajectory){

      reco::TrackRef trackRef(tracks,position++);
      reco::TransientTrack track(trackRef,&*magField,trackingGeometry);

//       hDPtIn->Fill(track.innermostMeasurementState().globalMomentum().perp() -
// 		   trajectory->lastMeasurement().updatedState().globalMomentum().perp());
//       hDPtOut->Fill(track.outermostMeasurementState().globalMomentum().perp() -
// 		    trajectory->firstMeasurement().updatedState().globalMomentum().perp());

      hDPtIn->Fill(track.innermostMeasurementState().globalMomentum().perp() -
 		   trajectory->firstMeasurement().updatedState().globalMomentum().perp());
      hDPtOut->Fill(track.outermostMeasurementState().globalMomentum().perp() -
 		    trajectory->lastMeasurement().updatedState().globalMomentum().perp());

      LogDebug(metname)<< "Difference: " <<track.found()- trajectory->recHits().size();
      
    }     
  }
}
