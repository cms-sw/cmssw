#include "TrackingTools/TrackRefitter/test/TrajectoryReader.h"

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

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include "TH1F.h"


using namespace std;
using namespace edm;

#include "FWCore/PluginManager/interface/ModuleDef.h"
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
void TrajectoryReader::beginJob(){

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hDPtIn = new TH1F("DeltaPtIn","P_{t}^{Track}-P_{t}^{Traj} inner state",10000,-20,20);
  hDPtOut = new TH1F("DeltaPtOut","P_{t}^{Track}-P_{t}^{Traj} outer state",10000,-20,20);

  hNHitLost = new TH1F("NHitLost","Number of lost hits",100,0,100);
  hFractionHitLost = new TH1F("FractionHitLost","Fraction of lost hits",100,0,100);
  hSuccess = new TH1F("Success","Number of Success",2,0,2);

}

void TrajectoryReader::endJob(){
  theFile->cd();

  // Write the histos to file
  hDPtIn->Write();
  hDPtOut->Write();
  hNHitLost->Write();
  hFractionHitLost->Write();
  hSuccess->Write();
  theFile->Close();
}
 

void TrajectoryReader::printTrajectoryRecHits(const Trajectory &trajectory,
					      ESHandle<GlobalTrackingGeometry> trackingGeometry) const{

  const std::string metname = "Reco|TrackingTools|TrajectoryReader";
  
  TransientTrackingRecHit::ConstRecHitContainer rechits = trajectory.recHits();
  LogDebug(metname) << "Size of the RecHit container: " << rechits.size();
  
  int i = 0;
  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator recHit = rechits.begin(); 
      recHit != rechits.end(); ++recHit)
    if((*recHit)->isValid()){
      const GeomDet* geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->toGlobal((*recHit)->localPosition()).z();
      LogTrace(metname) <<  i++ <<" r: "<< r <<" z: "<<z <<" "<<geomDet->toGlobal((*recHit)->localPosition())
			<<endl;
    }
}

void TrajectoryReader::printTrackRecHits(const reco::Track &track, 
					 ESHandle<GlobalTrackingGeometry> trackingGeometry) const{

  const std::string metname = "Reco|TrackingTools|TrajectoryReader";

  LogTrace(metname) << "Valid RecHits: "<<track.found() << " invalid RecHits: " << track.lost();
  
  int i = 0;
  for(trackingRecHit_iterator recHit = track.recHitsBegin(); recHit != track.recHitsEnd(); ++recHit)
    if((*recHit)->isValid()){
      const GeomDet* geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->surface().position().z();
      LogTrace(metname) << i++ <<" GeomDet position r: "<< r <<" z: "<<z;
    }
}

void TrajectoryReader::analyze(const Event & event, const EventSetup& eventSetup){
  
  // Global Tracking Geometry
  ESHandle<GlobalTrackingGeometry> trackingGeometry; 
  eventSetup.get<GlobalTrackingGeometryRecord>().get(trackingGeometry); 
  
  // Magfield Field
  ESHandle<MagneticField> magField;    
  eventSetup.get<IdealMagneticFieldRecord>().get(magField);

  const std::string metname = "Reco|TrackingTools|TrajectoryReader";
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByLabel(theInputLabel.label(),tracks);

  if(tracks->empty()) return;
  
  // Get the Trajectory collection from the event
  Handle<Trajectories> trajectories;
  event.getByLabel(theInputLabel,trajectories);
  
  LogTrace(metname) << "looking at: " << theInputLabel;

  LogTrace(metname) << "All trajectories";
  for(Trajectories::const_iterator trajectory = trajectories->begin(); 
      trajectory != trajectories->end(); ++trajectory)
    printTrajectoryRecHits(*trajectory,trackingGeometry);

  LogTrace(metname) << "All tracks";
  for (reco::TrackCollection::const_iterator tr = tracks->begin(); 
       tr != tracks->end(); ++tr) 
    printTrackRecHits(*tr,trackingGeometry);
  
  
  Handle<TrajTrackAssociationCollection> assoMap;
  event.getByLabel(theInputLabel,assoMap);

  LogTrace(metname) << "Association";
  for(TrajTrackAssociationCollection::const_iterator it = assoMap->begin();
      it != assoMap->end(); ++it){

    const Ref<vector<Trajectory> > traj = it->key;
    const reco::TrackRef tk = it->val;

    printTrackRecHits(*tk,trackingGeometry);
    printTrajectoryRecHits(*traj,trackingGeometry);

    
    // Check the difference in Pt
    reco::TransientTrack track(tk,&*magField,trackingGeometry);
    
    hDPtIn->Fill(track.innermostMeasurementState().globalMomentum().perp() -
		 traj->lastMeasurement().updatedState().globalMomentum().perp());
    hDPtOut->Fill(track.outermostMeasurementState().globalMomentum().perp() -
		  traj->firstMeasurement().updatedState().globalMomentum().perp());
    
    int diff = track.recHitsSize()- traj->recHits().size();
    LogTrace(metname)<< "Difference: " << diff;
    hNHitLost->Fill(diff);
    hFractionHitLost->Fill(double(diff)/track.recHitsSize());
  }
  
  
  int traj_size = trajectories->size();
  int track_size = tracks->size();
  
  if(traj_size != track_size){
    LogTrace(metname)
      <<"Mismatch between the # of Tracks ("<<track_size<<") and the # of Trajectories! ("
      <<traj_size<<") in "<<event.id();
    hSuccess->Fill(0);
  }
  else
    hSuccess->Fill(1);
}
