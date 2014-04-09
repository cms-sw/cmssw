#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"


#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


using namespace std;
using namespace edm;

/// Constructor
TrackTransformerForCosmicMuons::TrackTransformerForCosmicMuons(const ParameterSet& parameterSet){
  
  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");

  theRPCInTheFit = parameterSet.getParameter<bool>("RefitRPCHits");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;
}

/// Destructor
TrackTransformerForCosmicMuons::~TrackTransformerForCosmicMuons(){}


void TrackTransformerForCosmicMuons::setServices(const EventSetup& setup){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";

  setup.get<TrajectoryFitter::Record>().get("KFFitterForRefitInsideOut",theFitterIO);
  setup.get<TrajectoryFitter::Record>().get("KFSmootherForRefitInsideOut",theSmootherIO);  
  setup.get<TrajectoryFitter::Record>().get("KFFitterForRefitOutsideIn",theFitterOI);
  setup.get<TrajectoryFitter::Record>().get("KFSmootherForRefitOutsideIn",theSmootherOI);

  unsigned long long newCacheId_TC = setup.get<TrackingComponentsRecord>().cacheIdentifier();

  if ( newCacheId_TC != theCacheId_TC ){
    LogTrace(metname) << "Tracking Component changed!";
    theCacheId_TC = newCacheId_TC;
    setup.get<TrackingComponentsRecord>().get("SmartPropagatorRK",thePropagatorIO);
    setup.get<TrackingComponentsRecord>().get("SmartPropagatorRKOpposite",thePropagatorOI);
  }

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != theCacheId_GTG ) {
    LogTrace(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  
  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if ( newCacheId_MG != theCacheId_MG ) {
    LogTrace(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    theCacheId_TRH = newCacheId_TRH;
    LogTrace(metname) << "TransientRecHitRecord changed!";
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  }
}


TransientTrackingRecHit::ConstRecHitContainer
TrackTransformerForCosmicMuons::getTransientRecHits(const reco::TransientTrack& track) const {

  TransientTrackingRecHit::ConstRecHitContainer tkHits;
  TransientTrackingRecHit::ConstRecHitContainer staHits;

  bool printout = false;
  
  bool quad1 = false;
  bool quad2 = false;
  bool quad3 = false;
  bool quad4 = false;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Muon ){
		if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  		LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  		continue;
		}
	 	staHits.push_back(theMuonRecHitBuilder->build(&**hit));
    	DetId hitId = staHits.back()->geographicalId();
    	GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();
		float gpy=glbpoint.y();	
		float gpz=glbpoint.z();	
//		if (gpy != 0 && gpz !=0) slopeSum += gpy / gpz;

		if (gpy > 0 && gpz > 0) quad1 = true; 
		else if (gpy > 0 && gpz < 0) quad2 = true; 
		else if (gpy < 0 && gpz < 0) quad3 = true; 
		else if (gpy < 0 && gpz > 0) quad4 = true; 
		else return tkHits;
      }

  if(staHits.empty()) return staHits;

  if (quad1 && quad2) return tkHits;
  if (quad1 && quad3) return tkHits;
  if (quad1 && quad4) return tkHits;
  if (quad2 && quad3) return tkHits;
  if (quad2 && quad4) return tkHits;
  if (quad3 && quad4) return tkHits;


  bool doReverse = staHits.front()->globalPosition().y()>0 ? true : false;

  if (quad1) doReverse = SlopeSum(staHits); 
  if (quad2) doReverse = SlopeSum(staHits); 
  if (quad3) doReverse = SlopeSum(staHits); 
  if (quad4) doReverse = SlopeSum(staHits); 
  if(doReverse){
    reverse(staHits.begin(),staHits.end());
  }

  copy(staHits.begin(),staHits.end(),back_inserter(tkHits));


///  if ( quad1 && slopeSum < 0) printout = true;
//  if ( quad2 && slopeSum > 0) printout = true;
///  if ( quad3 && slopeSum > 0) printout = true;
///  if ( quad4 && slopeSum < 0) printout = true;
//  swap = printout;

  printout = quad1;  

  if (printout) for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = tkHits.begin();
      hit !=tkHits.end(); ++hit){

    DetId hitId = (*hit)->geographicalId();
    GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();


    if(hitId.det() == DetId::Muon) {
      if(hitId.subdetId() == MuonSubdetId::DT)
		LogTrace("TrackFitters") << glbpoint << " I am DT " << DTWireId(hitId);
//		std::cout<< glbpoint << " I am DT " << DTWireId(hitId)<<std::endl;
      else if (hitId.subdetId() == MuonSubdetId::CSC )
		LogTrace("TrackFitters") << glbpoint << " I am CSC " << CSCDetId(hitId);
//		std::cout<< glbpoint << " I am CSC " << CSCDetId(hitId)<<std::endl;
      else if (hitId.subdetId() == MuonSubdetId::RPC )
		LogTrace("TrackFitters") << glbpoint << " I am RPC " << RPCDetId(hitId);
      else 
		LogTrace("TrackFitters") << " UNKNOWN MUON HIT TYPE ";
    } 
  } 
  return tkHits;
}


/// the refitter used to refit the reco::Track
ESHandle<TrajectoryFitter> TrackTransformerForCosmicMuons::fitter(bool up, int quad, float sumy) const{
  if(quad ==1) {if (sumy < 0) return theFitterOI; else return theFitterIO;}
  if(quad ==2) {if (sumy < 0) return theFitterOI; else return theFitterIO;}
  if(quad ==3) {if (sumy > 0) return theFitterOI; else return theFitterIO;}
  if(quad ==4) {if (sumy > 0) return theFitterOI; else return theFitterIO;}

  if(up) return theFitterOI;
  else return theFitterIO;
}
  
/// the smoother used to smooth the trajectory which came from the refitting step
ESHandle<TrajectorySmoother> TrackTransformerForCosmicMuons::smoother(bool up, int quad, float sumy) const{
  if(quad ==1){ if (sumy < 0) return theSmootherOI; else return theSmootherIO;}
  if(quad ==2){ if (sumy < 0) return theSmootherOI; else return theSmootherIO;}
  if(quad ==3){ if (sumy > 0) return theSmootherOI; else return theSmootherIO;}
  if(quad ==4){ if (sumy > 0) return theSmootherOI; else return theSmootherIO;}
  if(up) return theSmootherOI;
  else return theSmootherIO;
}

ESHandle<Propagator> TrackTransformerForCosmicMuons::propagator(bool up, int quad, float sumy) const{
  if(quad ==1) {if (sumy > 0) return thePropagatorOI; else return thePropagatorIO;}
  if(quad ==2) {if (sumy > 0) return thePropagatorOI; else return thePropagatorIO;}
  if(quad ==3) {if (sumy < 0) return thePropagatorOI; else return thePropagatorIO;}
  if(quad ==4) {if (sumy < 0) return thePropagatorOI; else return thePropagatorIO;}
  if(up) return thePropagatorIO;
  else return thePropagatorOI;
}



/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformerForCosmicMuons::transform(const reco::Track& tr) const {

  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(tr,magneticField(),trackingGeometry());   

  // Build the transient Rechits
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit;// = getTransientRecHits(track);
  TransientTrackingRecHit::ConstRecHitContainer staHits;


  bool quad1 = false;
  bool quad2 = false;
  bool quad3 = false;
  bool quad4 = false;
  int quadrant = 0;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Muon ){
		if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  		LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  		continue;
		}
	 	staHits.push_back(theMuonRecHitBuilder->build(&**hit));
    	DetId hitId = staHits.back()->geographicalId();
    	GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();
		float gpy=glbpoint.y();	
		float gpz=glbpoint.z();	
//		if (gpy != 0 && gpz !=0) slopeSum += gpy / gpz;

		if (gpy > 0 && gpz > 0) 	{quad1 = true; quadrant = 1;}
		else if (gpy > 0 && gpz < 0){quad2 = true; quadrant = 2;}
		else if (gpy < 0 && gpz < 0){quad3 = true; quadrant = 3;}
		else if (gpy < 0 && gpz > 0){quad4 = true; quadrant = 4;}
		else return vector<Trajectory>();
      }


  	if(staHits.empty()) return vector<Trajectory>();

  	if (quad1 && quad2) return vector<Trajectory>();
  	if (quad1 && quad3) return vector<Trajectory>();
  	if (quad1 && quad4) return vector<Trajectory>();
  	if (quad2 && quad3) return vector<Trajectory>();
  	if (quad2 && quad4) return vector<Trajectory>();
  	if (quad3 && quad4) return vector<Trajectory>();

	bool doReverse = false;

  	if (quad1) doReverse = SlopeSum(staHits); 
  	if (quad2) doReverse = SlopeSum(staHits); 
  	if (quad3) doReverse = SlopeSum(staHits); 
  	if (quad4) doReverse = SlopeSum(staHits); 


  	if(doReverse){
    	reverse(staHits.begin(),staHits.end());
  	}

  	copy(staHits.begin(),staHits.end(),back_inserter(recHitsForReFit));

///
///
///
///
///


  if(recHitsForReFit.size() < 2) return vector<Trajectory>();

  bool up = recHitsForReFit.back()->globalPosition().y()>0 ? true : false;
  LogTrace(metname) << "Up ? " << up;

  float sumdy = SumDy(recHitsForReFit);


  PropagationDirection propagationDirection = doReverse ? oppositeToMomentum : alongMomentum;
  TrajectoryStateOnSurface firstTSOS = doReverse ? track.outermostMeasurementState() : track.innermostMeasurementState();
  unsigned int innerId = doReverse ? track.track().outerDetId() : track.track().innerDetId();

  LogTrace(metname) << "Prop Dir: " << propagationDirection << " FirstId " << innerId << " firstTSOS " << firstTSOS;

  TrajectorySeed seed(PTrajectoryStateOnDet(),TrajectorySeed::recHitContainer(),propagationDirection);


  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogTrace(metname)<<"Propagation occurring"<<endl;
    firstTSOS = propagator(up, quadrant, sumdy)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    LogTrace(metname)<<"Final destination: " << recHitsForReFit.front()->det()->surface().position() << endl;
    if(!firstTSOS.isValid()){
	  	std::cout<<"Propagation error! Dumping RecHits..."<<std::endl;
		
  		for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = recHitsForReFit.begin();
      		hit !=recHitsForReFit.end(); ++hit){

    		DetId hitId = (*hit)->geographicalId();
    		GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();
      		if(hitId.subdetId() == MuonSubdetId::DT)
			std::cout<< glbpoint << " I am DT " << DTWireId(hitId)<<std::endl;
      		else if (hitId.subdetId() == MuonSubdetId::CSC )
			std::cout<< glbpoint << " I am CSC " << CSCDetId(hitId)<<std::endl;
	 	} 


      LogTrace(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }
  

  vector<Trajectory> && trajectories = fitter(up, quadrant, sumdy)->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogTrace(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory const & trajectoryBW = trajectories.front();
    
  vector<Trajectory> && trajectoriesSM = smoother(up, quadrant, sumdy)->trajectories(trajectoryBW);

  if(trajectoriesSM.empty()){
    LogTrace(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;

}



///decide if the track should be reversed
bool TrackTransformerForCosmicMuons::SlopeSum(const TransientTrackingRecHit::ConstRecHitContainer& tkHits) const{

	bool retval = false;
	float y1 = 0 ;
	float z1 = 0 ;

	bool first = true;

	std::vector<float> py;
	std::vector<float> pz;
//	int quadrant = -1;
	
	float sumdy = 0;
	float sumdz = 0;

  	for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = tkHits.begin();
      hit !=tkHits.end(); ++hit){

     	DetId hitId = (*hit)->geographicalId();
    	GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();
      	if ( hitId.det() != DetId::Muon || hitId.subdetId() == 3 )continue;

		float y2 = glbpoint.y();
		float z2 = glbpoint.z();
		float dslope = 0;
		float dy = y2 - y1;
		float dz = z2 - z1;

//		if (y2 > 0 && z2 > 0) quadrant = 1;
//		else if (y2 > 0 && z2 < 0) quadrant = 2;
//		else if (y2 < 0 && z2 < 0) quadrant = 3;
//		else if (y2 < 0 && z2 > 0) quadrant = 4;


		if (fabs(dz) > 1e-3) dslope = dy / dz;
		if ( !first) {
			retval+=dslope; 
			sumdy +=dy;
			sumdz +=dz;
		}
		first = false;
		py.push_back(y1);
		pz.push_back(z1);
		y1 = y2;
		z1 = z2;
  	}
//	std::cout<<"quadrant "<<quadrant;
//	std::cout<<"\tsum dy = "<<sumdy;
//	std::cout<<"\tsum dz = "<<sumdz;
//	std::cout<<std::endl;


	if ( sumdz < 0) retval = true;
	
	return retval;

}


///decide if the track should be reversed
float TrackTransformerForCosmicMuons::SumDy(const TransientTrackingRecHit::ConstRecHitContainer& tkHits) const{

	bool retval = false;
	float y1 = 0 ;
	float z1 = 0 ;

	bool first = true;

	std::vector<float> py;
	std::vector<float> pz;
//	int quadrant = -1;
	
	float sumdy = 0;
	float sumdz = 0;



  	for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = tkHits.begin();
      hit !=tkHits.end(); ++hit){

     	DetId hitId = (*hit)->geographicalId();
    	GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();
      	if ( hitId.det() != DetId::Muon || hitId.subdetId() == 3 )continue;

		float y2 = glbpoint.y();
		float z2 = glbpoint.z();
		float dslope = 0;
		float dy = y2 - y1;
		float dz = z2 - z1;

//		if (y2 > 0 && z2 > 0) quadrant = 1;
//		else if (y2 > 0 && z2 < 0) quadrant = 2;
//		else if (y2 < 0 && z2 < 0) quadrant = 3;
//		else if (y2 < 0 && z2 > 0) quadrant = 4;


		if (fabs(dz) > 1e-3) dslope = dy / dz;
		if ( !first) {
			retval+=dslope; 
			sumdy +=dy;
			sumdz +=dz;
		}
		first = false;
		py.push_back(y1);
		pz.push_back(z1);
		y1 = y2;
		z1 = z2;
  	}
//	std::cout<<"quadrant "<<quadrant;
//	std::cout<<"\tsum dy = "<<sumdy;
//	std::cout<<"\tsum dz = "<<sumdz;
//	std::cout<<std::endl;

	return sumdy;
}

