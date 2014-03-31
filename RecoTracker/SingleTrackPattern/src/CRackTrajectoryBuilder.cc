//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CRackTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CRackTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CRackTrajectoryBuilder.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"


//#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"

using namespace std;


CRackTrajectoryBuilder::CRackTrajectoryBuilder(const edm::ParameterSet& conf) : conf_(conf) { 
  //minimum number of hits per tracks

  theMinHits=conf_.getParameter<int>("MinHits");
  //cut on chi2
  chi2cut=conf_.getParameter<double>("Chi2Cut");
  edm::LogInfo("CosmicTrackFinder")<<"Minimum number of hits "<<theMinHits<<" Cut on Chi2= "<<chi2cut;

  debug_info=conf_.getUntrackedParameter<bool>("debug", false);
  fastPropagation=conf_.getUntrackedParameter<bool>("fastPropagation", false);
  useMatchedHits=conf_.getUntrackedParameter<bool>("useMatchedHits", true);


  geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");

  

}


CRackTrajectoryBuilder::~CRackTrajectoryBuilder() {
//  delete theInitialState;
}


void CRackTrajectoryBuilder::init(const edm::EventSetup& es, bool seedplus){


//  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
// theInitialState          = new TransientInitialStateEstimator( es,tise_params);

  //services
  es.get<IdealMagneticFieldRecord>().get(magfield);
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
 
  
  if (seedplus) { 	 
    seed_plus=true; 	 
    thePropagator=      new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) ); 	 
    thePropagatorOp=    new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );} 	 
  else {
    seed_plus=false;
    thePropagator=      new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) ); 	
    thePropagatorOp=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  }
  
  theUpdator=       new KFUpdator();
//  theUpdator=       new KFStripUpdator();
  theEstimator=     new Chi2MeasurementEstimator(chi2cut);
  

  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);
  

  RHBuilder=   theBuilder.product();




  theFitter=        new KFTrajectoryFitter(*thePropagator,
					   *theUpdator,	
					   *theEstimator) ;
  

  theSmoother=      new KFTrajectorySmoother(*thePropagatorOp,
					     *theUpdator,	
					     *theEstimator);
  
}


void CRackTrajectoryBuilder::run(const TrajectorySeedCollection &collseed,
				  const SiStripRecHit2DCollection &collstereo,
				  const SiStripRecHit2DCollection &collrphi ,
				  const SiStripMatchedRecHit2DCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const edm::EventSetup& es,
				  edm::Event& e,
				  vector<Trajectory> &trajoutput)
{

  std::vector<Trajectory> trajSmooth;
  std::vector<Trajectory>::iterator trajIter;
  
  TrajectorySeedCollection::const_iterator iseed;
  unsigned int IS=0;
  for(iseed=collseed.begin();iseed!=collseed.end();iseed++){
    bool seedplus=((*iseed).direction()==alongMomentum);
    init(es,seedplus);
    hits.clear();
    trajFit.clear();
    trajSmooth.clear();

    Trajectory startingTraj = createStartingTrajectory(*iseed);
    Trajectory trajTmp; // used for the opposite direction
    Trajectory traj = startingTraj;

   
    //first propagate track in opposite direction ...
    seed_plus = !seed_plus;
    vector<const TrackingRecHit*> allHitsOppsite = SortHits(collstereo,collrphi,collmatched,collpixel, *iseed, true);
    seed_plus = !seed_plus;
    if (allHitsOppsite.size())
      {
	//there are hits which are above the seed,
	//cout << "Number of hits higher than seed " <<allHitsOppsite.size() << endl;	
	
	AddHit( traj, allHitsOppsite, thePropagatorOp);
	
	
	if (debug_info)
	  {
	    cout << "Hits in opposite direction..." << endl;
	    TransientTrackingRecHit::RecHitContainer::const_iterator iHit;
	    for ( iHit = hits.begin(); iHit!=hits.end(); iHit++)
	      {
		cout <<  (**iHit).globalPosition() << endl;
	      }
	  }
	
	
	// now add hist opposite to seed
	//now crate new starting trajectory
	reverse(hits.begin(), hits.end());
	if (thePropagatorOp->propagate(traj.firstMeasurement().updatedState(),
				       tracker->idToDet((hits.front())->geographicalId())->surface()).isValid()){
	  TSOS startingStateNew =  //TrajectoryStateWithArbitraryError()//
	    (thePropagatorOp->propagate(traj.firstMeasurement().updatedState(),
					tracker->idToDet((hits.front())->geographicalId())->surface()));
	  
	  if (debug_info)
	    {
	      cout << "Hits in opposite direction reversed..." << endl;
	      TransientTrackingRecHit::RecHitContainer::const_iterator iHit;
	      for ( iHit = hits.begin(); iHit!=hits.end(); iHit++)
		{
		  cout <<  (**iHit).globalPosition() << endl;
		}
	    }
	  
	  const TrajectorySeed& tmpseed=traj.seed();
	  trajTmp = theFitter->fit(tmpseed,hits, startingStateNew ).front();
	  
	  if(debug_info){
	    cout << "Debugging show fitted hits" << endl;	    
	        std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> > hitsFit= trajTmp.recHits();
	        std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> >::const_iterator hit;
	        for(hit=hitsFit.begin();hit!=hitsFit.end();hit++){
		  
	          cout << RHBuilder->build( &(*(*hit)->hit()) )->globalPosition() << endl;
	        }
	  }
	  
	}
      }
    else 
      {
	if(debug_info) cout << "There are no hits in opposite direction ..." << endl;
      }

    vector<const TrackingRecHit*> allHits;
    if (trajTmp.foundHits())
      {
	traj = trajTmp;
	allHits= SortHits(collstereo,collrphi,collmatched,collpixel,*iseed, false);	  
      }
    else
      {
	traj = startingTraj;
	hits.clear();
	allHits= SortHits(collstereo,collrphi,collmatched,collpixel,*iseed, true);	  
      }
    
    AddHit( traj,allHits, thePropagator);
    
    
    if(debug_info){
      cout << "Debugging show All fitted hits" << endl;	    
      std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> > hits= traj.recHits();
	    std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> >::const_iterator hit;
	    for(hit=hits.begin();hit!=hits.end();hit++){
	      
	      cout << RHBuilder->build( &(*(*hit)->hit()) )->globalPosition() << endl;
	    }

	    cout << qualityFilter( traj) << " <- quality filter good?" << endl;
	  }


    if (debug_info) cout << "now do quality checks" << endl;
    if ( qualityFilter( traj) ) {
      const TrajectorySeed& tmpseed=traj.seed();
      std::pair<TrajectoryStateOnSurface, const GeomDet*> initState =  innerState(traj); //theInitialState->innerState(traj);
	if (initState.first.isValid())
	      trajFit = theFitter->fit(tmpseed,hits, initState.first);
    }

    for (trajIter=trajFit.begin(); trajIter!=trajFit.end();trajIter++){
      trajSmooth=theSmoother->trajectories((*trajIter));
    }
    for (trajIter= trajSmooth.begin(); trajIter!=trajSmooth.end();trajIter++){
      if((*trajIter).isValid()){

	if (debug_info) cout << "adding track ... " << endl;
	trajoutput.push_back((*trajIter));
      }
    }
    delete theUpdator;
    delete theEstimator;
    delete thePropagator;
    delete thePropagatorOp;
    delete theFitter;
    delete theSmoother;
    //Only the first 30 seeds are considered
    if (IS>30) return;
    IS++;

  }
}

Trajectory CRackTrajectoryBuilder::createStartingTrajectory( const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction());
  std::vector<TM> && seedMeas = seedMeasurements(seed);
  for (auto i : seedMeas) result.push(std::move(i));
  return result;
}


std::vector<TrajectoryMeasurement> 
CRackTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    //RC TransientTrackingRecHit* recHit = RHBuilder->build(&(*ihit));
    TransientTrackingRecHit::RecHitPointer recHit = RHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = (&(*tracker))->idToDet( ihit->geographicalId());
    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));

    if (ihit == hitRange.second - 1) {
      TSOS  updatedState=startingTSOS(seed);
      result.push_back(std::move(TM( invalidState, updatedState, recHit)));

    } 
    else {
      result.push_back(std::move(TM( invalidState, recHit)));
    }
    
  }

  return result;
}

vector<const TrackingRecHit*> 
CRackTrajectoryBuilder::SortHits(const SiStripRecHit2DCollection &collstereo,
				  const SiStripRecHit2DCollection &collrphi ,
				  const SiStripMatchedRecHit2DCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const TrajectorySeed &seed,
				  const bool bAddSeedHits
				  ){


  //The Hits with global y more than the seed are discarded
  //The Hits correspondign to the seed are discarded
  //At the end all the hits are sorted in y
  vector<const TrackingRecHit*> allHits;

  SiStripRecHit2DCollection::DataContainer::const_iterator istrip;
  TrajectorySeed::range hRange= seed.recHits();
  TrajectorySeed::const_iterator ihit;
  float yref=0.;

  if (debug_info) cout << "SEED " << startingTSOS(seed) << endl; 
  if (debug_info) cout << "seed hits size " << seed.nHits() << endl;


  //seed debugging:
 // GeomDet *seedDet =  tracker->idToDet(seed.startingState().detId());


//  edm::LogInfo("CRackTrajectoryBuilder::SortHits") << "SEED " << seed.startingState(); 

  edm::LogInfo("CRackTrajectoryBuilder::SortHits") << "SEED " << startingTSOS(seed); 
//  edm::LogInfo("CRackTrajectoryBuilder::SortHits" << "seed hits size " << seed.nHits();



  //  if (seed.nHits()<2)
  //  return allHits;

  float_t yMin=0.;
  float_t yMax=0.;

  int seedHitSize= hRange.second - hRange.first;

  vector <int> detIDSeedMatched (seedHitSize);
  vector <int> detIDSeedRphi (seedHitSize);
  vector <int> detIDSeedStereo (seedHitSize);

  for (ihit = hRange.first; 
       ihit != hRange.second; ihit++) {

     // need to find track with lowest (seed_plus)/ highest y (seed_minus)
    // split matched hits ...
   const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*ihit));

   yref=RHBuilder->build(&(*ihit))->globalPosition().y();
   if (ihit == hRange.first)
     {
        yMin = yref;
	yMax = yref; 
     }

   if (matchedhit)
     {
       auto m = matchedhit->monoHit();
       auto s = matchedhit->stereoHit();
       float_t yGlobRPhi   = RHBuilder->build(&m)->globalPosition().y();
       float_t yGlobStereo = RHBuilder->build(&s)->globalPosition().y();

       if (debug_info) cout << "Rphi ..." << yGlobRPhi << endl;
       if (debug_info) cout << "Stereo ..." << yGlobStereo << endl;

       if ( yGlobStereo < yMin ) yMin = yGlobStereo;
       if ( yGlobRPhi   < yMin ) yMin = yGlobRPhi;

       if ( yGlobStereo > yMax ) yMax = yGlobStereo;
       if ( yGlobRPhi   > yMax ) yMax = yGlobRPhi;

       detIDSeedMatched.push_back (  matchedhit->geographicalId().rawId() );
       detIDSeedRphi.push_back ( m.geographicalId().rawId() );
       detIDSeedStereo.push_back ( s.geographicalId().rawId() );

      if (bAddSeedHits)
	{
	  if (useMatchedHits) 
	    {
	      
	      hits.push_back((RHBuilder->build(&(*ihit)))); 
	    }
	  else
	    {
	      if ( ( (yGlobRPhi > yGlobStereo ) && seed_plus ) || ((yGlobRPhi < yGlobStereo ) && !seed_plus ))  
		{
		  hits.push_back((RHBuilder->build(&m)));
		  hits.push_back((RHBuilder->build(&s)));
		}
	      else
		{
		  hits.push_back((RHBuilder->build(&s)));     
		  hits.push_back((RHBuilder->build(&m)));
		  
		}
	    }
	}
     }
   else if (bAddSeedHits)
     {
       hits.push_back((RHBuilder->build(&(*ihit)))); 
       detIDSeedRphi.push_back ( ihit->geographicalId().rawId() );
       detIDSeedMatched.push_back ( -1 );
       detIDSeedStereo.push_back ( -1 );
       
     }
       
   if ( yref < yMin ) yMin = yref;
   if ( yref > yMax ) yMax = yref;
   
//    if (bAddSeedHits)
//      hits.push_back((RHBuilder->build(&(*ihit)))); 

    LogDebug("CosmicTrackFinder")<<"SEED HITS"<<RHBuilder->build(&(*ihit))->globalPosition();
    if (debug_info) cout <<"SEED HITS"<<RHBuilder->build(&(*ihit))->globalPosition() << endl;
//    if (debug_info) cout <<"SEED HITS"<< seed.startingState().parameters() << endl;


  }
  
  yref = (seed_plus) ? yMin : yMax;
  
  if ((&collpixel)!=0){
    SiPixelRecHitCollection::DataContainer::const_iterator ipix;
    for(ipix=collpixel.data().begin();ipix!=collpixel.data().end();ipix++){
      float ych= RHBuilder->build(&(*ipix))->globalPosition().y();
      if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	allHits.push_back(&(*ipix));
    }
  } 
  

  if (useMatchedHits) // use matched
    {
        //add the matched hits ...
      SiStripMatchedRecHit2DCollection::DataContainer::const_iterator istripm;

      if ((&collmatched)!=0){
	for(istripm=collmatched.data().begin();istripm!=collmatched.data().end();istripm++){
	  float ych= RHBuilder->build(&(*istripm))->globalPosition().y();

	  int cDetId=istripm->geographicalId().rawId();
	  bool noSeedDet = ( detIDSeedMatched.end() == find (detIDSeedMatched.begin(), detIDSeedMatched.end(), cDetId ) ) ;

	  if ( noSeedDet )
	  if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	    {
	  //if (debug_info) cout << "adding matched hit " << &(*istripm) << endl; 
	      allHits.push_back(&(*istripm));
	}
	}
      }

   //add the rpi hits, but only accept hits that are not matched hits
  if ((&collrphi)!=0){
    for(istrip=collrphi.data().begin();istrip!=collrphi.data().end();istrip++){
      float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
      StripSubdetector monoDetId(istrip->geographicalId());
      if (monoDetId.partnerDetId())
	{
	  edm::LogInfo("CRackTrajectoryBuilder::SortHits")  << "this det belongs to a glued det " << ych << endl;
	  continue;
	}
    	  int cDetId=istrip->geographicalId().rawId();
	  bool noSeedDet = ( detIDSeedRphi.end()== find (detIDSeedRphi.begin(), detIDSeedRphi.end(), cDetId ) ) ;
	  if (noSeedDet)
      if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	{
	 
	  bool hitIsUnique = true;
	  //now 
	   if ((&collmatched)!=0)
	     for(istripm=collmatched.data().begin();istripm!=collmatched.data().end();istripm++)
	       {
		 //		 if ( isDifferentStripReHit2D ( *istrip, (istripm->stereoHit() ) ) == false)
		   if ( isDifferentStripReHit2D ( *istrip, (istripm->monoHit() ) ) == false)
		   {
		     hitIsUnique = false;
		     edm::LogInfo("CRackTrajectoryBuilder::SortHits")  << "rphi hit is in matched hits; y: " << ych << endl;
		     break;
		   }
	       } //end loop over all matched
	   if (hitIsUnique)
	     {
	   	 //      if (debug_info) cout << "adding rphi hit " << &(*istrip) << endl; 
	        allHits.push_back(&(*istrip));   
	     }
	}
    }
  }

  
  //add the stereo hits except the hits that are in the matched collection
  //update do not use unmatched rphi hist due to limitation of alignment framework
  //if (!useMatchedHits)
  //if ((&collstereo)!=0){
  //  for(istrip=collstereo.data().begin();istrip!=collstereo.data().end();istrip++){
  //    float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
  //
  //
  // 	  int cDetId = istrip->geographicalId().rawId();
  //        bool noSeedDet = ( detIDSeedStereo.end()== find (detIDSeedStereo.begin(), detIDSeedStereo.end(), cDetId ) ) ;
  //
  //        if (noSeedDet)
  //    if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
  //      {
  //
  //        bool hitIsUnique = true;
  //        //now 
  //         if ((&collmatched)!=0)
  //           for(istripm=collmatched.data().begin();istripm!=collmatched.data().end();istripm++)
  //             {
  //      	 if ( isDifferentStripReHit2D ( *istrip, * (istripm->stereoHit() ) ) == false)
  //      	   {
  //      	     hitIsUnique = false;
  //      	     edm::LogInfo("CRackTrajectoryBuilder::SortHits") << "stereo hit already in matched hits; y:  " << ych << endl;
  //      	     break;
  //      	   }
  //             } //end loop over all stereo
  //         if (hitIsUnique)
  //           {
  //             
  //          //   if (debug_info) cout << "now I am adding a stero hit, either noise or not in overlap ...!!!!" << endl;
  //                allHits.push_back(&(*istrip));   
  //           }
  //      }
  //  }
  //}
    }
  else // dont use matched ...
    {
      
      if ((&collrphi)!=0){
	for(istrip=collrphi.data().begin();istrip!=collrphi.data().end();istrip++){
	  float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
	  if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	    allHits.push_back(&(*istrip));   
	}
      }


      if ((&collstereo)!=0){
	for(istrip=collstereo.data().begin();istrip!=collstereo.data().end();istrip++){
	  float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
	  if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	    allHits.push_back(&(*istrip));
	}
      }

    }


//  if (seed_plus){
//    stable_sort(allHits.begin(),allHits.end(),CompareDetY_plus(*tracker));
//  }
//  else {
//    stable_sort(allHits.begin(),allHits.end(),CompareDetY_minus(*tracker));
//  }


  if (seed_plus){
      stable_sort(allHits.begin(),allHits.end(),CompareHitY_plus(*tracker));
  }
  else {
      stable_sort(allHits.begin(),allHits.end(),CompareHitY(*tracker));
  }



  if (debug_info) 
    {
      if (debug_info) cout << "all hits" << endl;

      //starting trajectory
      Trajectory startingTraj = createStartingTrajectory(seed);

      if (debug_info) cout << "START " << startingTraj.lastMeasurement().updatedState() << endl;
      if (debug_info) cout << "START  Err" << startingTraj.lastMeasurement().updatedState().localError().matrix() << endl;


      vector<const TrackingRecHit*>::iterator iHit;
      for (iHit=allHits.begin(); iHit<allHits.end(); iHit++)
	{
	      GlobalPoint gphit=RHBuilder->build(*iHit)->globalPosition();
	      if (debug_info) cout << "GH " << gphit << endl;

	      //      tracker->idToDet((*iHit)->geographicalId())->surface();

      TSOS prSt = thePropagator->propagate(startingTraj.lastMeasurement().updatedState(),
      			  tracker->idToDet((*iHit)->geographicalId())->surface());
	      
	      if(prSt.isValid()) 
		{
      if (debug_info) cout << "PR " << prSt.globalPosition() << endl;
      //if (debug_info) cout << "PR  Err" << prSt.localError().matrix() << endl;
		  
		}
	      else
		{
		  if (debug_info) cout << "not valid" << endl;
		}
	}
	      if (debug_info) cout << "all hits end" << endl;
    }


  return allHits;
}

TrajectoryStateOnSurface
CRackTrajectoryBuilder::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  TSOS  State= trajectoryStateTransform::transientState( pState, &(gdet->surface()), 
					   &(*magfield));
  return State;

}

void CRackTrajectoryBuilder::AddHit(Trajectory &traj,
                                     const vector<const TrackingRecHit*>& _Hits, Propagator *currPropagator){
   vector<const TrackingRecHit*> Hits = _Hits;
   if ( Hits.size() == 0 )
     return;
  
   if (debug_info) cout << "CRackTrajectoryBuilder::AddHit" << endl;
   if (debug_info) cout << "START " << traj.lastMeasurement().updatedState() << endl;
 
   vector <TrackingRecHitRange> hitRangeByDet;
   TrackingRecHitIterator prevDet;
 
   prevDet = Hits.begin();
   for( TrackingRecHitIterator iHit = Hits.begin(); iHit != Hits.end(); iHit++ )
     {
       if ( (*prevDet)->geographicalId() == (*iHit)->geographicalId() ) 
 	continue;
        
       hitRangeByDet.push_back( make_pair( prevDet, iHit ) );
       prevDet = iHit;
      }
   hitRangeByDet.push_back( make_pair( prevDet, Hits.end() ) );
 
   /// do the old version ....
 
     if (fastPropagation) {
       for( TrackingRecHitRangeIterator iHitRange = hitRangeByDet.begin(); iHitRange != hitRangeByDet.end(); iHitRange++ )
         {
            const TrackingRecHit *currHit = *(iHitRange->first);
 	  DetId      currDet = currHit->geographicalId();
  
           TSOS prSt= currPropagator->propagate(traj.lastMeasurement().updatedState(),
 					      tracker->idToDet(currDet)->surface());
  
           if ( !prSt.isValid())
 	    {
 	      if (debug_info) cout << "Not Valid: PRST" << prSt.globalPosition();
 //	      if (debug_info) cout << "Not Valid: HIT" << *currHit;
 
 
 	      continue;
 	    }
 
           TransientTrackingRecHit::RecHitPointer bestHit = RHBuilder->build( currHit );
           double chi2min = theEstimator->estimate( prSt, *bestHit).second;
 
           if (debug_info) cout << "Size " << iHitRange->first - (*iHitRange).second << endl;
           for( TrackingRecHitIterator iHit = (*iHitRange).first+1; iHit != iHitRange->second; iHit++ )
             {
               if (debug_info) cout << "loop3 " <<" "<< Hits.end() - iHit << endl;
 
               TransientTrackingRecHit::RecHitPointer tmpHit = RHBuilder->build( *iHit );
               double currChi2 = theEstimator->estimate(prSt, *tmpHit).second;
               if ( currChi2 < chi2min )
                 {
                   currChi2 = chi2min;
                   bestHit = tmpHit;
                 }
             }
           //now we have check if the track can be added to the trajectory
           if (debug_info) cout << chi2min << endl;
           if (chi2min < chi2cut)
             {
               if (debug_info) cout << "chi2 fine : " << chi2min << endl;
               TSOS UpdatedState= theUpdator->update( prSt, *bestHit );
               if (UpdatedState.isValid()){
 		hits.push_back(&(*bestHit));
                 traj.push( TM(prSt,UpdatedState, bestHit, chi2min) );
 		if (debug_info) edm::LogInfo("CosmicTrackFinder") <<
                   "STATE UPDATED WITH HIT AT POSITION "
 		  
 					      << bestHit->globalPosition()
                                               <<UpdatedState<<" "
                                               <<traj.chiSquared();
 		if (debug_info) cout  <<
                   "STATE UPDATED WITH HIT AT POSITION "
 		  
 					      << bestHit->globalPosition()
                                               <<UpdatedState<<" "
                                               <<traj.chiSquared();
                 if (debug_info) cout << "State is valid ..." << endl;
 		break; // now we need to 
               }
               else
                 {
 		  edm::LogWarning("CosmicTrackFinder")<<" State can not be updated with hit at position " << endl;
 		  TSOS UpdatedState= theUpdator->update( prSt, *bestHit );
 		  if (UpdatedState.isValid()){
 		    cout  <<
 		      "NOT! UPDATED WITH HIT AT POSITION "
 		      
 			  << bestHit->globalPosition()
 			  <<UpdatedState<<" "
 			  <<traj.chiSquared();
 		    
 		  }
 		}
 	    } 
         }
     } //simple version end
     else 
     {
       //first sort the dets in the order they are traversed by the trajectory
       // we need three loops:
       // 1: loop as long as there can be an new hit added to the trajectory
       // 2: loop over all dets that might be hit
       // 3: loop over all hits on a certain det
       
       
       std::vector < std::pair<TrackingRecHitRangeIterator, TSOS> > trackHitCandidates;
       std::vector <std::pair<TrackingRecHitRangeIterator, TSOS> >::iterator iHitRange;
       std::vector <uint32_t> processedDets;
       do
 	{
 	  
 	  //create vector of possibly hit detectors...
 	  trackHitCandidates.clear();      
 	  DetId currDet;
 	  for( TrackingRecHitRangeIterator iHit = hitRangeByDet.begin(); iHit != hitRangeByDet.end(); iHit++ )
 	    {
 	      const TrackingRecHit *currHit = *(iHit->first);
 	      currDet = currHit->geographicalId();
 
 	      if ( find(processedDets.begin(), processedDets.end(), currDet.rawId()) != processedDets.end() )
 		continue;
 	      
 	      TSOS prSt= currPropagator->propagate(traj.lastMeasurement().updatedState(),
 						tracker->idToDet(currDet)->surface());
 	      if ( ( !prSt.isValid() ) ||  (theEstimator->Chi2MeasurementEstimatorBase::estimate(prSt,tracker->idToDet(currDet)->surface() ) == false) )
 //	      if ( ( !prSt.isValid() ) ||  (theEstimator->estimate(prSt,tracker->idToDet(currDet)->surface() ) == false) )  
 		continue;
     
 	      trackHitCandidates.push_back(  make_pair(iHit, prSt) );
 	    }
 	  
 	  if (!trackHitCandidates.size())
 	  break;
 
 	  if (debug_info) cout << Hits.size() << " (int) trackHitCandidates.begin() " << trackHitCandidates.size() << endl;
 	  if (debug_info) cout << "Before sorting ... " << endl; 
 
 	  if (debug_info)	  
 	  for( iHitRange = trackHitCandidates.begin(); iHitRange != trackHitCandidates.end(); iHitRange++ )
 	    {
 	      if (debug_info) cout << (tracker->idToDet((*(iHitRange->first->first))->geographicalId()))->position();
 	    }
 	  if (debug_info) cout << endl;
 	  
 
 	  stable_sort( trackHitCandidates.begin(), trackHitCandidates.end(), CompareDetByTraj(traj.lastMeasurement().updatedState())  );
 	  
 	  if (debug_info) cout << "After sorting ... " << endl; 
 	  if (debug_info)
 	    {
 	    for( iHitRange = trackHitCandidates.begin(); iHitRange != trackHitCandidates.end(); iHitRange++ )
 	      {
 		if (debug_info) cout << (tracker->idToDet((*(iHitRange->first->first))->geographicalId()))->position();
 	      }
 	    cout << endl;
 	    }
 
 	for( iHitRange = trackHitCandidates.begin(); iHitRange != trackHitCandidates.end(); iHitRange++ )      //loop over dets
 	  {
 	    
 	    //now find the best hit of the detector
 	    if (debug_info) cout << "loop2 " << trackHitCandidates.size()  <<" " << trackHitCandidates.end() - iHitRange << endl;
 	    const TrackingRecHit *currHit = *(iHitRange->first->first);
 	    
 	    TransientTrackingRecHit::RecHitPointer bestHit = RHBuilder->build( currHit );
 	    TSOS currPrSt = (*iHitRange).second;
 	
 	    if (debug_info) cout << "curr position" << bestHit->globalPosition();
 	    for( TrackingRecHitIterator iHit = (*iHitRange).first->first+1; iHit != iHitRange->first->second; iHit++ )
 	      {
 		TransientTrackingRecHit::RecHitPointer tmpHit = RHBuilder->build( *iHit );
 		if (debug_info) cout << "curr position" << tmpHit->globalPosition() ;
 	  
 	      }
 	  }
 	if (debug_info)	cout << "Cross check end ..." << endl;
 
 
 	//just a simple test if the same hit can be added twice ...
 	//      for( iHitRange = trackHitCandidates.begin(); iHitRange != trackHitCandidates.end(); iHitRange++ )      //loop over all hits
 	
 	//      break;
 	
 	for( iHitRange = trackHitCandidates.begin(); iHitRange != trackHitCandidates.end(); iHitRange++ )      //loop over detsall hits
 	  {
 	    
 	    //now find the best hit of the detector
 	    if (debug_info) cout << "loop2 " << trackHitCandidates.size()  <<" " << trackHitCandidates.end() - iHitRange << endl;
 	    
 	    const TrackingRecHit *currHit = *(iHitRange->first->first);
 
 	    processedDets.push_back(currHit->geographicalId().rawId());
 	    
 	    
 	    TransientTrackingRecHit::RecHitPointer bestHit = RHBuilder->build( currHit );
 
 	    if (debug_info) cout << "curr position A" << bestHit->globalPosition() << endl;
 	    TSOS currPrSt = (*iHitRange).second;
 	    double chi2min = theEstimator->estimate( currPrSt, *bestHit).second;
 
 	    if (debug_info) cout << "Size " << iHitRange->first->second - (*iHitRange).first->first << endl; 
 	    for( TrackingRecHitIterator iHit = (*iHitRange).first->first+1; iHit != iHitRange->first->second; iHit++ )
 	      {
 		if (debug_info) cout << "loop3 " <<" "<< Hits.end() - iHit << endl;
 	      
 		TransientTrackingRecHit::RecHitPointer tmpHit = RHBuilder->build( *iHit );
 		if (debug_info) cout << "curr position B" << tmpHit->globalPosition() << endl;
 		double currChi2 = theEstimator->estimate(currPrSt, *tmpHit).second;
 		if ( currChi2 < chi2min )
 		  {
 		    if (debug_info) cout << "Is best hit" << endl;
 		    chi2min = currChi2;
 		    bestHit = tmpHit;
 		  }
 	      }
 	    //now we have checked the det and can remove the entry from the vector...
 	  
 	    //if (debug_info) cout << "before erase ..." << endl;
 	    //this is to slow ...
 	    // hitRangeByDet.erase( (*iHitRange).first,(*iHitRange).first+1 );	  
 	    //if (debug_info) cout << "after erase ..." << endl;
 
 	    if (debug_info) cout << chi2min << endl;
 	    //if the track can be added to the trajectory 
 	    if (chi2min < chi2cut)
 	      {
 		if (debug_info) cout << "chi2 fine : " << chi2min << endl;
 
 		//  if (debug_info) cout << "previaous state " << traj.lastMeasurement().updatedState() <<endl;
 		TSOS UpdatedState= theUpdator->update( currPrSt, *bestHit );
 		if (UpdatedState.isValid()){
 
 		  hits.push_back(&(*bestHit));
 		  traj.push( TM(currPrSt,UpdatedState, bestHit, chi2min) );
 		  if (debug_info) edm::LogInfo("CosmicTrackFinder") <<
 		    "STATE UPDATED WITH HIT AT POSITION "
 		    //   <<tmphitbestdet->globalPosition()
 						<<UpdatedState<<" "
 						<<traj.chiSquared();
 		  if (debug_info) cout << "Added Hit" << bestHit->globalPosition() << endl;
 		  if (debug_info) cout << "State is valid ..." << UpdatedState << endl;
 		  //cout << "updated state " << traj.lastMeasurement().updatedState() <<endl;
 
 		  //	      return; //break;
 		  //
 		  //	      TSOS prSt= currPropagator->propagate(traj.lastMeasurement().updatedState(),
 		  //					      tracker->idToDet( bestHit->geographicalId() )->surface());
 		  //	  
 		  //	      if ( prSt.isValid())
 		  //		cout << "the same hit can be added twice ..." << endl;	
 		  //
 		  
 		  break;
 		}
 		else
 		  {
 		    if (debug_info) edm::LogWarning("CosmicTrackFinder")<<" State can not be updated with hit at position "
 							<< bestHit->globalPosition();  
 		    //		cout << "State can not be updated with hit at " << bestHit->globalPosition() << endl;
 		  }
 		continue;
 	      }
 	    else
 	      {
 		//      cout << "chi2 to big : " << chi2min << endl;
 	      }
 	    if (debug_info) cout << " continue 1 " << endl;
 	  }
 	//now we remove all already processed dets from the list ...
 	// hitRangeByDet.erase( (*trackHitCandidates.begin()).first,(*iHitRange).first+1 );	  
 
 	if (debug_info) cout << " continue 2 " << endl;
       }
     //if this was the last exit
     while ( iHitRange != trackHitCandidates.end() );
     }
 
 
}


bool 
CRackTrajectoryBuilder::qualityFilter(const Trajectory& traj){
  int ngoodhits=0;
  if(geometry=="MTCC"){
    std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> > hits= traj.recHits();
    std::vector< ConstReferenceCountingPointer< TransientTrackingRecHit> >::const_iterator hit;
    for(hit=hits.begin();hit!=hits.end();hit++){
      unsigned int iid=(*hit)->hit()->geographicalId().rawId();
      //CHECK FOR 3 hits r-phi
      if(((iid>>0)&0x3)!=1) ngoodhits++;
    }
  }
  else ngoodhits=traj.foundHits();
  
  if ( ngoodhits >= theMinHits) {
    return true;
  }
  else {
    return false;
  }
}

//----------------------------------------------------------------------------------------------------------
// little helper function that returns false if both hits conatin the same information

bool CRackTrajectoryBuilder::isDifferentStripReHit2D  (const SiStripRecHit2D& hitA, const  SiStripRecHit2D& hitB )
{
  if ( hitA.geographicalId() != hitB.geographicalId() )
    return true;
  if ( hitA.localPosition().x() != hitB.localPosition().x() )
    return true;
  if ( hitA.localPosition().y() != hitB.localPosition().y() )
    return true;
  if ( hitA.localPosition().z() != hitB.localPosition().z() )
    return true;

  //  if (debug_info) cout << hitA.localPosition() << endl;
  //  if (debug_info) cout << hitB << endl;

  return false;
}



//----backfitting
std::pair<TrajectoryStateOnSurface, const GeomDet*> 
CRackTrajectoryBuilder::innerState( const Trajectory& traj) const
{
  int lastFitted = 999;
  int nhits = traj.foundHits();
  if (nhits < lastFitted+1) lastFitted = nhits-1;

  std::vector<TrajectoryMeasurement> measvec = traj.measurements();
  TransientTrackingRecHit::ConstRecHitContainer firstHits;

  bool foundLast = false;
  int actualLast = -99;
  for (int i=lastFitted; i >= 0; i--) {
    if(measvec[i].recHit()->isValid()){
      if(!foundLast){
	actualLast = i; 
	foundLast = true;
      }
      firstHits.push_back( measvec[i].recHit());
    }
  }
  TSOS unscaledState = measvec[actualLast].updatedState();
  AlgebraicSymMatrix55 C=ROOT::Math::SMatrixIdentity();
  // C *= 100.;

  TSOS startingState( unscaledState.localParameters(), LocalTrajectoryError(C),
		      unscaledState.surface(),
		      thePropagator->magneticField());

  // cout << endl << "FitTester starts with state " << startingState << endl;

  KFTrajectoryFitter backFitter( *thePropagator,
				 KFUpdator(),
				 Chi2MeasurementEstimator( 100., 3));

  PropagationDirection backFitDirection = traj.direction() == alongMomentum ? oppositeToMomentum: alongMomentum;

  // only direction matters in this contest
  TrajectorySeed fakeSeed = TrajectorySeed(PTrajectoryStateOnDet() , 
					   edm::OwnVector<TrackingRecHit>(),
					   backFitDirection);

  vector<Trajectory> fitres = backFitter.fit( fakeSeed, firstHits, startingState);

  if (fitres.size() != 1) {
    // cout << "FitTester: first hits fit failed!" << endl;
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement firstMeas = fitres[0].lastMeasurement();
  TSOS firstState = firstMeas.updatedState();

  //  cout << "FitTester: Fitted first state " << firstState << endl;
  //cout << "FitTester: chi2 = " << fitres[0].chiSquared() << endl;

  TSOS initialState( firstState.localParameters(), LocalTrajectoryError(C),
		     firstState.surface(),
		     thePropagator->magneticField());

  return std::pair<TrajectoryStateOnSurface, const GeomDet*>( initialState, 
							      firstMeas.recHit()->det());
}
