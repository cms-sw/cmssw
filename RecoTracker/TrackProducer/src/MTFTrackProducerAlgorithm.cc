#include "RecoTracker/TrackProducer/interface/MTFTrackProducerAlgorithm.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdatorMTF.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MeasurementByLayerGrouper.h"  //added
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrackFilterHitCollector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Utilities/General/interface/CMSexception.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"

void MTFTrackProducerAlgorithm::runWithCandidate(const TrackingGeometry * theG,
					         const MagneticField * theMF,
					         //const TrackCandidateCollection& theTCCollection,
						 const std::vector<Trajectory>& theTrajectoryCollection,

					         const TrajectoryFitter * theFitter,
					         const TransientTrackingRecHitBuilder* builder,
						 const MultiTrackFilterHitCollector* measurementCollector,
						 const SiTrackerMultiRecHitUpdatorMTF* updator,
						 const reco::BeamSpot& bs,
					         AlgoProductCollection& algoResults) const
{
  edm::LogInfo("MTFTrackProducer") << "Number of Trajectories: " << theTrajectoryCollection.size() << "\n";

  int cont = 0;
  float ndof = 0;
  //a for cicle to get a vector of trajectories, for building the vector<MTM> 
  std::vector<Trajectory> mvtraj=theTrajectoryCollection; 


  
  //now we create a map, in which we store a vector<TrajMeas> for each trajectory in the event
  std::map<int, std::vector<TrajectoryMeasurement> > mvtm;
  int a=0;
  
  for(std::vector<Trajectory>::iterator imvtraj=mvtraj.begin(); imvtraj!=mvtraj.end(); imvtraj++)
    //for(int a=1; a<=mvtraj.size(); a++)
    {
      
      Trajectory *traj = &(*imvtraj);
      mvtm.insert( make_pair(a, traj->measurements()) );
      a++;
    }
  
  LogDebug("MTFTrackProducerAlgorithm") << "after the cicle found " << mvtraj.size() << " trajectories"  << std::endl;
  LogDebug("MTFTrackProducerAlgorithm") << "built a map of " << mvtm.size() << " elements (trajectories, yeah)"  << std::endl;
  
  //here we do a first fit with annealing factor 1, and we collect the hits for the first time 
  std::vector<std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> > vecmrhits;
  std::vector<Trajectory> transientmvtraj;
  int b=0;
  for(std::vector<Trajectory>::const_iterator im=mvtraj.begin(); im!=mvtraj.end(); im++)
    
    {
      
      LogDebug("MTFTrackProducerAlgorithm") << "about to collect hits for the trajectory number " << b << std::endl;
      
      //collect hits and put in the vector vecmrhits to initialize it
      std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> hits = collectHits(mvtm, measurementCollector, b);
      vecmrhits.push_back(hits);
      
      LogDebug("MTFTrackProducerAlgorithm") << "hits collected";
      
      //build a vector of 1 element, to re-use the same method of DAF 
      std::vector<Trajectory> vtraj(1,(*im)); 
      
      //do the fit, taking a vector of trajectories with only an element 
      bool fitresult = fit(hits, theFitter, vtraj);
      LogDebug("MTFTrackProducerAlgorithm") << "fit done";
      
      //extract the element trajectory from the vector,if it's not empty
      if(fitresult == true)
	{
	  LogDebug("MTFTrackProducerAlgorithm") << "fit was good for trajectory number" << b << "\n"
						<< "the trajectory has size" << vtraj.size() << "\n"
						<< "and its number of measurements is: " << vtraj.front().measurements().size() << "\n";
	  Trajectory thetraj = vtraj.front();
	
	  //put this trajectory in the vector of trajectories that will become the new mvtraj after the for cicle
	  transientmvtraj.push_back(thetraj);
	}
      
      //else
      //{
      //  LogDebug("MTFTrackProducerAlgorithm") << "KFSmoother returned a broken trajectory, keeping the old one" << b << "\n"; 
      //    transientmvtraj.push_back(*im);
      //}
      
      b++;
    }
  
  LogDebug("MTFTrackProducerAlgorithm") << "cicle finished";
  
  //replace the mvtraj with a new one done with MultiRecHits...
  mvtraj.swap(transientmvtraj);
  LogDebug("MTFTrackProducerAlgorithm") << " with " << mvtraj.size() << "trajectories\n"; 

  
 

  //for cicle upon the annealing factor, with an update of the MRH every annealing step
  for(std::vector<double>::const_iterator ian = updator->getAnnealingProgram().begin(); ian != updator->getAnnealingProgram().end(); ian++)
    
    {
      
      //creates a transientmvtm taking infos from the mvtraj vector
      std::map<int, std::vector<TrajectoryMeasurement> > transientmvtm1;
      int b=0;
      
      for(std::vector<Trajectory>::iterator imv=mvtraj.begin(); imv!=mvtraj.end(); imv++)
	{
	  Trajectory *traj = &(*imv);
	  transientmvtm1.insert( make_pair(b,traj->measurements()) );
	  b++;
	}
      
      //subs the old mvtm with a new one which takes into account infos from the previous iteration step 
      mvtm.swap(transientmvtm1);
      
      //update the vector with the hits to be used at succesive annealing step
      vecmrhits.clear();

      
      for (uint d=0; d<mvtraj.size(); d++)
	{
	  
	  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> 
	    curiterationhits = updateHits(mvtm, measurementCollector, updator, *ian, builder, d);
	  
	  vecmrhits.push_back(curiterationhits);
	  
	}
      
      

      LogDebug("MTFTrackProducerAlgorithm") << "vector vecmrhits has size " 
					    << vecmrhits.size() << std::endl; 
      




      //define a transient vector to replace the original mvmtraj with
      std::vector<Trajectory> transientmvtrajone;
      
      int n=0;
      //for cicle on the trajectories of the vector mvtraj
      for(std::vector<Trajectory>::const_iterator j=mvtraj.begin(); j!=mvtraj.end(); j++)
	{
	  
	  //create a vector of 1 element vtraj
	  std::vector<Trajectory> vtraj(1,(*j));
	  
	  if(vtraj.size()){
	    
	    LogDebug("MTFTrackProducerAlgorithm") << "Seed direction is " << vtraj.front().seed().direction() 
						  << "Traj direction is " << vtraj.front().direction(); 
	    
	    
	  
	    //~~~~update with annealing factor (updator modified with respect to the DAF)~~~~ 
	    //std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> 
	    //  curiterationhits = updateHits(mvtm, measurementCollector, updator, *ian, builder, n);
	    
	    //fit with multirechits
	    fit(vecmrhits[n], theFitter, vtraj);
	    
	    //std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> 
	    // curiterationhits = updateHits(mvtm, measurementCollector, updator, *ian, builder, n);
	  
	    //LogDebug("MTFTrackProducerAlgorithm") << "done annealing value "  <<  (*ian) << " with " << vtraj.size() << " trajectories";
	  } 
	  
	  else 
	    
	    {
	      LogDebug("MTFTrackProducerAlgorithm") << "in map making skipping trajectory number: "  << n <<"\n";
	      continue;
	    }
	  
	  //extract the trajectory from the vector vtraj
	  Trajectory thetraj = vtraj.front();
	  
	  //check if the fit has succeded
	  //if(!vtraj.empty())
	  //{	
	  //  LogDebug("MTFTrackProducerAlgorithm") << "Size correct " << "\n";
	  
	  //if the vector vtraj is not empty add a trajectory to the vector
	  transientmvtrajone.push_back(thetraj);
	  //  }
	  //if vtraj is empty fill the vector with the trajectory coming from the previous annealing step
	  //else 
	  //{
	  //LogDebug("MTFTrackProducerAlgorithm") << "trajectory broken by the smoother, keeping the old one " <<"\n";
	  //transientmvtrajone.push_back(*j);
	  //}
	  
	  n++;
	}
      
      //substitute the vector mvtraj with the transient one
      mvtraj.swap(transientmvtrajone);
      
      
      
      LogDebug("MTFTrackProducerAlgorithm") << "number of trajectories after annealing value " 
					    << (*ian) 
					    << " annealing step " 
					    << mvtraj.size() << std::endl;
    }
  
  //std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> 
  //  curiterationhits = updateHits(mvtm, measurementCollector, updator, *ian, builder, n);
  
  LogDebug("MTFTrackProducerAlgorithm") << "Ended annealing program with " << mvtraj.size() << " trajectories" << std::endl;
  
  
  //define a transient vector to replace the original mvmtm with
//std::vector<Trajectory> transientmvtm1;
  
//for on the trajectories of the vector mvtraj
  for(std::vector<Trajectory>::iterator it=mvtraj.begin(); it!=mvtraj.end();it++){
    
    std::vector<Trajectory> vtraj(1,(*it));
    
    //check the trajectory to see that the number of valid hits with 
    //reasonable weight (>1e-6) is higher than the minimum set in the DAFTrackProducer.
    //This is a kind of filter to remove tracks with too many outliers.
    //Outliers are mrhits with all components with weight less than 1e-6 
    
    //std::vector<Trajectory> filtered;
    //filter(theFitter, vtraj, conf_.getParameter<int>("MinHits"), filtered);				
    //ndof = calculateNdof(filtered);
    ndof = calculateNdof(vtraj);
    //bool ok = buildTrack(filtered, algoResults, ndof, bs);
    bool ok = buildTrack(vtraj, algoResults, ndof, bs);
    if(ok) cont++;
    
  }
  
  edm::LogInfo("TrackProducer") << "Number of Tracks found: " << cont << "\n";
  
}





//modified
std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> 
MTFTrackProducerAlgorithm::collectHits(const std::map<int, std::vector<TrajectoryMeasurement> >& vtm, 
				       const MultiTrackFilterHitCollector* measurementCollector,
				       int i) const{
  
  TransientTrackingRecHit::RecHitContainer hits;
  
  //build a vector of recHits with a particular mesurementCollector...in this case the SimpleMTFCollector
  std::vector<TrajectoryMeasurement> collectedmeas = measurementCollector->recHits(vtm,i,1.);
  LogDebug("MTFTrackProducerAlgorithm") << "hits collected by the MTF Measurement Collector " << std::endl
					<< "trajectory number " << i << "has measurements" << collectedmeas.size();
  
  if (collectedmeas.empty()) {
    LogDebug("MTFTrackProducerAlgorithm") << "In method collectHits, we had a problem and no measurements were collected "  
					  <<"for trajectory number " << i << std::endl;
    return std::make_pair(TransientTrackingRecHit::RecHitContainer(), TrajectoryStateOnSurface());
  }  
  //put the collected MultiRecHits in a vector
  for (std::vector<TrajectoryMeasurement>::const_iterator iter = collectedmeas.begin(); iter!=collectedmeas.end(); iter++){
    hits.push_back(iter->recHit());
    if(iter->recHit()->isValid())
    LogDebug("MTFTrackProducerAlgorithm") << "this MultiRecHit has size: " << iter->recHit()->recHits().size();  
  }
  //make a pair with the infos about tsos and hit, with the tsos with arbitrary error (to do the fit better)
  return std::make_pair(hits,TrajectoryStateWithArbitraryError()(collectedmeas.front().predictedState()));	
}

//modified
std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>
MTFTrackProducerAlgorithm::updateHits(const std::map<int, std::vector<TrajectoryMeasurement> >& mapvtm,
				      const MultiTrackFilterHitCollector* measurementCollector, 
				      const SiTrackerMultiRecHitUpdatorMTF* updator,
				      double annealing,
				      const TransientTrackingRecHitBuilder* builder,
				      int i) const {
  using namespace std;
  
  std::map< int, std::vector<TrajectoryMeasurement> >::const_iterator itmeas = mapvtm.find(i);
  LogDebug("SimpleMTFHitCollector") << "found the element "<< i 
				    << "in the map" << std::endl;
  
  std::vector<TrajectoryMeasurement> meas = itmeas->second;
  TransientTrackingRecHit::RecHitContainer multirechits;
  
  if (meas.empty()) {LogDebug("MTFTrackProducerAlgorithm::updateHits") << "Warning!!!The trajectory measurement vector is empty" << "\n";}
  
  for(vector<TrajectoryMeasurement>::reverse_iterator imeas = meas.rbegin(); imeas != meas.rend(); imeas++)
    
    {
      if( imeas->recHit()->isValid() ) //&& imeas->recHit()->recHits().size() )
	
	{
	  
	  
	  std::vector<const TrackingRecHit*> trechit = imeas->recHit()->recHits();
	  // if ( typeid( *(imeas->recHit()) ) == typeid(TSiTrackerMultiRecHit) )  
	  //  LogDebug("SimpleMTFHitCollector") << "Hi, I am a MultiRecHit, and you? "<< "\n"; 
	  // else LogDebug("MTFTrackProducerAlgorithm::updateHits") << "rechit type: " << imeas->recHit()->getType() << "\n";
	  
	  LogDebug("MTFTrackProducerAlgorithm::updateHits") << "multirechit vector size: " << trechit.size() << "\n";
	  
	  TransientTrackingRecHit::RecHitContainer hits;
	  
	  for (vector<const TrackingRecHit*>::iterator itrechit=trechit.begin(); itrechit!=trechit.end(); itrechit++)
	    {
	      
	      hits.push_back( builder->build(*itrechit));
	      //  LogDebug("MTFTrackProducerAlgorithm") << "In updateHits method: RecHits transformed from tracking to transient "
	      //					    << "in the detector with detid: " << (*itrechit)->geographicalId().rawId() << "\n";
	    }
	  
	  MultiTrajectoryMeasurement multitm =  measurementCollector->TSOSfinder(mapvtm, *imeas, i);
	  
	  TrajectoryStateCombiner statecombiner;
	  TrajectoryStateOnSurface combinedtsos = statecombiner.combine(imeas->predictedState(), imeas->backwardPredictedState());
	  TrajectoryStateOnSurface predictedtsos = imeas->predictedState();	  
	  
	  //the method is the same of collectHits, but with an annealing factor, different from 1.0
	  //TransientTrackingRecHit::RecHitContainer hits;
	  TransientTrackingRecHit::RecHitPointer mrh = updator->buildMultiRecHit(combinedtsos, hits, &multitm, annealing);
	  
	  //passing the predicted state, should be the combined one (to be modified)
	  //TransientTrackingRecHit::RecHitPointer mrh = updator->buildMultiRecHit(predictedtsos, hits, &multitm, annealing);
	  
	  multirechits.push_back(mrh);
	  
	  
	}
      else 
	{
	  
	  multirechits.push_back(imeas->recHit());
	  
	}
      
      
    }

  
  return std::make_pair(multirechits,TrajectoryStateWithArbitraryError()(itmeas->second.back().predictedState()));
  
}


//method that make fit
bool MTFTrackProducerAlgorithm::fit(const std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>& hits, 
				    const TrajectoryFitter * theFitter,
				    std::vector<Trajectory>& vtraj) const {
  
  std::vector<Trajectory> newVec = theFitter->fit(TrajectorySeed(PTrajectoryStateOnDet(),
								 BasicTrajectorySeed::recHitContainer(),
								 vtraj.front().seed().direction()),
						  hits.first,
						  hits.second);

  //here we control if the fit-smooth round doesn't return an empty trajectory; if not we store the trajectory in vtraj
  if(newVec.size())
    {
      vtraj.reserve(newVec.size());
      vtraj.swap(newVec);
      LogDebug("MTFTrackProducerAlgorithm") << "swapped!" << std::endl;
    }
  
  //if the size of the trajectory is 0 we don't do anything leaving the old trajectory
  else
    {
      LogDebug("MTFTrackProducerAlgorithm") <<" somewhwere, something went terribly wrong...in fitting or smoothing trajectory with measurements:" 
					    << vtraj.front().measurements().size()
					    <<" was broken\n. We keep the old trajectory"
					    << std::endl;
      return false;
    }

  return true;  

}


bool MTFTrackProducerAlgorithm::buildTrack(const std::vector<Trajectory>& vtraj,
					   AlgoProductCollection& algoResults,
					   float ndof,
					   const reco::BeamSpot& bs) const{
  //variable declarations
  reco::Track * theTrack;
  Trajectory * theTraj; 
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  
  
  //LogDebug("TrackProducer") <<" FITTER FOUND "<< vtraj.size() << " TRAJECTORIES" <<"\n";
  LogDebug("MTFTrackProducerAlgorithm") <<" FITTER FOUND "<< vtraj.size() << " TRAJECTORIES" << std::endl;;
  TrajectoryStateOnSurface innertsos;
  
  if (vtraj.size() != 0){
    
    theTraj = new Trajectory( vtraj.front() );
    
    if (theTraj->direction() == alongMomentum) {
      //if (theTraj->direction() == oppositeToMomentum) {
      innertsos = theTraj->firstMeasurement().updatedState();
      //std::cout << "Inner momentum " << innertsos.globalParameters().momentum().mag() << std::endl;	
    } else { 
      innertsos = theTraj->lastMeasurement().updatedState();
    }
       
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

    if (tscbl.isValid()==false) return false;

    GlobalPoint v = tscbl.trackStateAtPCA().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscbl.trackStateAtPCA().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    LogDebug("TrackProducer") <<v<<p<<std::endl;

    theTrack = new reco::Track(theTraj->chiSquared(),
			       ndof, //in the DAF the ndof is not-integer
			       pos, mom, tscbl.trackStateAtPCA().charge(), tscbl.trackStateAtPCA().curvilinearError());


    LogDebug("TrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,std::make_pair(theTrack,theTraj->direction()));
    LogDebug("TrackProducer") <<"track done1\n";
    algoResults.push_back(aProduct);
    LogDebug("TrackProducer") <<"track done2\n";
    
    return true;
  } 
  else  return false;
}

void MTFTrackProducerAlgorithm::filter(const TrajectoryFitter* fitter, 
				       std::vector<Trajectory>& input, 
				       int minhits, 
				       std::vector<Trajectory>& output) const {
  if (input.empty()) return;
  
  int ngoodhits = 0;
  
  std::vector<TrajectoryMeasurement> vtm = input[0].measurements();	
  
  TransientTrackingRecHit::RecHitContainer hits;
  
  //count the number of non-outlier and non-invalid hits	
  for (std::vector<TrajectoryMeasurement>::reverse_iterator tm=vtm.rbegin(); tm!=vtm.rend();tm++){
	  //if the rechit is valid
    if (tm->recHit()->isValid()) {
      TransientTrackingRecHit::ConstRecHitContainer components = tm->recHit()->transientHits();
      bool isGood = false;
      for (TransientTrackingRecHit::ConstRecHitContainer::iterator rechit = components.begin(); rechit != components.end(); rechit++){
	//if there is at least one component with weight higher than 1e-6 then the hit is not an outlier
	if ((*rechit)->weight()>1e-6) {ngoodhits++; isGood = true; break;}
      }
      if (isGood) hits.push_back(tm->recHit()->clone(tm->updatedState()));
	    else hits.push_back(InvalidTransientRecHit::build(tm->recHit()->det(), TrackingRecHit::missing));
    } else {
      hits.push_back(tm->recHit()->clone(tm->updatedState()));
    }
	}
  
  
  LogDebug("DAFTrackProducerAlgorithm") << "Original number of valid hits " << input[0].foundHits() << "; after filtering " << ngoodhits;
  //debug
  if (ngoodhits>input[0].foundHits()) edm::LogError("DAFTrackProducerAlgorithm") << "Something wrong: the number of good hits from DAFTrackProducerAlgorithm::filter " << ngoodhits << " is higher than the original one " << input[0].foundHits();
  
  if (ngoodhits < minhits) return;	
  
  TrajectoryStateOnSurface curstartingTSOS = input.front().lastMeasurement().updatedState();
  LogDebug("DAFTrackProducerAlgorithm") << "starting tsos for final refitting " << curstartingTSOS ;
        //curstartingTSOS.rescaleError(100);
  
  output = fitter->fit(TrajectorySeed(PTrajectoryStateOnDet(),
				      BasicTrajectorySeed::recHitContainer(),
				      input.front().seed().direction()),
		       hits,
		       TrajectoryStateWithArbitraryError()(curstartingTSOS));
  LogDebug("DAFTrackProducerAlgorithm") << "After filtering " << output.size() << " trajectories";
  
}

float MTFTrackProducerAlgorithm::calculateNdof(const std::vector<Trajectory>& vtraj) const {
	if (vtraj.empty()) return 0;
	float ndof = 0;
	int nhits=0;
	const std::vector<TrajectoryMeasurement>& meas = vtraj.front().measurements();
	for (std::vector<TrajectoryMeasurement>::const_iterator iter = meas.begin(); iter != meas.end(); iter++){
		if (iter->recHit()->isValid()){
		  TransientTrackingRecHit::ConstRecHitContainer components = iter->recHit()->transientHits();
		  TransientTrackingRecHit::ConstRecHitContainer::const_iterator iter2;
		  for (iter2 = components.begin(); iter2 != components.end(); iter2++){
		    if ((*iter2)->isValid()){ndof+=((*iter2)->dimension())*(*iter2)->weight();
			LogDebug("DAFTrackProducerAlgorithm") << "hit dimension: "<<(*iter2)->dimension()
			<<" and weight: "<<(*iter2)->weight();
			nhits++;
			}
		  }
		}
	}
	
	LogDebug("DAFTrackProducerAlgorithm") <<"nhits: "<<nhits<< " ndof: "<<ndof-5;
	return ndof-5;
}
