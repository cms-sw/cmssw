#include "RecoTracker/SpecialSeedGenerators/interface/SeedFromGenericPairOrTriplet.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

SeedFromGenericPairOrTriplet::SeedFromGenericPairOrTriplet(const edm::ParameterSet& conf,
                                     			   const MagneticField* mf,
                                                           const TrackerGeometry* geom,
							   const TransientTrackingRecHitBuilder* builder,
							   bool momFromPSet):theMagfield(mf), theTracker(geom), theBuilder(builder), theSetMomentum(momFromPSet){}


TrajectorySeed* SeedFromGenericPairOrTriplet::seed(const SeedingHitSet& hits,
                                        	   const PropagationDirection& dir,
                                                   const NavigationDirection&  seedDir,
                                                   const edm::EventSetup& iSetup){
	if (hits.hits().size() == 3) {
		return seedFromTriplet(hits, dir, seedDir, iSetup);
	} else if (hits.hits().size() == 2){
		return seedFromPair(hits, dir, seedDir);
	} else {
		throw cms::Exception("CombinatorialSeedGeneratorForCosmics") << " Wrong number of hits in Set: "
                                                        << hits.hits().size() << ", should be 2 or 3 ";	
	}
}

TrajectorySeed* SeedFromGenericPairOrTriplet::seedFromTriplet(const SeedingHitSet& hits,
							      const PropagationDirection& dir,
							      const NavigationDirection&  seedDir,
							      const edm::EventSetup& iSetup){
	if (hits.hits().size() != 3) {
		throw cms::Exception("CombinatorialSeedGeneratorForCosmics") <<
			"call to SeedFromGenericPairOrTriplet::seedFromTriplet with " << hits.hits().size() << " hits ";
	}

        const TrackingRecHit* innerHit  = (hits.hits())[0].RecHit();
        const TrackingRecHit* middleHit = (hits.hits())[1].RecHit();
        const TrackingRecHit* outerHit  = (hits.hits())[2].RecHit();
        GlobalPoint inner  = theTracker->idToDet(innerHit->geographicalId() )->surface().toGlobal(innerHit->localPosition() );
        GlobalPoint middle = theTracker->idToDet(middleHit->geographicalId())->surface().toGlobal(middleHit->localPosition());
        GlobalPoint outer  = theTracker->idToDet(outerHit->geographicalId() )->surface().toGlobal(outerHit->localPosition() );
	if (theSetMomentum){
		LogDebug("SeedFromGenericPairOrTriplet") << 
			"Using the following hits: outer(r, phi, theta) (" << outer.perp() 
			<< ", " << outer.phi() 
			<< "," << outer.theta() 
			<< ")   middle (" 
			<< middle.perp() << ", " 
			<< middle.phi() << "," 
			<< middle.theta() 
			<<")    inner (" 
			<< inner.perp() 
			<< ", " << inner.phi() 
			<< "," << inner.theta() <<")"
			<< "   (x, y, z)   outer ("  
			<< inner.x() << ", " 
			<< inner.y() << ", " 
			<< inner.z() << ")    middle ("  
			<< middle.x() << ", " 
                        << middle.y() << ", " 
                        << middle.z() << ")";
		SeedingHitSet newSet;
		if (seedDir == outsideIn){
			newSet.add((hits.hits())[1]);
			newSet.add((hits.hits())[2]);
		} else {
			newSet.add((hits.hits())[0]);
                        newSet.add((hits.hits())[1]);
		}
		TrajectorySeed* seed = seedFromPair(newSet, dir, seedDir);
		if (!seed) return 0;
		LogDebug("SeedFromGenericPairOrTriplet") << "about to retrieve free state";
		TrajectoryStateTransform theTransformer;
		PTrajectoryStateOnDet startingState = seed->startingState();
		TrajectoryStateOnSurface theTSOS = theTransformer.transientState(startingState,
                                                               &(theTracker->idToDet(DetId(startingState.detId()))->surface()), 
                                                               &(*theMagfield));
		if (!theTSOS.isValid()){
			edm::LogError("SeedFromGenericPairOrTriplet::seedFromTriplet") << 
					"something wrong: starting TSOS not valid";
			return 0;
		}
		FreeTrajectoryState* freeState = theTSOS.freeState();
		if (!qualityFilter(freeState, hits)) return 0;
		return seed;   
						
	}
	GlobalPoint* firstPoint  = 0;
	GlobalPoint* secondPoint = 0;
	GlobalPoint* thirdPoint  = 0;
	int momentumSign         = 1;
	//const TrackingRecHit* firstHit  = 0;
	//const TrackingRecHit* secondHit = 0;
	//choose the prop dir and hit order accordingly to where the seed is made
	std::vector<const TrackingRecHit*> trHits;
	if (seedDir == outsideIn){
		LogDebug("SeedFromGenericPairOrTriplet") 
			<< "Seed from outsideIn alongMomentum OR insideOut oppositeToMomentum";
		firstPoint = &outer;
		secondPoint = &middle;
		thirdPoint = &inner;
		trHits.push_back(outerHit);
		trHits.push_back(middleHit);
		//firstHit  = outerHit;
		//secondHit = middleHit;
	} else {
		LogDebug("SeedFromGenericPairOrTriplet") 
			<< "Seed from outsideIn oppositeToMomentum OR insideOut alongMomentum";
		firstPoint = &inner;
		secondPoint = &middle;
		thirdPoint = &outer;
		trHits.push_back(innerHit);
		trHits.push_back(middleHit);
		//firstHit  = innerHit;
		//secondHit = middleHit;
        }
	if (dir == oppositeToMomentum) momentumSign = -1; 
	FastHelix helix(*thirdPoint, *secondPoint, *firstPoint, iSetup);
        FreeTrajectoryState originalStartingState = helix.stateAtVertex();
	LogDebug("SeedFromGenericPairOrTriplet") << "originalStartingState " << originalStartingState;
        GlobalTrajectoryParameters originalPar = originalStartingState.parameters();
        GlobalTrajectoryParameters newPar = GlobalTrajectoryParameters(*secondPoint, //originalPar.position(),
                                                                       momentumSign*originalPar.momentum(),
                                                                       originalPar.charge(),
                                                                       &originalPar.magneticField());
	
	/*FastCircle helix(*thirdPoint, *secondPoint, *firstPoint);
	GlobalTrajectoryParameters newPar = GlobalTrajectoryParameters(*secondPoint,
                                                                       momentumSign*originalPar.momentum(),
                                                                       originalPar.charge(),
                                                                       &originalPar.magneticField());*/
        FreeTrajectoryState* startingState = new FreeTrajectoryState(newPar, initialError(trHits[1]));
	if (!qualityFilter(startingState, hits)) return 0;
	TrajectorySeed* seed = buildSeed(startingState, trHits, dir);
	delete startingState;
	return seed;	
}

TrajectorySeed* SeedFromGenericPairOrTriplet::seedFromPair(const SeedingHitSet& hits,
                                                           const PropagationDirection& dir,
							   const NavigationDirection&  seedDir){
	if (hits.hits().size() != 2) {
                throw cms::Exception("CombinatorialSeedGeneratorForCosmics") <<
                        "call to SeedFromGenericPairOrTriplet::seedFromPair with " << hits.hits().size() << " hits ";
        }
	const TrackingRecHit* innerHit = (hits.hits())[0].RecHit();
        const TrackingRecHit* outerHit = (hits.hits())[1].RecHit();
        GlobalPoint inner  = theTracker->idToDet(innerHit->geographicalId() )->surface().toGlobal(innerHit->localPosition() );
        GlobalPoint outer  = theTracker->idToDet(outerHit->geographicalId() )->surface().toGlobal(outerHit->localPosition() );
	LogDebug("SeedFromGenericPairOrTriplet") <<
                        "Using the following hits: outer(r, phi, theta) (" << outer.perp()
                        << ", " << outer.phi()
                        << "," << outer.theta()
                        <<")    inner ("
                        << inner.perp()
                        << ", " << inner.phi()
                        << "," << inner.theta() <<")";
	GlobalPoint* firstPoint  = 0;
        GlobalPoint* secondPoint = 0;
        int momentumSign         = 1;
        //const TrackingRecHit* firstHit  = 0;
        //const TrackingRecHit* secondHit = 0;
	std::vector<const TrackingRecHit*> trHits;
        //choose the prop dir and hit order accordingly to where the seed is made
        if (seedDir == outsideIn){
		LogDebug("SeedFromGenericPairOrTriplet")
                        << "Seed from outsideIn alongMomentum OR insideOut oppositeToMomentum";
                firstPoint = &outer;
                secondPoint = &inner;
                //firstHit  = outerHit;
		//secondHit = innerHit;
		trHits.push_back(outerHit);
		trHits.push_back(innerHit);
        } else {
		LogDebug("SeedFromGenericPairOrTriplet")
                        << "Seed from outsideIn oppositeToMomentum OR insideOut alongMomentum";
                firstPoint = &inner;
                secondPoint = &outer;
                momentumSign = -1;
                //firstHit  = innerHit;
		//secondHit = outerHit;
		trHits.push_back(innerHit);
		trHits.push_back(outerHit);
        }
	if (dir == oppositeToMomentum) momentumSign = -1;
	GlobalVector momentum = momentumSign*theP*(*secondPoint-*firstPoint).unit();
        GlobalTrajectoryParameters gtp(*secondPoint,
                                       momentum,
                                       -1,
                                       &(*theMagfield));
        FreeTrajectoryState* startingState = new FreeTrajectoryState(gtp,initialError(trHits[1]));
	if (!qualityFilter(startingState, hits)) return 0;
        //TrajectorySeed* seed = buildSeed(startingState, firstHit, dir);
        TrajectorySeed* seed = buildSeed(startingState, trHits, dir);
        delete startingState;
        return seed;
}


TrajectorySeed* SeedFromGenericPairOrTriplet::buildSeed(const FreeTrajectoryState* startingState, 
							//const TrackingRecHit* firsthit,
							std::vector<const TrackingRecHit*>& trHits,
							const PropagationDirection& dir){
	//retrieve the surface of the last hit
	const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(trHits[1]);
	const SiStripRecHit2D* hit =               dynamic_cast<const SiStripRecHit2D*>(trHits[1]);
	const BoundPlane* plane = 0;
	if (matchedhit){
		const GluedGeomDet * stripdet=(const GluedGeomDet*)theTracker->idToDet(matchedhit->geographicalId());
		plane = &(stripdet->surface());
	} else if (hit){
		const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)theTracker->idToDetUnit(hit->geographicalId());
		plane = &(stripdet->surface());
	}
	if (!plane) {
		edm::LogError("SeedFromGenericPairOrTriplet::seeds") << 
			"Not a SiStripMatchedRecHit2D or SiStripRecHit2D used";
		return 0;
	}
	//debug
	GlobalPoint first = theTracker->idToDet(trHits[0]->geographicalId() )->surface().toGlobal(trHits[0]->localPosition() );
        GlobalPoint second  = theTracker->idToDet(trHits[1]->geographicalId() )->surface().toGlobal(trHits[1]->localPosition() );
        LogDebug("SeedFromGenericPairOrTriplet") <<
                        "Using the following hits: first(r, phi, theta) (" << first.perp()
                        << ", " << first.phi()
                        << "," << first.theta()
                        <<")    second ("
                        << second.perp()
                        << ", " << second.phi()
                        << "," << second.theta() <<")";

	TrajectoryStateOnSurface seedTSOS(*startingState, *plane);
	LogDebug("SeedFromGenericPairOrTriplet") << "starting TSOS " << seedTSOS ;
	PTrajectoryStateOnDet *PTraj=
		theTransformer.persistentState(seedTSOS, trHits[1]->geographicalId().rawId());
	edm::OwnVector<TrackingRecHit> seed_hits;
	//build the transientTrackingRecHit for the starting hit, then call the method clone to rematch if needed
	std::vector<const TrackingRecHit*>::const_iterator ihits;
	for (ihits = trHits.begin(); ihits != trHits.end(); ihits++){
		seed_hits.push_back((*ihits)->clone());
	}
	TrajectorySeed* seed = new TrajectorySeed(*PTraj,seed_hits,dir);
	delete PTraj;
	return seed;
}

CurvilinearTrajectoryError SeedFromGenericPairOrTriplet::initialError(const TrackingRecHit* rechit) {
        TransientTrackingRecHit::RecHitPointer transHit =  theBuilder->build(rechit);
        AlgebraicSymMatrix C(5,1);
        C*=0.1;
        if (theSetMomentum){
                C[0][0]=1/theP;
                C[3][3]=transHit->globalPositionError().cxx();
                C[4][4]=transHit->globalPositionError().czz();
        } else {
                float zErr = transHit->globalPositionError().czz();
                float transverseErr = transHit->globalPositionError().cxx(); // assume equal cxx cyy
                C[3][3] = transverseErr;
                C[4][4] = zErr;
        }

        return CurvilinearTrajectoryError(C);
}

bool SeedFromGenericPairOrTriplet::qualityFilter(const FreeTrajectoryState* startingState,
                                                         const SeedingHitSet& hits){
        if (theSetMomentum){
                if (hits.hits().size()==3){
                        std::vector<GlobalPoint> gPoints;
                        SeedingHitSet::Hits::const_iterator iHit;
                        for (iHit = hits.hits().begin(); iHit != hits.hits().end(); iHit++){
                                gPoints.push_back(theTracker->idToDet(iHit->RecHit()->geographicalId() )->surface().toGlobal(iHit->RecHit()->localPosition() ));
                        }
                        unsigned int subid=hits.hits().front().RecHit()->geographicalId().subdetId();
			if(subid == StripSubdetector::TEC || subid == StripSubdetector::TID){
                                LogDebug("SeedFromGenericPairOrTriplet") 
					<< "In the endcaps we cannot decide if hits are aligned with only phi and z";
				return true;
                        }
                        FastCircle circle(gPoints[0],
                                          gPoints[1],
                                          gPoints[2]);
                        if (circle.rho() < 500 && circle.rho() != 0) {
                                edm::LogVerbatim("SeedFromGenericPairOrTriplet::qualityFilter") <<
                                        "Seed qualityFilter rejected because rho = " << circle.rho();
                                return false;
                        }

                }
        } else {
                if (startingState->momentum().perp() < theP){
                        edm::LogVerbatim("SeedFromGenericPairOrTriplet::qualityFilter") <<
                                        "Seed qualityFilter rejected because too low pt";
                        return false;
                }
        }
        return true;
}

