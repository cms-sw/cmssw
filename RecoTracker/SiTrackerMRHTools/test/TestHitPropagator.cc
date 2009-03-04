#include "RecoTracker/SiTrackerMRHTools/test/TestHitPropagator.h"
#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"

using namespace std;
using namespace edm;

TestHitPropagator::TestHitPropagator(const edm::ParameterSet& conf): theConf(conf){
}

void TestHitPropagator::beginRun(edm::Run & run, const edm::EventSetup& c){}

TestHitPropagator::~TestHitPropagator(){}

void TestHitPropagator::endJob(){}

void TestHitPropagator::analyze(const edm::Event& e, const edm::EventSetup& c){
	//get the trajectory collection 
	Handle<std::vector<Trajectory> > trajectoryCollectionHandle;
	InputTag trajcollname = theConf.getParameter<InputTag>("Tracks");
	e.getByLabel(trajcollname, trajectoryCollectionHandle);
	//get the hit propagator
	ESHandle<TrackingRecHitPropagator> hitPropagatorHandle;
	string hitPropagatorName = theConf.getParameter<string>("HitPropagator");
	c.get<TrackingComponentsRecord>().get(hitPropagatorName, hitPropagatorHandle);	
	//to test the hit propagator:
	//for each trajectory project all the hits
	//on the first hit surface, using the first TrajectoryState
	std::vector<Trajectory>::const_iterator itraj;
	for (itraj = trajectoryCollectionHandle->begin(); itraj != trajectoryCollectionHandle->end(); itraj++){
		LogInfo("TestHitPropagator")<< "------Begin Trajectory------";
		testHitPropagator(*itraj, hitPropagatorHandle.product());	
		LogInfo("TestHitPropagator")<< "-------End Trajectory-------";
	}
}

void TestHitPropagator::testHitPropagator(const Trajectory& traj, const TrackingRecHitPropagator* hprop) const{
	const std::vector<TrajectoryMeasurement>& measurements = traj.measurements();
	if (measurements.size() == 0) return;
	TrajectoryStateOnSurface state = measurements.front().updatedState();
	if (!state.isValid()) {
		LogError("TestHitPropagator") << "first state not valid";
		return;
	}
	const GeomDet* det = measurements.front().recHit()->det();
	std::vector<TrajectoryMeasurement>::const_iterator imeas;
	for (imeas = measurements.begin(); imeas != measurements.end(); imeas++){
		if (!(imeas->recHit()->isValid())) continue;
		TransientTrackingRecHit::RecHitPointer propHit = hprop->project<GenericProjectedRecHit2D>(imeas->recHit(),
													    *det,
													    state);
		const GeomDet* currentDet = imeas->recHit()->det();
		LogVerbatim("TestHitPropagator")<< "Original Hit on surface (x, y, z) (" << 
			currentDet->surface().position().x() << ", " << 
			currentDet->surface().position().y() << ", " <<
			currentDet->surface().position().z() << "), global position (" <<
			imeas->recHit()->globalPosition().x() << ", " <<
			imeas->recHit()->globalPosition().y() << ", " <<
			imeas->recHit()->globalPosition().z() << ") " <<
			"Projected hit on surface (x, y, z) (" << 
			propHit->det()->surface().position().x() << ", " <<
			propHit->det()->surface().position().y() << ", " <<
			propHit->det()->surface().position().z() << "), global position ( " <<
			propHit->globalPosition().x() << ", " <<
			propHit->globalPosition().y() << ", " <<
			propHit->globalPosition().z() << ")";
	}
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
DEFINE_ANOTHER_FWK_MODULE(TestHitPropagator);
