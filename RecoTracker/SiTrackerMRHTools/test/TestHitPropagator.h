#ifndef TestHitPropagator_h
#define TestHitPropagator_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
/* 
creates TransientTrackingRecHit out of a rechit collection 
and projects all of them to the surface of the first rechit.
*/
class Trajectory;
class TrackingRecHitPropagator;	


class TestHitPropagator : public edm::EDAnalyzer 
{
	public:
	TestHitPropagator(const edm::ParameterSet& conf);
	virtual ~TestHitPropagator();
	
	virtual void beginRun(edm::Run & run, const edm::EventSetup& c);
	virtual void endJob();
	virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

	private:
	void testHitPropagator(const Trajectory& traj, const TrackingRecHitPropagator* hprop) const;

	edm::ParameterSet theConf;
 
};

#endif
