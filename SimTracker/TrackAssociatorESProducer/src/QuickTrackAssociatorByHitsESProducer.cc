#include "SimTracker/TrackAssociatorESProducer/src/QuickTrackAssociatorByHitsESProducer.hh"

#include <FWCore/Framework/interface/ESHandle.h>
#include <MagneticField/Engine/interface/MagneticField.h>
#include <SimTracker/Records/interface/TrackAssociatorRecord.h>
#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"

QuickTrackAssociatorByHitsESProducer::QuickTrackAssociatorByHitsESProducer( const edm::ParameterSet& config )
{
	// Tell the framework what's being produced
	std::string componentName=config.getParameter<std::string>("ComponentName");
	setWhatProduced(this,componentName);

	//now do what ever other initialization is needed
	config_=config;
}

std::auto_ptr<TrackAssociatorBase> QuickTrackAssociatorByHitsESProducer::produce( const TrackAssociatorRecord& record )
{
	//using namespace edm::es;
	edm::ESHandle<MagneticField> magneticField;
	record.getRecord<IdealMagneticFieldRecord>().get(magneticField);
	std::auto_ptr<TrackAssociatorBase> pTrackAssociatorBase( new QuickTrackAssociatorByHits(config_) );
	return pTrackAssociatorBase ;
}
