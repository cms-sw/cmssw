#ifndef QuickTrackAssociatorByHitsESProducer_h
#define QuickTrackAssociatorByHitsESProducer_h

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

// Forward declarations
class TrackAssociatorRecord;
class TrackAssociatorBase;

/** @brief ESProducer to create QuickTrackAssociatorByHits instances.
 *
 * This just makes the QuickTrackAssociatorByHits class available to the configuration
 * parser. Have a look at the QuickTrackAssociatorByHits documentation for any further
 * details.
 *
 * @author Mark Grimes (mark.grimes@cern.ch)
 * @date 12/Nov/2010
 */
class QuickTrackAssociatorByHitsESProducer : public edm::ESProducer
{
public:
	QuickTrackAssociatorByHitsESProducer( const edm::ParameterSet& config );
	std::auto_ptr<TrackAssociatorBase> produce( const TrackAssociatorRecord& record );
private:
	edm::ParameterSet config_;
};

//do in SealModule instead DEFINE_FWK_EVENTSETUP_MODULE(QuickTrackAssociatorByHitsESProducer);

#endif // end of ifndef slhctools_QuickTrackAssociatorByHitsESProducer_h
