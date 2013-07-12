#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "FWCore/Framework/interface/ESProducer.h"

/** \class TrackAssociatorByHitsESProducer
 *  ESProducer for TrackAssociatorByHits
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author magni
 */

class TrackAssociatorByHitsESProducer : public edm::ESProducer {
   public:
      TrackAssociatorByHitsESProducer(const edm::ParameterSet&);
      ~TrackAssociatorByHitsESProducer();

      typedef std::auto_ptr<TrackAssociatorBase> ReturnType;

      ReturnType produce(const TrackAssociatorRecord&);
   private:
      // ----------member data ---------------------------
  edm::ParameterSet conf_;
};
