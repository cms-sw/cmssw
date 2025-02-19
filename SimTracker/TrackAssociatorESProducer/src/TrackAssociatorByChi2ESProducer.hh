#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "FWCore/Framework/interface/ESProducer.h"


/** \class TrackAssociatorByChi2ESProducer
 *  ESProducer for TrackAssociatorByChi2
 *
 *  $Date: 2007/03/26 15:48:38 $
 *  $Revision: 1.2 $
 *  \author magni
 */

class TrackAssociatorByChi2ESProducer : public edm::ESProducer {
   public:
      TrackAssociatorByChi2ESProducer(const edm::ParameterSet&);
      ~TrackAssociatorByChi2ESProducer();

      typedef std::auto_ptr<TrackAssociatorBase> ReturnType;

      ReturnType produce(const TrackAssociatorRecord&);
   private:
      // ----------member data ---------------------------
  edm::ParameterSet conf_;
};

