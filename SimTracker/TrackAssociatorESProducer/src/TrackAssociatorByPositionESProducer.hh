#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByPosition.h"
#include "FWCore/Framework/interface/ESProducer.h"

//
// class decleration
//

class TrackAssociatorByPositionESProducer : public edm::ESProducer {
   public:
      TrackAssociatorByPositionESProducer(const edm::ParameterSet&);
      ~TrackAssociatorByPositionESProducer();

      typedef std::auto_ptr<TrackAssociatorBase> ReturnType;

      ReturnType produce(const TrackAssociatorRecord&);
   private:
      // ----------member data ---------------------------
  edm::ParameterSet conf_;
  std::string thePname;
};

