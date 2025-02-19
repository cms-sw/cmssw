#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorByTracks.h"
#include "FWCore/Framework/interface/ESProducer.h"

//
// class decleration
//

class VertexAssociatorByTracksESProducer : public edm::ESProducer {
   public:
      VertexAssociatorByTracksESProducer(const edm::ParameterSet&);
      ~VertexAssociatorByTracksESProducer();

      typedef std::auto_ptr<VertexAssociatorBase> ReturnType;

      ReturnType produce(const VertexAssociatorRecord&);
   private:
      // ----------member data ---------------------------
  edm::ParameterSet conf_;
};
