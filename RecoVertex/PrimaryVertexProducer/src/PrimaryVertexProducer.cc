#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& iConfig)
{
  using namespace reco;
   //register your products
  //#ifdef THIS_IS_AN_EVENT_EXAMPLE
  //   produces<VertexCollection>();

   //if do put with a label
   produces<VertexCollection>("PrimaryVertex");
   //#endif

   //now do what ever other initialization is needed

}


PrimaryVertexProducer::~PrimaryVertexProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   //Read 'ExampleData' from the Event
   //   Handle<ExampleData> pIn;
   //   iEvent.getByLabel("example",pIn);

   reco::Vertex::Point pos(-1, -1, -1);
   double e[6]; reco::Vertex::Error err(e);
   double chi2 = -1; double ndof = 1; double ntks = 0;
   std::auto_ptr<reco::Vertex> pOut(new reco::Vertex(pos, err, chi2, ndof, ntks));
   iEvent.put(pOut);

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer)
