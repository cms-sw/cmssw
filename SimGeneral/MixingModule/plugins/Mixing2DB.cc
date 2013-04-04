#include "Mixing2DB.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Mixing2DB::Mixing2DB(const edm::ParameterSet& iConfig)
{
  //cfi_=iConfig.getParameter<edm::ParameterSet>("input");
  cfi_=iConfig;
}


Mixing2DB::~Mixing2DB()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
Mixing2DB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
Mixing2DB::beginJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Mixing2DB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Mixing2DB::endJob() 
{
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  MixingModuleConfig * config = new MixingModuleConfig();
  config->read(cfi_);
  poolDbService->writeOne<MixingModuleConfig>(config,
					      poolDbService->currentTime(),
					      "MixingRcd");
}
