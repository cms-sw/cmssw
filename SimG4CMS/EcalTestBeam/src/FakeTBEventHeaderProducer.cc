/*
 * \file FakeTBEventHeaderProducer.cc
 *
 * $Id: FakeTBEventHeaderProducer.cc,v 1.3 2007/01/19 09:58:08 fabiocos Exp $
 *
 */

#include "SimG4CMS/EcalTestBeam/interface/FakeTBEventHeaderProducer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

using namespace cms;
using namespace std;


FakeTBEventHeaderProducer::FakeTBEventHeaderProducer(const edm::ParameterSet& ps) {
  produces<EcalTBEventHeader>();
  ecalTBInfoLabel_ = ps.getUntrackedParameter<string>("EcalTBInfoLabel","SimEcalTBG4Object");

}

 
FakeTBEventHeaderProducer::~FakeTBEventHeaderProducer() 
{
}

 void FakeTBEventHeaderProducer::produce(edm::Event & event, const edm::EventSetup& eventSetup)
{
  auto_ptr<EcalTBEventHeader> product(new EcalTBEventHeader());

  // get the vertex information from the event

  const PEcalTBInfo* theEcalTBInfo=0;
  // try {
  edm::Handle<PEcalTBInfo> EcalTBInfo;
  event.getByLabel(ecalTBInfoLabel_,EcalTBInfo);
  theEcalTBInfo = EcalTBInfo.product(); 
  //  }
  // catch ( std::exception& ex ) { }

  if (!theEcalTBInfo)
    return;
  
  product->setEventNumber(event.id().event());
  product->setRunNumber(event.id().run());
  product->setBurstNumber(1);
  product->setTriggerMask(0x1);
  product->setCrystalInBeam(EBDetId(1,theEcalTBInfo->nCrystal(),EBDetId::SMCRYSTALMODE));
  
//   LogDebug("FakeTBHeader") << (*product);
//   LogDebug("FakeTBHeader") << (*product).eventType();
//   LogDebug("FakeTBHeader") << (*product).crystalInBeam();
  event.put(product);
  
}
