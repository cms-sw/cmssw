#include "SimCalorimetry/HcalSimProducers/src/HcalTrigPrimDigiProducer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



HcalTrigPrimDigiProducer::HcalTrigPrimDigiProducer(const edm::ParameterSet& ps)
: theCoderFactory(HcalCoderFactory::DB),
  theAlgo(&theCoderFactory)
{
  produces<HcalTrigPrimDigiCollection>();
}


void HcalTrigPrimDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {

  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection>   hfDigis;

  // TODO make labels for unsuppressed
  e.getByType(hbheDigis);
  e.getByType(hfDigis);

  // get the conditions, for the decoding
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theCoderFactory.setDbService(conditions.product());



  // Step B: Create empty output
  std::auto_ptr<HcalTrigPrimDigiCollection> result(new HcalTrigPrimDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theAlgo.run(*hbheDigis,  *hfDigis, *result);

  edm::LogInfo("HcalTrigPrimDigiProducer") << "HcalTrigPrims: " << result->size();

  // Step D: Put outputs into event
  e.put(result);

}


