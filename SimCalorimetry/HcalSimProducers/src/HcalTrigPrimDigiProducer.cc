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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalNominalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGSimpleTranscoder.h"
#include <algorithm>


HcalTrigPrimDigiProducer::HcalTrigPrimDigiProducer(const edm::ParameterSet& ps)
: theCoderFactory(HcalCoderFactory::DB),
  theAlgo(&theCoderFactory),
  inputLabel_(ps.getParameter<std::string>("inputLabel")),
  emScale_(ps.getParameter<int>("egammaScaleMeV")),
  jetScale_(ps.getParameter<int>("jetScaleMeV"))  
{
  hcalScale_=std::min(emScale_,jetScale_);
  
  produces<HcalTrigPrimDigiCollection>();
}


void HcalTrigPrimDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  HcalNominalTPGCoder nc(hcalScale_/1000.0);
  HcalTPGSimpleTranscoder tcode(hcalScale_,emScale_,jetScale_);

  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection>   hfDigis;

  e.getByLabel(inputLabel_,hbheDigis);
  e.getByLabel(inputLabel_,hfDigis);

  // get the conditions, for the decoding
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  edm::ESHandle<CaloGeometry> geom;
  eventSetup.get<IdealGeometryRecord>().get(geom);

  theCoderFactory.setDbService(conditions.product());
  nc.setupGeometry(*(geom.product()));
  nc.setupForAuto(conditions.product());
  theCoderFactory.setTPGCoder(&nc);
  theCoderFactory.setCompressionLUTcoder(&tcode);

  // Step B: Create empty output
  std::auto_ptr<HcalTrigPrimDigiCollection> result(new HcalTrigPrimDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theAlgo.run(*hbheDigis,  *hfDigis, *result);

  edm::LogInfo("HcalTrigPrimDigiProducer") << "HcalTrigPrims: " << result->size();

  // Step D: Put outputs into event
  e.put(result);

  theCoderFactory.setTPGCoder(0);
  theCoderFactory.setCompressionLUTcoder(0);
}


