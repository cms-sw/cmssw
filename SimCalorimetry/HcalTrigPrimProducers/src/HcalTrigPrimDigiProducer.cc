#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTrigPrimDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

#include <algorithm>

HcalTrigPrimDigiProducer::HcalTrigPrimDigiProducer(const edm::ParameterSet& ps)
: 
  theAlgo(ps.getParameter<bool>("peakFilter"),
	  ps.getParameter<std::vector<double> >("weights"),
	  ps.getParameter<int>("latency")),
  inputLabel_(ps.getParameter<edm::InputTag>("inputLabel"))
{
   produces<HcalTrigPrimDigiCollection>();
}


void HcalTrigPrimDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection>   hfDigis;

  e.getByLabel(inputLabel_,hbheDigis);
  e.getByLabel(inputLabel_,hfDigis);

  // get the conditions, for the decoding
  edm::ESHandle<HcalTPGCoder> inputCoder;
  eventSetup.get<HcalTPGRecord>().get(inputCoder);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  eventSetup.get<CaloTPGRecord>().get(outTranscoder);
  outTranscoder->setup(eventSetup,CaloTPGTranscoder::HcalTPG);

  // Step B: Create empty output
  std::auto_ptr<HcalTrigPrimDigiCollection> result(new HcalTrigPrimDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theAlgo.run(inputCoder.product(),outTranscoder->getHcalCompressor().get(),
	      *hbheDigis,  *hfDigis, *result);

  //  edm::LogInfo("HcalTrigPrimDigiProducer") << "HcalTrigPrims: " << result->size();

  // Step D: Put outputs into event
  e.put(result);

  outTranscoder->releaseSetup();
}


