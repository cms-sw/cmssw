#ifndef HcalTrigPrimDigiProducer_h
#define HcalTrigPrimDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <vector>

class HcalTrigPrimDigiProducer : public edm::EDProducer
{
public:

  explicit HcalTrigPrimDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalTrigPrimDigiProducer() {}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  HcalTriggerPrimitiveAlgo theAlgo_;

  /// input tags for HCAL digis
  std::vector<edm::InputTag> inputLabel_;
  // this seems a strange way of doing things
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;

  /// input tag for FEDRawDataCollection
  edm::InputTag inputTagFEDRaw_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  double MinLongEnergy_, MinShortEnergy_, LongShortSlope_, LongShortOffset_;
  
  bool runZS_;

  bool runFrontEndFormatError_;

  bool HFEMB_;
  edm::ParameterSet LongShortCut_;
};

#endif

