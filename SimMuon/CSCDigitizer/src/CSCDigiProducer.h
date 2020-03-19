#ifndef CSCDigiProducer_h
#define CSCDigiProducer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"

class CSCStripConditions;

class CSCDigiProducer : public edm::stream::EDProducer<> {
public:
  typedef CSCDigitizer::DigiSimLinks DigiSimLinks;

  explicit CSCDigiProducer(const edm::ParameterSet &ps);
  ~CSCDigiProducer() override;

  /**Produces the EDM products,*/
  void produce(edm::Event &e, const edm::EventSetup &c) override;

private:
  CSCDigitizer theDigitizer;
  CSCStripConditions *theStripConditions;
  std::string geometryType;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> cf_token;
};

#endif
