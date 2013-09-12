#ifndef CSCDigiProducer_h
#define CSCDigiProducer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class CSCStripConditions;

class CSCDigiProducer : public edm::EDProducer
{
public:
  typedef CSCDigitizer::DigiSimLinks DigiSimLinks;

  explicit CSCDigiProducer(const edm::ParameterSet& ps);
  virtual ~CSCDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  CSCDigitizer theDigitizer;
  CSCStripConditions * theStripConditions;
  std::string geometryType;
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token; 
};

#endif

