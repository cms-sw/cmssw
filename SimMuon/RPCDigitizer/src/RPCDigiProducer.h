#ifndef RPCDigiProducer_h
#define RPCDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class RPCDigitizer;
class RPCGeometry;
class RPCSimSetUp;
class RPCSynchronizer;

class RPCDigiProducer : public edm::EDProducer
{
public:

  explicit RPCDigiProducer(const edm::ParameterSet& ps);
  virtual ~RPCDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  RPCDigitizer* theDigitizer;
  RPCSimSetUp* theRPCSimSetUp;

};

#endif

