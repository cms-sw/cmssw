#ifndef GEMDigiProducer_h
#define GEMDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimMuon/GEMDigitizer/src/GEMDigitizer.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"

class GEMGeometry;
class GEMSimSetUp;
class GEMSynchronizer;

class GEMDigiProducer : public edm::EDProducer
{
public:

  typedef GEMDigitizer::StripDigiSimLinks StripDigiSimLinks;

  explicit GEMDigiProducer(const edm::ParameterSet& ps);
  virtual ~GEMDigiProducer();

  virtual void beginRun( edm::Run&, const edm::EventSetup& );
  virtual void endRun( edm::Run&, const edm::EventSetup& ) {}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  void setGEMSetUp(std::vector<RPCStripNoises::NoiseItem>, std::vector<double>);

private:

  GEMDigitizer* digitizer_;
  GEMSimSetUp* gemSimSetUp_;

  //Name of Collection used for create the XF 
  std::string collectionXF_;
};

#endif

