#ifndef CSCDigiProducer_h
#define CSCDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"
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
  //Name of Collection used for create the XF 
  std::string mix_;
  std::string collection_for_XF;
};

#endif

