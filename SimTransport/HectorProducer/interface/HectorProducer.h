#ifndef SimTransport_HectorProducer_H
#define SimTransport_HectorProducer_H
 
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"

class HectorManager;

class Hector;

namespace HepMC {
  class GenEvent;
}
class HectorProducer : public edm::EDProducer
{
 public:
  HectorProducer(edm::ParameterSet const & p);    //!< default constructor
  virtual ~HectorProducer();   //!< default destructor
  virtual void beginJob() {}
  virtual void endJob() {}
  void produce(edm::Event & iEvent, const edm::EventSetup & es);   //!< this method will do the user analysis
 private:
  int eventsAnalysed; //!< just to count events that have been analysed
  HepMC::GenEvent * evt_;
  Hector * hector;
  
  std::string m_InTag;
  bool m_verbosity;
  bool m_FP420Transport;
  bool m_ZDCTransport;
};

#endif

