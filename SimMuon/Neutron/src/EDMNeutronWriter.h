#ifndef Neutron_EDMNeutronWriter_h
#define Neutron_EDMNeutronWriter_h

/** Writes an event made of neutron hits
*/

#include "SimMuon/Neutron/src/NeutronWriter.h"

class EDMNeutronWriter: public NeutronWriter {
public:
  EDMNeutronWriter();
  virtual ~EDMNeutronWriter();

  ///  writes out a list of SimHits. 
  virtual void writeCluster(int detType, const edm::PSimHitContainer & simHits);
  virtual void beginEvent(edm::Event & e, const edm::EventSetup & es);
  virtual void endEvent();
  virtual void initialize(int detType) {}

private:
  edm::Event * theEvent;
  std::auto_ptr<edm::PSimHitContainer> theHits;
};

#endif

