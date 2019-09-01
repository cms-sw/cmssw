#ifndef Neutron_EDMNeutronWriter_h
#define Neutron_EDMNeutronWriter_h

/** Writes an event made of neutron hits
*/

#include "SimMuon/Neutron/src/NeutronWriter.h"

class EDMNeutronWriter : public NeutronWriter {
public:
  EDMNeutronWriter();
  ~EDMNeutronWriter() override;

  ///  writes out a list of SimHits.
  void writeCluster(int detType, const edm::PSimHitContainer& simHits) override;
  void beginEvent(edm::Event& e, const edm::EventSetup& es) override;
  void endEvent() override;
  void initialize(int detType) override {}

private:
  edm::Event* theEvent;
  std::unique_ptr<edm::PSimHitContainer> theHits;
};

#endif
