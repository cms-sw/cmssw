#ifndef NeutronWriter_h
#define NeutronWriter_h

/** theNeutronWriter stores "events"
 * which consist of a list of SimHits,
 * grouped by detector type.  These can then
 * be read back to model neutron background
 * int muon chambers.
 *
 */

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class NeutronWriter {
public:
  ///  writes out a list of SimHits.
  virtual void writeCluster(int detType, const edm::PSimHitContainer& simHits) = 0;
  virtual void initialize(int detType) {}
  virtual void beginEvent(edm::Event& e, const edm::EventSetup& es) {}
  virtual void endEvent() {}
  virtual ~NeutronWriter() {}
};

#endif
