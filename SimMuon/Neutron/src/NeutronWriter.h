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

class NeutronWriter {
public:
  ///  makes an "event" from a list of SimHits.  Called by writeEvent
  virtual void writeEvent(int detType, const edm::PSimHitContainer & simHits) = 0;
  virtual ~NeutronWriter() {}

};

#endif

