#ifndef Neutron_NeutronProducer_h
#define Neutron_NeutronProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

/** Takes all hits with a high enough TOF, and writes
    them to a collection, with the TOF reset to be the current bunch
    crossing, so they can be treated as ordinary pileup
 */


class NeutronProducer: public edm::EDProducer
{
public:
  NeutronProducer(const edm::ParameterSet & pset);
  ~NeutronProducer();

  /// Handles the real EDM event
  virtual void produce(edm::Event& e, const edm::EventSetup& eventSetup);

  virtual int chamberId(int globalDetId) const = 0;

protected:

  /// helper to add time offsets and local det ID
  void adjust(PSimHit & h, float timeOffset);

private:

  edm::InputTag theInputTag;
  float theNeutronTimeCut;
  float theTimeWindow;
};

#endif

