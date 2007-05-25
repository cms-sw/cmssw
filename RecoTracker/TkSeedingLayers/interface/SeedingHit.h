#ifndef TkSeedingLayers_SeedingHit_H
#define TkSeedingLayers_SeedingHit_H

#include <boost/shared_ptr.hpp>

namespace edm { class EventSetup; }
class TrackingRecHit;
namespace ctfseeding { class SeedingLayer; }

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


namespace ctfseeding {

class SeedingHit {
public:
  SeedingHit(const TrackingRecHit * hit , const SeedingLayer &l,  const edm::EventSetup& iSetup);
  SeedingHit(const TransientTrackingRecHit::ConstRecHitPointer& ttrh, const SeedingLayer &l);

  // temporary FIX for BC!!! to be removed asap.
  SeedingHit(const TrackingRecHit * hit ,  const edm::EventSetup& iSetup);

  float phi() const;
  float rOrZ() const;
  float errorRZ() const;
  float errorRPhi() const;

  float r() const;
  float z() const;
  const TrackingRecHit * RecHit() const;

  operator const TrackingRecHit* () const;
  operator const TransientTrackingRecHit::ConstRecHitPointer& () const;

  const SeedingLayer & seedinglayer() const;

private:
  class SeedingHitImpl;
  boost::shared_ptr<SeedingHitImpl> theImpl;
};

}

#endif 
