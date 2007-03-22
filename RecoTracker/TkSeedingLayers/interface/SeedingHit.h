#ifndef TkSeedingLayers_SeedingHit_H
#define TkSeedingLayers_SeedingHit_H


namespace edm { class EventSetup; };
class TrackingRecHit;

namespace ctfseeding {

class SeedingHit {
public:
  SeedingHit(const TrackingRecHit * hit ,  const edm::EventSetup& iSetup);

  float phi() const {return thePhi;}
  float rOrZ() const { return theRZ; } 
  float r() const {return theR; }
  float z() const {return theZ; }

  const TrackingRecHit * RecHit() const { return theRecHit;}

private:
  const TrackingRecHit *theRecHit;
  float thePhi;
  float theR, theZ;
  float theRZ;
};

}

#endif 
