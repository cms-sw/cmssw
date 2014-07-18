#ifndef SimG4CMS_ShowerLibraryProducer_HFShowerG4Hit_h
#define SimG4CMS_ShowerLibraryProducer_HFShowerG4Hit_h

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"

#include <boost/cstdint.hpp>
#include <vector>

class HFShowerG4Hit : public G4VHit {

public:

  HFShowerG4Hit();
  HFShowerG4Hit(G4int hitId, G4int tkID, double edep, double time);
  virtual ~HFShowerG4Hit();
  HFShowerG4Hit(const HFShowerG4Hit &right);
  const HFShowerG4Hit& operator=(const HFShowerG4Hit &right);
  G4int operator==(const HFShowerG4Hit &right) const;

  inline void *operator new(size_t);
  inline void  operator delete(void *aHit);

private:

  G4int    theHitId;
  G4int    theTrackId;
  G4double theEdep;
  G4double theTime;
  G4ThreeVector localPos;
  G4ThreeVector globalPos;
  G4ThreeVector momDir;
  
public:

  inline void setHitId(G4int hitId)            {theHitId = hitId;}
  inline void setTrackId(G4int trackId)        {theTrackId = trackId;}
  inline void setEnergy(G4double edep)         {theEdep  = edep;}
  inline void updateEnergy(G4double edep)      {theEdep += edep;}
  inline void setTime(G4double t)              {theTime  = t;}
  inline void setLocalPos(const G4ThreeVector& xyz)   {localPos  = xyz;}
  inline void setGlobalPos(const G4ThreeVector& xyz)  {globalPos = xyz;}
  inline void setPrimMomDir(const G4ThreeVector& xyz) {momDir    = xyz;}
      
  inline G4int hitId()                  const {return theHitId;}
  inline G4int trackId()                const {return theTrackId;}
  inline G4double edep()                const {return theEdep;};
  inline G4double time()                const {return theTime;}
  inline G4ThreeVector localPosition()  const {return localPos;}
  inline G4ThreeVector globalPosition() const {return globalPos;}
  inline G4ThreeVector primaryMomDir()  const {return momDir;}
};

typedef G4THitsCollection<HFShowerG4Hit> HFShowerG4HitsCollection;

extern G4ThreadLocal G4Allocator<HFShowerG4Hit>* fHFShowerG4HitAllocator;

inline void* HFShowerG4Hit::operator new(size_t) {
  if (!fHFShowerG4HitAllocator) fHFShowerG4HitAllocator = 
    new G4Allocator<HFShowerG4Hit>;
  return (void*)fHFShowerG4HitAllocator->MallocSingle();
}

inline void HFShowerG4Hit::operator delete(void *aHit) {
  fHFShowerG4HitAllocator->FreeSingle((HFShowerG4Hit*) aHit);
}
#endif
