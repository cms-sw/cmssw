#ifndef SimG4CMS_ShowerLibraryProducer_FiberG4Hit_h
#define SimG4CMS_ShowerLibraryProducer_FiberG4Hit_h

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4LogicalVolume.hh"

#include <boost/cstdint.hpp>
#include <vector>

class FiberG4Hit : public G4VHit {

public:

  FiberG4Hit();
  FiberG4Hit(G4LogicalVolume* logVol, G4int tower, G4int depth, G4int tkID);
  virtual ~FiberG4Hit();
  FiberG4Hit(const FiberG4Hit &right);
  const FiberG4Hit& operator=(const FiberG4Hit &right);
  G4int operator==(const FiberG4Hit &right) const;

  inline void *operator new(size_t);
  inline void  operator delete(void *aHit);

private:

  G4int theTowerId;
  G4int theDepth;
  G4int theTrackId;
  G4int theNpe;
  G4double theTime;
  math::XYZPoint theHitPos;
  std::vector<HFShowerPhoton> thePhoton;
  const G4LogicalVolume* theLogV;
  
public:

  inline void setTowerId(G4int tower)    {theTowerId = tower;}
  inline void setDepth(G4int depth)      {theDepth = depth;}
  inline void setNpe(G4int npe)          {theNpe = npe;}
  inline void setPos(const math::XYZPoint& xyz) {theHitPos = xyz;}
  inline void setTime(G4double t)        {theTime = t; }
  inline void setPhoton(const std::vector<HFShowerPhoton>& photon) {thePhoton = photon; }
      
  inline G4int towerId()  const       {return theTowerId;}
  inline G4int depth()    const       {return theDepth;}
  inline G4int trackId()  const       {return theTrackId;}
  inline G4int npe()      const       {return theNpe;}
  math::XYZPoint hitPos() const       {return theHitPos;};
  inline G4double time()  const       {return theTime;}
  std::vector<HFShowerPhoton> photon() const {return thePhoton;}
  inline void add(G4int npe)      {theNpe +=npe;}
};

typedef G4THitsCollection<FiberG4Hit> FiberG4HitsCollection;

extern G4ThreadLocal G4Allocator<FiberG4Hit> *fFiberG4HitAllocator;

inline void* FiberG4Hit::operator new(size_t) {
  if (!fFiberG4HitAllocator) fFiberG4HitAllocator = 
    new G4Allocator<FiberG4Hit>;
  return (void*)fFiberG4HitAllocator->MallocSingle();
}

inline void FiberG4Hit::operator delete(void *aHit) {
  fFiberG4HitAllocator->FreeSingle((FiberG4Hit*) aHit);
}
#endif
