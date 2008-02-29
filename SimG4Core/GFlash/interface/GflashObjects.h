#ifndef GflashObjects_H
#define GflashObjects_H

#include <TObject.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <vector>

class GflashHit : public TObject {

 public:
  GflashHit() { Init(); }
  ~GflashHit() { Init(); }

  double energy;
  TVector3 position;

  void Init() {
    energy = 0.0;
    position.SetXYZ(0,0,0);
  }

  ClassDef(GflashHit,1)
};

class GflashObject : public TObject {

 public:
  GflashObject() { Init(); }
  ~GflashObject() { Init(); }
  double   energy;
  TVector3 direction;
  TVector3 position;
  std::vector<GflashHit> hits;

  void Init() {
    energy = 0.0;
    direction.SetXYZ(0,0,0);
    position.SetXYZ(0,0,0);
    hits.clear();
  }
  ClassDef(GflashObject,1)
};

#endif
