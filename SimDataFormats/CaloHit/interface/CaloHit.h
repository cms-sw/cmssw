#ifndef SimDataFormats_CaloHit_H
#define SimDataFormats_CaloHit_H
///////////////////////////////////////////////////////////////////////////////
// File: CaloHit.h
// Hit class for QIE analysis
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>
#include <cstdint>

class CaloHit {
public:
  CaloHit(int deti, int layi, double ei, double etai, double phii, double timi, uint32_t idi = 0);
  CaloHit();
  CaloHit(const CaloHit&);
  virtual ~CaloHit();

  int det() const { return deth; }
  int layer() const { return layerh; }
  double e() const { return eh; }
  double eta() const { return etah; }
  double phi() const { return phih; }
  double t() const { return timeh; }
  uint32_t id() const { return idh; }
  void setId(const uint32_t idi) { idh = idi; }

  bool operator<(const CaloHit& hit) const;

private:
  int deth, layerh;
  double eh, etah, phih, timeh;
  uint32_t idh;
};

class CaloHitMore {
public:
  bool operator()(const CaloHit* a, const CaloHit* b) { return (a->t() < b->t()); }
};

class CaloHitIdMore {
public:
  bool operator()(const CaloHit* a, const CaloHit* b) {
    if (a->id() < b->id()) {
      return true;
    } else if (a->id() > b->id()) {
      return false;
    } else {
      return (a->t() < b->t());
    }
  }
};

class CaloHitELess {
public:
  bool operator()(const CaloHit* a, const CaloHit* b) { return (a->e() > b->e()); }
};

class CaloHitEtLess {
public:
  bool operator()(const CaloHit* a, const CaloHit* b) {
    return (a->e() / std::cosh(a->eta()) > b->e() / std::cosh(b->eta()));
  }
};

std::ostream& operator<<(std::ostream&, const CaloHit&);
#endif
