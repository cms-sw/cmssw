#ifndef SimG4CMS_Calo_CaloDetInfo_H
#define SimG4CMS_Calo_CaloDetInfo_H
#include <iostream>
#include <string>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4VSolid.hh"

class CaloDetInfo {
public:
  CaloDetInfo(uint32_t id,
              uint32_t depth,
              double rho,
              const std::string& name,
              G4ThreeVector pos,
              const G4VSolid* sol,
              bool flag = false);
  CaloDetInfo();
  CaloDetInfo(const CaloDetInfo&);
  ~CaloDetInfo() = default;

  uint32_t id() const { return id_; }
  uint32_t depth() const { return depth_; }
  double rho() const { return rho_; }
  std::string name() const { return name_; }
  G4ThreeVector pos() const { return pos_; }
  const G4VSolid* solid() const { return solid_; }
  bool flag() const { return flag_; }

  bool operator<(const CaloDetInfo& info) const;

private:
  uint32_t id_;
  uint32_t depth_;
  double rho_;
  std::string name_;
  G4ThreeVector pos_;
  const G4VSolid* solid_;
  bool flag_;
};

class CaloDetInfoLess {
public:
  bool operator()(const CaloDetInfo* a, const CaloDetInfo* b) {
    if (a->id() == b->id()) {
      if (a->depth() == b->depth()) {
        return (a->rho() < b->rho());
      } else {
        return (a->depth() < b->depth());
      }
    } else {
      return (a->id() < b->id());
    }
  }
  bool operator()(const CaloDetInfo a, const CaloDetInfo b) {
    if (a.id() == b.id()) {
      if (a.depth() == b.depth()) {
        return (a.rho() < b.rho());
      } else {
        return (a.depth() < b.depth());
      }
    } else {
      return (a.id() < b.id());
    }
  }
};

std::ostream& operator<<(std::ostream&, const CaloDetInfo&);
#endif
