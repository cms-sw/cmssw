#ifndef SimG4CMS_Calo_CaloDetInfo_H
#define SimG4CMS_Calo_CaloDetInfo_H
#include <iostream>
#include <string>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4VSolid.hh"

class CaloDetInfo {
public:
  CaloDetInfo(unsigned int id, const std::string& name, G4ThreeVector pos, const G4VSolid* sol, bool flag = false);
  CaloDetInfo();
  CaloDetInfo(const CaloDetInfo&);
  ~CaloDetInfo() = default;

  uint32_t id() const { return id_; }
  std::string name() const { return name_; }
  G4ThreeVector pos() const { return pos_; }
  const G4VSolid* solid() const { return solid_; }
  bool flag() const { return flag_; }

  bool operator<(const CaloDetInfo& info) const;

private:
  uint32_t id_;
  std::string name_;
  G4ThreeVector pos_;
  const G4VSolid* solid_;
  bool flag_;
};

class CaloDetInfoLess {
public:
  bool operator()(const CaloDetInfo* a, const CaloDetInfo* b) { return (a->id() < b->id()); }
  bool operator()(const CaloDetInfo a, const CaloDetInfo b) { return (a.id() < b.id()); }
};

std::ostream& operator<<(std::ostream&, const CaloDetInfo&);
#endif
