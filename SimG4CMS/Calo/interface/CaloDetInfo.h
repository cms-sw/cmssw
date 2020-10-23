#ifndef SimG4CMS_Calo_CaloDetInfo_H
#define SimG4CMS_Calo_CaloDetInfo_H
#include <iostream>
#include <string>
#include <vector>
#include "G4ThreeVector.hh"

class CaloDetInfo {
public:
  CaloDetInfo(unsigned int id, std::string name, G4ThreeVector pos, std::vector<double> par);
  CaloDetInfo();
  CaloDetInfo(const CaloDetInfo&);
  ~CaloDetInfo() = default;

  uint32_t id() const { return id_; }
  std::string name() const { return name_; }
  G4ThreeVector pos() const { return pos_; }
  std::vector<double> par() const { return par_; }

  bool operator<(const CaloDetInfo& info) const;

private:
  uint32_t id_;
  std::string name_;
  G4ThreeVector pos_;
  std::vector<double> par_;
};

class CaloDetInfoLess {
public:
  bool operator()(const CaloDetInfo* a, const CaloDetInfo* b) { return (a->id() < b->id()); }
  bool operator()(const CaloDetInfo a, const CaloDetInfo b) { return (a.id() < b.id()); }
};

std::ostream& operator<<(std::ostream&, const CaloDetInfo&);
#endif
