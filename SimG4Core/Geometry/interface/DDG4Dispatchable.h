#ifndef SimG4Core_DDG4Dispatchable_H
#define SimG4Core_DDG4Dispatchable_H

#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "G4LogicalVolume.hh"

class DDG4Dispatchable {
public:
  DDG4Dispatchable(const DDLogicalPart* ddL, G4LogicalVolume* g4L) : ddLogical(ddL), g4Logical(g4L) {}
  const DDLogicalPart* getDDLogicalPart() const { return ddLogical; }
  G4LogicalVolume* getG4LogicalVolume() const { return g4Logical; }

private:
  const DDLogicalPart* ddLogical;
  G4LogicalVolume* g4Logical;
};

#endif
