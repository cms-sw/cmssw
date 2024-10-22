#ifndef SimG4Core_DDG4SensitiveConverter_h
#define SimG4Core_DDG4SensitiveConverter_h

#include "SimG4Core/Geometry/interface/DDG4DispContainer.h"

#include <iostream>
#include <string>
#include <vector>

class SensitiveDetectorCatalog;
class DDLogicalPart;

class DDG4SensitiveConverter {
public:
  DDG4SensitiveConverter();
  virtual ~DDG4SensitiveConverter();
  void upDate(const DDG4DispContainer &ddg4s, SensitiveDetectorCatalog &);

private:
  std::string getString(const std::string &, const DDLogicalPart *);
};

#endif
