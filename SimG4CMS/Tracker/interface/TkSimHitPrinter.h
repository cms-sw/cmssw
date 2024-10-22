#ifndef SimG4CMS_TkSimHitPrinter_H
#define SimG4CMS_TkSimHitPrinter_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include <string>
#include <fstream>

class G4Track;

class TkSimHitPrinter {
public:
  TkSimHitPrinter(const std::string&);
  ~TkSimHitPrinter();
  void startNewSimHit(const std::string&, const std::string&, int, int, int, int);
  void printLocal(const Local3DPoint&, const Local3DPoint&) const;
  void printGlobal(const Local3DPoint&, const Local3DPoint&) const;
  void printHitData(const std::string&, float, float, float) const;
  void printGlobalMomentum(float, float, float) const;

private:
  static std::ofstream* theFile;
};

#endif
