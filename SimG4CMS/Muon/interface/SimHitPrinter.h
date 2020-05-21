#ifndef SimG4CMS_Muon_SimHitPrinter_H
#define SimG4CMS_Muon_SimHitPrinter_H

/** \class SimHitPrinter
 *
 * class to print sim hits for validation and debugging
 * 
 * \author Tommaso Boccali <Tommaso.Boccali@cern.ch>
 *         Arno Straesser <Arno.Straessner@cern.ch>
 *
 * Modification:
 *
 */

#include <string>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include <fstream>
#include <atomic>

class SimHitPrinter {
public:
  SimHitPrinter(const std::string&);
  ~SimHitPrinter();

  void startNewSimHit(const std::string&);
  void startNewEvent(int);

  void printId(int) const;
  void printTrack(int) const;
  void printPabs(float) const;
  void printEloss(float) const;
  void printLocal(const LocalPoint&, const LocalPoint&) const;
  void printGlobal(const GlobalPoint&) const;

private:
  static std::atomic<std::ofstream*> theFile;
};

#endif
