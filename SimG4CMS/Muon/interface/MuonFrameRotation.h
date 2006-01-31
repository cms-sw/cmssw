#ifndef MuonSim_MuonFrameRotation_H
#define MuonSim_MuonFrameRotation_H

/** \class MuonFrameRotation
 *
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 * Pedro Arce 31/01/2006
 * Make it base class of each detector FrameRotation  
 */

#include "G4Step.hh"
#include "Geometry/Vector/interface/LocalPoint.h"

class MuonSubDetector;

class MuonFrameRotation {
 public:
  MuonFrameRotation(){};
  virtual ~MuonFrameRotation(){};
  virtual Local3DPoint transformPoint(Local3DPoint &,G4Step *) const = 0;

 private:
};

#endif
