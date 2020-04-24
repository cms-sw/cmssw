#ifndef SimG4CMS_Muon_MuonFrameRotation_H
#define SimG4CMS_Muon_MuonFrameRotation_H

/** \class MuonFrameRotation
 *
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 * Pedro Arce 31/01/2006
 * Make it base class of each detector FrameRotation  
 */

#include "G4Step.hh"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class MuonSubDetector;
class DDCompactView;

class MuonFrameRotation {
 public:
  MuonFrameRotation( ) { };
  virtual ~MuonFrameRotation(){};
  virtual Local3DPoint transformPoint(const Local3DPoint &,const G4Step *) const = 0;

 private:
};

#endif
