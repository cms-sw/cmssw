#ifndef MuonEndcapFrameRotation_H
#define MuonEndcapFrameRotation_H

/** \class MuonEndcapFrameRotation
 *
 * implementation of FrameRotation for the muon endcap
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"

#include "G4Step.hh"

class MuonEndcapFrameRotation : public MuonFrameRotation {
 public:
  virtual ~MuonEndcapFrameRotation() {};
  virtual Local3DPoint transformPoint(Local3DPoint &,G4Step *) const;
 private:
};


#endif
