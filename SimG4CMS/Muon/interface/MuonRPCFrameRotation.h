#ifndef SimG4CMS_Muon_MuonRPCFrameRotation_H
#define SimG4CMS_Muon_MuonRPCFrameRotation_H

/** \class MuonRPCFrameRotation
 *
 * implementation of FrameRotation for the muon rpc
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"

#include "G4Step.hh"

class MuonG4Numbering;
class MuonDDDConstants;

class MuonRPCFrameRotation : public MuonFrameRotation {
public:
  MuonRPCFrameRotation(const MuonDDDConstants& constants);
  ~MuonRPCFrameRotation() override;
  Local3DPoint transformPoint(const Local3DPoint&, const G4Step*) const override;

private:
  MuonG4Numbering* g4numbering;
  int theRegion;
};

#endif
