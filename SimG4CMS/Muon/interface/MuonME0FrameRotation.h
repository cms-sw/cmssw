#ifndef SimG4CMS_Muon_MuonME0FrameRotation_H
#define SimG4CMS_Muon_MuonME0FrameRotation_H

/** \class MuonME0FrameRotation
 *
 * implementation of FrameRotation for the muon ME0
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"

#include "G4Step.hh"

class MuonGeometryConstants;

class MuonME0FrameRotation : public MuonFrameRotation {
public:
  MuonME0FrameRotation(const MuonGeometryConstants& muonConstants);
  ~MuonME0FrameRotation() override;
  Local3DPoint transformPoint(const Local3DPoint&, const G4Step*) const override;
};

#endif
