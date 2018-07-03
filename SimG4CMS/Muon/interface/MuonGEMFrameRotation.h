#ifndef SimG4CMS_Muon_MuonGEMFrameRotation_H
#define SimG4CMS_Muon_MuonGEMFrameRotation_H

/** \class MuonGEMFrameRotation
 *
 * implementation of FrameRotation for the muon gem
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"

class MuonDDDConstants;
class G4Step;

class MuonGEMFrameRotation : public MuonFrameRotation {

public:
  MuonGEMFrameRotation( const MuonDDDConstants& muonConstants );
  ~MuonGEMFrameRotation() override;
  Local3DPoint transformPoint(const Local3DPoint &, const G4Step *) const override;

private:
  MuonG4Numbering* g4numbering;
  int              theSectorLevel;
};


#endif
