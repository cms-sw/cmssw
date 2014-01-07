#ifndef MuonMe0FrameRotation_H
#define MuonMe0FrameRotation_H

/** \class MuonMe0FrameRotation
 *
 * implementation of FrameRotation for the muon ME0
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 *
 */

#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"

#include "G4Step.hh"

class DDCompactView;

class MuonMe0FrameRotation : public MuonFrameRotation {

public:
  MuonMe0FrameRotation( const DDCompactView& cpv );
  virtual ~MuonMe0FrameRotation();
  virtual Local3DPoint transformPoint(const Local3DPoint &, const G4Step *) const;

private:
  MuonG4Numbering* g4numbering;
  int              theSectorLevel;
};


#endif
