#ifndef MuonGemFrameRotation_H
#define MuonGemFrameRotation_H

/** \class MuonGemFrameRotation
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

#include "G4Step.hh"

class DDCompactView;

class MuonGemFrameRotation : public MuonFrameRotation {

public:
  MuonGemFrameRotation( const DDCompactView& cpv );
  virtual ~MuonGemFrameRotation();
  virtual Local3DPoint transformPoint(const Local3DPoint &, const G4Step *) const;

private:
  MuonG4Numbering* g4numbering;
  int              theSectorLevel;
};


#endif
