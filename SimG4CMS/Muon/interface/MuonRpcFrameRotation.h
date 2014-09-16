#ifndef MuonRpcFrameRotation_H
#define MuonRpcFrameRotation_H

/** \class MuonRpcFrameRotation
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
class DDCompactView;

class MuonRpcFrameRotation : public MuonFrameRotation {
 public:
  MuonRpcFrameRotation( const DDCompactView& cpv );
  virtual ~MuonRpcFrameRotation();
  virtual Local3DPoint transformPoint(const Local3DPoint &, const G4Step *) const;
 private:
  MuonG4Numbering* g4numbering;
  int theRegion;
};


#endif
