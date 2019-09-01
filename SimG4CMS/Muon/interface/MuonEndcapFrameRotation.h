#ifndef SimG4CMS_Muon_MuonEndcapFrameRotation_H
#define SimG4CMS_Muon_MuonEndcapFrameRotation_H

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
  ~MuonEndcapFrameRotation() override{};
  Local3DPoint transformPoint(const Local3DPoint &, const G4Step *) const override;

private:
};

#endif
