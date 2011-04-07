#ifndef TrackAssociator_TAMuonChamberMatch_h
#define TrackAssociator_TAMuonChamberMatch_h

/**
 * 
 *  Description:
 *   An auxiliary class to store a muon trajetory matching to chambers.
 *   It's used to store information of about crossed muon detector 
 *   elements regardless of whether a segment was reconstructed or not
 *   for a given chamber.
 * 
 */

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/TrackAssociator/interface/TAMuonSegmentMatch.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class TAMuonChamberMatch {
 public:
   int station() const;
   std::string info() const;
   int detector() const { return id.subdetId(); }

   /// distance sign convention: negative - crossed chamber, positive - missed chamber
   std::vector<TAMuonSegmentMatch> segments;
   float localDistanceX;
   float localDistanceY;
   TrajectoryStateOnSurface tState;
   DetId id;
};
#endif
