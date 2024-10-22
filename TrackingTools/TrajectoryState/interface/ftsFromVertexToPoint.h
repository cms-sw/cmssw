#ifndef TrackingTools_TrajectoryState_ftsFromVertexToPoint_h
#define TrackingTools_TrajectoryState_ftsFromVertexToPoint_h
// -*- C++ -*-
//
//
/**
 *  Generates a FreeTrajectoryState from a given measured point, vertex momentum and charge.
 *  auto fts = trackingTools::ftsFromVertexToPoint(magField, xmeas, xvert, momentum, charge);
 *  gives a FreeTrajectoryState of a track which comes from xvert to xmeas. 
 *  The curvature of fts is computed taken into account the bend in the magnetic field.       

*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//     Modified by:  Jonas Rembser
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

class MagneticField;

namespace trackingTools {
  FreeTrajectoryState ftsFromVertexToPoint(MagneticField const& magField,
                                           GlobalPoint const& xmeas,
                                           GlobalPoint const& xvert,
                                           float momentum,
                                           TrackCharge charge);
}

#endif
