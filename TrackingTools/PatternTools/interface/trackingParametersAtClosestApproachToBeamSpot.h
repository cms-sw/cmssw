#ifndef TrackingTools_PatternTools_trackingParametersAtClosestApproachToBeamSpot_h
#define TrackingTools_PatternTools_trackingParametersAtClosestApproachToBeamSpot_h
// -*- C++ -*-
//
// Package:     TrackingTools/PatternTools
// Class  :     trackingParametersAtClosestApproachToBeamSpot
// 
/**\function trackingParametersAtClosestApproachToBeamSpot "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"

 Description: Given the momentum and origin of a particle, calculate the tracking parameters at its closest approach to the beam spot

 Usage:
   Value of first in return value is true if parameters were properly calculated.
*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 02 Jan 2015 19:32:32 GMT
//

// system include files

// user include files
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

// forward declarations

namespace reco {
  std::pair<bool,reco::TrackBase::ParameterVector> 
    trackingParametersAtClosestApproachToBeamSpot(const Basic3DVector<double>& vertex,
                                                  const Basic3DVector<double>& momAtVtx,
                                                  float charge,
                                                  const MagneticField& magField,
                                                  const BeamSpot& bs);
}

#endif
