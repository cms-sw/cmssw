// -*- C++ -*-
//
// Package:     TrackingTools/PatternTools
// Class  :     trackingParametersAtClosestApproachToBeamSpot
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 02 Jan 2015 19:32:37 GMT
//

// system include files

// user include files
#include "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "DataFormats/GeometryVector/interface/Pi.h"


std::pair<bool,reco::TrackBase::ParameterVector> 
reco::trackingParametersAtClosestApproachToBeamSpot(const Basic3DVector<double>& vertex,
                                                    const Basic3DVector<double>& momAtVtx,
                                                    float charge,
                                                    const MagneticField& magField,
                                                    const BeamSpot& bs) {
  TrackBase::ParameterVector sParameters;
  try {
    FreeTrajectoryState ftsAtProduction(GlobalPoint(vertex.x(),vertex.y(),vertex.z()),
					GlobalVector(momAtVtx.x(),momAtVtx.y(),momAtVtx.z()),
					TrackCharge(charge),
					&magField);
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,bs);//as in TrackProducerAlgorithm
    
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    sParameters[0] = tsAtClosestApproach.trackStateAtPCA().charge()/p.mag();
    sParameters[1] = Geom::halfPi() - p.theta();
    sParameters[2] = p.phi();
    sParameters[3] = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
    sParameters[4] = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
    
    return std::make_pair(true,sParameters);
  } catch ( cms::Exception const& ) {
    return std::make_pair(false,sParameters);
  }
}

