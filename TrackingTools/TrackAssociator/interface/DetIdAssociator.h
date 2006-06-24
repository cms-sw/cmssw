#ifndef MuonTest_MuonTest_DetIdAssociator_h
#define MuonTest_MuonTest_DetIdAssociator_h 1

// -*- C++ -*-
//
// Package:    MuonTest
// Class:      MuonTest
// 
/**\class MuonTest MuonTest.cc src/MuonTest/src/MuonTest.cc

 Description: Abstract base class for 3D point -> std::set<DetId>

 Implementation:
     It is expected that the mapping is performed using a 2D array of 
     DetId sets, to get fast a set of possible DetIds for a given 
     direction. Since all methods are virtual a practical 
     implementation can use other approaches.
**/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: DetIdAssociator.h,v 1.1 2006/06/09 17:30:20 dmytro Exp $
//
//

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>
#include <vector>


class DetIdAssociator{
 public:
   DetIdAssociator():theMap_(0),nPhi_(0),nEta_(0),etaBinSize_(0),ivProp_(0){};
   DetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :theMap_(0),nPhi_(nPhi),nEta_(nEta),etaBinSize_(etaBinSize),ivProp_(0){};
   
   virtual ~DetIdAssociator(){};
   virtual std::vector<GlobalPoint> getTrajectory( const FreeTrajectoryState&,
						   const std::vector<GlobalPoint>&);
   // find DetIds arround given direction
   // idR is a number of the adjacent bins to retrieve 
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint&, 
						  const int idR = 0);
   // dR is a cone radius in eta-phi
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& point,
						  const double dR = 0)
     {
	int etaIdR = int(dR/etaBinSize_); 
	int phiIdR = int(dR/(2*3.1416)*nPhi_);
	if (etaIdR>phiIdR)
	  return getDetIdsCloseToAPoint(point, 1+etaIdR);
	else
	  return getDetIdsCloseToAPoint(point, 1+phiIdR);
     }
   
   virtual std::set<DetId> getDetIdsInACone(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory,
					    const double );
   virtual std::set<DetId> getCrossedDetIds(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory);

   virtual int iEta (const GlobalPoint&);
   virtual int iPhi (const GlobalPoint&);
   virtual void setPropagator(Propagator* ptr){	ivProp_ = ptr; };
 
 protected:
   virtual void check_setup()
     {
	if (nEta_==0) throw cms::Exception("FatalError") << "Number of eta bins is not set.\n";
	if (nPhi_==0) throw cms::Exception("FatalError") << "Number of phi bins is not set.\n";
	if (ivProp_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
	if (etaBinSize_==0) throw cms::Exception("FatalError") << "Eta bin size is not set.\n";
     }
   
   virtual void buildMap();
   virtual GlobalPoint getPosition(const DetId&) = 0;
   virtual std::set<DetId> getASetOfValidDetIds() = 0;
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId&) = 0;
   
   virtual bool insideElement(const GlobalPoint&, const DetId&) = 0;
   virtual bool nearElement(const GlobalPoint& point, const DetId& id, const double distance)
     {
	GlobalPoint center = getPosition(id);
	return sqrt(pow(point.eta()-center.eta(),2)+pow(point.phi()-center.phi(),2)) < distance;
     };
   
   std::vector<std::vector<std::set<DetId> > >* theMap_;
   const int nPhi_;
   const int nEta_;
   const double etaBinSize_;
   Propagator *ivProp_;
};
#endif
