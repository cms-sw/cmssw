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
// $Id: DetIdAssociator.h,v 1.4 2006/12/19 01:01:00 dmytro Exp $
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
   enum PropagationTarget { Barrel, ForwardEndcap, BackwardEndcap };
	
   DetIdAssociator():theMap_(0),nPhi_(0),nEta_(0),etaBinSize_(0),ivProp_(0){};
   DetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :theMap_(0),nPhi_(nPhi),nEta_(nEta),etaBinSize_(etaBinSize),ivProp_(0){};
   
   virtual ~DetIdAssociator(){};
   // get track trajectory for a set of limiting surfaces of given radius and Z.
   // thetaOverlap defines a limit at which a track is propagated to both
   // barrel and endcap if close to the edge between them.
   virtual std::vector<GlobalPoint> getTrajectory( FreeTrajectoryState& ftsStart,
						   const std::vector<GlobalPoint>& surfaces,
						   const double etaOverlap = 0.1);
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
   int nPhiBins(){ return nPhi_;};
   int nEtaBins(){ return nEta_;};
   double etaBinSize(){ return etaBinSize_;};
   virtual void buildMap();
   
 protected:
   virtual void check_setup()
     {
	if (nEta_==0) throw cms::Exception("FatalError") << "Number of eta bins is not set.\n";
	if (nPhi_==0) throw cms::Exception("FatalError") << "Number of phi bins is not set.\n";
	if (ivProp_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
	if (etaBinSize_==0) throw cms::Exception("FatalError") << "Eta bin size is not set.\n";
     }
   
   virtual void dumpMapContent( int, int );
   virtual void dumpMapContent( int, int, int, int );
   
   virtual GlobalPoint getPosition(const DetId&) = 0;
   virtual std::set<DetId> getASetOfValidDetIds() = 0;
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId&) = 0;
   
   virtual bool insideElement(const GlobalPoint&, const DetId&) = 0;
   virtual bool nearElement(const GlobalPoint& point, const DetId& id, const double distance) {
     GlobalPoint center = getPosition(id);

     double pi = 3.1415926535;

     double deltaPhi(fabs(point.phi()-center.phi()));
     if(deltaPhi>pi) deltaPhi = fabs(deltaPhi-pi*2.);

     return sqrt(pow(point.eta()-center.eta(),2)+deltaPhi*deltaPhi) < distance;
   };
   
   std::vector<std::vector<std::set<DetId> > >* theMap_;
   const int nPhi_;
   const int nEta_;
   const double etaBinSize_;
   Propagator *ivProp_;
};
#endif
