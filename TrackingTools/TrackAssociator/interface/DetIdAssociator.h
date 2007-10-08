#ifndef TrackingTools_TrackAssociator_DetIdAssociator_h
#define TrackingTools_TrackAssociator_DetIdAssociator_h 1

// -*- C++ -*-
//
// Package:    TrackingTools/TrackAssociator
// Class:      DetIdAssociator
// 
/**\

 Description: Abstract base class for 3D point -> std::set<DetId>

 Implementation:
     A look up map of active detector elements in eta-phi space is 
     built to speed up access to the detector element geometry as well 
     as associated hits. The map is uniformly binned in eta and phi 
     dimensions. It is expected that the map is used to find a set of
     DetIds close to a given point, but since all methods are virtual 
     implementation may vary for various subdetectors.
**/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: DetIdAssociator.h,v 1.12.6.1 2007/10/06 05:50:12 jribnik Exp $
//
//

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include <set>
#include <vector>

class DetIdAssociator{
 public:
   enum PropagationTarget { Barrel, ForwardEndcap, BackwardEndcap };
   struct MapRange {
      float dThetaPlus;
      float dThetaMinus;
      float dPhiPlus;
      float dPhiMinus;
   };
	
   DetIdAssociator();
   DetIdAssociator(const int nPhi, const int nEta, const double etaBinSize);
   
   virtual ~DetIdAssociator();
   
   /// Preselect DetIds close to a point on the inner surface of the detector. 
   /// "iN" is a number of the adjacent bins of the map to retrieve 
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint&,
						  const int iN = 0) const;
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& direction,
						  const unsigned int iNEtaPlus,
						  const unsigned int iNEtaMinus,
						  const unsigned int iNPhiPlus,
						  const unsigned int iNPhiMinus) const;
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& direction,
						  const MapRange& mapRange) const;
   /// Preselect DetIds close to a point on the inner surface of the detector. 
   /// "d" defines the allowed range in theta-phi space:
   /// - theta is in [point.theta()-d, point.theta()+d]
   /// - phi is in [point.phi()-d, point.phi()+d]
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& point,
						  const double d = 0) const;
   /// - theta is in [point.theta()-dThetaMinus, point.theta()+dThetaPlus]
   /// - phi is in [point.phi()-dPhiMinus, point.phi()+dPhiPlus]
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& point,
						  const double dThetaPlus,
						  const double dThetaMinus,
						  const double dPhiPlus,
						  const double dPhiMinus) const;
   /// Find DetIds that satisfy given requirements
   /// - inside eta-phi cone of radius dR
   virtual std::set<DetId> getDetIdsInACone(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory,
					    const double dR) const;
   /// - DetIds crossed by the track
   virtual std::set<DetId> getCrossedDetIds(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory) const;
   /// - DetIds crossed by the track, ordered according to the order
   ///   that they were crossed by the track flying outside the detector
   virtual std::vector<DetId> getCrossedDetIdsOrdered(const std::set<DetId>&,
						      const std::vector<GlobalPoint>& trajectory) const;
   /// look-up map eta index
   virtual int iEta (const GlobalPoint&) const;
   /// look-up map phi index
   virtual int iPhi (const GlobalPoint&) const;
   /// set a specific track propagator to be used
   virtual void setPropagator(Propagator* ptr){	ivProp_ = ptr; };
   /// number of bins of the look-up map in phi dimension
   int nPhiBins() const { return nPhi_;}
   /// number of bins of the look-up map in eta dimension
   int nEtaBins() const { return nEta_;}
   /// look-up map bin size in eta dimension
   double etaBinSize() const { return etaBinSize_;};
   /// make the look-up map
   virtual void buildMap();
   /// get active detector volume
   const FiducialVolume& volume() const;

   virtual void setGeometry(const DetIdAssociatorRecord&) = 0;
   virtual const GeomDet* getGeomDet(const DetId&) const = 0;
   
 protected:
   virtual void check_setup() const
     {
	if (nEta_==0) throw cms::Exception("FatalError") << "Number of eta bins is not set.\n";
	if (nPhi_==0) throw cms::Exception("FatalError") << "Number of phi bins is not set.\n";
	// if (ivProp_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
	if (etaBinSize_==0) throw cms::Exception("FatalError") << "Eta bin size is not set.\n";
     }
   
   virtual void dumpMapContent( int, int ) const;
   virtual void dumpMapContent( int, int, int, int ) const;
   
   virtual GlobalPoint getPosition(const DetId&) const = 0;
   virtual std::set<DetId> getASetOfValidDetIds() const = 0;
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId&) const = 0;
   
   virtual bool insideElement(const GlobalPoint&, const DetId&) const = 0;
   virtual bool nearElement(const GlobalPoint& point, const DetId& id, const double distance) const {
     GlobalPoint center = getPosition(id);

     double deltaPhi(fabs(point.phi()-center.phi()));
     if(deltaPhi>M_PI) deltaPhi = fabs(deltaPhi-M_PI*2.);

     return (point.eta()-center.eta())*(point.eta()-center.eta()) + deltaPhi*deltaPhi < distance*distance;
   };
   
   // map parameters
   const int nPhi_;
   const int nEta_;
   std::set<DetId> **theMap_;
   bool theMapIsValid_;
   const double etaBinSize_;
   double maxEta_;
   double minTheta_;
   
   Propagator *ivProp_;
   // struct greater_energy : public binary_function<const CaloRecHit, const CaloRecHit, bool>
   //  {
   //	bool operator()(const CaloRecHit& x, const CaloRecHit& y) const
   //	  {  return x.energy() > y.energy();  }
   //  };
   // sort(v.begin(),v.end(), greater_energy())
   
   // Detector fiducial volume 
   // approximated as a closed cylinder with non-zero width.
   // Parameters are extracted from the active detector elements.
   FiducialVolume volume_;
};
#endif
