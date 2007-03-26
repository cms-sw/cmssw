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
// $Id: DetIdAssociator.h,v 1.8 2007/03/08 04:19:26 dmytro Exp $
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
#include <set>
#include <vector>


class DetIdAssociator{
 public:
   enum PropagationTarget { Barrel, ForwardEndcap, BackwardEndcap };
	
   DetIdAssociator():theMap_(0),nPhi_(0),nEta_(0),etaBinSize_(0),ivProp_(0){};
   DetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :theMap_(0),nPhi_(nPhi),nEta_(nEta),etaBinSize_(etaBinSize),ivProp_(0){};
   
   virtual ~DetIdAssociator(){};
   
   /// Preselect DetIds in a given direction using look-up maps
   /// idR is a number of the adjacent bins to retrieve 
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint&, 
						  const int idR = 0);
   /// dR is a cone radius in eta-phi
   virtual std::set<DetId> getDetIdsCloseToAPoint(const GlobalPoint& point,
						  const double dR = 0);
   /// Find DetIds that satisfy given requirements
   /// - cone radius
   virtual std::set<DetId> getDetIdsInACone(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory,
					    const double );
   /// - DetIds crossed by the track
   virtual std::set<DetId> getCrossedDetIds(const std::set<DetId>&,
					    const std::vector<GlobalPoint>& trajectory);
   /// - DetIds crossed by the track, ordered according to the order
   ///   that they were crossed by the track flying outside the detector
   virtual std::vector<DetId> getCrossedDetIdsOrdered(const std::set<DetId>&,
						      const std::vector<GlobalPoint>& trajectory);
   /// look-up map eta index
   virtual int iEta (const GlobalPoint&);
   /// look-up map phi index
   virtual int iPhi (const GlobalPoint&);
   /// set a specific track propagator to be used
   virtual void setPropagator(Propagator* ptr){	ivProp_ = ptr; };
   /// number of bins of the look-up map in phi dimension
   int nPhiBins(){ return nPhi_;}
   /// number of bins of the look-up map in eta dimension
   int nEtaBins(){ return nEta_;}
   /// look-up map bin size in eta dimension
   double etaBinSize(){ return etaBinSize_;};
   /// make the look-up map
   virtual void buildMap();
   /// get active detector volume
   FiducialVolume volume();
   
 protected:
   virtual void check_setup()
     {
	if (nEta_==0) throw cms::Exception("FatalError") << "Number of eta bins is not set.\n";
	if (nPhi_==0) throw cms::Exception("FatalError") << "Number of phi bins is not set.\n";
	// if (ivProp_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
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
