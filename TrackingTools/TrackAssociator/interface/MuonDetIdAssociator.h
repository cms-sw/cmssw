#ifndef TrackAssociator_MuonDetIdAssociator_h
#define TrackAssociator_MuonDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      MuonDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: MuonDetIdAssociator.h,v 1.1 2006/12/19 01:01:00 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class MuonDetIdAssociator: public DetIdAssociator{
 public:
   MuonDetIdAssociator():DetIdAssociator(48, 48 , 0.125),geometry_(0){};
   MuonDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :DetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0){};
   
   virtual void setGeometry(const GlobalTrackingGeometry* ptr){ geometry_ = ptr; }
   
   virtual const GeomDet* getGeomDet( const DetId& id );

 protected:
   
   virtual void check_setup();
   
   virtual GlobalPoint getPosition(const DetId& id);
   
   virtual std::set<DetId> getASetOfValidDetIds();
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id);

   virtual bool insideElement(const GlobalPoint& point, const DetId& id);

   const GlobalTrackingGeometry* geometry_;
};
#endif
