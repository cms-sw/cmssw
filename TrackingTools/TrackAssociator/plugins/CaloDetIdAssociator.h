#ifndef TrackAssociator_CaloDetIdAssociator_h
#define TrackAssociator_CaloDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CaloDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: CaloDetIdAssociator.h,v 1.1 2011/04/07 09:12:02 innocent Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CaloDetIdAssociator: public DetIdAssociator{
 public:
   CaloDetIdAssociator():DetIdAssociator(72, 70 ,0.087),geometry_(0){};
   CaloDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :DetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0){};

   CaloDetIdAssociator(const edm::ParameterSet& pSet)
     :DetIdAssociator(pSet.getParameter<int>("nPhi"),pSet.getParameter<int>("nEta"),pSet.getParameter<double>("etaBinSize")),geometry_(0){};
   
   virtual void setGeometry(const CaloGeometry* ptr){ geometry_ = ptr; };

   virtual void setGeometry(const DetIdAssociatorRecord& iRecord);

   virtual const GeomDet* getGeomDet(const DetId& id) const { return 0; };

   virtual const char* name() const { return "CaloTowers"; }

 protected:
   virtual void check_setup() const;
   
   virtual GlobalPoint getPosition(const DetId& id) const;
   
   virtual const std::vector<DetId>& getValidDetIds( unsigned int subDetectorIndex ) const;
   
   virtual std::pair<const_iterator, const_iterator> getDetIdPoints(const DetId& id) const;

   virtual bool insideElement(const GlobalPoint& point, const DetId& id) const {
      return  geometry_->getSubdetectorGeometry(id)->getGeometry(id)->inside(point);
   };

   virtual bool crossedElement(const GlobalPoint&, 
			       const GlobalPoint&, 
			       const DetId& id,
			       const double tolerance = -1,
			       const SteppingHelixStateInfo* = 0 ) const;
   const CaloGeometry* geometry_;
   std::vector<GlobalPoint> dummy_;
};
#endif
