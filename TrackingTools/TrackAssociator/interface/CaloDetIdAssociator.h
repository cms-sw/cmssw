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
// $Id: CaloDetIdAssociator.h,v 1.1 2006/06/09 17:30:20 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

class CaloDetIdAssociator: public DetIdAssociator{
 public:
   CaloDetIdAssociator():DetIdAssociator(72, 70 ,0.087),geometry_(0){};
   CaloDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :DetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0){};
   
   virtual void setGeometry(const CaloGeometry* ptr){ geometry_ = ptr; };
   
 protected:
   virtual void check_setup()
     {
	DetIdAssociator::check_setup();
	if (geometry_==0) throw cms::Exception("CaloGeometry is not set");
     };
   
   virtual GlobalPoint getPosition(const DetId& id){
      return geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
   };
   
   virtual std::set<DetId> getASetOfValidDetIds(){
      std::set<DetId> validIds;
      uint32_t detIdPrefix = 41 << 25; // CaloTower det=5, subdet = 1
      for(uint32_t z=0; z<=1; z++) // eta sign
	for(uint32_t i=1; i<=72;i++) // phi index range
	  for(uint32_t j=1; j<=32;j++){ //eta index range
	     uint32_t rawDetId = detIdPrefix+i+(j<<7)+(z<<13);
	     validIds.insert(DetId(rawDetId));
	  }
      return validIds;
   };
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id){
      std::vector<GlobalPoint> points;
      if(! geometry_->getSubdetectorGeometry(id)){
	 LogDebug("CaloDetIdAssociator") << "Cannot find sub-detector geometry for " << id.rawId() <<"\n";
      } else {
	 if(! geometry_->getSubdetectorGeometry(id)->getGeometry(id)) {
	    LogDebug("CaloDetIdAssociator") << "Cannot find CaloCell geometry for " << id.rawId() <<"\n";
	 } else {
	    points = geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getCorners();
	    points.push_back(getPosition(id));
	 }
      }
      
      return  points;
   };

   virtual bool insideElement(const GlobalPoint& point, const DetId& id){
      return  geometry_->getSubdetectorGeometry(id)->getGeometry(id)->inside(point);
   };

   const CaloGeometry* geometry_;
};
#endif
