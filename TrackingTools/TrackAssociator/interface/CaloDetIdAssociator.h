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
// $Id: CaloDetIdAssociator.h,v 1.2 2006/08/25 17:35:40 jribnik Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/DetIdInfo.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"

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
      std::set<DetId> setOfValidIds;
      std::vector<DetId> vectOfValidIds = geometry_->getValidDetIds(DetId::Calo, 1);
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id){
      if(! geometry_->getSubdetectorGeometry(id)){
         LogDebug("CaloDetIdAssociator") << "Cannot find sub-detector geometry for " << id.rawId() <<"\n";
      } else {
         if(! geometry_->getSubdetectorGeometry(id)->getGeometry(id)) {
            LogDebug("CaloDetIdAssociator") << "Cannot find CaloCell geometry for " << id.rawId() <<"\n";
         } else {
            const std::vector<GlobalPoint>& points( geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getCorners() );
	    for(std::vector<GlobalPoint>::const_iterator itr=points.begin();itr!=points.end();itr++)
	      {
		 //FIX ME
		 // the following is a protection from the NaN bug in CaloGeometry
		 if(isnan(itr->mag())||itr->mag()>1e5) { //Detector parts cannot be 1 km away or be NaN
		    edm::LogWarning("DetIdAssociator") << "Critical error! Bad calo detector unit geometry:\n\tDetId:" 
		      << id.rawId() << "\t mag(): " << itr->mag() << "\n" << DetIdInfo::info( id )
			<< "\nSkipped the element";
		    return std::vector<GlobalPoint>();
		 }
	      }
	    return points;
            // points.push_back(getPosition(id));
         }
      }
      return std::vector<GlobalPoint>();
   };

   virtual bool insideElement(const GlobalPoint& point, const DetId& id){
      return  geometry_->getSubdetectorGeometry(id)->getGeometry(id)->inside(point);
   };

   const CaloGeometry* geometry_;
};
#endif
