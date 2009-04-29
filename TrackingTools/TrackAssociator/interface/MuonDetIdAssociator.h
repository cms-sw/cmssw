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
// $Id: MuonDetIdAssociator.h,v 1.5 2009/04/29 12:15:08 jribnik Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TAMuonChamberMatch.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonDetIdAssociator: public DetIdAssociator{
 public:
   MuonDetIdAssociator():DetIdAssociator(48, 48 , 0.125),geometry_(0),cscbadchambers_(0),includeBadChambers_(0){};
   MuonDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :DetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0),cscbadchambers_(0),includeBadChambers_(0){};

   MuonDetIdAssociator(const edm::ParameterSet& pSet)
     :DetIdAssociator(pSet.getParameter<int>("nPhi"),pSet.getParameter<int>("nEta"),pSet.getParameter<double>("etaBinSize")),geometry_(0),cscbadchambers_(0),includeBadChambers_(pSet.getParameter<bool>("includeBadChambers")){};
   
   virtual void setGeometry(const GlobalTrackingGeometry* ptr){ geometry_ = ptr; }

   virtual void setGeometry(const DetIdAssociatorRecord& iRecord){
      edm::ESHandle<GlobalTrackingGeometry> geometryH;
      iRecord.getRecord<GlobalTrackingGeometryRecord>().get(geometryH);
      setGeometry(geometryH.product());
   };

   virtual void setCSCBadChambers(const CSCBadChambers* ptr){ cscbadchambers_ = ptr; }

   virtual void setConditions(const DetIdAssociatorRecord& iRecord){
      edm::ESHandle<CSCBadChambers> cscbadchambersH;
      iRecord.getRecord<CSCBadChambersRcd>().get(cscbadchambersH);
      setCSCBadChambers(cscbadchambersH.product());
   };

   // This method was taken from CalibMuon/CSCCalibration/src/CSCConditions.cc
   // This copy and paste job will have to do until the CSC guys provide a
   // more proper CSCConditionsRcd, or basically just move the isInBadChamber
   // method into the CSCBadChambers class.
   virtual bool isBadCSCChamber( const CSCDetId& id ) const {
      if (cscbadchambers_->numberOfBadChambers == 0) return false;

      short int iri = id.ring();
      if ( iri == 4 ) iri = 1; // reset ME1A to ME11
      CSCIndexer indexer;
      int ilin = indexer.chamberIndex( id.endcap(), id.station(), iri, id.chamber() );
      std::vector<int>::const_iterator badbegin = cscbadchambers_->chambers.begin();
      std::vector<int>::const_iterator badend = cscbadchambers_->chambers.end();
      std::vector<int>::const_iterator it = std::find( badbegin, badend, ilin );
      if ( it != badend ) return true; // id is in the list of bad chambers
      else return false;
   };
   
   virtual const GeomDet* getGeomDet( const DetId& id ) const;

 protected:
   
   virtual void check_setup() const;
   
   virtual GlobalPoint getPosition(const DetId& id) const;
   
   virtual std::set<DetId> getASetOfValidDetIds() const;
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id) const;

   virtual bool insideElement(const GlobalPoint& point, const DetId& id) const;

   const GlobalTrackingGeometry* geometry_;

   const CSCBadChambers* cscbadchambers_;
   bool includeBadChambers_;
};
#endif
