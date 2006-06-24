#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "TrackingTools/TrackAssociator/interface/MuonSegmentMatch.h"

class TrackDetMatchInfo {
 public:
   /// ECAL energy 
   double ecalEnergy();
   /// HCAL energy 
   double hcalEnergy();
   
   double ecalConeEnergy();
   double hcalConeEnergy();
   
   double outerHcalEnergy();
   double outerHcalConeEnergy();
   
   int numberOfSegments(){ return segments.size(); }
     
   double dX(int i){ 
      if (numberOfSegments()<=i) return -999.;
      return segments[i].segmentLocalPosition.x()-segments[i].trajectoryLocalPosition.x();
   }
   
   double dY(int i){ 
      if (numberOfSegments()<=i) return -999.;
      return segments[i].segmentLocalPosition.y()-segments[i].trajectoryLocalPosition.y();
   }
   
   double errXX(int i){ 
      if (numberOfSegments()<=i) return -999.;
      return segments[i].trajectoryLocalErrorXX+segments[i].segmentLocalErrorXX;
   }
    
   double errYY(int i){ 
      if (numberOfSegments()<=i) return -999.;
      return segments[i].trajectoryLocalErrorYY+segments[i].segmentLocalErrorYY;
   }
     
   math::XYZPoint trkGlobPosAtEcal;
   std::vector<EcalRecHit> ecalRecHits;
   std::vector<EcalRecHit> crossedEcalRecHits;
   
   math::XYZPoint trkGlobPosAtHcal;
   std::vector<CaloTower> towers;
   std::vector<CaloTower> crossedTowers;
   
   std::vector<MuonSegmentMatch> segments;
};

#endif
