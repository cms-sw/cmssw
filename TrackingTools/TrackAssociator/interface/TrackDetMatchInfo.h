#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrackAssociator/interface/MuonSegmentMatch.h"

class TrackDetMatchInfo {
 public:
   /// ECAL energy 
   double ecalEnergy();
   /// HCAL energy 
   double hcalEnergy();
   
   double ecalConeEnergy();
   double hcalConeEnergy();

   double ecalNeighborHitEnergy(int gridSize);
   double ecalNeighborTowerEnergy(int gridSize);
   double hcalNeighborEnergy(int gridSize);
   
   double outerHcalEnergy();
   double outerHcalConeEnergy();

   double ecalTowerEnergy();
   double ecalTowerConeEnergy();

   int numberOfSegments(){ return segments.size(); }
   int numberOfSegmentsInStation(int station) const;
   int numberOfSegmentsInStation(int station, int detector) const;
   int numberOfSegmentsInDetector(int detector) const;
     
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

   SimTrackRef simTrackRef_;
   reco::TrackRef trackRef_;

   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodMuon;
};
#endif
