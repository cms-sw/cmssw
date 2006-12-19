#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"

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

   int numberOfSegments() const;
   int numberOfSegmentsInStation(int station) const;
   int numberOfSegmentsInStation(int station, int detector) const;
   int numberOfSegmentsInDetector(int detector) const;
     
   math::XYZPoint trkGlobPosAtEcal;
   std::vector<EcalRecHit> ecalRecHits;
   std::vector<EcalRecHit> crossedEcalRecHits;
   
   math::XYZPoint trkGlobPosAtHcal;
   std::vector<CaloTower> towers;
   std::vector<CaloTower> crossedTowers;
   
   std::vector<MuonChamberMatch> chambers;

   SimTrackRef simTrackRef_;
   reco::TrackRef trackRef_;

   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodMuon;
};
#endif
