#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"

class TrackDetMatchInfo {
 public:
   TrackDetMatchInfo();
   /// ECAL energy 
   double ecalEnergy();
   double ecalConeEnergy();
   
   /// HCAL energy (HE+HB)
   double hcalEnergy();
   double hcalConeEnergy();
   
   /// HO energy
   double hoEnergy();
   double hoConeEnergy();

   /// Calo tower info
   double ecalTowerEnergy();
   double ecalTowerConeEnergy();
   double hcalTowerEnergy();
   double hcalTowerConeEnergy();
   double hoTowerEnergy();
   double hoTowerConeEnergy();

   /// Find detector elements with highest energy deposition
   DetId findEcalMaxDeposition();
   DetId findHcalMaxDeposition();
   DetId findTowerMaxDeposition();
   
   /// get energy of the NxN matrix around the maximal deposition
   /// N = 2*gridSize + 1
   double ecalNxNEnergy(const DetId&, int gridSize = 1);
   double hcalNxNEnergy(const DetId&, int gridSize = 1);
   double towerNxNEnergy(const DetId&, int gridSize = 1);

   int numberOfSegments() const;
   int numberOfSegmentsInStation(int station) const;
   int numberOfSegmentsInStation(int station, int detector) const;
   int numberOfSegmentsInDetector(int detector) const;
     
   math::XYZPoint trkGlobPosAtEcal;
   math::XYZPoint trkGlobPosAtHcal;
   math::XYZPoint trkGlobPosAtHO;

   std::vector<EcalRecHit> ecalRecHits;
   std::vector<EcalRecHit> crossedEcalRecHits;

   std::vector<HBHERecHit> hcalRecHits;
   std::vector<HBHERecHit> crossedHcalRecHits;
   
   std::vector<HORecHit> hoRecHits;
   std::vector<HORecHit> crossedHORecHits;

   std::vector<CaloTower> towers;
   std::vector<CaloTower> crossedTowers;
   
   std::vector<MuonChamberMatch> chambers;

   SimTrackRef simTrackRef_;
   reco::TrackRef trackRef_;

   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodCalo;
   bool isGoodHO;
   bool isGoodMuon;
   
};
#endif
