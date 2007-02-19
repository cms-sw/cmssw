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
   enum TowerEnergyType { Total, Ecal, Hcal, HO };
   
   TrackDetMatchInfo();
   
   /// ECAL energy 
   double ecalCrossedEnergy();
   double ecalConeEnergy();
   
   /// HCAL energy (HE+HB)
   double hcalCrossedEnergy();
   double hcalConeEnergy();
   
   /// HO energy
   double hoCrossedEnergy();
   double hoConeEnergy();

   /// Calo tower energy
   double towerCrossedEnergy( TowerEnergyType type = Total );
   double towerConeEnergy( TowerEnergyType type = Total );

   /// Find detector elements with highest energy deposition
   DetId findEcalMaxDeposition();
   DetId findHcalMaxDeposition();
   DetId findTowerMaxDeposition( TowerEnergyType type = Total );
   
   /// get energy of the NxN matrix around the maximal deposition
   /// N = 2*gridSize + 1
   double ecalNxNEnergy(const DetId&, int gridSize = 1);
   double hcalNxNEnergy(const DetId&, int gridSize = 1);
   double towerNxNEnergy(const DetId&, int gridSize = 1, TowerEnergyType type = Total);

   /// Track position at different parts of the calorimeter
   math::XYZPoint trkGlobPosAtEcal;
   math::XYZPoint trkGlobPosAtHcal;
   math::XYZPoint trkGlobPosAtHO;
   
   std::vector<EcalRecHit> coneEcalRecHits;
   std::vector<EcalRecHit> crossedEcalRecHits;
   std::vector<DetId>      crossedEcalIds;

   std::vector<HBHERecHit> coneHcalRecHits;
   std::vector<HBHERecHit> crossedHcalRecHits;
   std::vector<DetId>      crossedHcalIds;
   
   std::vector<HORecHit> coneHORecHits;
   std::vector<HORecHit> crossedHORecHits;
   std::vector<DetId>    crossedHOIds;

   std::vector<CaloTower> coneTowers;
   std::vector<CaloTower> crossedTowers;
   std::vector<DetId>     crossedTowerIds;
   
   std::vector<MuonChamberMatch> chambers;

   SimTrackRef simTrackRef_;
   reco::TrackRef trackRef_;

   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodCalo;
   bool isGoodHO;
   bool isGoodMuon;
   
   /// Obsolete methods and data members for backward compatibility.
   /// Will be removed in future releases.
   
   double ecalTowerEnergy() { return towerCrossedEnergy(Ecal); }
   double ecalTowerConeEnergy() { return towerConeEnergy(Ecal); }
   double hcalTowerEnergy() { return towerCrossedEnergy(Hcal); }
   double hcalTowerConeEnergy() { return towerConeEnergy(Hcal); }
   double hoTowerEnergy() { return towerCrossedEnergy(HO); }
   double hoTowerConeEnergy() { return towerConeEnergy(HO); }

   double ecalEnergy() { return ecalCrossedEnergy(); }
   double hcalEnergy() { return hcalCrossedEnergy(); }
   double hoEnergy() { return hoCrossedEnergy(); }
   
   int numberOfSegments() const;
   int numberOfSegmentsInStation(int station) const;
   int numberOfSegmentsInStation(int station, int detector) const;
   int numberOfSegmentsInDetector(int detector) const;

   std::vector<EcalRecHit>& ecalRecHits;
   std::vector<HBHERecHit>& hcalRecHits;
   std::vector<HORecHit>& hoRecHits;
   std::vector<CaloTower> towers;
   
   
};
#endif
