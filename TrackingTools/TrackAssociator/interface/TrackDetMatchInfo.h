#ifndef TrackAssociator_TrackDetMatchInfo_h
#define TrackAssociator_TrackDetMatchInfo_h

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class TrackDetMatchInfo {
 public:
   enum EnergyType { EcalRecHits, HcalRecHits, HORecHits, TowerTotal, TowerEcal, TowerHcal, TowerHO };
   
   TrackDetMatchInfo();
   
   /// energy in detector elements crossed by the track by types
   double crossedEnergy( EnergyType );
   
   /// cone energy around the track direction at the origin (0,0,0)
   /// ( not well defined for tracks originating away from IP)
   double coneEnergy( double dR, EnergyType );
   
   /// Find detector elements with highest energy deposition
   DetId findMaxDeposition( EnergyType );
   
   /// get energy of the NxN shape (N = 2*gridSize + 1) around given detector element
   double nXnEnergy(const DetId&, EnergyType, int gridSize = 1);
   
   /// get energy of the NxN shape (N = 2*gridSize + 1) around track projection
   double nXnEnergy(EnergyType, int gridSize = 1);

   /// Track position at different parts of the calorimeter
   math::XYZPoint trkGlobPosAtEcal;
   math::XYZPoint trkGlobPosAtHcal;
   math::XYZPoint trkGlobPosAtHO;
   
   bool isGoodEcal;
   bool isGoodHcal;
   bool isGoodCalo;
   bool isGoodHO;
   bool isGoodMuon;
   
   /// hits in the cone
   std::vector<EcalRecHit> ecalRecHits;
   std::vector<HBHERecHit> hcalRecHits;
   std::vector<HORecHit>   hoRecHits;
   std::vector<CaloTower>  towers;

   /// hits in detector elements crossed by a track
   std::vector<EcalRecHit> crossedEcalRecHits;
   std::vector<HBHERecHit> crossedHcalRecHits;
   std::vector<HORecHit>   crossedHORecHits;
   std::vector<CaloTower>  crossedTowers;

   /// detector elements crossed by a track 
   /// (regardless of whether energy was deposited or not)
   std::vector<DetId>      crossedEcalIds;
   std::vector<DetId>      crossedHcalIds;
   std::vector<DetId>      crossedHOIds;
   std::vector<DetId>      crossedTowerIds;
   
   std::vector<MuonChamberMatch> chambers;

   /// track info
   FreeTrajectoryState stateAtIP;
   
   /// MC truth info
   const SimTrack* simTrack;
   double ecalTrueEnergy;
   double hcalTrueEnergy;
   
   /// Obsolete methods and data members for backward compatibility.
   /// Will be removed in future releases.
   reco::TrackRef trackRef_;
   SimTrackRef simTrackRef_;
   
   double ecalCrossedEnergy();
   double ecalConeEnergy();
   double hcalCrossedEnergy();
   double hcalConeEnergy();
   double hoCrossedEnergy();
   double hoConeEnergy();

   double ecalTowerEnergy() { return crossedEnergy(TowerEcal); }
   double ecalTowerConeEnergy() { return coneEnergy(999,TowerEcal); }
   double hcalTowerEnergy() { return crossedEnergy(TowerHcal); }
   double hcalTowerConeEnergy() { return coneEnergy(999, TowerHcal); }
   double hoTowerEnergy() { return crossedEnergy(TowerHO); }
   double hoTowerConeEnergy() { return coneEnergy(999, TowerHO); }

   double ecalEnergy() { return ecalCrossedEnergy(); }
   double hcalEnergy() { return hcalCrossedEnergy(); }
   double hoEnergy() { return hoCrossedEnergy(); }
   
   int numberOfSegments() const;
   int numberOfSegmentsInStation(int station) const;
   int numberOfSegmentsInStation(int station, int detector) const;
   int numberOfSegmentsInDetector(int detector) const;
   
   void setGeometry( edm::ESHandle<CaloGeometry> geometry ) { caloGeometry = geometry; }
   void setMomentumAtIP( const GlobalVector& p4 ) { momentum = p4; }
   GlobalVector getMomentumAtIP() { return momentum; }

 private:
   bool insideCone(const DetId&, const double);
   edm::ESHandle<CaloGeometry> caloGeometry;
   GlobalVector momentum;
   
};
#endif
