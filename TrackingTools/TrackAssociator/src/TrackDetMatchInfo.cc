#include <map>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/DetIdInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

int TrackDetMatchInfo::numberOfSegments() const {
   int numSegments = 0;
   for(std::vector<MuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station) const {
   int numSegments = 0;
   for(std::vector<MuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->station()==station) numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station, int detector) const {
   int numSegments = 0;
   for(std::vector<MuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->station()==station&&chamber->detector()==detector) numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInDetector(int detector) const {
   int numSegments = 0;
   for(std::vector<MuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->detector()==detector) numSegments += chamber->segments.size();
   return numSegments;
}

///////////////////////////

double TrackDetMatchInfo::ecalCrossedEnergy()
{
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=crossedEcalRecHits.begin(); hit!=crossedEcalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::hcalCrossedEnergy()
{
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit = crossedHcalRecHits.begin(); hit != crossedHcalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::hoCrossedEnergy()
{
   double energy(0);
   for(std::vector<HORecHit>::const_iterator hit = crossedHORecHits.begin(); hit != crossedHORecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::ecalConeEnergy()
{
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::hcalConeEnergy()
{
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::hoConeEnergy()
{
   double energy(0);
   for(std::vector<HORecHit>::const_iterator hit=hoRecHits.begin(); hit!=hoRecHits.end(); hit++)
     energy += hit->energy();
   return energy;
}

double TrackDetMatchInfo::towerConeEnergy( TowerEnergyType type )
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
     {
	switch (type) {
	 case Total:
	   energy += hit->energy();
	   break;
	 case Ecal:
	   energy += hit->emEnergy();
	   break;
	 case Hcal:
	   energy += hit->hadEnergy();
	   break;
	 case HO:
	   energy += hit->energy();
	   break;
	 default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo tower energy type: " << type;
	}
     }
   return energy;
}

double TrackDetMatchInfo::towerCrossedEnergy( TowerEnergyType type )
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
     {
	switch (type) {
	 case Total:
	   energy += hit->energy();
	   break;
	 case Ecal:
	   energy += hit->emEnergy();
	   break;
	 case Hcal:
	   energy += hit->hadEnergy();
	   break;
	 case HO:
	   energy += hit->energy();
	   break;
	 default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo tower energy type: " << type;
	}
     }
   return energy;
}

//////////////////////////////////////////////////

double TrackDetMatchInfo::towerNxNEnergy(const DetId& id, int gridSize, TowerEnergyType type )
{
   if ( id.rawId() == 0 ) return -9999;
   if ( id.det() != DetId::Calo ) {
      edm::LogWarning("TrackAssociator") << "Wrong DetId. Expected CaloTower, but found:\n" <<
	DetIdInfo::info(id)<<"\n";
      return -99999;
   }
   CaloTowerDetId centerId(id);
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++) {
      CaloTowerDetId neighborId(hit->id());
      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
		      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
      if(  dEta <= gridSize && dPhi <= gridSize ) {
	switch (type) {
	 case Total:
	   energy += hit->energy();
	   break;
	 case Ecal:
	   energy += hit->emEnergy();
	   break;
	 case Hcal:
	   energy += hit->hadEnergy();
	   break;
	 case HO:
	   energy += hit->energy();
	   break;
	 default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo tower energy type: " << type;
	}
      }
   }
   return energy;
}

double TrackDetMatchInfo::hcalNxNEnergy(const DetId& id, int gridSize)
{
   if ( id.rawId() == 0 ) return -9999;
   if( id.det() != DetId::Hcal || (id.subdetId() != HcalBarrel && id.subdetId() != HcalEndcap) ) {
      edm::LogWarning("TrackAssociator") << "Wrong DetId. Expected HE or HB, but found:\n" <<
	DetIdInfo::info(id)<<"\n";
      return -99999;
   }
   HcalDetId centerId(id);
   double energy(0);
   for(std::vector<HBHERecHit>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++) {
      HcalDetId neighborId(hit->id());
      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
		      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
      if(  dEta <= gridSize && dPhi <= gridSize ) energy += hit->energy();
   }
   return energy;
}

double TrackDetMatchInfo::ecalNxNEnergy(const DetId& id, int gridSize)
{
   if ( id.rawId() == 0 ) return -9999;
   if( id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap) ) {
      edm::LogWarning("TrackAssociator") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n" <<
	DetIdInfo::info(id)<<"\n";
      return -99999;
   }
   // Since the ECAL granularity is small and the gap between EE and EB is significant,
   // energy is computed only within the system that contains the element with maximal
   // energy deposition
   if( id.subdetId() == EcalBarrel ) {
      EBDetId centerId(id);
      double energy(0);
      for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
	 if (hit->id().subdetId() != EcalBarrel) continue;
	 EBDetId neighborId(hit->id());
	 int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			 -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	 int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	 if ( abs(360-dPhi) < dPhi ) dPhi = 360-dPhi;
	 if(  dEta <= gridSize && dPhi <= gridSize ) energy += hit->energy();
      }
      return energy;
   }

   // Endcap
   EEDetId centerId(id);
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
      if (hit->id().subdetId() != EcalEndcap) continue;
      EEDetId neighborId(hit->id());
      if(  centerId.zside() == neighborId.zside() && 
	   abs(centerId.ix()-neighborId.ix()) <= gridSize && 
	   abs(centerId.iy()-neighborId.iy()) <= gridSize ) energy += hit->energy();
   }
   return energy;
}


TrackDetMatchInfo::TrackDetMatchInfo():
       trkGlobPosAtEcal(0,0,0)
     , trkGlobPosAtHcal(0,0,0)
     , trkGlobPosAtHO(0,0,0)
     , isGoodEcal(false)
     , isGoodHcal(false)
     , isGoodCalo(false)
     , isGoodHO(false)
     , isGoodMuon(false)
     , ecalRecHits(coneEcalRecHits)
     , hcalRecHits(coneHcalRecHits)
     , hoRecHits(coneHORecHits)
     , towers(coneTowers)
{
}

DetId TrackDetMatchInfo::findEcalMaxDeposition()
{
   DetId id;
   float maxEnergy = -9999;
   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
     if ( hit->energy() > maxEnergy ) {
	maxEnergy = hit->energy();
	id = hit->detid();
     }
   return id;
}

DetId TrackDetMatchInfo::findHcalMaxDeposition()
{
   DetId id;
   float maxEnergy = -9999;
   for(std::vector<HBHERecHit>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++)
     if ( hit->energy() > maxEnergy ) {
	maxEnergy = hit->energy();
	id = hit->detid();
     }
   return id;
}

DetId TrackDetMatchInfo::findTowerMaxDeposition(TowerEnergyType type)
{
   DetId id;
   float maxEnergy = -9999;
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
     {
	double energy = 0;
	switch (type) {
	 case Total:
	   energy = hit->energy();
	   break;
	 case Ecal:
	   energy = hit->emEnergy();
	   break;
	 case Hcal:
	   energy = hit->hadEnergy();
	   break;
	 case HO:
	   energy = hit->energy();
	   break;
	 default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo tower energy type: " << type;
	}
	if ( energy > maxEnergy ) {
	   maxEnergy = energy;
	   id = hit->id();
	}
     }
   return id;
}
