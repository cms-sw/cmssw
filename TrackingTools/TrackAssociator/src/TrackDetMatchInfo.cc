#include <map>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/DetIdInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


///////////////////////////

double TrackDetMatchInfo::crossedEnergy( EnergyType type )
{
   double energy(0);
   switch (type) {
    case EcalRecHits:
	{
	   for(std::vector<EcalRecHit>::const_iterator hit=crossedEcalRecHits.begin(); hit!=crossedEcalRecHits.end(); hit++)
	     energy += hit->energy();
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<HBHERecHit>::const_iterator hit = crossedHcalRecHits.begin(); hit != crossedHcalRecHits.end(); hit++)
	     energy += hit->energy();
	}
      break;
    case HORecHits:
	{
	   for(std::vector<HORecHit>::const_iterator hit = crossedHORecHits.begin(); hit != crossedHORecHits.end(); hit++)
	     energy += hit->energy();
	}
      break;
    case TowerTotal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += hit->energy();
	}
      break;
    case TowerEcal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += hit->emEnergy();
	}
      break;
    case TowerHcal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += hit->hadEnergy();
	}
      break;
    case TowerHO:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += hit->outerEnergy();
	}
      break;
    default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo energy type: " << type;
   }
   return energy;
}

bool TrackDetMatchInfo::insideCone(const DetId& id, const double dR) {
   // this part might be slow
   if ( ! caloGeometry.isValid() ) return false;
   if ( ! caloGeometry->getSubdetectorGeometry(id) ) return false;
   if ( ! caloGeometry->getSubdetectorGeometry(id)->getGeometry(id) ) return false;
   
   GlobalPoint center = caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
   double pi = 3.1415926535;

   double deltaPhi(fabs(momentum.phi()-center.phi()));
   if(deltaPhi>pi) deltaPhi = fabs(deltaPhi-pi*2.);

   return sqrt(pow(momentum.eta()-center.eta(),2)+pow(deltaPhi,2)) < dR;
}

double TrackDetMatchInfo::coneEnergy( double dR, EnergyType type )
{
   double energy(0);
   switch (type) {
    case EcalRecHits:
	{
	   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
	     if (insideCone(hit->detid(),dR)) energy += hit->energy();
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<HBHERecHit>::const_iterator hit = hcalRecHits.begin(); hit != hcalRecHits.end(); hit++)
	     if (insideCone(hit->detid(),dR)) energy += hit->energy();
	}
      break;
    case HORecHits:
	{
	   for(std::vector<HORecHit>::const_iterator hit = hoRecHits.begin(); hit != hoRecHits.end(); hit++)
	     if (insideCone(hit->detid(),dR)) energy += hit->energy();
	}
      break;
    case TowerTotal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone(hit->id(),dR)) energy += hit->energy();
	}
      break;
    case TowerEcal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone(hit->id(),dR)) energy += hit->energy();
	}
      break;
    case TowerHcal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone(hit->id(),dR)) energy += hit->energy();
	}
      break;
    case TowerHO:
	{
	   for(std::vector<CaloTower>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone(hit->id(),dR)) energy += hit->energy();
	}
      break;
    default:
	   edm::LogWarning("TrackAssociator") << "Unknown calo energy type: " << type;
   }
   return energy;
}


//////////////////////////////////////////////////

double TrackDetMatchInfo::nXnEnergy(const DetId& id, EnergyType type, int gridSize )
{
   if ( id.rawId() == 0 ) return -9999;
   switch (type)  {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
	{
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
		  case TowerTotal:
		    energy += hit->energy();
		    break;
		  case TowerEcal:
		    energy += hit->emEnergy();
		    break;
		  case TowerHcal:
		    energy += hit->hadEnergy();
		    break;
		  case TowerHO:
		    energy += hit->energy();
		    break;
		  default:
		    edm::LogWarning("TrackAssociator") << "Unknown calo tower energy type: " << type;
		 }
	      }
	   }
	   return energy;
	}
      break;
    case EcalRecHits:
	{
	   if( id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap) ) {
	      edm::LogWarning("TrackAssociator") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	      return -99999;
	   }
	   double energy(0);
	   // Since the ECAL granularity is small and the gap between EE and EB is significant,
	   // energy is computed only within the system that contains the central element
	   if( id.subdetId() == EcalBarrel ) {
	      EBDetId centerId(id);
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
	   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
	      if (hit->id().subdetId() != EcalEndcap) continue;
	      EEDetId neighborId(hit->id());
	      if(  centerId.zside() == neighborId.zside() && 
		   abs(centerId.ix()-neighborId.ix()) <= gridSize && 
		   abs(centerId.iy()-neighborId.iy()) <= gridSize ) energy += hit->energy();
	   }
	   return energy;
	}
      break;
    case HcalRecHits:
	{
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
      break;
    default:
      edm::LogWarning("TrackAssociator") << "Unkown or not implemented energy type requested, type:" << type;
   }
   return 0;
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
     , simTrack(0)
     , ecalTrueEnergy(-999)
     , hcalTrueEnergy(-999)
{
}

DetId TrackDetMatchInfo::findMaxDeposition( EnergyType type )
{
   DetId id;
   float maxEnergy = -9999;
   switch (type) {
    case EcalRecHits:
	{
	   for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
	     if ( hit->energy() > maxEnergy ) {
		maxEnergy = hit->energy();
		id = hit->detid();
	     }
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<HBHERecHit>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++)
	     if ( hit->energy() > maxEnergy ) {
		maxEnergy = hit->energy();
		id = hit->detid();
	     }
	}
      break;
    case TowerTotal:
    case TowerEcal:
    case TowerHcal:
	{
	   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
	     {
		double energy = 0;
		switch (type) {
		 case TowerTotal:
		   energy = hit->energy();
		   break;
		 case TowerEcal:
		   energy = hit->emEnergy();
		   break;
		 case TowerHcal:
		   energy = hit->hadEnergy();
		   break;
		 case TowerHO:
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
	}
    default:
      edm::LogWarning("TrackAssociator") << "Maximal energy deposition: unkown or not implemented energy type requested, type:" << type;
   }
   return id;
}

////////////////////////////////////////////////////////////////////////
// Obsolete
//



double TrackDetMatchInfo::ecalConeEnergy()
{
   return coneEnergy (999, EcalRecHits );
}

double TrackDetMatchInfo::hcalConeEnergy()
{
   return coneEnergy (999, HcalRecHits );
}

double TrackDetMatchInfo::hoConeEnergy()
{
   return coneEnergy (999, HcalRecHits );
}

double TrackDetMatchInfo::ecalCrossedEnergy()
{
   return crossedEnergy( EcalRecHits );
}

double TrackDetMatchInfo::hcalCrossedEnergy()
{
   return crossedEnergy( HcalRecHits );
}

double TrackDetMatchInfo::hoCrossedEnergy()
{
   return crossedEnergy( HORecHits );
}
      
      
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
