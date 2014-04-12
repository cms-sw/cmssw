#include <map>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "DetIdInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/VectorUtil.h"
#include <algorithm>


///////////////////////////

std::string TrackDetMatchInfo::dumpGeometry( const DetId& id )
{
   if ( ! caloGeometry.isValid() || 
	! caloGeometry->getSubdetectorGeometry(id) ||
	! caloGeometry->getSubdetectorGeometry(id)->getGeometry(id) ) {
      throw cms::Exception("FatalError")  << "Failed to access geometry for DetId: " << id.rawId();
   }
   std::ostringstream oss;

   const CaloCellGeometry::CornersVec& points = caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)->getCorners();
   for( CaloCellGeometry::CornersVec::const_iterator point = points.begin();
       point != points.end(); ++point)
     oss << "(" << point->z() << ", " << point->perp() << ", " << point->eta() << ", " << point->phi() << "), \t";
   return oss.str();
}


GlobalPoint TrackDetMatchInfo::getPosition( const DetId& id)
{
   // this part might be slow
   if ( ! caloGeometry.isValid() || 
	! caloGeometry->getSubdetectorGeometry(id) ||
	! caloGeometry->getSubdetectorGeometry(id)->getGeometry(id) ) {
      throw cms::Exception("FatalError") << "Failed to access geometry for DetId: " << id.rawId();
      return GlobalPoint(0,0,0);
   }
   return caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
}


double TrackDetMatchInfo::crossedEnergy( EnergyType type )
{
   double energy(0);
   switch (type) {
    case EcalRecHits:
	{
	   for(std::vector<const EcalRecHit*>::const_iterator hit=crossedEcalRecHits.begin(); hit!=crossedEcalRecHits.end(); hit++)
	     energy += (*hit)->energy();
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<const HBHERecHit*>::const_iterator hit = crossedHcalRecHits.begin(); hit != crossedHcalRecHits.end(); hit++)
	     energy += (*hit)->energy();
	}
      break;
    case HORecHits:
	{
	   for(std::vector<const HORecHit*>::const_iterator hit = crossedHORecHits.begin(); hit != crossedHORecHits.end(); hit++)
	     energy += (*hit)->energy();
	}
      break;
    case TowerTotal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += (*hit)->energy();
	}
      break;
    case TowerEcal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += (*hit)->emEnergy();
	}
      break;
    case TowerHcal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += (*hit)->hadEnergy();
	}
      break;
    case TowerHO:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     energy += (*hit)->outerEnergy();
	}
      break;
    default:
      throw cms::Exception("FatalError") << "Unknown calo energy type: " << type;
   }
   return energy;
}

bool TrackDetMatchInfo::insideCone(const DetId& id, const double dR) {
   GlobalPoint idPosition = getPosition(id);
   if (idPosition.mag()<0.01) return false;
   
   math::XYZVector idPositionRoot( idPosition.x(), idPosition.y(), idPosition.z() );
   math::XYZVector trackP3( stateAtIP.momentum().x(), stateAtIP.momentum().y(), stateAtIP.momentum().z() );
   return ROOT::Math::VectorUtil::DeltaR(trackP3, idPositionRoot) < 0.5;
}

double TrackDetMatchInfo::coneEnergy( double dR, EnergyType type )
{
   double energy(0);
   switch (type) {
    case EcalRecHits:
	{
	   for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
	     if (insideCone((*hit)->detid(),dR)) energy += (*hit)->energy();
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<const HBHERecHit*>::const_iterator hit = hcalRecHits.begin(); hit != hcalRecHits.end(); hit++)
	     if (insideCone((*hit)->detid(),dR)) energy += (*hit)->energy();
	}
      break;
    case HORecHits:
	{
	   for(std::vector<const HORecHit*>::const_iterator hit = hoRecHits.begin(); hit != hoRecHits.end(); hit++)
	     if (insideCone((*hit)->detid(),dR)) energy += (*hit)->energy();
	}
      break;
    case TowerTotal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone((*hit)->id(),dR)) energy += (*hit)->energy();
	}
      break;
    case TowerEcal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone((*hit)->id(),dR)) energy += (*hit)->emEnergy();
	}
      break;
    case TowerHcal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone((*hit)->id(),dR)) energy += (*hit)->hadEnergy();
	}
      break;
    case TowerHO:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=crossedTowers.begin(); hit!=crossedTowers.end(); hit++)
	     if (insideCone((*hit)->id(),dR)) energy += (*hit)->outerEnergy();
	}
      break;
    default:
      throw cms::Exception("FatalError") << "Unknown calo energy type: " << type;
   }
   return energy;
}


//////////////////////////////////////////////////

double TrackDetMatchInfo::nXnEnergy(const DetId& id, EnergyType type, int gridSize)
{
   double energy(0);
   if ( id.rawId() == 0 ) return 0.;
   switch (type)  {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
	{
	   if ( id.det() != DetId::Calo ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected CaloTower, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   CaloTowerDetId centerId(id);
	   for(std::vector<const CaloTower*>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++) {
	      CaloTowerDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ) {
		 switch (type) {
		  case TowerTotal:
		    energy += (*hit)->energy();
		    break;
		  case TowerEcal:
		    energy += (*hit)->emEnergy();
		    break;
		  case TowerHcal:
		    energy += (*hit)->hadEnergy();
		    break;
		  case TowerHO:
		    energy += (*hit)->outerEnergy();
		    break;
		  default:
		    throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
		 }
	      }
	   }
	}
      break;
    case EcalRecHits:
	{
	   if( id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   // Since the ECAL granularity is small and the gap between EE and EB is significant,
	   // energy is computed only within the system that contains the central element
	   if( id.subdetId() == EcalBarrel ) {
	      EBDetId centerId(id);
	      for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
		 if ((*hit)->id().subdetId() != EcalBarrel) continue;
		 EBDetId neighborId((*hit)->id());
		 int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
				 -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
		 int dPhi = abs( centerId.iphi()-neighborId.iphi() );
		 if ( abs(360-dPhi) < dPhi ) dPhi = 360-dPhi;
		 if(  dEta <= gridSize && dPhi <= gridSize ) {
		    energy += (*hit)->energy();
		 }
	      }
	   } else {
	      // Endcap
	      EEDetId centerId(id);
	      for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
		 if ((*hit)->id().subdetId() != EcalEndcap) continue;
		 EEDetId neighborId((*hit)->id());
		 if(  centerId.zside() == neighborId.zside() && 
		      abs(centerId.ix()-neighborId.ix()) <= gridSize && 
		      abs(centerId.iy()-neighborId.iy()) <= gridSize ) {
		    energy += (*hit)->energy();
		 }
	      }
	   }
	}
      break;
    case HcalRecHits:
	{
	   if( id.det() != DetId::Hcal || (id.subdetId() != HcalBarrel && id.subdetId() != HcalEndcap) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected HE or HB, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   HcalDetId centerId(id);
	   for(std::vector<const HBHERecHit*>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++) {
	      HcalDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ){
		 energy += (*hit)->energy();
	      }
	   }
	}
      break;
    case HORecHits:
	{
	   if( id.det() != DetId::Hcal || (id.subdetId() != HcalOuter) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected HO, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   HcalDetId centerId(id);
	   for(std::vector<const HORecHit*>::const_iterator hit=hoRecHits.begin(); hit!=hoRecHits.end(); hit++) {
	      HcalDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ) {
		 energy += (*hit)->energy();
	      }
	   }
	}
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
   }
   return energy;
}

double TrackDetMatchInfo::nXnEnergy(EnergyType type, int gridSize)
{
   switch (type)  {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
      if( crossedTowerIds.empty() ) return 0;
      return nXnEnergy(crossedTowerIds.front(), type, gridSize);
      break;
    case EcalRecHits:
      if( crossedEcalIds.empty() ) return 0;
      return nXnEnergy(crossedEcalIds.front(), type, gridSize);
      break;
    case HcalRecHits:
      if( crossedHcalIds.empty() ) return 0;
      return nXnEnergy(crossedHcalIds.front(), type, gridSize);
      break;
    case HORecHits:
      if( crossedHOIds.empty() ) return 0;
      return nXnEnergy(crossedHOIds.front(), type, gridSize);
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
   }
   return -999;
}


TrackDetMatchInfo::TrackDetMatchInfo():
       trkGlobPosAtEcal(0,0,0)
     , trkGlobPosAtHcal(0,0,0)
     , trkGlobPosAtHO(0,0,0)
     , trkMomAtEcal(0,0,0)
     , trkMomAtHcal(0,0,0)
     , trkMomAtHO(0,0,0)
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
	   for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++)
	     if ( (*hit)->energy() > maxEnergy ) {
		maxEnergy = (*hit)->energy();
		id = (*hit)->detid();
	     }
	}
      break;
    case HcalRecHits:
	{
	   for(std::vector<const HBHERecHit*>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++)
	     if ( (*hit)->energy() > maxEnergy ) {
		maxEnergy = (*hit)->energy();
		id = (*hit)->detid();
	     }
	}
      break;
    case HORecHits:
	{
	   for(std::vector<const HORecHit*>::const_iterator hit=hoRecHits.begin(); hit!=hoRecHits.end(); hit++)
	     if ( (*hit)->energy() > maxEnergy ) {
		maxEnergy = (*hit)->energy();
		id = (*hit)->detid();
	     }
	}
      break;
    case TowerTotal:
    case TowerEcal:
    case TowerHcal:
	{
	   for(std::vector<const CaloTower*>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
	     {
		double energy = 0;
		switch (type) {
		 case TowerTotal:
		   energy = (*hit)->energy();
		   break;
		 case TowerEcal:
		   energy = (*hit)->emEnergy();
		   break;
		 case TowerHcal:
		   energy = (*hit)->hadEnergy();
		   break;
		 case TowerHO:
		   energy = (*hit)->energy();
		   break;
		 default:
		   throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
		}
		if ( energy > maxEnergy ) {
		   maxEnergy = energy;
		   id = (*hit)->id();
		}
	     }
	}
    default:
      throw cms::Exception("FatalError") << "Maximal energy deposition: unkown or not implemented energy type requested, type:" << type;
   }
   return id;
}

DetId TrackDetMatchInfo::findMaxDeposition(const DetId& id, EnergyType type, int gridSize)
{
   double energy_max(0);
   DetId id_max;
   if ( id.rawId() == 0 ) return id_max;
   switch (type)  {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
	{
	   if ( id.det() != DetId::Calo ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected CaloTower, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   CaloTowerDetId centerId(id);
	   for(std::vector<const CaloTower*>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++) {
	      CaloTowerDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ) {
		 switch (type) {
		  case TowerTotal:
		    if ( energy_max < (*hit)->energy() ){
		       energy_max = (*hit)->energy();
		       id_max = (*hit)->id();
		    }
		    break;
		  case TowerEcal:
		    if ( energy_max < (*hit)->emEnergy() ){
		       energy_max = (*hit)->emEnergy();
		       id_max = (*hit)->id();
		    }
		    break;
		  case TowerHcal:
		    if ( energy_max < (*hit)->hadEnergy() ){
		       energy_max = (*hit)->hadEnergy();
		       id_max = (*hit)->id();
		    }
		    break;
		  case TowerHO:
		    if ( energy_max < (*hit)->outerEnergy() ){
		       energy_max = (*hit)->outerEnergy();
		       id_max = (*hit)->id();
		    }
		    break;
		  default:
		    throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
		 }
	      }
	   }
	}
      break;
    case EcalRecHits:
	{
	   if( id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   // Since the ECAL granularity is small and the gap between EE and EB is significant,
	   // energy is computed only within the system that contains the central element
	   if( id.subdetId() == EcalBarrel ) {
	      EBDetId centerId(id);
	      for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
		 if ((*hit)->id().subdetId() != EcalBarrel) continue;
		 EBDetId neighborId((*hit)->id());
		 int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
				 -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
		 int dPhi = abs( centerId.iphi()-neighborId.iphi() );
		 if ( abs(360-dPhi) < dPhi ) dPhi = 360-dPhi;
		 if(  dEta <= gridSize && dPhi <= gridSize ) {
		    if ( energy_max < (*hit)->energy() ){
		       energy_max = (*hit)->energy();
		       id_max = (*hit)->id();
		    }
		 }
	      }
	   } else {
	      // Endcap
	      EEDetId centerId(id);
	      for(std::vector<const EcalRecHit*>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
		 if ((*hit)->id().subdetId() != EcalEndcap) continue;
		 EEDetId neighborId((*hit)->id());
		 if(  centerId.zside() == neighborId.zside() && 
		      abs(centerId.ix()-neighborId.ix()) <= gridSize && 
		      abs(centerId.iy()-neighborId.iy()) <= gridSize ) {
		    if ( energy_max < (*hit)->energy() ){
		       energy_max = (*hit)->energy();
		       id_max = (*hit)->id();
		    }
		 }
	      }
	   }
	}
      break;
    case HcalRecHits:
	{
	   if( id.det() != DetId::Hcal || (id.subdetId() != HcalBarrel && id.subdetId() != HcalEndcap) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected HE or HB, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   HcalDetId centerId(id);
	   for(std::vector<const HBHERecHit*>::const_iterator hit=hcalRecHits.begin(); hit!=hcalRecHits.end(); hit++) {
	      HcalDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ){
		 if ( energy_max < (*hit)->energy() ){
		    energy_max = (*hit)->energy();
		    id_max = (*hit)->id();
		 }
	      }
	   }
	}
      break;
    case HORecHits:
	{
	   if( id.det() != DetId::Hcal || (id.subdetId() != HcalOuter) ) {
	      throw cms::Exception("FatalError") << "Wrong DetId. Expected HO, but found:\n" <<
		DetIdInfo::info(id)<<"\n";
	   }
	   HcalDetId centerId(id);
	   for(std::vector<const HORecHit*>::const_iterator hit=hoRecHits.begin(); hit!=hoRecHits.end(); hit++) {
	      HcalDetId neighborId((*hit)->id());
	      int dEta = abs( (centerId.ieta()<0?centerId.ieta()+1:centerId.ieta() )
			      -(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta() ) ) ;
	      int dPhi = abs( centerId.iphi()-neighborId.iphi() );
	      if ( abs(72-dPhi) < dPhi ) dPhi = 72-dPhi;
	      if(  dEta <= gridSize && dPhi <= gridSize ) {
		 if ( energy_max < (*hit)->energy() ){
		    energy_max = (*hit)->energy();
		    id_max = (*hit)->id();
		 }
	      }
	   }
	}
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
   }
   return id_max;
}

DetId TrackDetMatchInfo::findMaxDeposition(EnergyType type, int gridSize)
{
   DetId id_max;
   switch (type)  {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
      if( crossedTowerIds.empty() ) return id_max;
      return findMaxDeposition(crossedTowerIds.front(), type, gridSize);
      break;
    case EcalRecHits:
      if( crossedEcalIds.empty() ) return id_max;
      return findMaxDeposition(crossedEcalIds.front(), type, gridSize);
      break;
    case HcalRecHits:
      if( crossedHcalIds.empty() ) return id_max;
      return findMaxDeposition(crossedHcalIds.front(), type, gridSize);
      break;
    case HORecHits:
      if( crossedHOIds.empty() ) return id_max;
      return findMaxDeposition(crossedHOIds.front(), type, gridSize);
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
   }
   return id_max;
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
   for(std::vector<TAMuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station) const {
   int numSegments = 0;
   for(std::vector<TAMuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->station()==station) numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station, int detector) const {
   int numSegments = 0;
   for(std::vector<TAMuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->station()==station&&chamber->detector()==detector) numSegments += chamber->segments.size();
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInDetector(int detector) const {
   int numSegments = 0;
   for(std::vector<TAMuonChamberMatch>::const_iterator chamber=chambers.begin(); chamber!=chambers.end(); chamber++)
     if(chamber->detector()==detector) numSegments += chamber->segments.size();
   return numSegments;
}
