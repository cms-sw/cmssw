#include <map>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

int TrackDetMatchInfo::numberOfSegmentsInStation(int station) const {
	int numSegments = 0;
	for(std::vector<MuonSegmentMatch>::const_iterator segment=segments.begin(); segment!=segments.end(); segment++)
		if(segment->station()==station) numSegments++;
	return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station, int detector) const {
   int numSegments = 0;
   for(std::vector<MuonSegmentMatch>::const_iterator segment=segments.begin(); segment!=segments.end(); segment++)
      if(segment->station()==station&&segment->detector()==detector) numSegments++;
   return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInDetector(int detector) const {
	int numSegments = 0;
	for(std::vector<MuonSegmentMatch>::const_iterator segment=segments.begin(); segment!=segments.end(); segment++)
		if(segment->detector()==detector) numSegments++;
	return numSegments;
}

double TrackDetMatchInfo::ecalEnergy()
{
   double energy(0);
   for(std::vector<EcalRecHit>::const_iterator hit=crossedEcalRecHits.begin(); hit!=crossedEcalRecHits.end(); hit++)
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

double TrackDetMatchInfo::ecalNeighborHitEnergy(int gridSize)
{
   double energy(0);
   std::map<DetId, EcalRecHit> neighbors;
   for(std::vector<EcalRecHit>::const_iterator crossedHit=crossedEcalRecHits.begin(); crossedHit!=crossedEcalRecHits.end(); crossedHit++) {
      if(crossedHit->id().subdetId()==1) {//EB
         EBDetId crossedId(crossedHit->id());
         for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
            if(hit->id().subdetId()==1) {//EB
               EBDetId neighborId(hit->id());

               if(abs((crossedId.ieta()<0?crossedId.ieta()+1:crossedId.ieta())-(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta()))<gridSize-1 && abs(crossedId.iphi()-neighborId.iphi())%359<gridSize-1)
                  neighbors.insert(std::make_pair(hit->id(), *hit));
            }
         }
         continue;
      }
      if(crossedHit->id().subdetId()==2) {//EE
         EEDetId crossedId(crossedHit->id());
         for(std::vector<EcalRecHit>::const_iterator hit=ecalRecHits.begin(); hit!=ecalRecHits.end(); hit++) {
            if(hit->id().subdetId()==2) {//EE
               EEDetId neighborId(hit->id());
               if(crossedId.zside()==neighborId.zside() && abs(crossedId.ix()-neighborId.ix())<gridSize-1 && abs(crossedId.iy()-neighborId.iy())<gridSize-1)
                  neighbors.insert(std::make_pair(hit->id(), *hit));
            }
         }
         continue;
      }
   }

   for(std::map<DetId, EcalRecHit>::const_iterator hit=neighbors.begin(); hit!=neighbors.end(); hit++)
      energy += hit->second.energy();

   return energy;
}

double TrackDetMatchInfo::ecalNeighborTowerEnergy(int gridSize)
{
   double energy(0);
   std::map<DetId, CaloTower> neighbors;
   for(std::vector<CaloTower>::const_iterator crossedTower=crossedTowers.begin(); crossedTower!=crossedTowers.end(); crossedTower++) {
      CaloTowerDetId crossedId(crossedTower->id());
      for(std::vector<CaloTower>::const_iterator tower=towers.begin(); tower!=towers.end(); tower++) {
         CaloTowerDetId neighborId(tower->id());
         if(abs((crossedId.ieta()<0?crossedId.ieta()+1:crossedId.ieta())-(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta()))<gridSize-1 && abs(crossedId.iphi()-neighborId.iphi())%71<gridSize-1)
            neighbors.insert(std::make_pair(tower->id(), *tower));
      }
      continue;
   }

   for(std::map<DetId, CaloTower>::const_iterator tower=neighbors.begin(); tower!=neighbors.end(); tower++)
      energy += tower->second.emEnergy();

   return energy;
}

double TrackDetMatchInfo::hcalConeEnergy()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
     energy += hit->hadEnergy();
   return energy;
}

double TrackDetMatchInfo::hcalEnergy()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator tower=crossedTowers.begin(); tower!=crossedTowers.end(); tower++)
     energy += tower->hadEnergy();
   return energy;
}

double TrackDetMatchInfo::hcalNeighborEnergy(int gridSize)
{
   double energy(0);
   std::map<DetId, CaloTower> neighbors;
   for(std::vector<CaloTower>::const_iterator crossedTower=crossedTowers.begin(); crossedTower!=crossedTowers.end(); crossedTower++) {
      CaloTowerDetId crossedId(crossedTower->id());
      for(std::vector<CaloTower>::const_iterator tower=towers.begin(); tower!=towers.end(); tower++) {
         CaloTowerDetId neighborId(tower->id());
         if(abs((crossedId.ieta()<0?crossedId.ieta()+1:crossedId.ieta())-(neighborId.ieta()<0?neighborId.ieta()+1:neighborId.ieta()))<gridSize-1 && abs(crossedId.iphi()-neighborId.iphi())%71<gridSize-1)
            neighbors.insert(std::make_pair(tower->id(), *tower));
      }
      continue;
   }

   for(std::map<DetId, CaloTower>::const_iterator tower=neighbors.begin(); tower!=neighbors.end(); tower++)
      energy += tower->second.hadEnergy();

   return energy;
}

double TrackDetMatchInfo::outerHcalConeEnergy()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
     energy += hit->outerEnergy();
   return energy;
}

double TrackDetMatchInfo::outerHcalEnergy()
{
   double energy(0);
   for(std::vector<CaloTower>::const_iterator tower=crossedTowers.begin(); tower!=crossedTowers.end(); tower++)
     energy += tower->outerEnergy();
   return energy;
}

