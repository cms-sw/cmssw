#include <algorithm>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

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
   for(std::vector<CaloTower>::const_iterator hit=towers.begin(); hit!=towers.end(); hit++)
     energy += hit->emEnergy();
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

