//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.cc,v 1.2 2006/04/25 01:52:58 rpw Exp $
 */

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
using std::vector;

EcalSelectiveReadout::EcalSelectiveReadout(const vector<double>& thr,
                                                       int dEta_,
                                                       int dPhi_):
  threshold(thr), dEta(dEta_), dPhi(dPhi_)
{
  if(threshold.size()!=2){
      throw cms::Exception("EcalSelectiveReadout") 
         << "Argument 'thresholds' of "
      << "EcalSelectiveReadout::EcalSelectiveReadout(vector<int> thresholds) "
      << "method must a vector of two elements (with the low and the high "
      << "trigger tower Et for the selective readout flags. Aborts.";
  }
  resetSupercrystalInterest();
}

void EcalSelectiveReadout::resetSupercrystalInterest(){
  //init superCrystalInterest (sets all elts to 'unknown'):
  for(size_t iCap=0; iCap < nEndcaps; ++iCap){
    for(size_t iSCX = 0; iSCX < nSupercrystalXBins; ++iSCX){
      for(size_t iSCY = 0; iSCY < nSupercrystalYBins; ++iSCY){
        supercrystalInterest[iCap][iSCX][iSCY] = UNKNOWN;
      }
    }
  }
}


void
EcalSelectiveReadout::runSelectiveReadout0(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]) {
  //classifies the trigger towers (single,center,neighbor,low interest)
  classifyTriggerTowers(towerEt);

  //count number of TT in each interest class for debugging display
  int nTriggerTowerE[] = {0, 0, 0, 0};
  int nTriggerTowerB[] = {0, 0, 0, 0};
  
  for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
    for(size_t iEta = 0; iEta < nTriggerTowersInEta; ++iEta){
      if(iEta < nEndcapTriggerTowersInEta
         || iEta >= nBarrelTriggerTowersInEta + nEndcapTriggerTowersInEta){
        //in endcaps
        ++nTriggerTowerE[towerInterest[iEta][iPhi]];
      } else{//in barrel
        ++nTriggerTowerB[towerInterest[iEta][iPhi]];
      }
    }
  }

  static int ncall = 0;
  if(ncall < 10){
    ++ncall;
    edm::LogInfo("EcalSelectiveReadout")
         << nTriggerTowerB[LOWINTEREST] << " low interest TT in barrel\n"
         << nTriggerTowerB[SINGLE]      << " single TT in barrel\n"
         << nTriggerTowerB[NEIGHBOUR]   << " neighbor interest TT in barrel\n"
         << nTriggerTowerB[CENTER]      << " centre interest TT in barrel\n"
         << nTriggerTowerE[LOWINTEREST] << " low interest TT in endcap\n"
         << nTriggerTowerE[SINGLE]      << " single TT in endcap\n"
         << nTriggerTowerE[NEIGHBOUR]   << " neighbor TT in endcap\n"
         << nTriggerTowerE[CENTER]      << " center TT in endcap\n";
  }
  //end TT interest class composition debugging display
  
  //For the endcap the TT classification must be mapped to the SC:
  std::vector<DetId> endcapDetIds = theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  for(std::vector<DetId>::const_iterator eeDetIdItr = endcapDetIds.begin();
      eeDetIdItr != endcapDetIds.end(); ++eeDetIdItr)
  {
    // for each superCrystal, the interest is the highest interest
    // of any trigger tower associated with any crystal in this SC
    EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(*eeDetIdItr);
    assert(trigTower.rawId() != 0); 
    EEDetId eeDetId(*eeDetIdItr);
    int iz = (eeDetId.zside() == 1) ? 1 : 0;
    int superCrystalX = eeDetId.ix() / 5;
    int superCrystalY = eeDetId.iy() / 5;
    setHigher(supercrystalInterest[iz][superCrystalX][superCrystalY], 
              getTowerInterest(trigTower));
  }
}


EcalSelectiveReadout::towerInterest_t 
EcalSelectiveReadout::getCrystalInterest(const EBDetId & ebDetId) const
{
  EcalTrigTowerDetId thisTower = theTriggerMap->towerOf(ebDetId);
  return getTowerInterest(thisTower);
}


EcalSelectiveReadout::towerInterest_t 
EcalSelectiveReadout::getCrystalInterest(const EEDetId & eeDetId) const 
{
  int iz = (eeDetId.zside() == 1) ? 1 : 0;
  int superCrystalX = eeDetId.ix() / 5;
  int superCrystalY = eeDetId.iy() / 5;
  return supercrystalInterest[iz][superCrystalX][superCrystalY];
}



EcalSelectiveReadout::towerInterest_t
EcalSelectiveReadout::getTowerInterest(const EcalTrigTowerDetId & tower) const 
{
  // remember, array indeces start at zero
  int iEta = tower.ieta() + nTriggerTowersInEta/2;
  int iPhi = tower.iphi();
  return towerInterest[iEta][iPhi];
}


void
EcalSelectiveReadout::classifyTriggerTowers(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi])
{
  for(int iEta=0; iEta < (int)nTriggerTowersInEta; ++iEta){
    for(int iPhi=0; iPhi < (int)nTriggerTowersInPhi; ++iPhi){
      if(towerEt[iEta][iPhi] > threshold[1]){
        //flags this tower as a center tower
        towerInterest[iEta][iPhi] = CENTER;
        //flags the neighbours of this tower
        for(int iEtaNeigh = std::max<int>(0,iEta-dEta);
            iEtaNeigh <= std::min<int>(nTriggerTowersInEta-1, iEta+dEta);
            ++iEtaNeigh){
          for(int iPhiNeigh = iPhi-dPhi;
              iPhiNeigh <= iPhi+dPhi;
              ++iPhiNeigh){
            //beware, iPhiNeigh must be moved to [0,72] interval
            //=> %nTriggerTowersInPhi required
            int iPhiNeigh_ = iPhiNeigh%(int)nTriggerTowersInPhi;
            if(iPhiNeigh_<0) {
              iPhiNeigh_ += nTriggerTowersInPhi;
            }
            setHigher(towerInterest[iEtaNeigh][iPhiNeigh_],
                      NEIGHBOUR);
          }
        }
      } else if(towerEt[iEta][iPhi] > threshold[0]){
	setHigher(towerInterest[iEta][iPhi], SINGLE);
      }
    }
  }
}

