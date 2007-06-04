//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.cc,v 1.10 2007/02/14 18:37:04 pgras Exp $
 */

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

#include <iostream> //for debugging

using std::vector;

const char EcalSelectiveReadout::srpFlagMarker[] = {'.', 'S', 'N', 'C', 'F'};


EcalSelectiveReadout::EcalSelectiveReadout(int dEta_, int dPhi_):
  dEta(dEta_), dPhi(dPhi_){
}

void EcalSelectiveReadout::resetSupercrystalInterest(){
  //init superCrystalInterest (sets all elts to 'UNKNOWN'):
  for(size_t iCap=0; iCap < nEndcaps; ++iCap){
    for(size_t iSCX = 0; iSCX < nSupercrystalXBins; ++iSCX){
      for(size_t iSCY = 0; iSCY < nSupercrystalYBins; ++iSCY){
        supercrystalInterest[iCap][iSCX][iSCY] = UNKNOWN;
      }
    }
  }
}

void
EcalSelectiveReadout::runSelectiveReadout0(const ttFlag_t ttFlags[nTriggerTowersInEta][nTriggerTowersInPhi]) {
  //classifies the trigger towers (single,center,neighbor,low interest)
  classifyTriggerTowers(ttFlags);

  //count number of TT in each interest class for debugging display
  int nTriggerTowerE[] = {0, 0, 0, 0};
  int nTriggerTowerB[] = {0, 0, 0, 0};

   static int ncall = 0;
   if(ncall < 10){
     ++ncall;
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
  resetSupercrystalInterest();
  
#ifndef ECALSELECTIVEREADOUT_NOGEOM
  const std::vector<DetId>& endcapDetIds = theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  for(std::vector<DetId>::const_iterator eeDetIdItr = endcapDetIds.begin();
      eeDetIdItr != endcapDetIds.end(); ++eeDetIdItr){
    // for each superCrystal, the interest is the highest interest
    // of any trigger tower associated with any crystal in this SC
    EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(*eeDetIdItr);
    assert(trigTower.rawId() != 0); 
    EEDetId eeDetId(*eeDetIdItr);
    int iz = (eeDetId.zside() > 0) ? 1 : 0;
    int superCrystalX = (eeDetId.ix()-1) / 5;
    int superCrystalY = (eeDetId.iy()-1) / 5;
    setHigher(supercrystalInterest[iz][superCrystalX][superCrystalY], 
              getTowerInterest(trigTower));
  }
#else //ECALSELECTIVEREADOUT_NOGEOM not defined
  EEDetId xtal;
  for(int iZ0=0; iZ0<2; ++iZ0){//0->EE-, 1->EE+
    for(unsigned iX0=0; iX0<nEndcapXBins; ++iX0){
      for(unsigned iY0=0; iY0<nEndcapYBins; ++iY0){
        try{
          xtal = EEDetId(iX0+1, iY0+1, (iZ0>0?1:-1));
        } catch(cms::Exception e){//exception thrown if no crystal at
          //                        this position
          continue;
        }
        //works around a EEDetId bug. To remove once the bug fixed.
        if(39 <= iX0 && iX0 <= 60 && 45 <= iY0 && iY0 <= 54){
          continue;
        }
        // for each superCrystal, the interest is the highest interest
        // of any trigger tower associated with any crystal in this SC
        EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(xtal);
        assert(trigTower.rawId() != 0); 
        int superCrystalX = iX0 / 5;
        int superCrystalY = iY0 / 5;
        setHigher(supercrystalInterest[iZ0][superCrystalX][superCrystalY], 
                  getTowerInterest(trigTower));
      }
    }
  }
#endif
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
  int superCrystalX = (eeDetId.ix()-1) / 5;
  int superCrystalY = (eeDetId.iy()-1) / 5;
  return supercrystalInterest[iz][superCrystalX][superCrystalY];
}
EcalSelectiveReadout::towerInterest_t 

EcalSelectiveReadout::getSuperCrystalInterest(const EcalScDetId& scDetId) const 
{
  int iz = (scDetId.zside() == 1) ? 1 : 0;
  int superCrystalX = scDetId.ix()-1;
  int superCrystalY = scDetId.iy()-1;
  return supercrystalInterest[iz][superCrystalX][superCrystalY];
}



EcalSelectiveReadout::towerInterest_t
EcalSelectiveReadout::getTowerInterest(const EcalTrigTowerDetId & tower) const 
{
  // remember, array indices start at zero
  int iEta = tower.ieta()<0? tower.ieta() + nTriggerTowersInEta/2
    : tower.ieta() + nTriggerTowersInEta/2 -1;
  int iPhi = tower.iphi() - 1;
  return towerInterest[iEta][iPhi];
}

void
EcalSelectiveReadout::classifyTriggerTowers(const ttFlag_t ttFlags[nTriggerTowersInEta][nTriggerTowersInPhi])
{
  //starts with a all low interest map:
  for(int iEta=0; iEta < (int)nTriggerTowersInEta; ++iEta){
    for(int iPhi=0; iPhi < (int)nTriggerTowersInPhi; ++iPhi){
      towerInterest[iEta][iPhi] = LOWINTEREST;
    }
  }

  for(int iEta=0; iEta < (int)nTriggerTowersInEta; ++iEta){
    for(int iPhi=0; iPhi < (int)nTriggerTowersInPhi; ++iPhi){
      if(ttFlags[iEta][iPhi] == TTF_HIGH_INTEREST){
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
      } else if(ttFlags[iEta][iPhi] == TTF_MID_INTEREST){
	setHigher(towerInterest[iEta][iPhi], SINGLE);
      } else if(ttFlags[iEta][iPhi] & TTF_FORCED_RO_MASK){
        setHigher(towerInterest[iEta][iPhi], FORCED_RO);
      }
    }
  }

  //dealing with pseudo-TT in the two innest eta-ring of the endcaps
  //=>choose the highest priority  SRF of the 2 pseudo-TT constituting
  //a TT. Note that for S and C, the 2 pseudo-TT must already have the
  //same mask.
  const size_t innerEtas[] = {0, 1,
                              nTriggerTowersInEta-2, nTriggerTowersInEta-1};
  for(size_t i=0; i < 4; ++i){
    size_t iEta = innerEtas[i];
    for(size_t iPhi = 0 ; iPhi < nTriggerTowersInPhi; iPhi+=2){
      const towerInterest_t srf = std::max(towerInterest[iEta][iPhi],
                                           towerInterest[iEta][iPhi+1]);
      if(srf==NEIGHBOUR){
        std::cerr << "towerInterest[" << iEta << "][" << iPhi << "] = "
                  << towerInterest [iEta][iPhi] <<"\t"
                  << "towerInterest[" << iEta << "][" << (iPhi+1) << "] = "
                  << towerInterest [iEta][iPhi+1] <<" \t-> "
                  << srf << "\n";
      }
      towerInterest[iEta][iPhi] = srf;
      towerInterest[iEta][iPhi+1] = srf;
    }
  }
}

void EcalSelectiveReadout::printHeader(std::ostream & os) const{
  os << "#SRP flag map\n#\n"
    "# +-->Phi/Y " << srpFlagMarker[0] << ": low interest\n"
    "# |         " << srpFlagMarker[1] << ": single\n"
    "# |         " << srpFlagMarker[2] << ": neighbour\n"
    "# V Eta/X   " << srpFlagMarker[3] << ": center\n"
    "#           " << srpFlagMarker[4] << ": forced readout\n"
    "#\n";
}

void EcalSelectiveReadout::print(std::ostream & os) const
{
  //EE-
  printEndcap(0, os);

  //EB
  printBarrel(os);

  //EE+
  printEndcap(1, os);
}


void EcalSelectiveReadout::printBarrel(std::ostream & os) const
{
  for(size_t iEta = nEndcapTriggerTowersInEta;
      iEta < nEndcapTriggerTowersInEta
        + nBarrelTriggerTowersInEta;
      ++iEta){
    for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
      towerInterest_t srFlag
        = towerInterest[iEta][iPhi];
      os << srpFlagMarker[srFlag];
    }
    os << "\n"; //one phi per line
  }
}


void EcalSelectiveReadout::printEndcap(int endcap, std::ostream & os) const
{
    for(size_t iX=0; iX<nSupercrystalXBins; ++iX){
    for(size_t iY=0; iY<nSupercrystalYBins; ++iY){
      towerInterest_t srFlag
        = supercrystalInterest[endcap][iX][iY];
        os << (srFlag==UNKNOWN?
                    ' ':srpFlagMarker[srFlag]);
    }
    os << "\n"; //one Y supercystal column per line
  } //next supercrystal X-index
}


std::ostream & operator<<(std::ostream & os, const EcalSelectiveReadout & selectiveReadout)
{
  selectiveReadout.print(os);
  return os;
}
