//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.cc,v 1.18 2012/09/21 09:17:28 eulisse Exp $
 */

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <iomanip>
#include <cassert>
//#include <iostream> //for debugging

using std::vector;

const char EcalSelectiveReadout::srpFlagMarker[] = {'.', 'S', 'N', 'C',
                                                    '4', '5', '6', '7'};

EcalSelectiveReadout::EcalSelectiveReadout(int dEta_, int dPhi_):
  theTriggerMap(0), theElecMap(0), dEta(dEta_), dPhi(dPhi_) {
}

void EcalSelectiveReadout::resetEeRuInterest(){
  //init superCrystalInterest (sets all elts to 'UNKNOWN'):
  for(size_t iCap=0; iCap < nEndcaps; ++iCap){
    for(int iDccPhi = 0; iDccPhi < nDccPerEe; ++iDccPhi){
      for(int iDccCh = 0; iDccCh < maxDccChs; ++iDccCh){
        eeRuInterest_[iCap][iDccPhi][iDccCh] = UNKNOWN;
      }
    }
  }
}

void
EcalSelectiveReadout::runSelectiveReadout0(const ttFlag_t ttFlags[nTriggerTowersInEta][nTriggerTowersInPhi]) {

  //printDccChMap(std::cout);
  
  //classifies the trigger towers (single,center,neighbor,low interest)
  classifyTriggerTowers(ttFlags);
  
  //count number of TT in each interest class for debugging display
  int nTriggerTowerE[] = {0, 0, 0, 0, 0, 0, 0, 0};
  int nTriggerTowerB[] = {0, 0, 0, 0, 0, 0, 0, 0};

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

        assert(towerInterest[iEta][iPhi] >= 0
               && towerInterest[iEta][iPhi] <= 0x7);
      }
    }
    edm::LogInfo("EcalSelectiveReadout")
      << "without forced bit + with forced bit set:\n"
      << nTriggerTowerB[LOWINTEREST] << " + "
      << nTriggerTowerB[LOWINTEREST | FORCED_MASK]
      << " low interest TT(s) in barrel\n"
      << nTriggerTowerB[SINGLE]       << " + "
      << nTriggerTowerB[SINGLE | FORCED_MASK]
      << " single TT(s) in barrel\n"
      << nTriggerTowerB[NEIGHBOUR]   << " + "
      << nTriggerTowerB[NEIGHBOUR | FORCED_MASK]
      << " neighbor interest TT(s) in barrel\n"
      << nTriggerTowerB[CENTER]      << " + "
      << nTriggerTowerB[CENTER | FORCED_MASK]
      << " centre interest TT(s) in barrel\n"
      << nTriggerTowerE[LOWINTEREST] << " + "
      << nTriggerTowerE[LOWINTEREST | FORCED_MASK]
      << " low interest TT(s) in endcap\n"
      << nTriggerTowerE[SINGLE]      << " + "
      << nTriggerTowerE[SINGLE | FORCED_MASK]
      << " single TT(s) in endcap\n"
      << nTriggerTowerE[NEIGHBOUR]   << " + "
      << nTriggerTowerE[NEIGHBOUR | FORCED_MASK]
      << " neighbor TT(s) in endcap\n"
      << nTriggerTowerE[CENTER]      << " + "
      << nTriggerTowerE[CENTER | FORCED_MASK]
      << " center TT(s) in endcap\n";
  }
  //end TT interest class composition debugging display
  
  //For the endcap the TT classification must be mapped to the SC:  
  resetEeRuInterest();
  
#ifndef ECALSELECTIVEREADOUT_NOGEOM
  const std::vector<DetId>& endcapDetIds = theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  for(std::vector<DetId>::const_iterator eeDetIdItr = endcapDetIds.begin();
      eeDetIdItr != endcapDetIds.end(); ++eeDetIdItr){
    // for each superCrystal, the interest is the highest interest
    // of any trigger tower associated with at least one crystal from this SC.
    // The forced bit must be set if the flag of one of these trigger towers has
    // the forced bit set.
    EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(*eeDetIdItr);
    assert(trigTower.rawId() != 0); 
    EEDetId eeDetId(*eeDetIdItr);
    int iz = (eeDetId.zside() > 0) ? 1 : 0;
    //Following statement will set properly the actual 2-bit flag value
    //and the forced bit: TTF forced bit is propagated to every RU that
    //overlaps with the corresponding TT.
    combineFlags(eeRuInterest(eeDetId), getTowerInterest(trigTower));
  }
#else //ECALSELECTIVEREADOUT_NOGEOM defined
  EEDetId xtal;
  for(int iZ0=0; iZ0<2; ++iZ0){//0->EE-, 1->EE+
    for(unsigned iX0=0; iX0<nEndcapXBins; ++iX0){
      for(unsigned iY0=0; iY0<nEndcapYBins; ++iY0){

        if (!(xtal.validDetId(iX0+1, iY0+1, (iZ0>0?1:-1)))){ 
          continue; 
        }
        xtal = EEDetId(iX0+1, iY0+1, (iZ0>0?1:-1));
        //works around a EEDetId bug. To remove once the bug fixed.
        if(39 <= iX0 && iX0 <= 60 && 45 <= iY0 && iY0 <= 54){
          continue;
        }
        // for each superCrystal, the interest is the highest interest
        // of any trigger tower associated with any crystal in this SC
        EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(xtal);
        assert(trigTower.rawId() != 0); 
        //Following statement will set properly the actual 2-bit flag value
        //and the forced bit: TTF forced bit is propagated to every RU that
        //overlaps with the corresponding TT.
        combineFlags(eeRuInterest(xtal), getTowerInterest(trigTower));

        assert(0<= eeRuInterest(xtal)  && eeRuInterest(xtal) <= 0x7);
        
      } //next iY0
    } //next iX0
  } //next iZ0
#endif //ECALSELECTIVEREADOUT_NOGEOM not defined
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
  //   int iz = (eeDetId.zside() > 0) ? 1 : 0;
  //   int superCrystalX = (eeDetId.ix()-1) / 5;
  //   int superCrystalY = (eeDetId.iy()-1) / 5;
  //   return supercrystalInterest[iz][superCrystalX][superCrystalY];
  return const_cast<EcalSelectiveReadout*>(this)->eeRuInterest(eeDetId);
}

EcalSelectiveReadout::towerInterest_t 
EcalSelectiveReadout::getSuperCrystalInterest(const EcalScDetId& scDetId) const 
{
  return const_cast<EcalSelectiveReadout*>(this)->eeRuInterest(scDetId);
}

EcalSelectiveReadout::towerInterest_t&
EcalSelectiveReadout::eeRuInterest(const EEDetId& eeDetId){
  const EcalElectronicsId& id = theElecMap->getElectronicsId(eeDetId);
  const int iZ0 = id.zside()>0 ? 1 : 0;
  const int iDcc0 = id.dccId()-1;
  const int iDccPhi0 = (iDcc0<9)?iDcc0:(iDcc0-45);
  const int iDccCh0 = id.towerId()-1;
  assert(0 <= iDccPhi0 && iDccPhi0 < nDccPerEe);
  assert(0 <= iDccCh0  && iDccCh0 < maxDccChs);

  assert(eeRuInterest_[iZ0][iDccPhi0][iDccCh0] == UNKNOWN
         || (0<= eeRuInterest_[iZ0][iDccPhi0][iDccCh0]
             && eeRuInterest_[iZ0][iDccPhi0][iDccCh0] <=7));
  
  return eeRuInterest_[iZ0][iDccPhi0][iDccCh0];
}

EcalSelectiveReadout::towerInterest_t&
EcalSelectiveReadout::eeRuInterest(const EcalScDetId& scDetId){
  std::pair<int, int> dccAndDccCh = theElecMap->getDCCandSC(scDetId);
  const int iZ0 = (scDetId.zside()>0) ? 1: 0;
  const int iDcc0 = dccAndDccCh.first-1;
  const int iDccPhi0 = (iDcc0<9)?iDcc0:(iDcc0-45);
  const int iDccCh0 = dccAndDccCh.second-1;
  assert(0 <= iDccPhi0 && iDccPhi0 <= nDccPerEe);
  assert(0 <= iDccCh0  && iDccCh0 <= maxDccChs);

  assert(-1<= eeRuInterest_[iZ0][iDccPhi0][iDccCh0] && eeRuInterest_[iZ0][iDccPhi0][iDccCh0] <=7);
  
  return eeRuInterest_[iZ0][iDccPhi0][iDccCh0];
}


EcalSelectiveReadout::towerInterest_t
EcalSelectiveReadout::getTowerInterest(const EcalTrigTowerDetId & tower) const 
{
  // remember, array indices start at zero
  int iEta = tower.ieta()<0? tower.ieta() + nTriggerTowersInEta/2
    : tower.ieta() + nTriggerTowersInEta/2 -1;
  int iPhi = tower.iphi() - 1;

  assert(-1 <= towerInterest[iEta][iPhi] && towerInterest[iEta][iPhi] < 8);
  
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
      //copy forced bit from ttFlags to towerInterests:
      towerInterest[iEta][iPhi] =  (towerInterest_t) (towerInterest[iEta][iPhi]
                                                      | (ttFlags[iEta][iPhi] & FORCED_MASK)) ;
      if((ttFlags[iEta][iPhi] & ~FORCED_MASK) == TTF_HIGH_INTEREST){
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
            combineFlags(towerInterest[iEtaNeigh][iPhiNeigh_],
                         NEIGHBOUR);
          }
        }
      } else if((ttFlags[iEta][iPhi] & ~FORCED_MASK) == TTF_MID_INTEREST){
        combineFlags(towerInterest[iEta][iPhi], SINGLE);
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
    "#           " << srpFlagMarker[4] << ": forced low interest\n"
    "#           " << srpFlagMarker[5] << ": forced single\n"
    "#           " << srpFlagMarker[6] << ": forced neighbout\n"
    "#           " << srpFlagMarker[7] << ": forced center\n"
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
      towerInterest_t srFlag;
      char c;
      if(!EcalScDetId::validDetId(iX+1,iY+1,endcap>=1?1:-1)){
        //        srFlag = UNKNOWN;
        c = ' ';
      } else{
        srFlag
          = getSuperCrystalInterest(EcalScDetId(iX+1, iY+1, endcap>=1?1:-1));//supercrystalInterest[endcap][iX][iY];
        c = srFlag==UNKNOWN ? '?' : srpFlagMarker[srFlag];
      }
      os << c;
    }
    os << "\n"; //one Y supercystal column per line
  } //next supercrystal X-index
}


std::ostream & operator<<(std::ostream & os, const EcalSelectiveReadout & selectiveReadout)
{
  selectiveReadout.print(os);
  return os;
}


void
EcalSelectiveReadout::printDccChMap(std::ostream& os) const{
  for(int i=-1; i<=68; ++i){
    if((i+1)%10==0) os << "//";
    os << std::setw(2) << i << ": " << (char)('0'+i);
    if(i%10==9) os << "\n"; else os << " ";
  }
  
  os << "\n";
  
  for(int endcap = 0; endcap < 2; ++endcap){
    os << "Sc2DCCch0: " << (endcap?"EE+":"EE-") << "\n";
    for(size_t iY=0; iY<nSupercrystalYBins; ++iY){
      os << "Sc2DCCch0: ";
      for(size_t iX=0; iX<nSupercrystalXBins; ++iX){
        //if(iX) os << ",";
        if(!EcalScDetId::validDetId(iX+1,iY+1,endcap>=1?1:-1)){
          //os << std::setw(2) << -1;
          os << (char)('0'-1);
        } else{
          //os << std::setw(2) << theElecMap->getDCCandSC(EcalScDetId(iX+1, iY+1, endcap>0?1:-1)).second-1;
          os << (char)('0'+(theElecMap->getDCCandSC(EcalScDetId(iX+1, iY+1, endcap>0?1:-1)).second-1));
        }
      }
      os << "\n";
    }
    os << "\n";
  }
  os << "\n";
}
