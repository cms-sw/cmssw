//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id: EcalSelectiveReadout.cc,v 1.15 2009/06/07 22:38:10 pgras Exp $
 */

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <iomanip>
//#include <iostream> //for debugging


using std::vector;

const char EcalSelectiveReadout::srpFlagMarker[] = {'.', 'S', 'N', 'C', 'F'};

// //list of sets of partial supercrystal whose SR flags are forced to be identical
// //because they are grouped within a same DCC channel (same readout unit).
// //SC (x0,y0) in {(11,11),(11,8),(8,11),(8,8)} are not included here and are treated
// //separetly, because they can contains crystals read in different ZS/S/F mode.   
// //note: {0,0,0} = padding for the array size.
// //[iScGroup][iSc].ix: Supercrytal X-index
// //[iScGroup][iSc].iy: Supercrytal Y-index
// //[iScGroup][iSc].iz: Supercrytal Z-index
// static const int nScGroups = 12;
// static const int maxScInGroup = 4;
// static const struct Xyz {
//   int ix;
//   int iy;
//   int iz;
// } groupedSc[nScGroups][maxScInGroup] = {
//   {{18,15,0}, {19,12,0}, {16,17,0}, {17,16,0}}, //DCC  2, RUs 3 and 25
//   {{12,19,0}, {15,18,0}, { 0, 0,0}, { 0, 0,0}}, //DCC  3, RU 30
//   {{ 4,18,0}, { 7,19,0}, { 0, 0,0}, { 0, 0,0}}, //DCC  4, RU 30
//   {{ 0,12,0}, { 1,15,0}, { 2,16,0}, { 3,17,0}}, //DCC  5, RUs 3 and 25
//   {{ 1, 4,0}, { 2, 3,0}, { 3, 2,0}, { 4, 1,0}}, //DCC  7, RUs 14 and 21
//   {{17, 3,0}, {18, 4,0}, {15, 1,0}, {16, 2,0}}, //DCC  9, RUs 14 and 21
//   {{16,17,1}, {17,16,1}, {18,15,1}, {19,12,1}}, //DCC 47, RUs 3 and 25
//   {{12,19,1}, {15,18,1}, { 0, 0,0}, { 0, 0,0}}, //DCC 48, RU 30
//   {{ 4,18,1}, { 7,19,1}, { 0, 0,0}, { 0, 0,0}}, //DCC 49, RU 30
//   {{ 2,16,1}, { 3,17,1}, { 0,12,1}, { 1,15,1}}, //DCC 50, RUs 3 and 25
//   {{ 1, 4,1}, { 2, 3,1}, { 3, 2,1}, { 4, 1,1}}, //DCC 52, RUs 14 and 21
//   {{17, 3,1}, {18, 4,1}, {15, 1,1}, {16, 2,1}}, //DCC 54, RUs 14 and 21
// };

EcalSelectiveReadout::EcalSelectiveReadout(int dEta_, int dPhi_):
  theTriggerMap(0), theElecMap(0), dEta(dEta_), dPhi(dPhi_) {
}

// void EcalSelectiveReadout::resetSupercrystalInterest(){
//   //init superCrystalInterest (sets all elts to 'UNKNOWN'):
//   for(size_t iCap=0; iCap < nEndcaps; ++iCap){
//     for(size_t iSCX = 0; iSCX < nSupercrystalXBins; ++iSCX){
//       for(size_t iSCY = 0; iSCY < nSupercrystalYBins; ++iSCY){
//         supercrystalInterest[iCap][iSCX][iSCY] = UNKNOWN;
//       }
//     }
//   }
// }

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
  //resetSupercrystalInterest();
  resetEeRuInterest();
  
#ifndef ECALSELECTIVEREADOUT_NOGEOM
  const std::vector<DetId>& endcapDetIds = theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  for(std::vector<DetId>::const_iterator eeDetIdItr = endcapDetIds.begin();
      eeDetIdItr != endcapDetIds.end(); ++eeDetIdItr){
    // for each superCrystal, the interest is the highest interest
    // of any trigger tower associated with at least one crystal from this SC
    EcalTrigTowerDetId trigTower = theTriggerMap->towerOf(*eeDetIdItr);
    assert(trigTower.rawId() != 0); 
    EEDetId eeDetId(*eeDetIdItr);
    int iz = (eeDetId.zside() > 0) ? 1 : 0;
//     int superCrystalX = (eeDetId.ix()-1) / 5;
//     int superCrystalY = (eeDetId.iy()-1) / 5;
//    setHigher(supercrystalInterest[iz][superCrystalX][superCrystalY], 
//              getTowerInterest(trigTower));
    setHigher(eeRuInterest(eeDetId), getTowerInterest(trigTower));
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
        // int superCrystalX = iX0 / 5;
        // int superCrystalY = iY0 / 5;
        // setHigher(supercrystalInterest[iZ0][superCrystalX][superCrystalY], 
        //           getTowerInterest(trigTower));
        setHigher(eeRuInterest(xtal), getTowerInterest(trigTower));
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
//   int iz = (scDetId.zside() > 1) ? 1 : 0;
//   int superCrystalX = scDetId.ix()-1;
//   int superCrystalY = scDetId.iy()-1;
//   return supercrystalInterest[iz][superCrystalX][superCrystalY];
  int iScX0 = scDetId.ix() - 1;
  int iScY0 = scDetId.iy() - 1;
  if(8 <= iScX0 && iScX0 <=11
     && 8 <= iScY0 && iScY0 <=11){
    //an inner partial supercrystal
    // -> no interest flag, because it is not uniform
    // within a such SC
    //for debugging
    //std::cout << __FILE__ << ":" << __LINE__ << ": "
    //           <<  "inner partial SC ix0 = " << iScX0
    //           << " iy0 = " << iScY0 << " -> "
    //           << " DCC = " << theElecMap->getDCCandSC(scDetId).first
    //           << " DCC Ch = " << theElecMap->getDCCandSC(scDetId).second
    //           << "\n";
    return EcalSelectiveReadout::UNKNOWN;
  } else{
    return eeRuInterest(scDetId);
  }
}

EcalSelectiveReadout::towerInterest_t&
EcalSelectiveReadout::eeRuInterest(const EEDetId& eeDetId){
  const EcalElectronicsId& id = theElecMap->getElectronicsId(eeDetId);
  const int iZ0 = id.zside()>0 ? 1 : 0;
  const int iDcc0 = id.dccId()-1;
  const int iDccPhi0 = (iDcc0<9)?iDcc0:(iDcc0-45);
  const int iDccCh0 = id.towerId()-1;
  assert(0 <= iDccPhi0 && iDccPhi0 <= nDccPerEe);
  assert(0 <= iDccCh0  && iDccCh0 <= maxDccChs);
  return eeRuInterest_[iZ0][iDccPhi0][iDccCh0];
}

EcalSelectiveReadout::towerInterest_t
EcalSelectiveReadout::eeRuInterest(const EcalScDetId& scDetId) const{
  std::pair<int, int> dccAndDccCh = theElecMap->getDCCandSC(scDetId);
  const int iZ0 = scDetId.zside()>0 ? 1: 0;
  const int iDcc0 = dccAndDccCh.first-1;
  const int iDccPhi0 = iDcc0<9?iDcc0:(iDcc0-45);
  const int iDccCh0 = dccAndDccCh.second-1;
  assert(0 <= iDccPhi0 && iDccPhi0 <= nDccPerEe);
  assert(0 <= iDccCh0  && iDccCh0 <= maxDccChs);
  return eeRuInterest_[iZ0][iDccPhi0][iDccCh0];
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
      towerInterest_t srFlag;
      if(!EcalScDetId::validDetId(iX+1,iY+1,endcap>=1?1:-1)){
        srFlag = UNKNOWN;
      } else{
        srFlag
          = getSuperCrystalInterest(EcalScDetId(iX+1, iY+1, endcap>=1?1:-1));//supercrystalInterest[endcap][iX][iY];
      }
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
