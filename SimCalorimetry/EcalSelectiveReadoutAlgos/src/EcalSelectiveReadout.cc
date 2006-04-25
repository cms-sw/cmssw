//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-"
/*
 * $Id$
 */

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include <string>
#include <iostream>
#include <cstring> //for memcpy

using namespace std;

const string EcalSelectiveReadout::fileName("triggerTower.root");

EcalSelectiveReadout::EcalSelectiveReadout(const vector<double>& thr,
                                           int dEta_, int dPhi_):
  threshold(thr),dEta(dEta_), dPhi(dPhi_)
{
  if(threshold.size()!=2){
    cerr << __FILE__ << ":" << __LINE__ << ": "
         << "Argument 'thresholds' of "
      "EcalSelectiveReadout::EcalSelectiveReadout(vector<int> thresholds) "
      "method must a vector of two elements (with the low and the high "
      "trigger tower Et for the selective readout flags. Aborts."
         << endl;
    abort();
  }
  readEndcapTowerMap();
  //init superCrystalInterest (sets all elts to 'unknown'):
  resetSupercrystalInterest();
}

EcalSelectiveReadout::EcalSelectiveReadout(const vector<double>& thr,
                                                       const int*towerMap,
                                                       int dEta_,
                                                       int dPhi_):
  threshold(thr), dEta(dEta_), dPhi(dPhi_)
{
  memcpy(triggerTower, towerMap, sizeof(triggerTower));
  if(threshold.size()!=2){
    cerr << __FILE__ << ":" << __LINE__ << ": "
         << "Argument 'thresholds' of "
      "EcalSelectiveReadout::EcalSelectiveReadout(vector<int> thresholds) "
      "method must a vector of two elements (with the low and the high "
      "trigger tower Et for the selective readout flags. Aborts."
         << endl;
    abort();
  }
  //init superCrystalInterest (sets all elts to 'unknown'):
  resetSupercrystalInterest();
}

void EcalSelectiveReadout::dumpTriggerTowerMap(ostream& out) const{
  out << "iEndcap\tiSC\tCrystal\tTT iX\tTT iY\n";
  for(size_t iEndcap=0; iEndcap < nEndcaps; ++iEndcap){
    for(size_t iX=0; iX < nEndcapXBins;
          ++iX){
      for(size_t iY=0; iY < nEndcapYBins; ++iY){
        out << iEndcap << "\t"
            << iX << "\t"
            << iY << "\t"
            << triggerTower[iEndcap][iX][iY][0] << "\t"
            << triggerTower[iEndcap][iX][iY][1] << "\n";
      }
    }
  }
  out << flush;
}

void EcalSelectiveReadout::resetSupercrystalInterest(){
  //init superCrystalInterest (sets all elts to 'unknown'):
  for(size_t iCap=0; iCap < nEndcaps; ++iCap){
    for(size_t iSCX = 0; iSCX < nSupercrystalXBins; ++iSCX){
      for(size_t iSCY = 0; iSCY < nSupercrystalYBins; ++iSCY){
        supercrystalInterest[iCap][iSCX][iSCY] = unknown;
      }
    }
  }
}

void  EcalSelectiveReadout::readEndcapTowerMap(){
/*
  TFile f(fileName.c_str(), "READ");  
  TTree* tree = dynamic_cast<TTree*>(f.Get("TT"));
                                     
  if(tree==0){
    cerr << "Tree \"TT\" not found. Aborts." << endl;
    abort();
  }

  tree->SetBranchAddress("TT", &triggerTower);
  assert(tree->GetEntries()==1);
  tree->GetEntry(0);
  //  dumpTriggerTowerMap();
  f.Close();
*/
}

void
EcalSelectiveReadout::runSelectiveReadout0(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]) {
  //classifies the trigger towers (single,center,neighbor,low interest)
  vector<vector<towerInterest_t> > towerInterest_ =
    classifyTriggerTowers(towerEt);

  for(size_t i=0; i < towerInterest_.size(); ++i){
    for(size_t j=0; j < towerInterest_[i].size();++j){
      towerInterest[i][j] = towerInterest_[i][j];
    }
  }

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
    cout << __FILE__ << ":" << __LINE__ << ": \n" 
         << nTriggerTowerB[0] << " low interest TT in barrel\n"
         << nTriggerTowerB[1] << " single TT in barrel\n"
         << nTriggerTowerB[2] << " neighbor interest TT in barrel\n"
         << nTriggerTowerB[3] << " centre interest TT in barrel\n"
         << nTriggerTowerE[0] << " low interest TT in endcap\n"
         << nTriggerTowerE[1] << " single TT in endcap\n"
         << nTriggerTowerE[2] << " neighbor TT in endcap\n"
         << nTriggerTowerE[3] << " center TT in endcap\n";
  }
  //end TT interest class composition debugging display
  
  //For the endcap the TT classification must be mapped to the SC:
  for(size_t iEndcap = 0; iEndcap < nEndcaps; ++iEndcap){   
    for(size_t iSCX = 0; iSCX < nSupercrystalXBins; ++iSCX){
      for(size_t iSCY = 0; iSCY < nSupercrystalYBins; ++iSCY){
        towerInterest_t interest = lowInterest;
        //looks at the interest flag of each TT containing at least a crystal
        //of this SC and select the less restrictive one (i.e. with
        //the highest value).
        for(size_t iCrysX = 0; iCrysX < supercrystalEdge; ++iCrysX){
          for(size_t iCrysY = 0; iCrysY < supercrystalEdge; ++iCrysY){
            const int* tower;
            tower = triggerTower[iEndcap][iSCX*5+iCrysX][iSCY*5+iCrysY];
            if(tower[0] > 0){//crystal exits (check required because of
              //...............incomplete border supercrystals)
              setHigher(interest,
                        towerInterest[tower[0]][tower[1]]);
            }
          }//next crystal x
        }//next crystal y
        supercrystalInterest[iEndcap][iSCX][iSCY] = interest;
      }//next supercrystal x
    }//next supercrystal y
  }//next endcap
}



vector<vector<EcalSelectiveReadout::towerInterest_t> >
EcalSelectiveReadout::classifyTriggerTowers(const float towerEt[nTriggerTowersInEta][nTriggerTowersInPhi]) const{
  vector<towerInterest_t> tmp(nTriggerTowersInPhi, lowInterest);
  vector<vector<towerInterest_t> >towerFlags(nTriggerTowersInEta, tmp);
  for(int iEta=0; iEta < (int)nTriggerTowersInEta; ++iEta){
    for(int iPhi=0; iPhi < (int)nTriggerTowersInPhi; ++iPhi){
      if(towerEt[iEta][iPhi] > threshold[1]){
        //flags this tower as a center tower
        towerFlags[iEta][iPhi] = center;
        //flags the neighbours of this tower
        for(int iEtaNeigh = max<int>(0,iEta-dEta);
            iEtaNeigh <= min<int>(nTriggerTowersInEta-1, iEta+dEta);
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
            setHigher(towerFlags[iEtaNeigh][iPhiNeigh_],
                      neighbour);
          }
        }
      } else if(towerEt[iEta][iPhi] > threshold[0]){
	setHigher(towerFlags[iEta][iPhi], single);
      }
    }
  }
  return towerFlags;
}

vector<size_t> EcalSelectiveReadout::towerOfCrystal(size_t iEndcap,
                                                          size_t iX,
                                                          size_t iY) const{
  vector<size_t> result(2);
  const int *index = triggerTower[iEndcap][iX][iY];
  result[0] = index[0];
  result[1] = index[1];
  return result;
}



vector<vector<size_t> >
EcalSelectiveReadout::crystalsOfTower(size_t iEta, size_t iPhi) const{
  vector<vector<size_t> > result;
  if(iEta>10 && iEta < 45){//in barrel
    size_t iEtaSize = 5;
    size_t iPhiSize = 5;
    size_t iEtaMin = iEta*iEtaSize;
    size_t iPhiMin = iPhi*iPhiSize;
    vector<size_t> xtal(2);
    result.reserve(iEtaSize*iPhiSize);
    for(size_t eta = iEtaMin; eta < iEtaMin+iEtaSize; ++eta){
      for(size_t phi = iPhiMin; phi < iPhiMin+iPhiSize; ++phi){
        xtal[0] = eta;
        xtal[1] = phi;
        result.push_back(xtal);
      }
    }
  } else{//in endcap
    vector<size_t> xtal(3);
    for(xtal[0] = 0; xtal[0] < 2; ++xtal[0]){//iZ
      for(xtal[1] = 0; xtal[1] < nEndcapXBins; ++xtal[1]){
        for(xtal[2] = 0; xtal[2] < nEndcapYBins; ++xtal[2]){
          const int* TT;
          TT = triggerTower[xtal[0]][xtal[1]][xtal[2]];
          if(iEta == (size_t)TT[0] && iPhi == (size_t)TT[1]){
            result.push_back(xtal);
          }
        }
      }
    }
  }
  return result;
}
