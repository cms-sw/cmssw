#ifndef L1CaloRegion_h
#define L1CaloRegion_h

/*This ClassDescribes the 4x4 region

M.Bachtis, S.Dasu 
University of Wisconsin-Madison
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>

namespace l1slhc {

class L1CaloRegion
{
 public:
  L1CaloRegion();
  L1CaloRegion(int,int,int);
 ~L1CaloRegion();

  //Get Functions
  int iEta() const;//Eta of origin in integer coordinates
  int iPhi() const;//Phi of Origin in integer
  int E() const;//Compressed Et 


  
 private:
  int iEta_;
  int iPhi_;
  int E_;

};

}

#endif
