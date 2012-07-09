/* L1CaloTower class
Keeps both ECAL/HCAL Tower information .
Used as input for Clustering

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


#ifndef L1CaloTower_h
#define L1CaloTower_h


namespace l1slhc {

class L1CaloTower 
{

public:

  L1CaloTower();
  ~L1CaloTower();
  

  //Setters
  void setPos(int,int);//Set Tower position (iEta,iPhi)
  void setParams(int,int,bool); //Set ECAL,HCAL Energy and finegrain 

  //getters 
  int E() const; 
  int H() const;
  int iEta() const;
  int iPhi() const;
  bool fineGrain() const;

private:
  int E_;
  int H_;
  int iEta_;
  int iPhi_;
  bool fineGrain_;


};

}
#endif

