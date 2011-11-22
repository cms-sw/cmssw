#ifndef L1CaloCluster_h
#define L1CaloCluster_h


#include <ostream>
/*This ClassDescribes the 2x2 cluster thing
0|1
- -  The Cluster reference point is 0 (ieta,iphi)=0,0 
2|3

M.Bachtis, S.Dasu 
University of Wisconsin-Madison
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

namespace l1slhc {

class L1CaloCluster
{
 public:
  L1CaloCluster();
  L1CaloCluster(int,int);
 ~L1CaloCluster();

  void  setFg(bool); //Set FG Bit
  void  setEGamma(bool); //Set EGamma Bit
  void  setEGammaValue(int); //Set EGamma Value E/E+H (%)
  void  setIsoClusters(int,int); //Number of isolated objectsisolation Clusters
  void  setIsoEG(bool); //EG isolation
  void  setIsoTau(bool); //Tau isolation
  void  setCentral(bool); //Central Bit 
  void  setLeadTower(bool);//Lead Tower over threshold bit for taus
  void  setLorentzVector(const math::PtEtaPhiMLorentzVector&); //Central Bit 
  void  setPosBits(int,int);
  void  setConstituents(const L1CaloTowerRefVector&);
  void  setE(int);
  void  addConstituent(const L1CaloTowerRef&);
  int   hasConstituent(int,int);
  void  removeConstituent(int,int);

  //Get Functions
  int   iEta() const;//Eta of origin in integer coordinates
  int   iPhi() const;//Phi of Origin in integer
  int   E() const;//Compressed Et 
  int   innerEta() const; //Weighted position eta
  int   innerPhi() const; //Weighted position phi
  L1CaloTowerRefVector getConstituents() const;  
  L1CaloTowerRef getConstituent(int);

  
 

  
  //Electron Variables
  bool fg() const; //Finegrain bit
  bool eGamma() const; //Electron/Photon bit
  int eGammaValue() const; //Electron/Photon bit

  //isolation Variables
  bool isCentral() const;//Means that the cluster was not pruned during isolation
  bool isoEG() const; //Egamma Isolatioon
  bool isoTau() const; //Tau isolation
  int isoClustersEG() const; //2x2 isolation clusters for Egamma cuts
  int isoClustersTau() const; //2x2 isolation clusters for Tau Cut
  bool hasLeadTower() const;

  //Trigger Results
  bool isEGamma() const; //Returns the EGAMMA decision 
  bool isIsoEGamma() const; //Returns the iso EGAMMA decision 
  bool isTau() const;  //returns The Tau decison
  bool isIsoTau() const;  //returns The Tau decison

  math::PtEtaPhiMLorentzVector p4() const;//returns Physics wise LorentzVector in eta,phi continuous space


  
 private:
  //Refs to teh caloTowwers
  L1CaloTowerRefVector constituents_;

  //Coordinates of the reference Point  
  int iEta_;
  int iPhi_;
  int E_;
  //FineGrain / EGamma /Isolations

  bool fg_;
  bool eGamma_;
  bool central_;
  bool isoEG_;
  bool leadTowerTau_;
  bool isoTau_;
  int eGammaValue_;
  int innerEta_;
  int innerPhi_;
  int isoClustersEG_;
  int isoClustersTau_;

  math::PtEtaPhiMLorentzVector p4_;//Lorentz Vector of precise position



};



//Sorting class

struct HigherClusterEt
{
  bool operator()(L1CaloCluster cl1,L1CaloCluster cl2)
  {
    return (cl1.E()) > (cl2.E()); 
  }

};



}

std::ostream& operator<<(std::ostream& s, const l1slhc::L1CaloCluster&);

#endif

