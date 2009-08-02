#ifndef L1CaloCluster_h
#define L1CaloCluster_h

/*This ClassDescribes the 2x2 cluster thing
0|1
- -  The Cluster reference point is 0 (ieta,iphi)=0,0 
2|3

M.Bachtis, S.Dasu 
University of Wisconsin-Madison
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>

namespace l1slhc {

class L1CaloCluster
{
 public:
  L1CaloCluster();
  L1CaloCluster(int,int);
 ~L1CaloCluster();

  void setTower(int,int);//Adds a calotower to the cluster
  void setFg(bool); //Set FG Bit
  void setEGamma(bool); //Set EGamma Bit
  void setLeadTowerTau(bool); //Set EGamma Bit
  void setEGammaValue(int); //Set EGamma Bit
  void setIsoClusters(int,int); //isolation Clusters
  void setIsoEG(bool); //2x2 isolation
  void setIsoTau(bool); //2x3 isolation 
  void setCentral(bool); //Central Bit 
  void setLorentzVector(const math::PtEtaPhiMLorentzVector&); //Central Bit 
  void setPosBits(int,int);

  //Get Functions
  int iEta() const;//Eta of origin in integer coordinates
  int iPhi() const;//Phi of Origin in integer
  int E() const;//Compressed Et 
  int towerE(int) const; //Get The Tower Et
  int seedTowerE() const; //Get The seed Tower Et
                        
  int innerEta() const; //Weighted position eta
  int innerPhi() const; //Weighted position phi

  
  //Electron Variables
  bool fg() const; //Finegrain bit
  bool eGamma() const; //Electron/Photon bit
  bool hasLeadTowerTau() const; //Tau Bit
  int eGammaValue() const; //Electron/Photon bit

  //isolation Variables
  bool isCentral() const;//Means that the cluster was not pruned during isolation
  bool isoEG() const; //Egamma Isolatioon
  bool isoTau() const; //Tau isolation
  int isoClustersEG() const; //2x2 isolation
  int isoClustersTau() const; //2x2 isolation


  //Trigger Results
  bool isEGamma() const; //Returns the EGAMMA decision 
  bool isIsoEGamma() const; //Returns the iso EGAMMA decision 
  bool isTau() const;  //returns The Tau decison
  math::PtEtaPhiMLorentzVector p4() const;//returns Physics wise LorentzVector in eta,phi continuous space
  
 private:
  std::vector<int> towerE_;//Map of Tower Energies
  //Coordinates of the reference Point  
  int iEta_;
  int iPhi_;

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

typedef std::vector<L1CaloCluster> L1CaloClusterCollection;


//Sorting class

struct HigherClusterEt
{
  bool operator()(L1CaloCluster cl1,L1CaloCluster cl2)
  {
    return (cl1.E()) > (cl2.E()); 
  }

};

}

#endif
