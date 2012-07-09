/* SLHC Calo Trigger 
Class for Trigger configuration...Contains wiring and Cuts
M.Bachtis,S.Dasu. University of Wisconsin-Madison
*/


#ifndef L1CaloTriggerSetup_h
#define L1CaloTriggerSetup_h

#include <vector>
#include <map>
#include <string>


class L1CaloTriggerSetup
{
 public:
  int latticeDim_;  //Eta Dimension of the square lattice 
  int latticeEta0_; //Eta of the Square Lattice
  int latticePhi0_; //Phi of the Square Lattice
  int latticeEtaM_; //Eta of the Square Lattice
  int latticePhiM_; //Phi of the Square Lattice
  int ecalActivityCut_; //Ecal Activity Cut
  int hcalActivityCut_; //hcalActivity Cut
  int electronCutA_; //Electron ID Cut
  int electronCutB_; //Electron ID Cut
  int electronCutC_; //Electron ID Cut

  int tauSeedTower_; //Electron ID Cut

  int clusterCut_; //Cluster Threshold Cut
  int isolationEA_;//Isolation ratio Electron;
  int isolationEB_;//Isolation ratio Electron;

  int isolationTA_;//Isolation ratio Tau;
  int isolationTB_;//Isolation ratio Tau;
  int isolationThrEG_;//Isolation threshold EG;
  int isolationThrTau_;//Isolation threshold Tau;
  int isolationZone_;//Number of towers that define the isolation zone;
  //  int jetCenter_ ; //jet Center Deviation
  int jetET_ ; //jet Center Deviation
  int fineGrainPass_; //ignore fine grain bit (set it to 0)



 public:

  //Geometry Mapping between towers/lattice 
  std::map<int ,std::pair<int,int> > geoMap_;

  //Lattice Navigation helper Functions
  int getEta(int bin)   const  //get the ieta of a specific Bin
    {
      return bin%latticeDim_;
      
    } 
  int getPhi(int bin)   const  //get the iphi of a specific bin
    {  
      return bin/latticeDim_;
    }

  int getBin(int eta,int phi) const //get the bin for a ieta,iphi pair
    {
      return phi*latticeDim_+eta;
    }

    int etaMin() const
    {
      return latticeEta0_;
    }

  int etaMax() const
    {
      return latticeEtaM_;
    }

    int phiMin() const
    {
      return latticePhi0_;
    }

  int phiMax() const
    {
      return latticePhiM_;
    }


  int ecalActivityThr() const
    {
      return ecalActivityCut_;
    }

  int hcalActivityThr() const
    {
      return hcalActivityCut_;
    }

  int clusterThr() const
    {
      return clusterCut_;
    }

  int seedTowerThr() const
    {
      return tauSeedTower_;
    }

  std::vector<int> electronThr()
    {
      std::vector<int> a;
      a.push_back(electronCutA_);
      a.push_back(electronCutB_);
      a.push_back(electronCutC_);

      return a;
    }

  int nIsoTowers() const
    {
      return isolationZone_;
    }

/*   int jetCenterDev() */
/*     { */
/*       return jetCenter_; */
/*     } */

  int minJetET() const
    {
      return jetET_;
    }

  int fineGrainPass() const
  {
    return fineGrainPass_;
  }


  std::vector<int> isoThr()
    {
      std::vector<int> a;
      a.push_back(isolationThrEG_);
      a.push_back(isolationThrTau_);

      return a;
    }



  std::vector<int> isolationE()
    {
      std::vector<int> a;
      a.push_back(isolationEA_);
      a.push_back(isolationEB_);

      return a;
    }

  std::vector<int> isolationT()
    {
      std::vector<int> a;
      a.push_back(isolationTA_);
      a.push_back(isolationTB_);
      return a;
    }






  L1CaloTriggerSetup()
    {
       latticeDim_  = 1;
       latticeEta0_ = 1;
       latticePhi0_ = 1;
       latticeEtaM_ = -1000;
       latticePhiM_ = -1111;
       ecalActivityCut_=2; 
       hcalActivityCut_=6; 
       electronCutA_=8; 
       electronCutB_=0; 
       electronCutC_=0; 

       tauSeedTower_=0;
       clusterCut_=4; 
    }

  ~L1CaloTriggerSetup()
    {

    }

  void setGeometry(int eta0,int phi0,int etam,int phim,int dim)
    {
      latticeDim_  = dim;
      latticeEta0_ = eta0;
      latticePhi0_ = phi0;
      latticeEtaM_ = etam;
      latticePhiM_ = phim;

    }

  void addWire(int no,int eta,int phi) //Configure Wire Connection
    {
      std::pair<int,int> p = std::make_pair(eta,phi);
      geoMap_[no] = p;
    }

 
  void setThresholds(int ecal_a_c,int hcal_a_c,int egammaA,int egammaB,int egammaC,int tauSeed,int clusterCut,int isoRatioEA,int isoRatioEB,int isoRatioTA,int isoRatioTB,int isoZone,int isoThresEG,int isoThresTau,int jetet, int fgp)
    {

      ecalActivityCut_ = ecal_a_c;
      hcalActivityCut_ = hcal_a_c;
      electronCutA_ = egammaA;
      electronCutB_ = egammaB;
      electronCutC_ = egammaC;
      tauSeedTower_ = tauSeed;
      clusterCut_ = clusterCut;
      isolationEA_ = isoRatioEA;
      isolationEB_ = isoRatioEB;
      isolationTA_ = isoRatioTA;
      isolationTB_ = isoRatioTB;
      isolationZone_ = isoZone;
      isolationThrEG_ = isoThresEG;
      isolationThrTau_ = isoThresTau;
      //      jetCenter_ = jetc;
      jetET_ = jetet;
      fineGrainPass_ = fgp;
    }


};

#endif




