#ifndef PhysicsTools_PatAlgos_interface_SuperClusterHelper_h
#define PhysicsTools_PatAlgos_interface_SuperClusterHelper_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


class SuperClusterHelper {
 public:
  SuperClusterHelper(const reco::GsfElectron * electron, const EcalRecHitCollection * rechits, const CaloTopology*, const CaloGeometry* );
  SuperClusterHelper(const pat::Electron * electron, const EcalRecHitCollection * rechits, const CaloTopology*, const CaloGeometry* );
  ~SuperClusterHelper(){};
  
  float rawEnergy() const {return theSuperCluster_->rawEnergy();}
  float eta() const {return theSuperCluster_->eta();}
  float phi() const {return theSuperCluster_->phi();}
  float etaWidth() const {return theSuperCluster_->etaWidth();}
  float phiWidth() const {return theSuperCluster_->phiWidth();}
  float clustersSize() const {return theSuperCluster_->clustersSize();}
  float hadronicOverEm() const {return theElectron_->hadronicOverEm();}
  float sigmaIetaIeta() const {return theElectron_->sigmaIetaIeta();}
  float preshowerEnergy() const {return theSuperCluster_->preshowerEnergy();}
  float preshowerEnergyOverRaw() const {return theSuperCluster_->preshowerEnergy()/theSuperCluster_->rawEnergy();}
  float e3x3()      const { return EcalClusterTools::e3x3(*seedCluster_,rechits_,topology_); }
  float e5x5()      const { return EcalClusterTools::e5x5(*seedCluster_,rechits_,topology_); }  
  float eMax()      const { return EcalClusterTools::eMax(*seedCluster_,rechits_); }
  float e2nd()      const { return EcalClusterTools::e2nd(*seedCluster_,rechits_); }
  float eTop()      const { return EcalClusterTools::eTop(*seedCluster_,rechits_,topology_); }
  float eBottom()   const { return EcalClusterTools::eBottom(*seedCluster_,rechits_,topology_); }
  float eLeft()     const { return EcalClusterTools::eLeft(*seedCluster_,rechits_,topology_); }
  float eRight()    const { return EcalClusterTools::eRight(*seedCluster_,rechits_,topology_); }
  float e2x5Max()   const { return EcalClusterTools::e2x5Max(*seedCluster_,rechits_,topology_); }
  float e2x5Top()   const { return EcalClusterTools::e2x5Top(*seedCluster_,rechits_,topology_); }
  float e2x5Bottom()const { return EcalClusterTools::e2x5Bottom(*seedCluster_,rechits_,topology_); }
  float e2x5Left()  const { return EcalClusterTools::e2x5Left(*seedCluster_,rechits_,topology_); }
  float e2x5Right() const { return EcalClusterTools::e2x5Right(*seedCluster_,rechits_,topology_); } 
  float r9()        const { //std::cout << " E3x3 " << e3x3() << " raw "  << theSuperCluster_->rawEnergy() << std::endl;
    return e3x3()/theSuperCluster_->rawEnergy();}
  float spp();
  float sep();
  float seedEta()   const { return seedCluster_->eta(); }
  float seedPhi()   const { return seedCluster_->phi(); }
  float seedEnergy()const { return seedCluster_->energy();}
  int ietaSeed()  { localCoordinates(); return ietaSeed_;}
  int iphiSeed()  { localCoordinates(); return iphiSeed_;}
  float etaCrySeed()  { localCoordinates(); return etaCrySeed_;}
  float phiCrySeed()  { localCoordinates(); return phiCrySeed_;} 
  float thetaTilt() { localCoordinates(); return thetaTilt_;}
  float phiTilt() {localCoordinates(); return phiTilt_;}

  // i=1,2,3 (0 would be the seed)
  float subClusterEnergy(unsigned i) const;
  float subClusterEta(unsigned i) const;
  float subClusterPhi(unsigned i) const;
  float subClusterEmax(unsigned i) const;
  float subClusterE3x3(unsigned i) const;
  // does not include the seed energy
  float eSubClusters() const {return eSubClusters_;}

  float esClusterEnergy(unsigned i) const;
  float esClusterEta(unsigned i) const;
  float esClusterPhi(unsigned i) const;

  float nPreshowerClusters() const { return nESClusters_;}
  float eESClusters() const { return eESClusters_;}

 private:
  const reco::GsfElectron * theElectron_;
  const reco::SuperCluster * theSuperCluster_;
  std::vector<const reco::CaloCluster*> theBasicClusters_;
  std::vector<const reco::CaloCluster*> theESClusters_;
  const reco::CaloCluster * seedCluster_;
  const EcalRecHitCollection * rechits_;
  const CaloTopology* topology_;
  const CaloGeometry* geometry_;
  EcalClusterLocal local_;
  bool barrel_;

  
  /// cached variables 
  /// covariance matrix
  bool covComputed_;
  std::vector<float> vCov_;
  float spp_;
  float sep_;
  /// local coordinates
  bool localCoordinatesComputed_;
  int ietaSeed_; // ix in the endcaps
  int iphiSeed_; // iy in the endcaps
  float etaCrySeed_;
  float phiCrySeed_;
  float thetaTilt_;
  float phiTilt_;
  unsigned nBasicClusters_;
  unsigned nESClusters_;
  float eSubClusters_;
  float eESClusters_;

 private:
  void computeLocalCovariances();
  void localCoordinates();

  static bool sortClusters(const reco::CaloCluster* c1, const reco::CaloCluster* c2) 
  {
      return (c1->energy()>c2->energy());
  }
};


#endif
