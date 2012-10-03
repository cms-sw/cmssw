#ifndef PhysicsTools_PatAlgos_interface_SuperClusterHelper_h
#define PhysicsTools_PatAlgos_interface_SuperClusterHelper_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


class SuperClusterHelper {
 public:
  SuperClusterHelper(const reco::GsfElectron * electron, const EcalRecHitCollection * rechits, const CaloTopology*, const CaloGeometry* );
  SuperClusterHelper(const pat::Electron * electron, const EcalRecHitCollection * rechits, const CaloTopology*, const CaloGeometry* );
  ~SuperClusterHelper(){};
  
  float rawEnergy() const {return theElectron_->superCluster()->rawEnergy();}
  float eta() const {return theElectron_->superCluster()->eta();}
  float phi() const {return theElectron_->superCluster()->phi();}
  float etaWidth() const {return theElectron_->superCluster()->etaWidth();}
  float phiWidth() const {return theElectron_->superCluster()->phiWidth();}
  float clustersSize() const {return theElectron_->superCluster()->clustersSize();}
  float hadronicOverEm() const {return theElectron_->hadronicOverEm();}
  float sigmaIetaIeta() const {return theElectron_->sigmaIetaIeta();}
  float preshowerEnergy() const {return theElectron_->superCluster()->preshowerEnergy();}
  float preshowerEnergyOverRaw() const {return theElectron_->superCluster()->preshowerEnergy()/theElectron_->superCluster()->rawEnergy();}
  float e3x3()      const {return EcalClusterTools::e3x3(*seedCluster_,rechits_,topology_); }
  float e5x5()      const {return EcalClusterTools::e5x5(*seedCluster_,rechits_,topology_); }  
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
  float r9()        const { return e3x3()/theElectron_->superCluster()->rawEnergy();}
  float spp();
  float sep();
  float seedEta()   const { return seedCluster_->eta(); }
  float seedPhi()   const { return seedCluster_->phi(); }
  float seedEnergy()const { return seedCluster_->energy();}
  int ietaSeed() const { return ietaSeed_;}
  int iphiSeed() const { return iphiSeed_;}
  float etaCrySeed() const { return etaCrySeed_;}
  float phiCrySeed() const { return phiCrySeed_;} 

 private:
  const reco::GsfElectron * theElectron_;
  const EcalRecHitCollection * rechits_;
  const reco::CaloCluster * seedCluster_;
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
  int ietaSeed_;
  int iphiSeed_;
  float etaCrySeed_;
  float phiCrySeed_;
  float thetaTilt_;
  float phiTilt_;
  
 private:
  void computeLocalCovariances();
  void localCoordinates();
};


#endif
