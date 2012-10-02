#include "EgammaAnalysis/ElectronTools/interface/SuperClusterHelper.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"



SuperClusterHelper::SuperClusterHelper(const reco::GsfElectron * electron, const EcalRecHitCollection * rechits, const CaloTopology * topo, const CaloGeometry* geom) {
  theElectron_  = electron;
  rechits_ = rechits ;
  seedCluster_ = & (*(electron->superCluster()->seed()));
  topology_ = topo;
  geometry_ = geom;
  barrel_ = electron->isEB();
  covComputed_ = false;
  localCoordinatesComputed_ = false;
}

SuperClusterHelper::SuperClusterHelper(const pat::Electron * electron, const EcalRecHitCollection * rechits, const CaloTopology * topo, const CaloGeometry * geom) {
  theElectron_  = (const reco::GsfElectron*)electron;
  rechits_ = rechits ;
  seedCluster_ = & (*(electron->seed()));
  topology_ = topo;
  geometry_ = geom;
  barrel_ = electron->isEB();
  covComputed_ = false;
  localCoordinatesComputed_ = false; 
}

void SuperClusterHelper::computeLocalCovariances() {
  if (!covComputed_) {
    vCov_ = EcalClusterTools::localCovariances(*seedCluster_, rechits_, topology_, 4.7);
    covComputed_ = true;
    
    spp_ = 0;
    if (!isnan(vCov_[2])) spp_ = sqrt (vCov_[2]);
    
    if (theElectron_->sigmaIetaIeta()*spp_ > 0) {
      sep_ = vCov_[1] / theElectron_->sigmaIetaIeta()*spp_;
    } else if (vCov_[1] > 0) {
      sep_ = 1.0;
    } else {
      sep_ = -1.0;
    }
  }
}

float SuperClusterHelper::spp() {
  computeLocalCovariances();
  return spp_;
}

float SuperClusterHelper::sep() {
  computeLocalCovariances();
  return sep_;
}

void SuperClusterHelper::localCoordinates() {
  if (barrel_) {
    local_.localCoordsEB(*seedCluster_, *geometry_, etaCrySeed_ , phiCrySeed_ ,ietaSeed_ , iphiSeed_ , thetaTilt_ , phiTilt_);
  }
}
