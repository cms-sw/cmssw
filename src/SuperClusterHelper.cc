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
//  for(unsigned ir=0; ir<rechits_->size();++ir) {
//    std::cout << "RecHit " << (*rechits_)[ir].id().rawId() << " " << (*rechits_)[ir] << std::endl;
//  }
  seedCluster_ = & (*(electron->seed()));
//  std::vector< std::pair<DetId, float> >::const_iterator it=seedCluster_->hitsAndFractions().begin();
//  std::vector< std::pair<DetId, float> >::const_iterator itend=seedCluster_->hitsAndFractions().end();
//  for( ; it!=itend ; ++it) {
//    DetId id=it->first;
//    std::cout << " Basic cluster " << id.rawId() << std::endl;
//  }
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
      std::cout << "Computing sep "<< vCov_[1] << std::endl;
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
  std::cout << " SEP " << sep_ << std::endl;
  return sep_;
}

void SuperClusterHelper::localCoordinates() {
  if (localCoordinatesComputed_) return;

  if (barrel_) {
    local_.localCoordsEB(*seedCluster_, *geometry_, etaCrySeed_ , phiCrySeed_ ,ietaSeed_ , iphiSeed_ , thetaTilt_ , phiTilt_);
  } else {
    local_.localCoordsEE(*seedCluster_, *geometry_, etaCrySeed_ , phiCrySeed_ ,ietaSeed_ , iphiSeed_ , thetaTilt_ , phiTilt_);
  }
    localCoordinatesComputed_ = true;
}
