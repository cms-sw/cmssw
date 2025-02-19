///////////////////////////////////////////////////////////////////////////////
// File: SimG4HcalHitCluster.cc
// Description: Cluster class in SimG4HcalValidation
///////////////////////////////////////////////////////////////////////////////
#include "Validation/HcalHits/interface/SimG4HcalHitCluster.h"

#include "CLHEP/Vector/LorentzVector.h"

SimG4HcalHitCluster::SimG4HcalHitCluster(): ec(0), etac(0), phic(0) {}

SimG4HcalHitCluster::~SimG4HcalHitCluster() {}

bool SimG4HcalHitCluster::operator<(const SimG4HcalHitCluster& cluster) const {
  return (ec/cosh(etac) < cluster.e()/cosh(cluster.eta())) ? false : true ;
}

SimG4HcalHitCluster& SimG4HcalHitCluster::operator+=(const CaloHit& hit) {

  hitsc.push_back(hit);
 
  if (ec == 0. && etac == 0. && phic == 0.) {
    ec   = hit.e();
    etac = hit.eta();
    phic = hit.phi();
  } else {   
    // cluster px,py,pz
    double et = ec / my_cosh(etac);
    double px = et * cos(phic);
    double py = et * sin(phic);
    double pz = et * my_sinh(etac); 

    CLHEP::HepLorentzVector clusHLV(px,py,pz,ec);
      
    // hit px,py,pz
    double eh   = hit.e();
    double etah = hit.eta();
    double phih = hit.phi();
    et = eh / my_cosh(etah);
    px = et * cos(phih);
    py = et * sin(phih);
    pz = et * my_sinh(etah); 
      
    CLHEP::HepLorentzVector hitHLV(px,py,pz,eh);
      
    // clus + hit
    clusHLV += hitHLV;
      
    double theta  = clusHLV.theta();
    etac = -log(tan(theta/2.));
    phic = clusHLV.phi();
    ec   = clusHLV.t();
  }

  return *this;
}

double SimG4HcalHitCluster::collectEcalEnergyR() {

  double sum = 0.;
  std::vector<CaloHit>::iterator itr;

  for (itr = hitsc.begin(); itr < hitsc.end(); itr++) {
    if (itr->det() == 10 || itr->det() == 11 || itr->det() == 12) {
      sum += itr->e(); 
    }    
  }
  return sum;
}

std::ostream& operator<<(std::ostream& os, const SimG4HcalHitCluster& cluster){
  os << " SimG4HcalHitCluster:: E " << cluster.e() << "  eta " << cluster.eta()
     << "  phi " << cluster.phi();
  return os;
}
