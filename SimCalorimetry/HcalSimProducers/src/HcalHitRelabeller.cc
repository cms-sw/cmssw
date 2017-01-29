#include "SimCalorimetry/HcalSimProducers/interface/HcalHitRelabeller.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

HcalHitRelabeller::HcalHitRelabeller(const edm::ParameterSet& ps) : 
  theGeometry(0), theRecNumber(0),
  neutralDensity_(ps.getParameter<bool>("doNeutralDensityFilter")) { }

void HcalHitRelabeller::process(std::vector<PCaloHit>& hcalHits) {

  if (theRecNumber) {
    for (unsigned int ii=0; ii<hcalHits.size(); ++ii) {

#ifdef EDM_ML_DEBUG
      std::cout << "Hit[" << ii << "] " << std::hex << hcalHits[ii].id() << std::dec << '\n';
#endif
      double energy = (hcalHits[ii].energy());
      if (neutralDensity_) {
	energy *= (energyWt(hcalHits[ii].id()));
	hcalHits[ii].setEnergy(energy);
      }
      DetId newid = relabel(hcalHits[ii].id());
#ifdef EDM_ML_DEBUG
      std::cout << "Hit " << ii << " out of " << hcalHits.size() << " " 
		<< std::hex << newid.rawId() << std::dec << " E " << energy 
                << std::endl;
#endif
      hcalHits[ii].setID(newid.rawId());
#ifdef EDM_ML_DEBUG
      std::cout << "Modified Hit " << hcalHits[ii] << std::endl;
#endif
    }
  } else {
    edm::LogWarning("HcalSim") << "HcalHitRelabeller: no valid HcalDDDRecConstants";
  }
  
}


void HcalHitRelabeller::setGeometry(const CaloGeometry*& geom, 
				    const HcalDDDRecConstants *& recNum) {
  theGeometry  = geom;
  theRecNumber = recNum;
}

DetId HcalHitRelabeller::relabel(const uint32_t testId) const {

#ifdef EDM_ML_DEBUG
  std::cout << "Enter HcalHitRelabeller::relabel " << std::endl;
#endif
  HcalDetId hid;
  int       det, z, depth, eta, phi, layer, sign;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
#ifdef EDM_ML_DEBUG
  std::cout << "det: " << det << " "
  	    << "z: " << z << " "
   	    << "depth: " << depth << " "
   	    << "ieta: " << eta << " "
   	    << "iphi: " << phi << " "
   	    << "layer: " << layer << std::endl;
#endif
  HcalDDDRecConstants::HcalID id = theRecNumber->getHCID(det,eta,phi,layer,depth);
  sign=(z==0)?(-1):(1);

  if (id.subdet==int(HcalBarrel)) {
    hid=HcalDetId(HcalBarrel,sign*id.eta,id.phi,id.depth);        
  } else if (id.subdet==int(HcalEndcap)) {
    hid=HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalOuter)) {
    hid=HcalDetId(HcalOuter,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalForward)) {
    hid=HcalDetId(HcalForward,sign*id.eta,id.phi,id.depth);
  }
#ifdef EDM_ML_DEBUG
  std::cout << " new HcalDetId -> hex.RawID = "
	    << std::hex << hid.rawId() << std::dec;
  std::cout.flush();
  std::cout << " det, z, depth, eta, phi = " << det << " "
	    << z << " "<< id.depth << " " << id.eta << " "
	    << id.phi << " ---> " << hid << std::endl;  
#endif
  return hid;
}

double HcalHitRelabeller::energyWt(const uint32_t testId) const {

  HcalDetId hid;
  int       det, z, depth, eta, phi, layer;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
  int       zside = (z==0) ? (-1) : (1);
  double    wt    = (((det==1) || (det==2)) && (depth == 0)) ? 
    theRecNumber->getLayer0Wt(det,phi,zside) : 1.0;
#ifdef EDM_ML_DEBUG
  std::cout << "EnergyWT::det: " << det << " z: " << z  << ":" << zside
            << " depth: " << depth << " ieta: " << eta << " iphi: " << phi
            << " layer: " << layer << " wt " << wt << std::endl;
#endif
  return wt;
}
