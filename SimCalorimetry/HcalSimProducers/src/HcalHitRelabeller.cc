#include "SimCalorimetry/HcalSimProducers/interface/HcalHitRelabeller.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

HcalHitRelabeller::HcalHitRelabeller(const edm::ParameterSet&) : theGeometry(0), theRecNumber(0) { }

void HcalHitRelabeller::process(std::vector<PCaloHit>& hcalHits) {

  if (theRecNumber) {
    for (unsigned int ii=0; ii<hcalHits.size(); ++ii) {

#ifdef DebugLog
      std::cout << "Hit[" << ii << "] " << std::hex << hcalHits[ii].id() << std::dec << '\n';
#endif
      DetId newid = relabel(hcalHits[ii].id());
#ifdef DebugLog
      std::cout << "Hit " << ii << " out of " << hcalHits.size() << " " << std::hex << newid.rawId() << std::dec << '\n';
//      HcalDetId newcell(newid);
//      if (theGeometry) {
//	const CaloCellGeometry *cellGeometry =
//	  theGeometry->getSubdetectorGeometry(newcell)->getGeometry(newcell);
//	GlobalPoint globalposition =(GlobalPoint)(cellGeometry->getPosition());
//	std::cout << "PCaloHit " << newcell << " position: " << globalposition 
//		  << std::endl;
//      }
//      std::cout.flush();
#endif
      hcalHits[ii].setID(newid.rawId());
#ifdef DebugLog
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

#ifdef DebugLog
  std::cout << "Enter HcalHitRelabeller::relabel " << std::endl;
#endif
  HcalDetId hid;
  int       det, z, depth, eta, phi, layer, sign;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
  HcalDDDRecConstants::HcalID id = theRecNumber->getHCID(det,eta,phi,layer,depth);
  sign=(z==0)?(-1):(1);
#ifdef DebugLog
  std::cout << "det: " << det << " "
  	    << "z: " << z << " "
   	    << "depth: " << depth << " "
   	    << "ieta: " << eta << " "
   	    << "iphi: " << phi << " "
   	    << "layer: " << layer << " ";
  std::cout.flush();
#endif

  if (id.subdet==int(HcalBarrel)) {
    hid=HcalDetId(HcalBarrel,sign*id.eta,id.phi,id.depth);        
  } else if (id.subdet==int(HcalEndcap)) {
    hid=HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalOuter)) {
    hid=HcalDetId(HcalOuter,sign*id.eta,id.phi,id.depth);    
  } else if (id.subdet==int(HcalForward)) {
    hid=HcalDetId(HcalForward,sign*id.eta,id.phi,id.depth);
  }
#ifdef DebugLog
  std::cout << " new HcalDetId -> hex.RawID = "
	    << std::hex << hid.rawId() << std::dec;
  std::cout.flush();
  std::cout << " det, z, depth, eta, phi = " << det << " "
	    << z << " "<< id.depth << " " << id.eta << " "
	    << id.phi << " ---> " << hid << std::endl;  
#endif
  return hid;
}
