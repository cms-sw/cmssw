#include "SimCalorimetry/HcalSimProducers/interface/HcalHitRelabeller.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

//#define DEBUG

HcalHitRelabeller::HcalHitRelabeller(const edm::ParameterSet& ps) {
  // try to make sure the memory gets pinned in place
  m_segmentation.resize(29);
  m_CorrectPhi = ps.getUntrackedParameter<bool>("CorrectPhi",false);
  for (int i=0; i<29; i++) {
    char name[10];
    snprintf(name,10,"Eta%d",i+1);
    if (i>0) {
      m_segmentation[i]=ps.getUntrackedParameter<std::vector<int> >(name,m_segmentation[i-1]);
    } else {
      m_segmentation[i]=ps.getUntrackedParameter<std::vector<int> >(name);
    }
  }
#ifdef DEBUG
  for (int i=0; i<29; i++) {
    std::cout << "Segmentation[" << i << "] with " << m_segmentation[i].size() << " elements:";
    for (unsigned int k=0; k<m_segmentation[i].size(); ++k)
      std::cout << " " << m_segmentation[i][k];
    std::cout << std::endl;
  }
  std::cout << "correctPhi " << m_CorrectPhi << std::endl;
#endif
}

void HcalHitRelabeller::process(std::vector<PCaloHit>& hcalHits) {

  for (unsigned int ii=0; ii<hcalHits.size(); ++ii) {

#ifdef DEBUG
    std::cout << "Hit[" << ii << "] " << std::hex << hcalHits[ii].id() << std::dec << '\n';
#endif
    DetId newid = relabel(hcalHits[ii].id());
#ifdef DEBUG
    std::cout << "Hit " << ii << " out of " << hcalHits.size() << " " << std::hex << newid.rawId() << std::dec << '\n';
    HcalDetId newcell(newid);
    const CaloCellGeometry *cellGeometry =
      theGeometry->getSubdetectorGeometry(newcell)->getGeometry(newcell);
    GlobalPoint globalposition = (GlobalPoint)(cellGeometry->getPosition());
    
    std::cout << "PCaloHit " << newcell << " position: " << globalposition << std::endl;
    std::cout.flush();
#endif
    hcalHits[ii].setID(newid.rawId());
#ifdef DEBUG
    std::cout << "Modified Hit " << hcalHits[ii] << std::endl;
#endif
  }
  //End Change by Wetzel
  
}


void HcalHitRelabeller::setGeometry(const CaloGeometry*& geom) {
  theGeometry = geom;
}

DetId HcalHitRelabeller::relabel(const uint32_t testId) const {

#ifdef DEBUG
  std::cout << "Enter HcalHitRelabeller::relabel " << std::endl;
#endif
  HcalDetId hid;
  int det, z, depth, eta, phi, layer, sign;
  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);

  layer-=1; // one is added in the simulation, here used for indexing  

  sign=(z==0)?(-1):(1);
#ifdef DEBUG
  std::cout << "det: " << det << " "
  	    << "z: " << z << " "
   	    << "depth: " << depth << " "
   	    << "ieta: " << eta << " "
   	    << "iphi: " << phi << " "
   	    << "layer: " << layer << " ";
  std::cout.flush();
#endif
  int newDepth = 0; // moved out of if's just for printing purposes...
  int phi_skip = phi;
  if (m_CorrectPhi) {
    if      (eta >= 40) phi_skip  = (phi-1)*4 - 1;
    else if (eta >  20) phi_skip  = (phi-1)*2 + 1;
    if (phi_skip < 0)   phi_skip += 72;
  }

  if (det==int(HcalBarrel)) {
    newDepth=m_segmentation[eta-1][layer];
    if(eta==16 && newDepth > 2) newDepth=2;// tower 16 HACK to be watched out..
    hid=HcalDetId(HcalBarrel,eta*sign,phi_skip,newDepth);        
  }
  if (det==int(HcalEndcap)) {
    newDepth=m_segmentation[eta-1][layer];
    if (eta==16 && newDepth<3) newDepth=3; // tower 16 HACK to be watched out..
    hid=HcalDetId(HcalEndcap,eta*sign,phi_skip,newDepth);    
  }
  if (det==int(HcalOuter)) {
    hid=HcalDetId(HcalOuter,eta*sign,phi_skip,4);    
    newDepth = 4;
  }
  if (det==int(HcalForward)) {
    hid=HcalDetId(HcalForward,eta*sign,phi_skip,depth);
    newDepth = depth; 
  }
#ifdef DEBUG
  std::cout << " new HcalDetId -> hex.RawID = "
	    << std::hex << hid.rawId() << std::dec;
  std::cout.flush();
  std::cout << " det, z, depth, eta, phi = "
	    << det << " "
	    << z << " "
	    << newDepth << " "
	    << eta << " "
	    << phi << " " << phi_skip << " "
	    <<  " ---> " << hid << std::endl;  
#endif
  return hid;
}
