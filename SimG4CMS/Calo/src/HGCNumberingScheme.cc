///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/Math/interface/FastMath.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

#define DebugLog

HGCNumberingScheme::HGCNumberingScheme(std::vector<double> gp) :
  CaloNumberingScheme(0), gpar(gp) {
  edm::LogInfo("HGCSim") << "Creating HGCNumberingScheme";
}

HGCNumberingScheme::~HGCNumberingScheme() {
  edm::LogInfo("HGCSim") << "Deleting HGCNumberingScheme";
}

//
uint32_t HGCNumberingScheme::getUnitID(ForwardSubdetector &subdet, int &layer, int &module, int &iz, G4ThreeVector &pos, float &dz, float &bl1, float &tl1, float &h1)
{

  //an ugly way of doing this
  if(subdet == ForwardSubdetector::HGCEE)
    {
      if(layer>=int(gpar[1]) && layer<=int(gpar[2]))      layer = layer-gpar[1];
      else if(layer>=int(gpar[3]) && layer<=int(gpar[4])) layer = layer-gpar[3]+(gpar[2]-gpar[1]+1);
      else if(layer>=int(gpar[5]) && layer<=int(gpar[6])) layer = layer-gpar[5]+(gpar[4]-gpar[3]+1)+(gpar[2]-gpar[1]+1);
      else	                                          edm::LogError("HGCSim") << "[HGCNumberingScheme] can't find EE bounds for layer #" << layer;
    }
  else
    {
    }

  
  std::pair<float,float> phi_r=fastmath::atan2r(pos.x(),pos.y());
  int phiSector( phi_r.first>0 );

  int icell(0);
  float cellSize(gpar[0]);
  if(h1!=0  && tl1!=bl1){
 
    float a=2*h1/(tl1-bl1);
    float b=-(tl1+bl1)/(tl1-bl1)*h1;   
    if(phi_r.first<0)  { a*=-1; }

    int M = (int) 2*h1/cellSize;
    int iM = (int) (fabs(pos.y())-bl1)/cellSize;
    
    float xM=fabs((pos.y()-b)/a);
    int N=(int) xM/cellSize;
    int iN=(int) fabs(pos.x())/cellSize;

    if((iN>N || iM>M) && subdet == ForwardSubdetector::HGCEE)
      {
	edm::LogError("HGCSim") << "[HGCNumberingScheme] hit seems to be out of bounds cell=(" << iN << "," << iM << ")  while max is (" << N << "," << M << ")\n"
				<< "\tLocal position: (" << pos.x() << "," << pos.y() << "," << pos.z() << ")\n"
				<< "\tTrapezoid bl=" << bl1 << " tl=" << tl1 << " h=" << h1 << " @ layer " << layer;
	iN=0;	iM=0;
      }
    
    for(int iyM=0; iyM<iM; iyM++)
      icell += (int)(cellSize*iyM-b)/(a*cellSize);
    icell += iN;

    if(icell>0xffff && subdet == ForwardSubdetector::HGCEE)
      {
	edm::LogError("HGCSim") << "[HGCNumberingScheme] cell id seems to be out of bounds cell id=" << icell << "\n"
				<< "\tLocal position: (" << pos.x() << "," << pos.y() << "," << pos.z() << ")\n"
				<< "\tTrapezoid bl=" << bl1 << " tl=" << tl1 << " h=" << h1 << " @ layer " << layer;
      }    
  }
  else{
    edm::LogError("HGCSim") << "[HGCNumberingScheme] failed to determine trapezoid bounds...assigning 0 to cell number\n";
  }
  
  uint32_t index = (subdet == ForwardSubdetector::HGCEE ? 
		    HGCEEDetId(subdet,iz,layer,module,phiSector,icell).rawId() : 
		    HGCHEDetId(subdet,iz,layer,module,phiSector,icell).rawId() );


// #ifdef DebugLog
//   edm::LogInfo("HGCSim") << "HGCNumberingScheme det = " << subdet 
// 			 << " module = " << module << " layer = " << layer
// 			 << " zside = " << iz << " Cell = " << cellx << ":" 
// 			 << celly << " packed index = 0x" << std::hex << index 
// 			 << std::dec;
// #endif

  return index;
}
