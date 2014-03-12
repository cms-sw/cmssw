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
  edm::LogInfo("HGCSim") << "Creating HGCNumberingScheme with cellSize=" << gpar[HGCCellSize];
}

HGCNumberingScheme::~HGCNumberingScheme() {
  edm::LogInfo("HGCSim") << "Deleting HGCNumberingScheme";
}

//
uint32_t HGCNumberingScheme::getUnitID(ForwardSubdetector &subdet, int &layer, int &sector, int &iz, G4ThreeVector &pos, float &dz, float &bl1, float &tl1, float &h1)
{
  
  //check which phi sub-sector this hit belongs to
  std::pair<float,float> phi_r=fastmath::atan2r(pos.x(),pos.y());
  int phiSector( phi_r.first>0 );  

  //assign a cell number in the positive quadrant
  int icell=assignCell(fabs(pos.x()), pos.y(), gpar[HGCCellSize], h1, bl1, tl1);
  
  //check if it fits
  if(icell>0xffff)
    {
      edm::LogError("HGCSim") << "[HGCNumberingScheme] cell id seems to be out of bounds cell id=" << icell << "\n"
			      << "\tLocal position: (" << pos.x() << "," << pos.y() << "," << pos.z() << ")\n"
			      << "\tTrapezoid bl=" << bl1 << " tl=" << tl1 << " h=" << h1 << " @ layer " << layer;
    }    
  
  //build the index
  uint32_t index = (subdet == ForwardSubdetector::HGCEE ? 
		    HGCEEDetId(subdet,iz,layer,sector,phiSector,icell).rawId() : 
		    HGCHEDetId(subdet,iz,layer,sector,phiSector,icell).rawId() );

  return index;
}

//
int HGCNumberingScheme::assignCell(float x, float y, float cellSize, float h, float bl, float tl)
{
  //linear parameterization of the trapezoid
  float a=2*h/(tl-bl);
  float b=-h*(tl+bl)/(tl-bl);
  
  //this is the cell # in the row and column
  int kx=floor( fabs(x)/cellSize );
  int ky=floor((y+h)/cellSize);
  
  //find the cell sequentially in the trapezoid
  //notice the arithmetic sum can't be used as \sum floor(x) != floor( \sum x )
  int icell(0);
  for(int iky=0; iky<ky; iky++)
    icell += floor( (iky*cellSize-h-b)/(a*cellSize) );
  icell += kx;

  //all done here
  return icell;
}

//
std::pair<float,float> HGCNumberingScheme::getLocalCoords(int cell, float cellSize, float h, float bl, float tl)
{
  //linear parameterization of the trapezoid
  float a=2*h/(tl-bl);
  float b=-h*(tl+bl)/(tl-bl);
  
  //find the y-row iteratively
  int maxKy=floor(2*h/cellSize);
  int ky(0),testCell(0);
  for(int iky=0; iky<maxKy; iky++)
    {
      int deltay( floor( (iky*cellSize-h-b)/(a*cellSize) ) );
      if(testCell+deltay > cell) break;
      testCell+=deltay;
      ky++;
    }
  
  //find the x-column
  int kx=cell-testCell;

  //all done here (return centered at cell)
  return std::pair<float,float>((kx+0.5)*cellSize,(ky+0.5)*cellSize-h);
}
