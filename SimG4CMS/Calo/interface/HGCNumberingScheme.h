#ifndef SimG4CMS_HGCNumberingScheme_h
#define SimG4CMS_HGCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include <vector>

class HGCNumberingScheme : public CaloNumberingScheme {

public:

  enum HGCNumberingParameters { HGCCellSize };

  HGCNumberingScheme(std::vector<double> gpar);

  virtual ~HGCNumberingScheme();

  /**
     @short assigns the det id to a hit
   */
  virtual uint32_t getUnitID(ForwardSubdetector &subdet, int &layer, int &module, int &iz, G4ThreeVector &pos, float &dz, float &bl1, float &tl1, float &h1);

  /**
     @short maps a hit position to a sequential cell in a trapezoid surface defined by h,b,t
   */
  int assignCell(float x, float y, float cellSize, float h, float bl, float tl);

  /**
     @short inverts the cell number in a trapezoid surface to local coordinates
   */
  std::pair<float,float> getLocalCoords(int cell, float cellSize, float h, float bl, float tl);

  /**
     @short returns the cell size
   */
  inline float getCellSize() { return gpar.size()>0 ? gpar[HGCCellSize] : 0; }

private:
  
  HGCNumberingScheme();

  //a vector of parameters read from the xml
  std::vector<double>    gpar;
};

#endif
