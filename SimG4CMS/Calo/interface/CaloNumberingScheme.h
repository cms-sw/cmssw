///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for Calorimeters
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloNumberingScheme_h
#define CaloNumberingScheme_h

#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include <map>
//#include "Utilities/UI/interface/Verbosity.h"

class CaloNumberingScheme {

public:
  CaloNumberingScheme(){};
  virtual ~CaloNumberingScheme(){};
  virtual unsigned int getUnitID(const  G4Step* aStep) const =0;
	 
  // Utilities to get detector levels during a step
  virtual int  detectorLevel(const G4Step*) const;
  virtual void detectorLevel(const G4Step*, int&, int*, G4String*) const;
    
public:
      
/*   // additional tool: */
/*   int getUnitWithMaxEnergy(map<int,float,less<int> >& themap); */

/*   float  energyInMatrix(int nCellInEta, int nCellInPhi,  */
/* 			int crystalWithMaxEnergy,  */
/* 			map<int,float,less<int> >& themap);  */

protected:
  //  CaloNumberingPacker myPacker;

private:
  //  static UserVerbosity cout;

};

#endif
