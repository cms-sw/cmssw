///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.h
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalEndcapNumberingScheme_h
#define EcalEndcapNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"

class EcalEndcapNumberingScheme : public CaloNumberingScheme {

public:
  EcalEndcapNumberingScheme();
  ~EcalEndcapNumberingScheme();
	 
  virtual unsigned int getUnitID(const G4Step* aStep) const ;

private:
  //  static UserVerbosity cout;

};

#endif
