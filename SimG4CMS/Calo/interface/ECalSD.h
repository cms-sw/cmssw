///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.h
// Description: Stores hits of Electromagnetic calorimeters in appropriate
//              container
// Use in your sensitive detector builder:
//    ECalSD* ecalSD = new ECalSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////
#ifndef ECalSD_h
#define ECalSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"
#include <map>


class ECalSD : public CaloSD {

public:    

  ECalSD(G4String, const DDCompactView &, edm::ParameterSet const &);
  virtual ~ECalSD();
  virtual double getEnergyDeposit(G4Step*);

private:    

  void   initMap(G4String);
  double curve_LY(G4String& , G4StepPoint* ); 
  double crystalLength(G4String);

  bool                 useBirk;
  double               birk1, birk2;
  map<G4String,double> lengthMap;

};

#endif // ECalSD_h
