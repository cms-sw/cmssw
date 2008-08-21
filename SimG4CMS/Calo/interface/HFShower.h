#ifndef SimG4CMS_HFShower_h
#define SimG4CMS_HFShower_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShower.h
// Description: Generates hits for HF with Cerenkov photon code
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "G4String.hh"

class DDCompactView;    
class G4Step;

#include <map>
#include <vector>
 
class HFShower {

public:    

  HFShower(std::string & name, const DDCompactView & cpv, 
	   edm::ParameterSet const & p);
  virtual ~HFShower();

public:

  struct Hit {
    Hit() {}
    double                   time;
    double                   wavelength; 
  };

  std::vector<Hit>           getHits(G4Step * aStep);

private:    

  double                     fibreLength(G4String);

private:    

  HFCherenkov*               cherenkov;
  HFFibre*                   fibre;

  double                     probMax;
  std::map<G4String,double>  fibreDz2;

};

#endif // HFShower_h
