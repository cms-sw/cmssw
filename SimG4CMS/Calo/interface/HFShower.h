///////////////////////////////////////////////////////////////////////////////
// File: HFShower.h
// Description: Generates hits for HF with Cerenkov photon code
///////////////////////////////////////////////////////////////////////////////
#ifndef HFShower_h
#define HFShower_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "G4String.hh"

class DDCompactView;    
class G4Step;

#include <map>
#include <vector>
 
class HFShower {

public:    

  HFShower(const DDCompactView & cpv, edm::ParameterSet const & p);
  virtual ~HFShower();
  int                        getHits(G4Step * aStep);
  double                     getTSlice(int i);

private:    

  double                     fibreLength(G4String);
  void                       clearHits();

  HFCherenkov*               cherenkov;
  HFFibre*                   fibre;

  int                        verbosity;
  double                     cFibre;
  double                     probMax;
  std::map<G4String,double>  fibreDz2;

  int                        nHit;
  std::vector<double>        wlHit;
  std::vector<double>        timHit;

};

#endif // HFShower_h
