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
  int                        getHits(G4Step * aStep);
  double                     getTSlice(int i);

private:    

  double                     fibreLength(G4String);
  void                       clearHits();

private:    

  HFCherenkov*               cherenkov;
  HFFibre*                   fibre;

  double                     probMax;
  std::map<G4String,double>  fibreDz2;

  int                        nHit;
  std::vector<double>        wlHit;
  std::vector<double>        timHit;

};

#endif // HFShower_h
