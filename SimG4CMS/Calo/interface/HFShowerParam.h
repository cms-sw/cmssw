#ifndef SimG4CMS_HFShowerParam_h
#define SimG4CMS_HFShowerParam_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerParam.h
// Description: Generates hits for HF with some parametrized information
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "G4ThreeVector.hh"

class DDCompactView;
class G4Step;

#include <string>
#include <vector>
 
class HFShowerParam {

public:    

  HFShowerParam(std::string & name, const DDCompactView & cpv, 
		edm::ParameterSet const & p);
  virtual ~HFShowerParam();

public:    

  struct Hit {
    Hit() {}
    G4ThreeVector       position;
    int                 depth;
    double              time;
    double              edep;
  };

  std::vector<Hit>      getHits(G4Step * aStep);
  G4ThreeVector         getPosHit(int i);
  int                   getDepth(int i);
  double                getTSlice(int i);
  
private:    

  std::vector<double>   getDDDArray(const std::string&, const DDsvalues_type&);

private:    

  HFFibre*              fibre;
  double                pePerGeV;
  bool                  trackEM;
  std::vector<double>   gpar;
};

#endif // HFShowerParam_h
