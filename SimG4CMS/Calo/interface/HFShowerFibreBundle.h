#ifndef SimG4CMS_HFShowerFibreBundle_h
#define SimG4CMS_HFShowerFibreBundle_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerFibreBundle.h
// Description: Get energy deposits for HFShower PMT's
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"

class DDCompactView;    
class G4Step;

#include <string>
#include <vector>
 
class HFShowerFibreBundle {

public:    

  HFShowerFibreBundle(std::string & name, const DDCompactView & cpv, 
		      edm::ParameterSet const & p);
  virtual ~HFShowerFibreBundle();
  double                getHits(G4Step * aStep, bool type);
  double                getRadius();
  void                  initRun(G4ParticleTable *, HcalDDDSimConstants*);

private:    

  std::vector<double>   getDDDArray(const std::string&, const DDsvalues_type&);

private:    

  HFCherenkov           *cherenkov1, *cherenkov2;
  double                facTube, facCone; //Packing factors
  int                   indexR, indexF;
  std::vector<double>   rTable;          // R-table
  std::vector<int>      pmtR1, pmtFib1;  // R-index, fibre table for right box
  std::vector<int>      pmtR2, pmtFib2;  // R-index, fibre table for left box
};

#endif // HFShowerFibreBundle_h
