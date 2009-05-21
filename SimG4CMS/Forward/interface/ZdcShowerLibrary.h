#ifndef SimG4CMS_ZdcShowerLibrary_h
#define SimG4CMS_ZdcShowerLibrary_h 1
///////////////////////////////////////////////////////////////////////////////
// File: ZdcShowerLibrary.h
// Description: Gets information from a shower library
// E. Garcia June 2008
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/ForwardGeometry/src/ZdcHardcodeGeometryData.h"

#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
 
#include <string>
#include <memory>

class G4Step;
class DDCompactView;    
class ZdcShowerLibrary {

public:
  
  //Constructor and Destructor
  ZdcShowerLibrary(std::string & name, const DDCompactView & cpv, edm::ParameterSet const & p);
  ~ZdcShowerLibrary();
  
 public:
  
  struct Hit {
    Hit() {}
    G4ThreeVector             entryLocal;
    G4ThreeVector             position;
    int                       depth;
    double                    time;
    int                       detID;
    double                    DeHad;
    double                    DeEM;
  };


  void                        initRun(G4ParticleTable * theParticleTable);
  std::vector<Hit>&           getHits(G4Step * aStep, bool & ok);
  int                         getEnergyFromLibrary(G4ThreeVector posHit, G4ThreeVector momDir, double energy,
                                                   G4int parCode,HcalZDCDetId::Section section, bool side, int channel);
  int                         photonFluctuation(double eav, double esig,double edis);
  int                         encodePartID(G4int parCode);
  
 protected:

private:

  bool                        verbose;
  G4int                         emPDG, epPDG, gammaPDG;
  G4int                         pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  G4int                         anuePDG, anumuPDG, anutauPDG, geantinoPDG;

  int                         npe;
  std::vector<ZdcShowerLibrary::Hit> hits;
  
};
#endif
