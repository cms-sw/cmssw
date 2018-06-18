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
  ZdcShowerLibrary(const std::string & name, const DDCompactView & cpv, edm::ParameterSet const & p);
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

  std::vector<Hit>&           getHits(const G4Step * aStep, bool & ok);
  int                         getEnergyFromLibrary(const G4ThreeVector& posHit, const G4ThreeVector& momDir, double energy,
                                                   G4int parCode,HcalZDCDetId::Section section, bool side, int channel);
  int                         photonFluctuation(double eav, double esig,double edis);

private:

  bool                        verbose;

  int                         npe;
  std::vector<ZdcShowerLibrary::Hit> hits;
  
};
#endif
