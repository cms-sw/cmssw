#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4PhysicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4GeomTestVolume.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include <string>

G4CheckOverlap::G4CheckOverlap(const edm::ParameterSet &p) {

  std::vector<std::string> nodeNames 
    = p.getParameter<std::vector<std::string> >("NodeNames");
  double tolerance 
    = p.getUntrackedParameter<double>("Tolerance", 0.0)*CLHEP::mm;
  int nPoints = p.getUntrackedParameter<int>("Resolution", 10000);
  bool verbose = p.getUntrackedParameter<bool>("Verbose", true);
  int nPrints = p.getUntrackedParameter<int>("ErrorThreshold", 1);
  int level = p.getUntrackedParameter<int>("Level", 0);
  int depth = p.getUntrackedParameter<int>("Depth", -1);

  unsigned int nn = nodeNames.size();
  G4cout << "Initialised with " 
	 << nodeNames.size() << " nodes; " << " nPoints= " << nPoints
	 << "; tolerance= " << tolerance/mm << " mm; verbose: " 
	 << verbose << G4endl;
  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  if(0 < nn) { 
    for (unsigned int ii=0; ii<nn; ++ii) {
      if("" == nodeNames[ii] || "world" == nodeNames[ii] 
	 || "World" == nodeNames[ii] ) { 
	nodeNames[ii] = "DDDWorld"; 
      }
      G4cout << "Check overlaps for Node[" << ii << "] : " << nodeNames[ii] << G4endl; 
      G4VPhysicalVolume* pv = pvs->GetVolume((G4String)nodeNames[ii]);
      G4GeomTestVolume test(pv, tolerance, nPoints, verbose);
      test.SetErrorsThreshold(nPrints);
      test.TestRecursiveOverlap(level, depth);
    }
  }
}

G4CheckOverlap::~G4CheckOverlap() {}
  
