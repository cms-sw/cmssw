#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4GeomTestVolume.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4GDMLParser.hh"

#include <string>

G4CheckOverlap::G4CheckOverlap(const edm::ParameterSet &p) {

  std::vector<std::string> nodeNames 
    = p.getParameter<std::vector<std::string> >("NodeNames");
  double tolerance 
    = p.getUntrackedParameter<double>("Tolerance", 0.0)*CLHEP::mm;
  int nPoints = p.getUntrackedParameter<int>("Resolution", 10000);
  bool verbose = p.getUntrackedParameter<bool>("Verbose", true);
  bool regionFlag = p.getUntrackedParameter<bool>("RegionFlag", true);
  bool gdmlFlag = p.getUntrackedParameter<bool>("gdmlFlag", false);
  int nPrints = p.getUntrackedParameter<int>("ErrorThreshold", 1);
  int level = p.getUntrackedParameter<int>("Level", 0);
  int depth = p.getUntrackedParameter<int>("Depth", -1);

  const G4RegionStore* regStore =  G4RegionStore::GetInstance();

  G4LogicalVolume* lv;
  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();

  unsigned int nn = nodeNames.size();
  G4cout << "G4OverlapCheck is initialised with " 
	 << nodeNames.size() << " nodes; " << " nPoints= " << nPoints
	 << "; tolerance= " << tolerance/mm << " mm; verbose: " 
	 << verbose << " RegionFlag: " << regionFlag << G4endl;

  if(0 < nn) { 
    for (unsigned int ii=0; ii<nn; ++ii) {
      if("" == nodeNames[ii] || "world" == nodeNames[ii] || "World" == nodeNames[ii] ) { 
	nodeNames[ii] = "DDDWorld"; 
	G4cout << "### Check overlaps for DDDWorld " << G4endl; 
	G4VPhysicalVolume* pv = pvs->GetVolume("DDDWorld");
	G4GeomTestVolume test(pv, tolerance, nPoints, verbose);
	test.SetErrorsThreshold(nPrints);
	test.TestRecursiveOverlap(level, depth);
      } else if(regionFlag) {
        G4cout << "---------------------------------------------------------------" << G4endl;
	G4cout << "### Check overlaps for G4Region Node[" << ii << "] : " << nodeNames[ii] 
	       << G4endl; 
        G4Region* reg = regStore->GetRegion((G4String)nodeNames[ii]);
        if(!reg) {
	  G4cout << "### NO G4Region found - EXIT" << G4endl;
          return; 
	}
	std::vector<G4LogicalVolume*>::iterator rootLVItr
	  = reg->GetRootLogicalVolumeIterator();
	unsigned int numRootLV = reg->GetNumberOfRootVolumes();

	for(unsigned int iLV=0; iLV < numRootLV; ++iLV, ++rootLVItr ) {
	  // Cover each root logical volume in this region
	  lv = *rootLVItr;
	  G4cout << "### Check overlaps for G4LogicalVolume " << lv->GetName() << G4endl; 
          for(unsigned int i=0; i<numPV; ++i) {
            if(((*pvs)[i])->GetLogicalVolume() == lv) {
	      G4cout << "### Check overlaps for PhysVolume  " <<  ((*pvs)[i])->GetName()
		     << G4endl;
	      // gdml dump only for 1 volume
              if(gdmlFlag) {
		G4GDMLParser gdml;
		gdml.Write(((*pvs)[i])->GetName()+".gdml", (*pvs)[i], true);
                gdmlFlag = false;
	      }

	      G4GeomTestVolume test(((*pvs)[i]), tolerance, nPoints, verbose);
	      test.SetErrorsThreshold(nPrints);
	      test.TestRecursiveOverlap(level, depth);
	    }
	  }
	}

      } else {
	G4cout << "### Check overlaps for PhysVolume Node[" << ii << "] : " << nodeNames[ii] 
	       << G4endl; 
	G4VPhysicalVolume* pv = pvs->GetVolume((G4String)nodeNames[ii]);
	G4GeomTestVolume test(pv, tolerance, nPoints, verbose);
	test.SetErrorsThreshold(nPrints);
	test.TestRecursiveOverlap(level, depth);
      }
    }
  }
}

G4CheckOverlap::~G4CheckOverlap() {}
  
