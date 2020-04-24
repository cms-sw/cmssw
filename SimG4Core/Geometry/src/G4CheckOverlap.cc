#include "SimG4Core/Geometry/interface/G4CheckOverlap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4LogicalVolumeStore.hh"
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
  std::string PVname = p.getParameter<std::string>("PVname");
  std::string LVname = p.getParameter<std::string>("LVname");
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
  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();
  unsigned int numLV = lvs->size();
  unsigned int nn = nodeNames.size();

  std::vector<G4String> savedgdml;

  G4cout << "G4OverlapCheck is initialised with " 
	 << nodeNames.size() << " nodes; " << " nPoints= " << nPoints
	 << "; tolerance= " << tolerance/mm << " mm; verbose: " 
	 << verbose << "\n"
	 << "               RegionFlag: " << regionFlag
	 << "  PVname: " << PVname << "  LVname: " << LVname << "\n"
	 << "               Nlv= " << numLV << "   Npv= " << numPV << G4endl;

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
	G4cout << "      " << numRootLV << " Root Logical Volumes in this region" << G4endl; 

	for(unsigned int iLV=0; iLV < numRootLV; ++iLV, ++rootLVItr ) {
	  // Cover each root logical volume in this region
	  lv = *rootLVItr;
	  G4cout << "### Check overlaps for G4LogicalVolume " << lv->GetName() << G4endl; 
          for(unsigned int i=0; i<numPV; ++i) {
            if(((*pvs)[i])->GetLogicalVolume() == lv) {
              G4String pvname = ((*pvs)[i])->GetName();
              G4bool isNew = true;
              for(unsigned int k=0; k<savedgdml.size(); ++k) {
		if(pvname == savedgdml[k]) {
		  isNew = false;
		  break;
		}
	      }
              if(!isNew) { 
		G4cout << "### Check overlaps for PhysVolume  " << pvname
		       << " is skipted because was already done" << G4endl;
		continue; 
	      }
              savedgdml.push_back(pvname);
	      G4cout << "### Check overlaps for PhysVolume  " << pvname
		     << G4endl;
	      // gdml dump only for 1 volume
              if(gdmlFlag) {
		G4GDMLParser gdml;
		gdml.Write(pvname+".gdml", (*pvs)[i], true);
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
  if("" != PVname) {
    G4cout << "----------- List of PhysVolumes by name -----------------" << G4endl;
    for (unsigned int i=0; i<numPV; ++i) {
      if(PVname == ((*pvs)[i])->GetName()) {
	G4cout << " ##### PhysVolume " << PVname << " [" << ((*pvs)[i])->GetCopyNo() 
	       << "]  LV: " <<  ((*pvs)[i])->GetLogicalVolume()->GetName() 
	       << " Mother LV: " <<  ((*pvs)[i])->GetMotherLogical()->GetName()
	       << " Region: " << ((*pvs)[i])->GetLogicalVolume()->GetRegion()->GetName() 
	       << G4endl;
	G4cout << "       Translation: " << ((*pvs)[i])->GetObjectTranslation() << G4endl;
	G4cout << "       Rotation:    " << ((*pvs)[i])->GetObjectRotationValue() << G4endl;
	if(gdmlFlag) {
	  G4GDMLParser gdml;
	  gdml.Write(PVname+".gdml", (*pvs)[i], true);
	}
      }
    }
  }
  if("" != LVname) {
    G4cout << "---------- List of Logical Volumes by name ------------------" << G4endl;
    for (unsigned int i=0; i<numLV; ++i) {
      if(LVname == ((*lvs)[i])->GetName()) {
	G4int np = ((*lvs)[i])->GetNoDaughters();
	G4cout << " ##### LogVolume " << LVname << "  " << np << " daughters" 
	       << " Region: " << ((*lvs)[i])->GetRegion()->GetName() 
	       << G4endl;
	G4cout << *(((*lvs)[i])->GetSolid()) << G4endl;
	for (G4int j=0; j<np; ++j) {
	  G4VPhysicalVolume* pv = ((*lvs)[i])->GetDaughter(j);
	  if(pv) {
	    G4cout << "   PV: " << pv->GetName() << " [" << pv->GetCopyNo() << "]"
		   << " type: " << pv->VolumeType() << "  multiplicity: " 
		   << pv->GetMultiplicity() << " LV: " 
		   << pv->GetLogicalVolume()->GetName() << G4endl;
	    G4cout << "       Translation: " << pv->GetObjectTranslation() << G4endl;
	    G4cout << "       Rotation:    " << pv->GetObjectRotationValue() << G4endl;
            G4cout << *(pv->GetLogicalVolume()->GetSolid()) << G4endl;
	  }
	}
      }
    }
  }
  G4cout << "---------------- End of overlap checks ---------------------" << G4endl;
}

G4CheckOverlap::~G4CheckOverlap() {}
  
