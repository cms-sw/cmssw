#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimG4CMS/Calo/interface/HcalTestAnalysis.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>

#define ddebug
#define debugFill
#define debugqie
#define debugLayer
#define debugStep

HcalTestAnalysis::HcalTestAnalysis(const edm::ParameterSet &p): 
  myqie(0), addTower(3), tuples(0), numberingFromDDD(0), org(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("HcalTestAnalysis");
  verbosity    = m_Anal.getParameter<int>("Verbosity");
  eta0         = m_Anal.getParameter<double>("Eta0");
  phi0         = m_Anal.getParameter<double>("Phi0");
  int laygroup = m_Anal.getParameter<int>("LayerGrouping");
  centralTower = m_Anal.getParameter<int>("CentralTower");
  names        = m_Anal.getParameter<std::vector<std::string> >("Names");
  std::string file = m_Anal.getParameter<std::string>("FileName");
  verbosddd    = (verbosity/1000)%10;
  int verb1    = (verbosity/100)%10;
  int verb2    = (verbosity/10)%10;
  verbosity   %= 10;

  if (verbosity > 0)
    std::cout << "HcalTestAnalysis:: Initialised as observer of begin/end "
	      << "events and of G4step" << std::endl;

  count  = 0;
  group_ = layerGrouping(laygroup);
  nGroup = 0;
  for (unsigned int i=0; i<group_.size(); i++) 
    if (group_[i]>nGroup) nGroup = group_[i];
  tower_ = towersToAdd(centralTower, addTower);
  nTower = tower_.size()/2;

  if (verbosity > 0)
    std::cout << "HcalTestAnalysis:: initialised for " << nGroup 
	      << " Longitudinal groups and " << nTower << " towers" 
	      << std::endl;

  // qie
  myqie  = new HcalQie(p);

  // Ntuples
  tuples = new HcalTestHistoClass(verb1, file);

  // Numbering scheme
  org    = new HcalTestNumberingScheme(verb2);
} 
   
HcalTestAnalysis::~HcalTestAnalysis() {
  if (verbosity > 0) {
    std::cout << std::endl << " -------->  Total number of selected entries : "
	      << count << std::endl; 
    std::cout << "Pointers:: HcalQie " << myqie << ", HistoClass " << tuples
	      << ", Numbering Scheme " << org << " and FromDDD " 
	      << numberingFromDDD << std::endl;
  }
  if (myqie)  {
    if (verbosity > 0) std::cout << "Delete HcalQie" << std::endl;
    delete myqie;
  }
  if (tuples) {
    if (verbosity > 0) std::cout << "Delete HistogramClass" << std::endl;
    delete tuples;
  }
  if (numberingFromDDD) {
    if (verbosity > 0) std::cout << "Delete HcalNumberingFromDDD" << std::endl;
    delete numberingFromDDD;
  }
}

std::vector<int> HcalTestAnalysis::layerGrouping(int group) {

  std::vector<int> temp(19);
  if (group <= 1) {
    int grp[19] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2};
    for (int i=0; i<19; i++)
      temp[i] = grp[i];
  } else  if (group == 2) {
    int grp[19] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    for (int i=0; i<19; i++)
      temp[i] = grp[i];
  } else if (group == 3) {
    int grp[19] = {1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
    for (int i=0; i<19; i++)
      temp[i] = grp[i];
  } else if (group == 4) {
    int grp[19] = {1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7};
    for (int i=0; i<19; i++)
      temp[i] = grp[i];
  } else {
    int grp[19] = {1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7};
    for (int i=0; i<19; i++)
      temp[i] = grp[i];
  }

  if (verbosity > 0) {
    std::cout << "HcalTestAnalysis:: Layer Grouping ";
    for (int i=0; i<19; i++)
      std::cout << " " << temp[i];
    std::cout << std::endl;
  }
  return temp;
}

std::vector<int> HcalTestAnalysis::towersToAdd(int centre, int nadd) {

  int etac = (centre/100)%100;
  int phic = (centre%100);
  int etamin, etamax, phimin, phimax;
  if (etac>0) {
    etamin = etac-nadd;
    etamax = etac+nadd;
  } else {
    etamin = etac;
    etamax = etac;
  }
  if (phic>0) {
    phimin = phic-nadd;
    phimax = phic+nadd;
  } else {
    phimin = phic;
    phimax = phic;
  }

  int nbuf, kount=0;
  nbuf = (etamax-etamin+1)*(phimax-phimin+1);
  std::vector<int> temp(2*nbuf);
  for (int eta=etamin; eta<=etamax; eta++) {
    for (int phi=phimin; phi<=phimax; phi++) {
      temp[kount] = (eta*100 + phi);
      temp[kount+nbuf] = max(abs(eta-etac),abs(phi-phic));
      kount++;
    }
  }

  if (verbosity > 0) {
    std::cout << "HcalTestAnalysis:: Towers to be considered for Central " 
	      << centre << " and " << nadd << " on either side" << std::endl;
    for (int i=0; i<nbuf; i++) {
      std::cout << "\t" << setw(3) << i << " " << temp[i] << " " 
		<< temp[nbuf+i];
      if (i%6 == 5) std::cout << std::endl;
    }
    if (nbuf%6 != 5) std::cout << std::endl;
  }
  return temp;
}

//==================================================================== per JOB
void HcalTestAnalysis::update(const BeginOfJob * job) {

  edm::ESHandle<DDCompactView> pDD;
  (*job)()->get<IdealGeometryRecord>().get(pDD);
  if (verbosity > 0) 
    std::cout << "HcalTestAnalysis:: Initialise HcalNumberingFromDDD for "
	      << names[0] << std::endl;
  numberingFromDDD = new HcalNumberingFromDDD(verbosddd, names[0], (*pDD));
}

//==================================================================== per RUN
void HcalTestAnalysis::update(const BeginOfRun * run) {

  int irun = (*run)()->GetRunID();
  if (verbosity > 0) 
    std::cout <<" =====> Begin of Run = " << irun << std::endl;

  bool loop = true, eta = true, phi = true;
  int  etac = (centralTower/100)%100;
  if (etac == 0) {
    etac = 1;
    eta  = false;
  }
  int  phic = (centralTower%100);
  if (phic == 0) {
    phic = 1;
    phi  = false;
  }
  int  idet = static_cast<int>(HcalBarrel);
  while (loop) {
    HcalCellType::HcalCell tmp = numberingFromDDD->cell(idet,1,1,etac,phic);
    if (tmp.ok) {
      if (eta) eta0 = tmp.eta;
      if (phi) phi0 = tmp.phi;
      loop = false;
    } else if (idet == static_cast<int>(HcalBarrel)) {
      idet = static_cast<int>(HcalEndcap);
    } else if (idet == static_cast<int>(HcalEndcap)) {
      idet = static_cast<int>(HcalForward);
    } else {
      loop = false;
    }
  }

  if (verbosity > 0) 
    std::cout << "HcalTestAnalysis:: Central Tower " << centralTower 
	      << " corresponds to eta0 = " << eta0 << " phi0 = " << phi0
	      << std::endl;
 
  std::string sdname = names[0];
  G4SDManager* sd = G4SDManager::GetSDMpointerIfExist();
  if (sd != 0) {
    G4VSensitiveDetector* aSD = sd->FindSensitiveDetector(sdname);
    if (aSD==0) {
      if (verbosity > 0) 
	std::cout << "HcalTestAnalysis::beginOfRun: No SD with name " 
		  << sdname << " in this Setup " << std::endl;  
    } else {
      HCalSD* theCaloSD = dynamic_cast<HCalSD*>(aSD);
      if (verbosity > 0) 
	std::cout << "HcalTestAnalysis::beginOfRun: Finds SD with name " 
		  << theCaloSD->GetName() << " in this Setup " << std::endl;
      if (org) {
        theCaloSD->setNumberingScheme(org);
	if (verbosity > 0) 
	  std::cout << "HcalTestAnalysis::beginOfRun: set a new numbering "
		    << "scheme" << std::endl;
      }
    }
  } else {
    if (verbosity > 0) 
      std::cout << "HcalTestAnalysis::beginOfRun: Could not get SD Manager!" 
		<< std::endl;
  }

}

//=================================================================== per EVENT
void HcalTestAnalysis::update(const BeginOfEvent * evt) {
 
  // Reset counters
  tuples->setCounters();
 
  int i = 0;
  edepEB = edepEE = edepHB = edepHE = edepHO = 0.;
  for (i = 0; i < 20; i++) edepl[i] = 0.;
  for (i = 0; i < 20; i++) mudist[i] = -1.;

  int iev = (*evt)()->GetEventID();
  if (verbosity > 0) 
    std::cout <<" =====> Begin of event = " << iev << std::endl;
}

//=================================================================== each STEP
void HcalTestAnalysis::update(const G4Step * aStep) {

  if (aStep != NULL) {
    G4VPhysicalVolume* curPV  = aStep->GetPreStepPoint()->GetPhysicalVolume();
    G4String name = curPV->GetName();
    name.assign(name,0,3);
    double edeposit = aStep->GetTotalEnergyDeposit();
    int    layer=-1;
    if (name == "EBR") {
      edepEB += edeposit;
    } else if (name == "EFR") {
      edepEE += edeposit;
    } else if (name == "HBS") {
      layer = (curPV->GetCopyNo()/10)%100;
      if (layer >= 0 && layer < 17) {
	edepHB += edeposit;
      } else {
	if (verbosity > 0) 
	  std::cout << "Error " << curPV->GetName() << curPV->GetCopyNo() 
		    << std::endl;
	layer = -1;
      }
    } else if (name == "HES") {
      layer = (curPV->GetCopyNo()/10)%100;
      if (layer >= 0 && layer < 19) {
	edepHE += edeposit;
      } else {
	if (verbosity > 0) 
	  std::cout << "Error " << curPV->GetName() << curPV->GetCopyNo() 
		    << std::endl;
	layer = -1;
      }
    } else if (name == "HTS") {
      layer = (curPV->GetCopyNo()/10)%100;
      if (layer >= 17 && layer < 20) {
	edepHO += edeposit;
       } else {
	 if (verbosity > 0) 
	   std::cout << "Error " << curPV->GetName() << curPV->GetCopyNo()
		     << std::endl;
	layer = -1;
       }
    }
    if (layer >= 0 && layer < 20) {
      edepl[layer] += edeposit;

      // Calculate the distance if it is a muon
      G4String part = aStep->GetTrack()->GetDefinition()->GetParticleName();
      if ((part == "mu-" || part == "mu+") && mudist[layer] < 0) {
	Hep3Vector pos = aStep->GetPreStepPoint()->GetPosition();
	double theta   = pos.theta();
	double   eta   = -log(tan(theta * 0.5));
	double   phi   = pos.phi();
	double  dist   = sqrt ((eta-eta0)*(eta-eta0) + (phi-phi0)*(phi-phi0));
	mudist[layer]  = dist*pos.perp();
      }
    }

#ifdef debugStep
    if (layer >= 0 && layer < 20 && verbosity > 3) {
      std::cout << "HcalTestAnalysis:: G4Step: " << name << " Layer " 
		<< std::setw(3) << layer << " Edep " << std::setw(6) 
		<< edeposit/MeV << " MeV" << std::endl;
    }
#endif
  } else {
    if (verbosity > 3)
      std::cout << "HcalTestAnalysis:: G4Step: Null Step" << std::endl;
  }
}

//================================================================ End of EVENT
void HcalTestAnalysis::update(const EndOfEvent * evt) {

  count++;
  // Fill event input 
  fill(evt);
#ifdef ddebug
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis:: ---  after Fill " << std::endl;
#endif

  // Qie analysis
  qieAnalysis();
#ifdef ddebug
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis:: ---  after QieAnalysis " << std::endl;
#endif

  // Layers tuples filling
  layerAnalysis();
#ifdef ddebug
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis:: ---  after LayerAnalysis " << std::endl;
#endif

  // Writing the data to the Tree
  tuples->fillTree();
#ifdef ddebug
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis:: --- fillTree  " << std::endl;
#endif

}

//---------------------------------------------------
void HcalTestAnalysis::fill(const EndOfEvent * evt) {

#ifdef debugFill
  if (verbosity > 1) 
    std::cout << " Fill event " << (*evt)()->GetEventID() << std::endl;
#endif
  
  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  
  int nhc = 0, neb = 0, nef = 0, j = 0;    
  caloHitCache.erase (caloHitCache.begin(), caloHitCache.end());
 
  // Hcal
  int HCHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[0]);
  CaloG4HitCollection* theHCHC = (CaloG4HitCollection*) allHC->GetHC(HCHCid);
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis :: Hit Collection for " << names[0] 
	      << " of ID " << HCHCid << " is obtained at " << theHCHC 
	      << std::endl;
#endif
  if (HCHCid >= 0 && theHCHC > 0) {
    for (j = 0; j < theHCHC->entries(); j++) {

      CaloG4Hit* aHit = (*theHCHC)[j]; 
    
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      
      Hep3Vector pos  = aHit->getPosition();
      double theta    = pos.theta();
      double   eta    = -log(tan(theta * 0.5));
      double   phi    = pos.phi();
    
      uint32_t unitID = aHit->getUnitID();
      int subdet, zside, layer, etaIndex, phiIndex, lay;
      org->unpackHcalIndex(unitID,subdet,zside,layer,etaIndex,phiIndex,lay);
      double jitter   = time-timeOfFlight(subdet,lay,eta);
      if (jitter<0) jitter = 0;
      CaloHit hit(subdet,lay,e,eta,phi,jitter,unitID);
      caloHitCache.push_back(hit);
      nhc++;

      std::string det =  "HB";
      if (subdet == static_cast<int>(HcalForward)) {
	det = "HF";
      } else if (subdet == static_cast<int>(HcalEndcap)) {
	if (etaIndex <= 20) {
	  det  = "HES";
	} else {
	  det = "HED";
	}
      }

#ifdef debugFill
      if (verbosity > 2) 
	std::cout << " " << det << "  layer " << std::setw(2) << layer 
		  << " time " << std::setw(6) << time << " theta " 
		  << std::setw(8) << theta << " eta " << std::setw(8) << eta  
		  << " phi " << std::setw(8) << phi << " e " << std::setw(8) 
		  << e << std::endl;   
#endif      
    }
  }
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis::HCAL hits : " << nhc << std::endl 
	      << std::endl; 
#endif

  // EB
  int EBHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[1]);
  CaloG4HitCollection* theEBHC = (CaloG4HitCollection*) allHC->GetHC(EBHCid);
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis :: Hit Collection for " << names[1]
	      << " of ID " << EBHCid << " is obtained at " << theEBHC 
	      << std::endl;
#endif
  if (EBHCid >= 0 && theEBHC > 0) {
    for (j = 0; j < theEBHC->entries(); j++) {

      CaloG4Hit* aHit = (*theEBHC)[j]; 
    
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      std::string det =  "EB";
      
      Hep3Vector pos  = aHit->getPosition();
      double theta    = pos.theta();
      double   eta    = -log(tan(theta/2.));
      double   phi    = pos.phi();

      HcalNumberingFromDDD::HcalID id = numberingFromDDD->unitID(eta,phi,1,1);
      uint32_t unitID = org->getUnitID(id);
      int subdet, zside, layer, ieta, iphi, lay;
      org->unpackHcalIndex(unitID,subdet,zside,layer,ieta,iphi,lay);
      subdet = 10;
      layer  = 0;
      unitID = org->packHcalIndex(subdet,zside,layer,ieta,iphi,lay);
      CaloHit hit(subdet,lay,e,eta,phi,time,unitID);
      caloHitCache.push_back(hit);
      neb++;
#ifdef debugFill    
      if (verbosity > 2) 
	std::cout << " " << det << "  layer " << std::setw(2) << layer 
		  << " time " << std::setw(6) << time << " theta " 
		  << std::setw(8) << theta << " eta " << std::setw(8) << eta 
		  << " phi " << std::setw(8) << phi << " e " << std::setw(8) 
		  << e << std::endl;   
#endif
    }
  }
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis::EB hits : " << neb << std::endl 
	      << std::endl; 
#endif

  // EE
  int EEHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[2]);
  CaloG4HitCollection* theEEHC = (CaloG4HitCollection*) allHC->GetHC(EEHCid);
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis :: Hit Collection for " << names[2]
	      << " of ID " << EEHCid << " is obtained at " << theEEHC 
	      << std::endl;
#endif
  if (EEHCid >= 0 && theEEHC > 0) {
    for (j = 0; j < theEEHC->entries(); j++) {

      CaloG4Hit* aHit = (*theEEHC)[j]; 
    
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      std::string det = "EE";
      
      Hep3Vector pos  = aHit->getPosition();
      double theta    = pos.theta();
      double   eta    = -log(tan(theta/2.));
      double   phi    = pos.phi();

      HcalNumberingFromDDD::HcalID id = numberingFromDDD->unitID(eta,phi,1,1);
      uint32_t unitID = org->getUnitID(id);
      int subdet, zside, layer, ieta, iphi, lay;
      org->unpackHcalIndex(unitID,subdet,zside,layer,ieta,iphi,lay);
      subdet = 11;
      layer  = 0;
      unitID = org->packHcalIndex(subdet,zside,layer,ieta,iphi,lay);
      CaloHit hit(subdet,lay,e,eta,phi,time,unitID);
      caloHitCache.push_back(hit);
      nef++;
#ifdef debugFill
      if (verbosity > 2) 
	std::cout << " " << det << "  layer " << std::setw(2) << layer 
		  << " time " << std::setw(6) << time << " theta " 
		  << std::setw(8) << theta << " eta " << std::setw(8) << eta  
		  << " phi " << std::setw(8) << phi << " e " << std::setw(8) 
		  << e << std::endl;   
#endif
    }
  }
#ifdef debugFill
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis::EE hits : " << nef << std::endl 
	      << std::endl; 
#endif
}

//-----------------------------------------------------------------------------
void HcalTestAnalysis::qieAnalysis() {

  //Fill tuple with hit information
  int hittot = caloHitCache.size();
  tuples->fillHits(caloHitCache);

  //Get the index of the central tower
  HcalNumberingFromDDD::HcalID id = numberingFromDDD->unitID(eta0,phi0,1,1);
  uint32_t unitID = org->getUnitID(id);
  int subdet, zside, layer, ieta, iphi, lay;
  org->unpackHcalIndex(unitID,subdet,zside,layer,ieta,iphi,lay);
  int      laymax = 0;
  std::string det = "Unknown";
  if (subdet == static_cast<int>(HcalBarrel)) {
    laymax = 4; det = "HB";
  } else if (subdet == static_cast<int>(HcalEndcap)) {
    laymax = 2; det = "HES";
  }
#ifdef debugqie
  if (verbosity > 1) 
    std::cout << "HcalTestAnalysis::Qie: " << det << " Eta " << ieta << " Phi "
	      << iphi << " Laymax " << laymax << " Hits " << hittot 
	      << std::endl;
#endif

  if (laymax>0 && hittot>0) {
    std::vector<CaloHit>  hits(hittot);
    std::vector<double> eqielay(80,0.0),  esimlay(80,0.0),  esimtot(4,0.0);
    std::vector<double> eqietow(200,0.0), esimtow(200,0.0), eqietot(4,0.0);
    int etac = (centralTower/100)%100;
    int phic = (centralTower%100);

    for (int layr=0; layr<nGroup; layr++) {
      int layx, layy=20;
      for (int i=0; i<20; i++) 
	if (group_[i] == layr+1 && i < layy) layy = i+1;
      if (subdet == static_cast<int>(HcalBarrel)) {
	if      (layy < 2)    layx = 0;
	else if (layy < 17)   layx = 1;
	else if (layy == 17)  layx = 2;
	else                  layx = 3;
      } else {
	if      (layy < 2)    layx = 0;
	else                  layx = 1;
      }

      for (int it=0; it<nTower; it++) {
	int    nhit = 0;
	double esim = 0;
	for (int k1 = 0; k1 < hittot; k1++) {
	  CaloHit hit = caloHitCache[k1];
	  int     subdetc = hit.det();
	  int     layer   = hit.layer();
	  int     group   = 0;
	  if (layer > 0 && layer < 20) group = group_[layer];
	  if (subdetc == subdet && group == layr+1) {
	    int zsidec, ietac, iphic, idx;
	    unitID = hit.id();
	    org->unpackHcalIndex(unitID,subdetc,zsidec,layer,ietac,iphic,lay);
	    if (etac > 0 && phic > 0) {
	      idx = ietac*100 + iphic;
	    } else if (etac > 0) {
	      idx = ietac*100;
	    } else if (phic > 0) {
	      idx = iphic;
	    } else {
	      idx = 0;
	    }
	    if (zsidec==zside && idx==tower_[it]) {
	      hits[nhit] = hit;
#ifdef debugqie
	      if (verbosity > 2) std::cout << nhit << " " << hit << std::endl;
#endif
	      nhit++;
	      esim += hit.e();
	    }
	  }
	}

	std::vector<int> cd = myqie->getCode(nhit,hits);
	double         eqie = myqie->getEnergy(cd);

#ifdef debugqie
	if (verbosity > 1)
	  std::cout << "HcalTestAnalysis::Qie: Energy in layer " << layr 
		    << " Sim " << esim << " After QIE " << eqie << std::endl;
#endif
	for (int i=0; i<4; i++) {
	  if (tower_[nTower+it] <= i) {
	    esimtot[i]         += esim;
	    eqietot[i]         += eqie;
	    esimlay[20*i+layr] += esim;
	    eqielay[20*i+layr] += eqie;
	    esimtow[50*i+it]   += esim;
	    eqietow[50*i+it]   += eqie;
	  }
	}
      }
    }
#ifdef debugqie
    if (verbosity > 1)
      std::cout << "HcalTestAnalysis::Qie: Total energy " << esimtot[3] 
		<< " (SimHit) " << eqietot[3] << " (After QIE)" << std::endl;
#endif

    std::vector<double> latphi(10);
    int nt = 2*addTower + 1;
    for (int it=0; it<nt; it++)
      latphi[it] = it-addTower;
    for (int i=0; i<4; i++) {
      double scals=1, scalq=1;
      std::vector<double> latfs(10,0.), latfq(10,0.), longs(20), longq(20);
      if (esimtot[i]>0) scals = 1./esimtot[i];
      if (eqietot[i]>0) scalq = 1./eqietot[i];
      for (int it=0; it<nTower; it++) {
	int phib = it%nt; 
	latfs[phib] += scals*esimtow[50*i+it];
	latfq[phib] += scalq*eqietow[50*i+it];
      }
      for (int layr=0; layr<=nGroup; layr++) {
	longs[layr] = scals*esimlay[20*i+layr];
	longq[layr] = scalq*eqielay[20*i+layr];
      }
      tuples->fillQie(i,esimtot[i],eqietot[i],nGroup,longs,longq,
		      nt,latphi,latfs,latfq);
    }
  }
}

//---------------------------------------------------
void HcalTestAnalysis::layerAnalysis(){

#ifdef debugLayer
  if (verbosity > 1) {
    int i = 0;
    std::cout << std::endl
	      << " ===>>> HcalTestAnalysis: Energy deposit in MeV " <<std::endl
	      << " at EB : " << setw(6) << edepEB/MeV << std::endl
	      << " at EE : " << setw(6) << edepEE/MeV << std::endl
	      << " at HB : " << setw(6) << edepHB/MeV << std::endl
	      << " at HE : " << setw(6) << edepHE/MeV << std::endl
	      << " at HO : " << setw(6) << edepHO/MeV << std::endl
	      << std::endl
	      << " ---- HcalTestAnalysis: Energy deposit in Layers" 
	      << std::endl;
    for (i = 0; i < 20; i++) 
      std::cout << " Layer " << std::setw(2) << i << " E " << std::setw(8) 
		<< edepl[i]/MeV  << " MeV" << std::endl;
  }
#endif

  tuples->fillLayers(edepl, edepHO, edepHB+edepHE, mudist);  
}


//---------------------------------------------------
double HcalTestAnalysis::timeOfFlight(int det, int layer, double eta) {

  double theta = 2.0*atan(exp(-eta));
  double dist  = 0.;
  if (det == static_cast<int>(HcalBarrel)) {
    const double rLay[19] = {
      1836.0, 1902.0, 1962.0, 2022.0, 2082.0, 2142.0, 2202.0, 2262.0, 2322.0, 
      2382.0, 2448.0, 2514.0, 2580.0, 2646.0, 2712.0, 2776.0, 2862.5, 3847.0,
      4052.0};
    if (layer>0 && layer<20) dist += rLay[layer-1]*mm/sin(theta);
  } else {
    const double zLay[19] = {
      4034.0, 4032.0, 4123.0, 4210.0, 4297.0, 4384.0, 4471.0, 4558.0, 4645.0, 
      4732.0, 4819.0, 4906.0, 4993.0, 5080.0, 5167.0, 5254.0, 5341.0, 5428.0,
      5515.0};
    if (layer>0 && layer<20) dist += zLay[layer-1]*mm/cos(theta);
  }
  double tmp = dist/c_light/ns;
#ifdef ddebug
  if (verbosity > 2)
    std::cout << "HcalTestAnalysis::timeOfFlight " << tmp << " for det/lay " 
	      << det << " " << layer << " eta/theta " << eta << " " 
	      << theta/deg << " dist " << dist << std::endl;
#endif
  return tmp;
}
