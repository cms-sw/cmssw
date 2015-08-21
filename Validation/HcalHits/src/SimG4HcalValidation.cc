///////////////////////////////////////////////////////////////////////////////
// File: SimG4Validation.cc
// Description: Main analysis class for Hcal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////
#include "Validation/HcalHits/interface/SimG4HcalValidation.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <cmath>
#include <iostream>
#include <iomanip>

SimG4HcalValidation::SimG4HcalValidation(const edm::ParameterSet &p): 
  jetf(0), numberingFromDDD(0), org(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("SimG4HcalValidation");
  infolevel     = m_Anal.getParameter<int>("InfoLevel");
  hcalOnly      = m_Anal.getParameter<bool>("HcalClusterOnly");
  applySampling = m_Anal.getParameter<bool>("HcalSampling");
  coneSize      = m_Anal.getParameter<double>("ConeSize");
  ehitThreshold = m_Anal.getParameter<double>("EcalHitThreshold");
  hhitThreshold = m_Anal.getParameter<double>("HcalHitThreshold");
  timeLowlim    = m_Anal.getParameter<double>("TimeLowLimit");
  timeUplim     = m_Anal.getParameter<double>("TimeUpLimit");
  jetThreshold  = m_Anal.getParameter<double>("JetThreshold");
  eta0          = m_Anal.getParameter<double>("Eta0");
  phi0          = m_Anal.getParameter<double>("Phi0");
  names         = m_Anal.getParameter<std::vector<std::string> >("Names");
  labelLayer    = m_Anal.getUntrackedParameter<std::string>("LabelLayerInfo","HcalInfoLayer");
  labelNxN      = m_Anal.getUntrackedParameter<std::string>("LabelNxNInfo","HcalInfoNxN");
  labelJets     = m_Anal.getUntrackedParameter<std::string>("LabelJetsInfo","HcalInfoJets");

  produces<PHcalValidInfoLayer>(labelLayer);
  if (infolevel > 0) produces<PHcalValidInfoNxN>(labelNxN);
  if (infolevel > 1) produces<PHcalValidInfoJets>(labelJets);

  edm::LogInfo("ValidHcal") << "HcalTestAnalysis:: Initialised as observer of "
			    << "begin/end events and of G4step with Parameter "
			    << "values: \n\tInfoLevel     = " << infolevel
			    << "\n\thcalOnly      = " << hcalOnly 
			    << "\n\tapplySampling = " << applySampling 
			    << "\n\tconeSize      = " << coneSize
			    << "\n\tehitThreshold = " << ehitThreshold 
			    << "\n\thhitThreshold = " << hhitThreshold
			    << "\n\tttimeLowlim   = " << timeLowlim
			    << "\n\tttimeUplim    = " << timeUplim
			    << "\n\teta0          = " << eta0
			    << "\n\tphi0          = " << phi0
			    << "\nLabels (Layer): " << labelLayer
			    << " (NxN): " << labelNxN << " (Jets): "
			    << labelJets;

  init();
} 
   
SimG4HcalValidation::~SimG4HcalValidation() {

  edm::LogInfo("ValidHcal") << "\n -------->  Total number of selected entries"
			    << " : " << count << "\nPointers:: JettFinder " 
			    << jetf << ", Numbering Scheme " << org 
			    << " and FromDDD " << numberingFromDDD;
  if (jetf)   {
    edm::LogInfo("ValidHcal") << "Delete Jetfinder";
    delete jetf;
    jetf  = 0;
  }
  if (numberingFromDDD) {
    edm::LogInfo("ValidHcal") << "Delete HcalNumberingFromDDD";
    delete numberingFromDDD;
    numberingFromDDD = 0;
  }
}

void SimG4HcalValidation::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<PHcalValidInfoLayer> productLayer(new PHcalValidInfoLayer);
  layerAnalysis(*productLayer);
  e.put(productLayer,labelLayer);

  if (infolevel > 0) {
    std::auto_ptr<PHcalValidInfoNxN> productNxN(new PHcalValidInfoNxN);
    nxNAnalysis(*productNxN);
    e.put(productNxN,labelNxN);
  }

  if (infolevel > 1) {
    std::auto_ptr<PHcalValidInfoJets> productJets(new PHcalValidInfoJets);
    jetAnalysis(*productJets);
    e.put(productJets,labelJets);
  }
}

//==================================================================== per RUN
void SimG4HcalValidation::init() {

  float sHB[4] = { 117., 117., 178., 217.};
  float sHE[3] = { 178., 178., 178.};
  float sHF[3] = { 2.84, 2.09, 0.};     //

  float deta[4] = { 0.0435, 0.1305, 0.2175, 0.3045 };
  float dphi[4] = { 0.0436, 0.1309, 0.2182, 0.3054 };

  int i=0;
  for (i = 0; i < 4; i++) { 
    scaleHB.push_back(sHB[i]);
  }
  for (i = 0; i < 3; i++) { 
    scaleHE.push_back(sHE[i]);
  }
  for (i = 0; i < 3; i++) { 
    scaleHF.push_back(sHF[i]);
  }

  // window steps;
  for(i = 0; i < 4; i++) { 
    dEta.push_back(deta[i]);
    dPhi.push_back(dphi[i]);
  }

  // jetfinder conse size setting
  jetf   = new SimG4HcalHitJetFinder(coneSize);

  // counter 
  count = 0;
  
}

void SimG4HcalValidation::update(const BeginOfJob * job) {

  // Numbering From DDD
  edm::ESHandle<HcalDDDSimConstants>    hdc;
  (*job)()->get<HcalSimNumberingRecord>().get(hdc);
  HcalDDDSimConstants *hcons = (HcalDDDSimConstants*)(&(*hdc));
  edm::LogInfo("ValidHcal") << "HcalTestAnalysis:: Initialise "
			    << "HcalNumberingFromDDD";
  numberingFromDDD = new HcalNumberingFromDDD(hcons);

  // Numbering scheme
  org              = new HcalTestNumberingScheme(false);

}

void SimG4HcalValidation::update(const BeginOfRun * run) {

  int irun = (*run)()->GetRunID();
  
  edm::LogInfo("ValidHcal") <<" =====> Begin of Run = " << irun;
 
  std::string  sdname = names[0];
  G4SDManager* sd     = G4SDManager::GetSDMpointerIfExist();
  if (sd != 0) {
    G4VSensitiveDetector* aSD = sd->FindSensitiveDetector(sdname);
    if (aSD==0) {
      edm::LogWarning("ValidHcal") << "SimG4HcalValidation::beginOfRun: No SD"
				   << " with name " << sdname << " in this "
				   << "Setup";
    } else {
      HCalSD* theCaloSD = dynamic_cast<HCalSD*>(aSD);
      edm::LogInfo("ValidHcal") << "SimG4HcalValidation::beginOfRun: Finds SD"
				<< "with name " << theCaloSD->GetName() 
				<< " in this Setup";
      if (org) {
        theCaloSD->setNumberingScheme(org);
        edm::LogInfo("ValidHcal") << "SimG4HcalValidation::beginOfRun: set a "
				  << "new numbering scheme";
      }
    }
  } else {
    edm::LogWarning("ValidHcal") << "SimG4HcalValidation::beginOfRun: Could "
				 << "not get SD Manager!";
  }

}

//=================================================================== per EVENT
void SimG4HcalValidation::update(const BeginOfEvent * evt) {
 
  int i = 0;
  edepEB = edepEE = edepHB = edepHE = edepHO = 0.;
  for (i = 0; i < 5;  i++) edepd[i] = 0.;
  for (i = 0; i < 20; i++) edepl[i] = 0.;
  vhitec = vhithc = enEcal = enHcal = 0;  
  // Cache reset  
  clear();

  int iev = (*evt)()->GetEventID();
  LogDebug("ValidHcal") << "SimG4HcalValidation: =====> Begin of event = "
			<< iev;
}

//=================================================================== each STEP
void SimG4HcalValidation::update(const G4Step * aStep) {

  if (aStep != NULL) {
    G4VPhysicalVolume* curPV  = aStep->GetPreStepPoint()->GetPhysicalVolume();
    G4String name = curPV->GetName();
    name.assign(name,0,3);
    double edeposit = aStep->GetTotalEnergyDeposit();
    int    layer=-1,  depth=-1;
    if (name == "EBR") {
      depth = 0;
      edepEB += edeposit;
    } else if (name == "EFR") {
      depth = 0;
      edepEE += edeposit;
    } else if (name == "HBS") {
      layer = (curPV->GetCopyNo()/10)%100;
      depth = (curPV->GetCopyNo())%10 + 1;
      if (depth > 0 && depth < 4 && layer >= 0 && layer < 17) {
	edepHB += edeposit;
      } else {
	edm::LogWarning("ValidHcal") << "SimG4HcalValidation:Error " 
				     << curPV->GetName() << curPV->GetCopyNo();
	depth = -1; layer = -1;
      }
    } else if (name == "HES") {
      layer = (curPV->GetCopyNo()/10)%100;
      depth = (curPV->GetCopyNo())%10 + 1;
      if (depth > 0 && depth < 3 && layer >= 0 && layer < 19) {
	edepHE += edeposit;
      } else {
	edm::LogWarning("ValidHcal") << "SimG4HcalValidation:Error " 
				     << curPV->GetName() << curPV->GetCopyNo();
	depth = -1; layer = -1;
      }
    } else if (name == "HTS") {
      layer = (curPV->GetCopyNo()/10)%100;
      depth = (curPV->GetCopyNo())%10 + 1;
      if (depth > 3 && depth < 5 && layer >= 17 && layer < 20) {
	edepHO += edeposit;
       } else {
	 edm::LogWarning("ValidHcal") << "SimG4HcalValidation:Error " 
				      << curPV->GetName() <<curPV->GetCopyNo();
	 depth = -1; layer = -1;
       }
    }
    if (depth >= 0 && depth < 5)  edepd[depth] += edeposit;
    if (layer >= 0 && layer < 20) edepl[layer] += edeposit;

    if (layer >= 0 && layer < 20) {
      LogDebug("ValidHcal") << "SimG4HcalValidation:: G4Step: " << name 
			    << " Layer " << std::setw(3) << layer << " Depth "
			    << std::setw(2) << depth << " Edep " <<std::setw(6)
			    << edeposit/MeV << " MeV";
    }
  }
}

//================================================================ End of EVENT
void SimG4HcalValidation::update(const EndOfEvent * evt) {

  count++;

  // Fill hits cache for jetfinding etc.
  fill(evt);
  LogDebug("ValidHcal") << "SimG4HcalValidation:: ---  after Fill ";
}

//---------------------------------------------------
void SimG4HcalValidation::fill(const EndOfEvent * evt) {

  LogDebug("ValidHcal") << "SimG4HcalValidation:Fill event " 
			<< (*evt)()->GetEventID();
  
  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  
  int nhc = 0, j = 0;    
 
  // Hcal
  int HCHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[0]);
  CaloG4HitCollection* theHCHC = (CaloG4HitCollection*) allHC->GetHC(HCHCid);
  LogDebug("ValidHcal") << "SimG4HcalValidation :: Hit Collection for " 
			<< names[0] << " of ID " << HCHCid <<" is obtained at "
			<< theHCHC;
  if (HCHCid >= 0 && theHCHC > 0) {
    for (j = 0; j < theHCHC->entries(); j++) {

      CaloG4Hit* aHit = (*theHCHC)[j]; 
    
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      
      math::XYZPoint pos  = aHit->getPosition();
      double theta    = pos.theta();
      double   eta    = -log(tan(theta * 0.5));
      double   phi    = pos.phi();
    
      uint32_t unitID = aHit->getUnitID();
      int subdet, zside, layer, etaIndex, phiIndex, lay;
      org->unpackHcalIndex(unitID,subdet,zside,layer,etaIndex,phiIndex,lay);

      // some logic to separate HO ...
      layer--;
      std::string det =  "HB";
      if (subdet == static_cast<int>(HcalForward)) {
	det = "HF";
	uint16_t depth = aHit->getDepth();
	if (depth != 0) { layer += 2; lay += 2; }
	if      (layer == 1) vhithc += e;
	else if (layer == 0) vhitec += e;
	else
	  edm::LogInfo("ValidHcal") << "SimG4HcalValidation::HitPMT " 
				    << subdet << " " << (2*zside-1)*etaIndex 
				    << " " << phiIndex << " " << layer+1 
				    << " R " << pos.rho() << " Phi " << phi/deg
				    << " Edep " << e << " Time " << time;
      } else if (subdet == static_cast<int>(HcalEndcap)) {
	if (etaIndex <= 20) {
	  det  = "HES";
	} else {
	  det = "HED";
	}
      }
      LogDebug("ValidHcal") << "SimG4HcalValidation::debugFill Hcal " 
			    << det << " layer " << std::setw(2) << layer 
			    << " lay " << std::setw(2) << lay << " time " 
			    << std::setw(6) << time << " theta " 
			    << std::setw(8) << theta << " eta " << std::setw(8)
			    << eta << " phi " << std::setw(8) << phi << " e " 
			    << std::setw(8) << e << " ID 0x" << std::hex
			    << unitID << " ID dec " << std::dec << (int)unitID;

      // if desired, apply sampling factors in HCAL !!!
      if (applySampling) e *= getHcalScale(det,layer);    

      //    filter on time & energy 
      if (time >= timeLowlim && time <= timeUplim && e > hhitThreshold ) {
	enHcal += e;
	CaloHit ahit(subdet,lay,e,eta,phi,time,unitID);
	hitcache.push_back(ahit);    // fill cache
	++nhc;
      }    
    }
  }
  LogDebug("ValidHcal") << "SimG4HcalValidation:: HCAL hits : " << nhc;

  if (!hcalOnly) { //--------------------------  ECAL hits --------------------
    int ndets = names.size();
    if (ndets > 3) ndets = 3;
    for (int idty = 1; idty < ndets; idty++) {
      std::string          det = "EB";
      if      (idty ==  2) det = "EE";
      else if (idty > 2)   det = "ES";

      int nec    = 0;
      int ECHCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[idty]);
      CaloG4HitCollection* theECHC =(CaloG4HitCollection*)allHC->GetHC(ECHCid);
      LogDebug("ValidHcal") << "SimG4HcalValidation:: Hit Collection for "
			    << names[idty] << " of ID " << ECHCid 
			    << " is obtained at " << theECHC;
      if (ECHCid >= 0 && theECHC > 0) {
	for (j = 0; j < theECHC->entries(); j++) {

	  CaloG4Hit* aHit = (*theECHC)[j]; 
    
	  double e        = aHit->getEnergyDeposit()/GeV;
	  double time     = aHit->getTimeSlice();
      
	  math::XYZPoint pos  = aHit->getPosition();
	  double theta    = pos.theta();
	  double   eta    = -log(tan(theta/2.));
	  double   phi    = pos.phi();
	  HcalNumberingFromDDD::HcalID id = numberingFromDDD->unitID(eta,phi,1,1);
	  uint32_t unitID = org->getUnitID(id);
	  int subdet, zside, layer, ieta, iphi, lay;
	  org->unpackHcalIndex(unitID,subdet,zside,layer,ieta,iphi,lay);
	  subdet = idty+9;
	  layer  = 0;
	  unitID = org->packHcalIndex(subdet,zside,layer,ieta,iphi,lay);
	  
	  //    filter on time & energy 
	  if (time >= timeLowlim && time <= timeUplim && e > ehitThreshold ) {
	    enEcal += e;
	    CaloHit ahit(subdet,lay,e,eta,phi,time,unitID);
	    hitcache.push_back(ahit);    // fill cache
	    ++nec;
	  }

	  LogDebug("ValidHcal") << "SimG4HcalValidation::debugFill Ecal " 
				<< det << " layer " << std::setw(2) << layer 
				<< " lay " << std::setw(2) << lay << " time " 
				<< std::setw(6) << time << " theta " 
				<< std::setw(8) << theta << " eta " 
				<< std::setw(8) << eta  << " phi " 
				<< std::setw(8) << phi << " e " << std::setw(8)
				<< e << " ID 0x" << std::hex << unitID 
				<< " ID dec " << std::dec << (int)unitID;

	}
      }

      LogDebug("ValidHcal") << "SimG4HcalValidation:: " << det << " hits : "
			    << nec;
    }
  } // end of if(!hcalOnly)

}

void SimG4HcalValidation::layerAnalysis(PHcalValidInfoLayer& product) {

  int i = 0;
  LogDebug("ValidHcal") << "\n ===>>> SimG4HcalValidation: Energy deposit "
			<< "in MeV\n at EB : " << std::setw(6) << edepEB/MeV 
			<< "\n at EE : " << std::setw(6) << edepEE/MeV 
			<< "\n at HB : " << std::setw(6) << edepHB/MeV 
			<< "\n at HE : " << std::setw(6) << edepHE/MeV 
			<< "\n at HO : " << std::setw(6) << edepHO/MeV 
			<< "\n ---- SimG4HcalValidation: Energy deposit in";
  for (i = 0; i < 5; i++) 
    LogDebug("ValidHcal") << " Depth " << std::setw(2) << i << " E " 
			  << std::setw(8) << edepd[i]/MeV << " MeV";
  LogDebug("ValidHcal") << " ---- SimG4HcalValidation: Energy deposit in"
			<< "layers";
  for (i = 0; i < 20; i++) 
    LogDebug("ValidHcal") << " Layer " << std::setw(2) << i << " E " 
			  << std::setw(8) << edepl[i]/MeV  << " MeV";

  product.fillLayers(edepl, edepd, edepHO, edepHB+edepHE, edepEB+edepEE);  

  // Hits in HF
  product.fillHF(vhitec, vhithc, enEcal, enHcal);
  LogDebug("ValidHcal") << "SimG4HcalValidation::HF hits " << vhitec 
			<< " in EC and " << vhithc << " in HC\n"
			<< "                     HB/HE   " << enEcal 
			<< " in EC and " << enHcal << " in HC";

  // Another HCAL hist to porcess and store separately (a bit more complicated)
  fetchHits(product);

  LogDebug("ValidHcal") << " layerAnalysis ===> after fetchHits";

}

//-----------------------------------------------------------------------------
void SimG4HcalValidation::nxNAnalysis(PHcalValidInfoNxN& product) {

  std::vector<CaloHit> * hits = &hitcache;
  std::vector<CaloHit>::iterator hit_itr;

  LogDebug("ValidHcal") << "SimG4HcalValidation::NxNAnalysis : entrance ";

  collectEnergyRdir(eta0,phi0); // HCAL and ECAL energy in  SimHitCache
                                // around (eta0,phi0)

  LogDebug("ValidHcal") << " NxNAnalysis : coolectEnergyRdir - Ecal " << een 
			<< "   Hcal " << hen;

  double etot  = 0.;      // total e deposited     in "cluster"
  double ee    = 0.;      // ECAL  e deposited     in "cluster"
  double he    = 0.;      // HCAL  e deposited     in "cluster"
  double hoe   = 0.;      // HO    e deposited     in "cluster"     

  int    max   = dEta.size(); // 4
  
  for (hit_itr = hits->begin(); hit_itr < hits->end(); hit_itr++) {

    double e     = hit_itr->e();
    double t     = hit_itr->t();
    double eta   = hit_itr->eta();
    double phi   = hit_itr->phi();
    int    type  = hit_itr->det();
    int    layer = hit_itr->layer();
    
    // NxN calulation
    
    if (fabs(eta0-eta) <= dEta[max-1] && fabs(phi0-phi) <= dPhi[max-1]) {
      etot += e;
      if (type == 10 || type == 11 || type == 12)  ee += e;
      if (type  == static_cast<int>(HcalBarrel) ||
	  type  == static_cast<int>(HcalEndcap) ||
	  type  == static_cast<int>(HcalForward)) { 
	he += e; 
	if (type  == static_cast<int>(HcalBarrel) && layer > 17) hoe += e;

	// which concrete i-th square ?
	for (int i = 0; i < max; i++ ) { 
	  if ((eta0-eta) <= dEta[i] && (eta0-eta) >= -dEta[i] &&
	      (phi0-phi) <= dPhi[i] && (phi0-phi) >= -dPhi[i]) {

	    LogDebug("ValidHcal") << "SimG4HcalValidation:: nxNAnalysis eta0,"
				  << " phi0 = " << eta0 << " " << phi0 
				  << "   type, layer = " << type << "," 
				  << layer << "  eta, phi = " << eta << " " 
				  << phi;

	    product.fillTProfileNxN(e, i, t);  
	    break;
	  }
	}
        // here comes break ...
      }
    }
  }

  product.fillEcollectNxN(ee, he, hoe, etot);       
  product.fillHvsE(een, hen, hoen, een+hen);

  LogDebug("ValidHcal") << " nxNAnalysis ===> after fillHvsE";


}


//-----------------------------------------------------------------------------
void SimG4HcalValidation::jetAnalysis(PHcalValidInfoJets& product) {

  std::vector<CaloHit> * hhit = &hitcache;

  jetf->setInput(hhit);

  // zeroing cluster list, perfom clustering, fill cluster list & return pntr
  std::vector<SimG4HcalHitCluster> * result = jetf->getClusters(hcalOnly);  

  std::vector<SimG4HcalHitCluster>::iterator clus_itr;

  LogDebug("ValidHcal") << "\n ---------- Final list of " << (*result).size()
			<< " clusters ---------------";
  for (clus_itr = result->begin(); clus_itr < result->end(); clus_itr++) 
    LogDebug("ValidHcal") << (*clus_itr);

  std::vector<double> enevec, etavec, phivec; 

  if ((*result).size() > 0) {

    sort((*result).begin(),(*result).end()); 

    clus_itr = result->begin(); // first cluster only 
    double etac = clus_itr->eta();
    double phic = clus_itr->phi();

    double ecal_collect = 0.;    // collect Ecal energy in the cone
    if (!hcalOnly) {
      ecal_collect = clus_itr->collectEcalEnergyR();}
    else {
      collectEnergyRdir(etac, phic);
      ecal_collect = een;
    }
    LogDebug("ValidHcal") << " JetAnalysis ===> ecal_collect  = " 
			  << ecal_collect;

    // eta-phi deviation of the cluster from nominal (eta0, phi0) values
    double dist = jetf->rDist(eta0, phi0, etac, phic);
    LogDebug("ValidHcal") << " JetAnalysis ===> eta phi deviation  = " << dist;
    product.fillEtaPhiProfileJet(eta0, phi0, etac, phic, dist);
   
    LogDebug("ValidHcal") << " JetAnalysis ===> after fillEtaPhiProfileJet";
    
    std::vector<CaloHit> * hits = clus_itr->getHits() ;
    std::vector<CaloHit>::iterator hit_itr;

    double ee = 0., he = 0., hoe = 0., etot = 0.;
    
    // cycle over all hits in the FIRST cluster
    for (hit_itr = hits->begin(); hit_itr < hits->end(); hit_itr++) {
      double e   = hit_itr->e();
      double t   = hit_itr->t();
      double r   = jetf->rDist(&(*clus_itr), &(*hit_itr));

      // energy collection
      etot += e;
      if (hit_itr->det()  == 10 || hit_itr->det()  == 11 ||
	  hit_itr->det()  == 12) ee  += e;
      if (hit_itr->det()  == static_cast<int>(HcalBarrel) ||
	  hit_itr->det()  == static_cast<int>(HcalEndcap) ||
	  hit_itr->det()  == static_cast<int>(HcalForward)) { 
	he  += e; 
	if (hit_itr->det()  == static_cast<int>(HcalBarrel) &&
	    hit_itr->layer() > 17) 
	  hoe += e; 
      }

      if (hit_itr->det()  == static_cast<int>(HcalBarrel) ||
	  hit_itr->det()  == static_cast<int>(HcalEndcap) ||
	  hit_itr->det()  == static_cast<int>(HcalForward)) { 
	product.fillTProfileJet(he, r, t);
      }
    }
    
    product.fillEcollectJet(ee, he, hoe, etot);       

    LogDebug("ValidHcal") << " JetAnalysis ===> after fillEcollectJet: "
			  << "ee/he/hoe/etot " << ee << "/" << he << "/" 
			  << hoe << "/" << etot;

    // Loop over clusters
    for (clus_itr = result->begin(); clus_itr < result->end(); clus_itr++) {
      if (clus_itr->e() > jetThreshold) {
	enevec.push_back(clus_itr->e());
	etavec.push_back(clus_itr->eta());
	phivec.push_back(clus_itr->phi());
      }
    }
    product.fillJets(enevec, etavec, phivec);

    LogDebug("ValidHcal") << " JetAnalysis ===> after fillJets\n"
			  << " JetAnalysis ===> (*result).size() "
			  << (*result).size();
 
    // Di-jet mass
    if (etavec.size() > 1) {
      if (etavec[0] > -2.5 && etavec[0] < 2.5 && 
	  etavec[1] > -2.5 && etavec[1] < 2.5) {
 
	LogDebug("ValidHcal") << " JetAnalysis ===> Di-jet mass enter\n"
			      << " JetAnalysis ===> Di-jet vectors ";
	for (unsigned int i = 0; i < enevec.size(); i++)
	  LogDebug("ValidHcal") << "   e, eta, phi = " << enevec[i] << " "
				<<  etavec[i] << " " << phivec[i];

        double et0 = enevec[0] / cosh(etavec[0]);
        double px0 = et0  * cos(phivec[0]);
        double py0 = et0  * sin(phivec[0]);
        double pz0 = et0  * sinh(etavec[0]);
        double et1 = enevec[1] / cosh(etavec[1]);
        double px1 = et1  * cos(phivec[1]);
        double py1 = et1  * sin(phivec[1]);
        double pz1 = et1  * sinh(etavec[1]);
         
        double dijetmass2 = (enevec[0]+enevec[1])*(enevec[0]+enevec[1])
          - (px1+px0)*(px1+px0) - (py1+py0)*(py1+py0) - (pz1+pz0)*(pz1+pz0);
 
	LogDebug("ValidHcal") << " JetAnalysis ===> Di-jet massSQ "
			      << dijetmass2;

        double dijetmass;
        if(dijetmass2 >= 0.) dijetmass = sqrt(dijetmass2);
        else                 dijetmass = -sqrt(-1. * dijetmass2);

        product.fillDiJets(dijetmass);
 
	LogDebug("ValidHcal") << " JetAnalysis ===> after fillDiJets";
      }
    }
  }
}

//---------------------------------------------------
void SimG4HcalValidation::fetchHits(PHcalValidInfoLayer& product) {

  LogDebug("ValidHcal") << "Enter SimG4HcalValidation::fetchHits with "
			<< hitcache.size() << " hits";
  int nHit = hitcache.size();
  int hit  = 0;
  int i;
  std::vector<CaloHit>::iterator itr;
  std::vector<CaloHit*> lhits(nHit);
  for (i = 0, itr = hitcache.begin(); itr != hitcache.end(); i++, itr++) {
    uint32_t unitID=itr->id();
    int   subdet, zside, group, ieta, iphi, lay;
    HcalTestNumbering::unpackHcalIndex(unitID,subdet,zside,group,
				       ieta,iphi,lay);
    subdet = itr->det();
    lay    = itr->layer();
    group  = (subdet&15)<<20;
    group += ((lay-1)&31)<<15;
    group += (zside&1)<<14;
    group += (ieta&127)<<7;
    group += (iphi&127);
    itr->setId(group);
    lhits[i] = &hitcache[i];
    LogDebug("ValidHcal") << "SimG4HcalValidation::fetchHits:Original " << i 
			  << " "  << hitcache[i] << "\n"
			  << "SimG4HcalValidation::fetchHits:Copied   " << i 
			  << " " << *lhits[i];
  }
  sort(lhits.begin(),lhits.end(),CaloHitIdMore());
  std::vector<CaloHit*>::iterator k1, k2;
  for (i = 0, k1 = lhits.begin(); k1 != lhits.end(); i++, k1++)
    LogDebug("ValidHcal") << "SimG4HcalValidation::fetchHits:Sorted " << i 
			  << " " << **k1;
  int nHits = 0;
  for (i = 0, k1 = lhits.begin(); k1 != lhits.end(); i++, k1++) {
    double       ehit  = (**k1).e();
    double       t     = (**k1).t();
    uint32_t     unitID= (**k1).id();
    int          jump  = 0;
    LogDebug("ValidHcal") << "SimG4HcalValidation::fetchHits:Start " << i 
			  << " U/T/E" << " 0x" << std::hex << unitID 
			  << std::dec << " "  << t << " " << ehit;
    for (k2 = k1+1; k2 != lhits.end() && (t-(**k2).t()) < 1 &&
	   (t-(**k2).t()) > -1 && unitID == (**k2).id(); k2++) {
      ehit += (**k2).e();
      LogDebug("ValidHcal") << "\t + " << (**k2).e();
      jump++;
    }
    LogDebug("ValidHcal") << "\t = " << ehit << " in " << jump;

    double eta  = (*k1)->eta();
    double phi  = (*k1)->phi();
    int lay     = ((unitID>>15)&31) + 1;
    int subdet  = (unitID>>20)&15;
    int zside   = (unitID>>14)&1;
    int ieta    = (unitID>>7)&127;
    int iphi    = (unitID)&127;
 
    // All hits in cache
    product.fillHits(nHits, lay, subdet, eta, phi, ehit, t);
    nHits++;

    LogDebug("ValidHcal") << "SimG4HcalValidation::fetchHits:Hit " << nHits 
			  << " " << i << " ID 0x" << std::hex << unitID 
			  << "   det " <<  std::dec  << subdet << " " << lay 
			  << " " << zside << " " << ieta << " " << iphi 
			  << " Time " << t << " E " << ehit;

    i  += jump;
    k1 += jump;
  }

  LogDebug("ValidHcal") << "SimG4HcalValidation::fetchHits called with "
			<< nHit << " hits" << " and writes out " << nHits
			<< '(' << hit << ") hits";

}
//---------------------------------------------------
void SimG4HcalValidation::clear(){
   hitcache.erase( hitcache.begin(), hitcache.end()); 
}

//---------------------------------------------------
void SimG4HcalValidation::collectEnergyRdir (const double eta0, 
					     const double phi0) {

  std::vector<CaloHit> * hits = &hitcache;
  std::vector<CaloHit>::iterator hit_itr;

  double sume = 0., sumh = 0., sumho = 0.;  

  for (hit_itr = hits->begin(); hit_itr < hits->end(); hit_itr++) {

    double e   = hit_itr->e();
    double eta = hit_itr->eta();
    double phi = hit_itr->phi();
    
    int type  = hit_itr->det();

    double r =  jetf->rDist(eta0, phi0, eta, phi);
    if (r < coneSize) {
      if (type == 10 || type == 11 || type == 12) {
	sume += e;
      } else           {
	sumh += e;
	if (type  == static_cast<int>(HcalBarrel) &&
	    hit_itr->layer() > 17) sumho += e;
      }
    }
  }
  
  een  = sume;
  hen  = sumh;
  hoen = sumho;
}

//---------------------------------------------------
double SimG4HcalValidation::getHcalScale(std::string det, int layer) const {

  double tmp = 0.;
    
  if (det == "HB")                         {
    tmp = scaleHB[layer];
  } else if (det == "HES" || det == "HED") {
    tmp = scaleHE[layer];
  } else if (det == "HF")                  {
    tmp = scaleHF[layer];
  }    
  
  return tmp;
}
