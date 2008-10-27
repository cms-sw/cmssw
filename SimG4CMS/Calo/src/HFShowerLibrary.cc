///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"

HFShowerLibrary::HFShowerLibrary(std::string & name, const DDCompactView & cpv,
				 edm::ParameterSet const & p) : fibre(0),hf(0),
								emBranch(0),
								hadBranch(0),
								nHit(0), 
								npe(0) {
  

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  probMax                 = m_HF.getParameter<double>("ProbMax");

  edm::ParameterSet m_HS= p.getParameter<edm::ParameterSet>("HFShowerLibrary");
  edm::FileInPath fp       = m_HS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  backProb                 = m_HS.getParameter<double>("BackProbability");  
  std::string emName       = m_HS.getParameter<std::string>("TreeEMID");
  std::string hadName      = m_HS.getParameter<std::string>("TreeHadID");
  std::string branchEvInfo = m_HS.getUntrackedParameter<std::string>("BranchEvt","HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo");
  std::string branchPre    = m_HS.getUntrackedParameter<std::string>("BranchPre","HFShowerPhotons_hfshowerlib_");
  std::string branchPost   = m_HS.getUntrackedParameter<std::string>("BranchPost","_R.obj");
  verbose                  = m_HS.getUntrackedParameter<bool>("Verbosity",false);

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  if (!hf->IsOpen()) { 
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree 
			      << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogInfo("HFShower") << "HFShowerLibrary: opening " << nTree 
			     << " successfully"; 
  }

  TTree* event = (TTree *) hf ->Get("Events");
  if (event) {
    std::string info = branchEvInfo + branchPost;
    TBranch *evtInfo = event->GetBranch(info.c_str());
    if (evtInfo) {
      loadEventInfo(evtInfo);
    } else {
      edm::LogError("HFShower") << "HFShowerLibrary: HFShowerLibrayEventInfo"
				<< " Branch does not exist in Event";
      throw cms::Exception("Unknown", "HFShowerLibrary")
	<< "Event information absent\n";
    }
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: Events Tree does not "
			      << "exist";
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Events tree absent\n";
  }
  
  edm::LogInfo("HFShower") << "HFShowerLibrary: Library " << libVers 
			   << " ListVersion "	<< listVersion 
			   << " Events Total " << totEvents << " and "
			   << evtPerBin << " per bin";
  edm::LogInfo("HFShower") << "HFShowerLibrary: Energies (GeV) with " 
			   << nMomBin	<< " bins";
  for (int i=0; i<nMomBin; i++)
    edm::LogInfo("HFShower") << "HFShowerLibrary: pmom[" << i << "] = "
			     << pmom[i]/GeV << " GeV";

  std::string nameBr = branchPre + emName + branchPost;
  emBranch         = event->GetBranch(nameBr.c_str());
  if (verbose) emBranch->Print();
  nameBr           = branchPre + hadName + branchPost;
  hadBranch        = event->GetBranch(nameBr.c_str());
  if (verbose) hadBranch->Print();
  edm::LogInfo("HFShower") << "HFShowerLibrary:Branch " << emName 
			   << " has " << emBranch->GetEntries() 
			   << " entries and Branch " << hadName 
			   << " has " << hadBranch->GetEntries() 
			   << " entries";
  edm::LogInfo("HFShower") << "HFShowerLibrary::No packing information -"
			   << " Assume x, y, z are not in packed form";

  edm::LogInfo("HFShower") << "HFShowerLibrary: Maximum probability cut off " 
			   << probMax << "  Back propagation of light prob. "
                           << backProb ;
  
  G4String attribute = "ReadOutName";
  G4String value     = name;
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  if (dodet) {
    DDsvalues_type sv(fv.mergedSpecifics());

    //Radius (minimum and maximum)
    int nR     = -1;
    std::vector<double> rTable = getDDDArray("rTable",sv,nR);
    rMin = rTable[0];
    rMax = rTable[nR-1];
    edm::LogInfo("HFShower") << "HFShowerLibrary: rMIN " << rMin/cm 
			     << " cm and rMax " << rMax/cm;

    //Delta phi
    int nEta   = -1;
    std::vector<double> etaTable = getDDDArray("etaTable",sv,nEta);
    int nPhi   = nEta + nR - 2;
    std::vector<double> phibin   = getDDDArray("phibin",sv,nPhi);
    dphi       = phibin[nEta-1];
    edm::LogInfo("HFShower") << "HFShowerLibrary: (Half) Phi Width of wedge " 
			     << dphi/deg;

    //Special Geometry parameters
    int ngpar = 7;
    gpar      = getDDDArray("gparHF",sv,ngpar);
    edm::LogInfo("HFShower") << "HFShowerLibrary: " << ngpar << " gpar (cm)";
    for (int ig=0; ig<ngpar; ig++)
      edm::LogInfo("HFShower") << "HFShowerLibrary: gpar[" << ig << "] = "
			       << gpar[ig]/cm << " cm";
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: cannot get filtered "
			      << " view for " << attribute << " matching "
			      << name;
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "cannot match " << attribute << " to " << name <<"\n";
  }
  
  fibre = new HFFibre(name, cpv, p);
  emPDG = epPDG = gammaPDG = 0;
  pi0PDG = etaPDG = nuePDG = numuPDG = nutauPDG= 0;
  anuePDG= anumuPDG = anutauPDG = geantinoPDG = 0;
}

HFShowerLibrary::~HFShowerLibrary() {
  if (hf)     hf->Close();
  if (fibre)  delete   fibre;  fibre  = 0;
}

void HFShowerLibrary::initRun(G4ParticleTable * theParticleTable) {

  G4String parName;
  emPDG = theParticleTable->FindParticle(parName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(parName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(parName="gamma")->GetPDGEncoding();
  pi0PDG = theParticleTable->FindParticle(parName="pi0")->GetPDGEncoding();
  etaPDG = theParticleTable->FindParticle(parName="eta")->GetPDGEncoding();
  nuePDG = theParticleTable->FindParticle(parName="nu_e")->GetPDGEncoding();
  numuPDG = theParticleTable->FindParticle(parName="nu_mu")->GetPDGEncoding();
  nutauPDG= theParticleTable->FindParticle(parName="nu_tau")->GetPDGEncoding();
  anuePDG = theParticleTable->FindParticle(parName="anti_nu_e")->GetPDGEncoding();
  anumuPDG= theParticleTable->FindParticle(parName="anti_nu_mu")->GetPDGEncoding();
  anutauPDG= theParticleTable->FindParticle(parName="anti_nu_tau")->GetPDGEncoding();
  geantinoPDG= theParticleTable->FindParticle(parName="geantino")->GetPDGEncoding();
  LogDebug("HFShower") << "HFShowerLibrary: Particle codes for e- = " << emPDG
		       << ", e+ = " << epPDG << ", gamma = " << gammaPDG 
		       << ", pi0 = " << pi0PDG << ", eta = " << etaPDG
		       << ", geantino = " << geantinoPDG << "\n        nu_e = "
		       << nuePDG << ", nu_mu = " << numuPDG << ", nu_tau = "
		       << nutauPDG << ", anti_nu_e = " << anuePDG
		       << ", anti_nu_mu = " << anumuPDG << ", anti_nu_tau = "
		       << anutauPDG;
}

int HFShowerLibrary::getHits(G4Step * aStep) {


  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track    = aStep->GetTrack();
  // Get Z-direction 
  const G4DynamicParticle *aParticle = track->GetDynamicParticle();
  G4ThreeVector momDir = aParticle->GetMomentumDirection();
  //  double mom = aParticle->GetTotalMomentum();

  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();
  int           parCode  = track->GetDefinition()->GetPDGEncoding();

  double tSlice = (postStepPoint->GetGlobalTime())/nanosecond;
  double pin    = preStepPoint->GetTotalEnergy();
  
  double px   = momDir.x(); 
  double py   = momDir.y(); 
  double pz   = momDir.z(); 

  double xint = hitPoint.x(); 
  double yint = hitPoint.y(); 
  double zint = hitPoint.z(); 

  // if particle moves from interaction point or "backwards (halo)
  int backward = 0;
  if(pz * zint < 0.) backward = 1;
  
  double sphi   = sin(momDir.phi());
  double cphi   = cos(momDir.phi());
  double ctheta = cos(momDir.theta());
  double stheta = sin(momDir.theta());
 
  /*  // Obsolete piece, only valid if particle comes from IP...
  double sphi   = sin(hitPoint.phi());
  double cphi   = cos(hitPoint.phi());
  double ctheta = cos(hitPoint.theta());
  double stheta = sin(hitPoint.theta());
  */

  LogDebug("HFShower") << "HFShowerLibrary: getHits " << partType
		       << " of energy " << pin/GeV << " GeV"
		       << "  dir.orts " << px << ", " << py << ", " << pz
                       << "  Pos x,y,z = " << xint << "," << yint << "," 
                       << zint << "   sphi,cphi,stheta,ctheta  =" 
                       << sphi << "," << cphi << ","   
                       << stheta << "," << ctheta ; 
    
                       
  if (parCode == pi0PDG || parCode == etaPDG || parCode == nuePDG ||
      parCode == numuPDG || parCode == nutauPDG || parCode == anuePDG ||
      parCode == anumuPDG || parCode == anutauPDG || parCode == geantinoPDG) {
    return -1;
  } else if (parCode == emPDG || parCode == epPDG || parCode == gammaPDG ) {
    if (pin<pmom[nMomBin-1]) {
      interpolate(0, pin);
    } else {
      extrapolate(0, pin);
    }
  } else {
    if (pin<pmom[nMomBin-1]) {
      interpolate(1, pin);
    } else {
      extrapolate(1, pin);
    }
  }
    
  nHit = 0;
  if (npe > 0) {
    hit.clear(); hit.resize(npe);
  }
  for (int i = 0; i < npe; i++) {
    LogDebug("HFShower") << "HFShowerLibrary: Hit " << i << " " << pe[i];
    double zv = std::abs(pe[i].z()); // abs local z  
    if (zv <= gpar[1] && pe[i].lambda() > 0 &&
	(pe[i].z() >= 0 || pe[i].z() <= -gpar[0])) {
      int depth = 1;
      if(backward == 0) {            // fully valid only for "front" particles
	if (pe[i].z() < 0) depth = 2;// with "front"-simulated shower lib.
      }
      else {                         // for "backward" particles - almost equal
	double r = G4UniformRand();  // share between L and S fibers
        if(r > 0.5) depth = 2;
      } 
      

      // Updated coordinate transformation from local
      //  back to global using two Euler angles: phi and theta
      double pex = pe[i].x();
      double pey = pe[i].y();

      double xx = pex*ctheta*cphi - pey*sphi + zv*stheta*cphi; 
      double yy = pex*ctheta*sphi + pey*cphi + zv*stheta*sphi;
      double zz = -pex*stheta + zv*ctheta;

      G4ThreeVector pos = hitPoint + G4ThreeVector(xx,yy,zz);

      if(backward == 0) zv = gpar[1] - zv;      // remaining distance to PMT !

      double r  = pos.perp();
      double p  = fibre->attLength(pe[i].lambda());
      double fi = pos.phi();
      if (fi < 0) fi += twopi;
      int    isect = int(fi/dphi) + 1;
      isect        = (isect + 1) / 2;
      double dfi   = ((isect*2-1)*dphi - fi);
      if (dfi < 0) dfi = -dfi;
      double dfir  = r * sin(dfi);
      LogDebug("HFShower") << "HFShowerLibrary: Position shift " << xx 
			   << ", " << yy 
			   << ", "  << zz << ": " << pos << " R " << r 
			   << " Phi " << fi << " Section " << isect 
			   << " R*Dfi " << dfir;
      zz           = ((pos.z()) >= 0 ? (pos.z()) : -(pos.z()));
      double r1    = G4UniformRand();
      double r2    = G4UniformRand();
      double r3    = -9999.;
      if(backward) r3 = G4UniformRand();

      LogDebug("HFShower") << "  rLimits " << rInside(r)
			   << " attenuation " << r1 <<":" << exp(-p*zv) 
			   << " r2 " << r2 << " r3 " << r3 << " rDfi "  
                           << gpar[5] << " zz " 
			   << zz << " zLim " << gpar[4] << ":" 
			   << gpar[4]+gpar[1];
      LogDebug("HFShower") << "  rInside(r) :" << rInside(r) 
                           << "  r1 <= exp(-p*zv) :" <<  (r1 <= exp(-p*zv))
                           << "  r2 <= probMax :"    <<  (r2 <= probMax)
			   << "  r3 <= backProb :"   <<  (r3 <= backProb) 
                           << "  dfir > gpar[5] :"   <<  (dfir > gpar[5])
                           << "  zz >= gpar[4] :"    <<  (zz >= gpar[4])
			   << "  zz <= gpar[4]+gpar[1] :" 
			   << (zz <= gpar[4]+gpar[1]);   

      if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax && dfir > gpar[5] &&
	  zz >= gpar[4] && zz <= gpar[4]+gpar[1] && r3 <= backProb ){
	hit[nHit].position = pos;
	hit[nHit].depth    = depth;
	hit[nHit].time     = (tSlice+(pe[i].t())+(fibre->tShift(pos,depth,true)));
	LogDebug("HFShower") << "HFShowerLibrary: Final Hit " << nHit 
			     <<" position " << (hit[nHit].position) <<" Depth "
			     <<(hit[nHit].depth) <<" Time " <<(hit[nHit].time);
	nHit++;
      }
      else  LogDebug("HFShower") << " REJECTED !!!";
    }
  }

  LogDebug("HFShower") << "HFShowerLibrary: Total Hits " << nHit
		       << " out of " << npe << " PE";
  if (nHit > npe)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Hit buffer " << npe 
				<< " smaller than " << nHit << " Hits";
  return nHit;

}

G4ThreeVector HFShowerLibrary::getPosHit(int i) {

  G4ThreeVector pos;
  if (i < nHit) pos = (hit[i].position);
  LogDebug("HFShower") << " HFShowerLibrary: getPosHit (" << i << "/" << nHit 
		       << ") " << pos;
  return pos;
}

int HFShowerLibrary::getDepth(int i) {

  int depth = 0;
  if (i < nHit) depth = (hit[i].depth);
  LogDebug("HFShower") << " HFShowerLibrary: getDepth (" << i << "/" << nHit 
		       << ") "  << depth;
  return depth;
}

double HFShowerLibrary::getTSlice(int i) {
  
  double tim = 0.;
  if (i < nHit) tim = (hit[i].time);
  LogDebug("HFShower") << " HFShowerLibrary: Time (" << i << "/" << nHit 
		       << ") "  << tim;
  return tim;
}

bool HFShowerLibrary::rInside(double r) {

  if (r >= rMin && r <= rMax) return true;
  else                        return false;
}

void HFShowerLibrary::getRecord(int type, int record) {

  int nrc     = record-1;
  int nPhoton = 0;
  photon.clear();
  if (type > 0) {
    hadBranch->SetAddress(&photon);
    hadBranch->GetEntry(nrc);
  } else {
    emBranch->SetAddress(&photon);
    emBranch->GetEntry(nrc);
  }
  nPhoton = photon.size();
  LogDebug("HFShower") << "HFShowerLibrary::getRecord: Record " << record
		       << " of type " << type << " with " << nPhoton 
		       << " photons";
  for (int j = 0; j < nPhoton; j++) 
    LogDebug("HFShower") << "Photon " << j << photon[j];
}

void HFShowerLibrary::loadEventInfo(TBranch* branch) {

  std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
  branch->SetAddress(&eventInfoCollection);
  branch->GetEntry(0);
  edm::LogInfo("HFShower") << "HFShowerLibrary::loadEventInfo loads "
			   << " EventInfo Collection of size "
			   << eventInfoCollection.size() << " records";
  totEvents   = eventInfoCollection[0].totalEvents();
  nMomBin     = eventInfoCollection[0].numberOfBins();
  evtPerBin   = eventInfoCollection[0].eventsPerBin();
  libVers     = eventInfoCollection[0].showerLibraryVersion();
  listVersion = eventInfoCollection[0].physListVersion();
  pmom        = eventInfoCollection[0].energyBins();
  for (unsigned int i=0; i<pmom.size(); i++) 
    pmom[i] *= GeV;
}

void HFShowerLibrary::interpolate(int type, double pin) {

  LogDebug("HFShower") << "HFShowerLibrary:: Interpolate for Energy " <<pin/GeV
		       << " GeV with " << nMomBin << " momentum bins and " 
		       << evtPerBin << " entries/bin -- total " << totEvents;
  int irc[2];
  double w = 0.;
  double r = G4UniformRand();

  if (pin<pmom[0]) {
    w = pin/pmom[0];
    irc[1] = int(evtPerBin*r) + 1;
    irc[0] = 0;
  } else {
    for (int j=0; j<nMomBin-1; j++) {
      if (pin >= pmom[j] && pin < pmom[j+1]) {
	w = (pin-pmom[j])/(pmom[j+1]-pmom[j]);
	if (j == nMomBin-2) { 
	  irc[1] = int(evtPerBin*0.5*r);
	} else {
	  irc[1] = int(evtPerBin*r);
	}
	irc[1] += (j+1)*evtPerBin + 1;
	r = G4UniformRand();
	irc[0] = int(evtPerBin*r) + 1 + j*evtPerBin;
	if (irc[0]<0) {
	  edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = "
				      << irc[0] << " now set to 0";
	  irc[0] = 0;
	} else if (irc[0] > totEvents) {
	  edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = "
				      << irc[0] << " now set to "<< totEvents;
	  irc[0] = totEvents;
	}
      }
    }
  }
  if (irc[1]<1) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " 
				<< irc[1] << " now set to 1";
    irc[1] = 1;
  } else if (irc[1] > totEvents) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " 
				<< irc[1] << " now set to "<< totEvents;
    irc[1] = totEvents;
  }

  LogDebug("HFShower") << "HFShowerLibrary:: Select records " << irc[0] 
		       << " and " << irc[1] << " with weights " << 1-w 
		       << " and " << w;

  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir < 2; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
	r = G4UniformRand();
	if ((ir==0 && r > w) || (ir > 0 && r < w)) {
	  storePhoton (j);
	}
      }
    }
  }

  if (npe > npold || (npold == 0 && irc[0] > 0)) 
    edm::LogWarning("HFShower") << "HFShowerLibrary: Interpolation Warning =="
				<< " records " << irc[0] << " and " << irc[1]
				<< " gives a buffer of " << npold 
				<< " photons and fills " << npe << " *****";
  else
    LogDebug("HFShower") << "HFShowerLibrary: Interpolation == records " 
			 << irc[0] << " and " << irc[1] << " gives a "
			 << "buffer of " << npold << " photons and fills "
			 << npe << " PE";
  for (int j=0; j<npe; j++) {
    LogDebug("HFShower") << "Photon " << j << " " << pe[j];
  }
}

void HFShowerLibrary::extrapolate(int type, double pin) {

  int nrec   = int(pin/pmom[nMomBin-1]);
  double w   = (pin - pmom[nMomBin-1]*nrec)/pmom[nMomBin-1];
  nrec++;
  LogDebug("HFShower") << "HFShowerLibrary:: Extrapolate for Energy " << pin 
		       << " GeV with " << nMomBin << " momentum bins and " 
		       << evtPerBin << " entries/bin -- total " << totEvents 
		       << " using " << nrec << " records";
  std::vector<int> irc(nrec);

  for (int ir=0; ir<nrec; ir++) {
    double r = G4UniformRand();
    irc[ir] = int(evtPerBin*0.5*r) +(nMomBin-1)*evtPerBin + 1;
    if (irc[ir]<1) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to 1";
      irc[ir] = 1;
    } else if (irc[ir] > totEvents) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to "
				  << totEvents;
      irc[ir] = totEvents;
    } else {
      LogDebug("HFShower") << "HFShowerLibrary::Extrapolation use irc[" 
			   << ir  << "] = " << irc[ir];
    }
  }

  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir<nrec; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
	double r = G4UniformRand();
	if (ir != nrec-1 || r < w) {
	  storePhoton (j);
	}
      }
      LogDebug("HFShower") << "Record [" << ir << "] = " << irc[ir] 
			   << " npold = " << npold;
    }
  }
  LogDebug("HFShower") << "HFShowerLibrary:: uses " << npold << " photons";

  if (npe > npold || npold == 0)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Extrapolation Warning == "
				<< nrec << " records " << irc[0] << ", " 
				<< irc[1] << ", ... gives a buffer of " <<npold
				<< " photons and fills " << npe 
				<< " *****";
  else
    LogDebug("HFShower") << "HFShowerLibrary: Extrapolation == " << nrec
			 << " records " << irc[0] << ", " << irc[1] 
			 << ", ... gives a buffer of " << npold 
			 << " photons and fills " << npe << " PE";
  for (int j=0; j<npe; j++) {
    LogDebug("HFShower") << "Photon " << j << " " << pe[j];
  }
}

void HFShowerLibrary::storePhoton(int j) {

  pe.push_back(photon[j]);
  LogDebug("HFShower") << "HFShowerLibrary: storePhoton " << j << " npe " 
		       << npe << " " << pe[npe];
  npe++;
}

std::vector<double> HFShowerLibrary::getDDDArray(const std::string & str, 
						 const DDsvalues_type & sv, 
						 int & nmin) {

  LogDebug("HFShower") << "HFShowerLibrary:getDDDArray called for " << str 
		       << " with nMin " << nmin;

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HFShower") << value;
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HFShower") << "HFShowerLibrary : # of " << str 
				  << " bins " << nval << " < " << nmin 
				  << " ==> illegal";
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < nmin for array " << str << "\n";
      }
    } else {
      if (nval < 2) {
	edm::LogError("HFShower") << "HFShowerLibrary : # of " << str 
				  << " bins " << nval << " < 2 ==> illegal"
				  << " (nmin=" << nmin << ")";
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < 2 for array " << str << "\n";
      }
    }
    nmin = nval;

    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary : cannot get array " << str;
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "cannot get array " << str << "\n";
  }
}
