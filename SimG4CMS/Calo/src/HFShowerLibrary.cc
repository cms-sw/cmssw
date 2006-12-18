///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
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
								emTree(0),
								hadTree(0),
								nPhoton(0),
								nHit(0), 
								npe(0) {
  

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  probMax                 = m_HF.getParameter<double>("ProbMax");

  edm::ParameterSet m_HS= p.getParameter<edm::ParameterSet>("HFShowerLibrary");
  edm::FileInPath fp       = m_HS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  std::string emTree_name  = m_HS.getParameter<std::string>("TreeEMID");
  std::string hadTree_name = m_HS.getParameter<std::string>("TreeHadID");

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  bool format = true;
  if (fp.relativePath() == "vcal5x5.root") format = false;

  if (!hf->IsOpen()) { 
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree 
			      << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogInfo("HFShower") << "HFShowerLibrary: opening " << nTree 
			     << " successfully"; 
  }

  emTree  = (TTree *) hf->Get(emTree_name.c_str());
  emTree->Print();
  hadTree = (TTree *) hf->Get(hadTree_name.c_str());
  hadTree->Print();
  edm::LogInfo("HFShower") << "HFShowerLibrary:Ntuple " << emTree_name 
			   << " has " << emTree->GetEntries() 
			   << " entries and Ntuple "  << hadTree_name 
			   << " has " << hadTree->GetEntries() << " entries";

  //Packing parameters
  TTree * packing = (TTree *) hf->Get("Packing");
  if (packing) {
    loadPacking(packing);
    edm::LogInfo("HFShower") << "HFShowerLibrary::XOffset: " << xOffset 
			     << " XMultiplier: " << xMultiplier << " XScale: " 
			     << xScale << " YOffset: " << yOffset 
			     << " YMultiplier: " << yMultiplier << " YScale: " 
			     << yScale  << " ZOffset: " << zOffset 
			     << " ZMultiplier: " << zMultiplier << " ZScale: " 
			     << zScale;
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: Packing Branch does not"
			      << " exist" ;
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Packing information absent\n";
  } 

  TTree * evtinfo = (TTree *) hf->Get("EventInfo");
  if (evtinfo) {
    loadEventInfo(evtinfo, format);
    edm::LogInfo("HFShower") << "HFShowerLibrary: Library " << libVers 
			     << " ListVersion "	<< listVersion 
			     << " Events Total " << totEvents << " and "
			     << evtPerBin << " per bin";
    edm::LogInfo("HFShower") << "HFShowerLibrary: Energies (GeV) with " 
			     << nMomBin	<< " bins";
    for (int i=0; i<nMomBin; i++)
      edm::LogInfo("HFShower") << "HFShowerLibrary: pmom[" << i << "] = "
			       << pmom[i]/GeV << " GeV";
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: EvtInfo Branch does not"
			      << " exist";
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Event information absent\n";
  } 

  edm::LogInfo("HFShower") << "HFShowerLibrary: Maximum probability cut off " 
			   << probMax;
  
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
  
  fibre = new HFFibre(cpv);
}

HFShowerLibrary::~HFShowerLibrary() {
  if (hf)     hf->Close();
  if (fibre)  delete   fibre;  fibre  = 0;
}

int HFShowerLibrary::getHits(G4Step * aStep) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track    = aStep->GetTrack();   
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();

  double tSlice = (postStepPoint->GetGlobalTime())/nanosecond;
  double pin    = preStepPoint->GetTotalEnergy();
  double sphi   = sin(hitPoint.phi());
  double cphi   = cos(hitPoint.phi());
  double ctheta = cos(hitPoint.theta());
  double stheta = sin(hitPoint.theta());

  double xint =   hitPoint.x(); 
  double yint =   hitPoint.y(); 
  double zint =   hitPoint.z(); 

  LogDebug("HFShower") << "HFShowerLibrary: getHits " << partType
		       << " of energy " << pin/GeV << " GeV" 
                       << " in.Pos x,y,z = " << xint << "," << yint << "," 
                       << zint << "   sphi,cphi,stheta,ctheta  =" 
                       << sphi << "," << cphi << ","   
                       << stheta << "," << ctheta ; 
    
                       
  if (partType == "pi0" || partType == "eta" || partType == "nu_e" ||
      partType == "nu_mu" || partType == "nu_tau" || partType == "anti_nu_e" ||
      partType == "anti_nu_mu" || partType == "anti_nu_tau" || 
      partType == "geantino") {
    return -1;
  } else if (partType == "e-" || partType == "e+" || partType == "gamma" ) {
    if (pin<pmom[nMomBin-1]) {
      interpolate(emTree, pin);
    } else {
      extrapolate(emTree, pin);
    }
  } else {
    if (pin<pmom[nMomBin-1]) {
      interpolate(hadTree, pin);
    } else {
      extrapolate(hadTree, pin);
    }
  }
    
  nHit = 0;
  if (npe > 0) {
    hit.clear(); hit.resize(npe);
  }
  for (int i = 0; i < npe; i++) {
    LogDebug("HFShower") << "HFShowerLibrary: Hit " << i << " position " 
			 << pe[i].x << ", " << pe[i].y << ", " << pe[i].z 
			 << " Lambda " <<pe[i].lambda << " Time " <<pe[i].time;
    double zv = (pe[i].z >= 0 ? pe[i].z : -pe[i].z);
    if (zv <= gpar[1] && pe[i].lambda > 0 &&
	(pe[i].z >= 0 || pe[i].z <= -gpar[0])) {
      int depth = 1;
      if (pe[i].z < 0) depth = 2;


      // Updated coordinate transformation from local
      //  back to global using two Euler angles: phi and theta
      double pex = pe[i].x;
      double pey = pe[i].y;

      double xx = pex*ctheta*cphi - pey*sphi + zv*stheta*cphi; 
      double yy = pex*ctheta*sphi + pey*cphi + zv*stheta*sphi;
      double zz = -pex*stheta + zv*ctheta;

      // Original transformation
      /*
      double xx = (pe[i].x)*(ctheta + (1.-ctheta)*sphi*sphi) -
	(pe[i].y)*sphi*cphi*(1.-ctheta) + zv*cphi*stheta;
      double yy = (pe[i].y)*(ctheta + (1.-ctheta)*cphi*cphi) -
	(pe[i].x)*sphi*cphi*(1.-ctheta) + zv*sphi*stheta;
      double zz =-(pe[i].x)*cphi*stheta + (pe[i].y)*sphi*stheta +zv*ctheta;
      */

      G4ThreeVector pos = hitPoint + G4ThreeVector(xx,yy,zz);

      zv = gpar[1] - zv;
      double r  = pos.perp();
      double p  = fibre->attLength(pe[i].lambda);
      double fi = pos.phi();
      if (fi < 0) fi += twopi;
      int    isect = int(fi/dphi) + 1;
      isect        = (isect + 1) / 2;
      double dfi   = ((isect*2-1)*dphi - fi);
      if (dfi < 0) dfi = -dfi;
      double dfir  = r * sin(dfi);
      LogDebug("HFShower") << "HFShowerLibrary: Position " << xx << ", " << yy 
			   << ", "  << zz << ": " << pos << " R " << r 
			   << " Phi " << fi << " Section " << isect 
			   << " R*Dfi " << dfir;
      zz           = ((pos.z()) >= 0 ? (pos.z()) : -(pos.z()));
      double r1    = G4UniformRand();
      double r2    = G4UniformRand();
      LogDebug("HFShower") << "                   rLimits " << rInside(r)
			   << " attenuation " << r1 <<":" << exp(-p*zv) 
			   << " r2 " << r2 << " rDfi " << gpar[5] << " zz " 
			   << zz << " zLim " << gpar[4] << ":" 
			   << gpar[4]+gpar[1];

      LogDebug("HFShower") << "  rInside(r) :" << rInside(r) 
                           << "  r1 <= exp(-p*zv) :" <<  (r1 <= exp(-p*zv))
                           << "  r2 <= probMax :" << (r2 <= probMax)
                           << "  dfir > gpar[5] :" << (dfir > gpar[5])
                           << "  zz >= gpar[4] :" <<  (zz >= gpar[4])
			   << "  zz <= gpar[4]+gpar[1] :" 
			   << (zz <= gpar[4]+gpar[1]);   

      if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax && dfir > gpar[5] &&
	  zz >= gpar[4] && zz <= gpar[4]+gpar[1]) {
	hit[nHit].position = pos;
	hit[nHit].depth    = depth;
	hit[nHit].time     = (tSlice + (pe[i].time));
	LogDebug("HFShower") << "HFShowerLibrary: Final Hit " << nHit 
			     <<" position " << (hit[nHit].position) <<" Depth "
			     <<(hit[nHit].depth) <<" Time " <<(hit[nHit].time);
	nHit++;
      }
      else  LogDebug("HFShower") << " REJECTED !!!";
    }
  }

  LogDebug("HFShower") << "HFShowerLibrary: Total Hits " << nHit;
  if (nHit > npe)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Hit buffer " << npe 
				<< " smaller than " << nHit << " Hits";
  return nHit;

}

G4ThreeVector HFShowerLibrary::getPosHit(int i) {

  G4ThreeVector pos;
  if (i < nHit) pos = (hit[i].position);
  LogDebug("HFShower") << " HFShowerLibrary: PosHit (" << i << "/" << nHit 
		       << ") " << pos;
  return pos;
}

int HFShowerLibrary::getDepth(int i) {

  int depth = 0;
  if (i < nHit) depth = (hit[i].depth);
  LogDebug("HFShower") << " HFShowerLibrary: Depth (" << i << "/" << nHit 
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


int HFShowerLibrary::getPhoton(TTree* tree, int record) {

  int nph = 0;
  if (tree && record > 0) {
    tree->SetBranchAddress("NPH", &nph);
    int nrc = record-1;
    tree->GetEntry(nrc);
  }
  return nph;
}

void HFShowerLibrary::getRecord(TTree* tree, int record) {

  int nrc = record-1;
  nPhoton = getPhoton(tree, record);
  if (nPhoton > 0 && tree && nrc >= 0) {
    photon.clear(); photon.resize(nPhoton);
    LogDebug("HFShower") << "HFShowerLibrary: Record " << record << " with "
			 << nPhoton << " photons";
    int nph, coor[10000], wl[10000], time[10000];
    tree->SetBranchAddress("XYZ", &coor);
    tree->SetBranchAddress("L",   &wl);
    tree->SetBranchAddress("T",   &time);
    tree->SetBranchAddress("NPH", &nph);
    tree->GetEntry(nrc);
    for (int j = 0; j < nPhoton; j++) {
      photon[j].xyz    = coor[j];
      photon[j].lambda = wl[j];
      photon[j].time   = time[j];
      LogDebug("HFShower") << "Photon " << j << " xyz " << photon[j].xyz 
			   << " L " << photon[j].lambda << " Time " 
			   << photon[j].time;
    }
  }
}

void HFShowerLibrary::loadPacking(TTree* tree) {

  tree->SetBranchAddress("XOffset",     &xOffset);
  tree->SetBranchAddress("XMultiplier", &xMultiplier);
  tree->SetBranchAddress("XScale",      &xScale);
  tree->SetBranchAddress("YOffset",     &yOffset);
  tree->SetBranchAddress("YMultiplier", &yMultiplier);
  tree->SetBranchAddress("YScale",      &yScale);
  tree->SetBranchAddress("ZOffset",     &zOffset);
  tree->SetBranchAddress("ZMultiplier", &zMultiplier);
  tree->SetBranchAddress("ZScale",      &zScale);
  if (tree->GetEntries() > 0) {
    tree->GetEntry(0);
  }
}

void HFShowerLibrary::loadEventInfo(TTree* tree, bool format) {

  int v[200];
  libVers     = -1;
  listVersion = 0;
  if (format) {
    tree->SetBranchAddress("LIBVERS",     &libVers);
    tree->SetBranchAddress("PHYLISTVERS", &listVersion);
  }
  tree->SetBranchAddress("NUMBINS",     &nMomBin);
  tree->SetBranchAddress("EVTNUMPERBIN",&evtPerBin);
  tree->SetBranchAddress("TOTEVTS",     &totEvents);
  tree->SetBranchAddress("ENERGIES",    &v);
  if (tree->GetEntries() > 0) {
    tree->GetEntry(0);
    for (int i=0; i<nMomBin; i++) {
      double val = ((double)(v[i]))*GeV;
      pmom.push_back(val);
    }
  } 
}

void HFShowerLibrary::interpolate(TTree * tree, double pin) {

  int nentry = int(tree->GetEntries());
  int nevent = nentry/nMomBin;
  LogDebug("HFShower") << "HFShowerLibrary:: Interpolate for Energy " <<pin/GeV
		       << " GeV with " << nMomBin << " momentum bins and " 
		       << nevent << " entries/bin -- total " << nentry;
  int irc[2], j;
  double w = 0.;
  double r = G4UniformRand();

  if (pin<pmom[0]) {
    w = pin/pmom[0];
    irc[1] = int(nevent*r) + 1;
    irc[0] = 0;
  } else {
    for (j=0; j<nMomBin-1; j++) {
      if (pin >= pmom[j] && pin < pmom[j+1]) {
	w = (pin-pmom[j])/(pmom[j+1]-pmom[j]);
	if (j == nMomBin-2) { 
	  irc[1] = int(nevent*0.5*r);
	} else {
	  irc[1] = int(nevent*r);
	}
	irc[1] += (j+1)*nevent + 1;
	r = G4UniformRand();
	irc[0] = int(nevent*r) + 1 + j*nevent;
	if (irc[0]<0) {
	  edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = "
				      << irc[0] << " now set to 0";
	  irc[0] = 0;
	} else if (irc[0] > nentry) {
	  edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = "
				      << irc[0] << " now set to "<< nentry;
	  irc[0] = nentry;
	}
      }
    }
  }
  if (irc[1]<1) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " 
				<< irc[1] << " now set to 1";
    irc[1] = 1;
  } else if (irc[1] > nentry) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " 
				<< irc[1] << " now set to "<< nentry;
    irc[1] = nentry;
  }

  LogDebug("HFShower") << "HFShowerLibrary:: Select records " << irc[0] 
		       << " and " << irc[1] << " with weights " << 1-w 
		       << " and " << w;
  int npold = getPhoton (tree, irc[1]);
  if (irc[0]>0) {
    getRecord (tree, irc[0]);
    npold += nPhoton;
  }
  if (npold <= 0) npold = 1;
  pe.clear(); pe.resize(npold);

  npe = 0;
  if (irc[0]>0) {
    for (j=0; j<nPhoton; j++) {
      r = G4UniformRand();
      if (r > w) {
	storePhoton (j);
	npe++;
      }
    }
  }

  getRecord (tree, irc[1]);
  for (j=0; j<nPhoton; j++) {
    r = G4UniformRand();
    if (r < w) {
      storePhoton (j);
      npe++;
    }
  }

  if (npe > npold || npold == 0)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Interpolation error =="
				<< " buffer " << npold << " filled " << npe 
				<< " *****";
  LogDebug("HFShower") << "HFShowerLibrary: Interpolation gives " << npe
		       << " Photons == buffer " << npold;
  for (j=0; j<npe; j++) {
    LogDebug("HFShower") << "Photon " << j << " X " << (pe[j].x) << " Y " 
			 << (pe[j].y) << " Z " << (pe[j].z) << " Lam " 
			 << (pe[j].lambda) << " T " << (pe[j].time);
  }
}

void HFShowerLibrary::extrapolate(TTree * tree, double pin) {

  int nentry = int(tree->GetEntries());
  int nevent = nentry/nMomBin;
  int nrec   = int(pin/pmom[nMomBin-1]);
  double w   = (pin - pmom[nMomBin-1]*nrec)/pmom[nMomBin-1];
  nrec++;
  LogDebug("HFShower") << "HFShowerLibrary:: Extrapolate for Energy " << pin 
		       << " GeV with " << nMomBin << " momentum bins and " 
		       << nevent << " entries/bin -- total " << nentry 
		       << " using " << nrec << " records";
  std::vector<int> irc(nrec);
  int    j, ir;
  double r;

  npe = 0;
  int npold = 0;
  for (ir=0; ir<nrec; ir++) {
    r = G4UniformRand();
    irc[ir] = int(nevent*0.5*r) +(nMomBin-1)*nevent + 1;
    if (irc[ir]<1) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to 1";
      irc[ir] = 1;
    } else if (irc[ir] > nentry) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir 
				  << "] = " << irc[ir] << " now set to "
				  << nentry;
      irc[ir] = nentry;
    }
    LogDebug("HFShower") << "Record [" << ir << "] = " << irc[ir] 
			 << " npold = " << npold;
    npold += getPhoton (tree, irc[ir]);
  }
  LogDebug("HFShower") << "HFShowerLibrary:: uses " << npold << " photons";
  if (npold <= 0) npold = 1;
  pe.clear(); pe.resize(npold);

  for (ir=0; ir<nrec; ir++) {
    getRecord (tree, irc[ir]);
    for (j=0; j<nPhoton; j++) {
      r = G4UniformRand();
      if (ir != nrec-1 || r < w) {
	storePhoton (j);
	npe++;
      }
    }
  }

  if (npe > npold || npold == 0)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Extrapolation error =="
				<< " buffer " << npold << " filled " << npe 
				<< " *****";
  LogDebug("HFShower") << "HFShowerLibrary: Extrapolation gives " << npe
		       << " Photons == buffer "  << npold;
  for (j=0; j<npe; j++) {
    LogDebug("HFShower") << "Photon " << j << " X " << (pe[j].x) << " Y " 
			 << (pe[j].y) << " Z " << (pe[j].z) << " Lam " 
			 << (pe[j].lambda) << " T " << (pe[j].time);
  }
}

void HFShowerLibrary::storePhoton(int j) {

  int ix = (photon[j].xyz)/xMultiplier;
  int iy = (photon[j].xyz)/yMultiplier - ix*yMultiplier;
  int iz = (photon[j].xyz)/zMultiplier - ix*xMultiplier - iy*yMultiplier;
  pe[npe].x      = (ix/xScale - xOffset)*cm + 5.; //to account for wrong offset
  pe[npe].y      = (iy/yScale - yOffset)*cm + 35.;//idem 
  pe[npe].z      = (iz/zScale - zOffset)*cm;
  pe[npe].lambda = (photon[j].lambda);
  pe[npe].time   = (photon[j].time)/100.;
  LogDebug("HFShower") << "HFShowerLibrary: storePhoton " << j << " npe " <<npe
		       << " ixyz " << (photon[j].xyz) << " x " << (pe[npe].x)
		       << " y " << (pe[npe].y) << " z " << (pe[npe].z) << " l "
		       << (pe[npe].lambda) << " t " << (pe[npe].time);
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
