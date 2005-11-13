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

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Random/RandFlat.h"

#define debug

HFShowerLibrary::HFShowerLibrary(std::string & name, const DDCompactView & cpv,
				 edm::ParameterSet const & p) : fibre(0),hf(0),
								ixyz(0),l(0),
								it(0),nHit(0),
								posHit(0),
								depHit(0),
								timHit(0),
								xpe(0),ype(0),
								zpe(0),lpe(0),
								tpe(0) {

  //static SimpleConfigurable<string> pathN(envUtil("OSCAR_DATA_PATH",".").getEnv(),"HFShowerLibrary:FilePath");
  //static SimpleConfigurable<string> filename("hfshowerlibrary_g3.root","HFShowerLibrary:FileName");
  //FileInPath f1(pathN,filename);
  //string pTreeName = f1.name();
  //if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  //static SimpleConfigurable<string> pTreeEMID("h3","HFShowerLibrary:TreeEMID");
  //static SimpleConfigurable<string> pTreeHadID("h8","HFShowerLibrary:TreeHadID");
  //static SimpleConfigurable<float>  pProbmax(0.7268,"HFShowerLibrary:ProbMax");

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  verbosity        = m_HF.getParameter<int>("Verbosity");
  probMax          = m_HF.getParameter<double>("ProbMax");
  int iv           = verbosity/10;
  verbosity       %= 10;

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
    if (verbosity > 0)
      std::cout << "HFShowerLibrary: opening " << nTree << " failed " 
		<< std::endl; 
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Opening of " << pTreeName << " fails" <<"\n";
  } else {
    if (verbosity > 0)
      std::cout << "HFShowerLibrary: opening " << nTree << " successful " 
		<< std::endl;
  }

  emTree  = (TTree *) hf->Get(emTree_name.c_str());
  emTree->Print();
  hadTree = (TTree *) hf->Get(hadTree_name.c_str());
  hadTree->Print();
  if (verbosity > 0)
    std::cout << "HFShowerLibrary:Ntuple " << emTree_name << " has "
	      << emTree->GetEntries() << " entries and Ntuple " 
	      << hadTree_name << " has " << hadTree->GetEntries() 
	      << " entries " << std::endl;

  //Packing parameters
  TTree * packing = (TTree *) hf->Get("Packing");
  if (packing) {
    loadPacking(packing);
    if (verbosity > 0)
      std::cout << "HFShowerLibrary::XOffset: " << xOffset 
		<< " XMultiplier: " << xMultiplier << " XScale: " << xScale 
		<< " YOffset: " << yOffset << " YMultiplier: " << yMultiplier 
		<< " YScale: " << yScale  << " ZOffset: " << zOffset 
		<< " ZMultiplier: " << zMultiplier << " ZScale: " << zScale 
		<< std::endl;
  } else {
    if (verbosity > 0)
      std::cout << "HFShowerLibrary: Packing Branch does not exist" 
		<< std::endl;
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Packing information absent" <<"\n";
  } 

  TTree * evtinfo = (TTree *) hf->Get("EventInfo");
  if (evtinfo) {
    loadEventInfo(evtinfo, format);
    if (verbosity > 0) {
      std::cout << "HFShowerLibrary: Library " << libVers << " ListVersion "
		<< listVersion << " Events Total " << totEvents << " and "
		<< evtPerBin << " per bin" << std::endl;
      std::cout << "HFShowerLibrary: Energies (GeV) with " << nMomBin
		<< " bins" << std::endl << "                   ";
      for (int i=0; i<nMomBin; i++)
	std::cout << " " << pmom[i]/GeV;
      std::cout << std::endl;
    }
  } else {
    if (verbosity > 0)
      std::cout << "HFShowerLibrary: EvtInfo Branch does not exist" 
		<< std::endl;
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Event information absent" <<"\n";
  } 

  if (verbosity > 0)
    std::cout << "HFShowerLibrary: Maximum probability cut off " << probMax
	      << std::endl;

  G4String attribute = "ReadOutName";
  G4String value     = name;
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  //Radius (minimum and maximum)
  int nR     = -1;
  std::vector<double> rTable = getDDDArray("rTable",sv,nR);
  rMin = rTable[0];
  rMax = rTable[nR-1];
  if (verbosity > 0)
    std::cout << "HFShowerLibrary: rMIN " << rMin/cm << " cm and rMax "
	      << rMax/cm << std::endl;

  //Delta phi
  int nEta   = -1;
  std::vector<double> etaTable = getDDDArray("etaTable",sv,nEta);
  int nPhi   = nEta + nR - 2;
  std::vector<double> phibin   = getDDDArray("phibin",sv,nPhi);
  dphi       = phibin[nEta-1];
  if (verbosity > 0)
    std::cout << "HFShowerLibrary: (Half) Phi Width of wedge " << dphi/deg
	      << std::endl;

  //Special Geometry parameters
  int ngpar = 7;
  gpar      = getDDDArray("gparHF",sv,ngpar);
  if (verbosity > 0) {
    std::cout << "HFShowerLibrary: " << ngpar << " gpar (cm)";
    for (int ig=0; ig<ngpar; ig++) {
      std::cout << " " << gpar[ig]/cm;
    }
    std::cout << std::endl;
  }

  fibre = new HFFibre(iv, cpv);
}

HFShowerLibrary::~HFShowerLibrary() {
  if (hf)     hf->Close();
  if (ixyz)   delete[] ixyz;
  if (l)      delete[] l;
  if (it)     delete[] it;
  if (posHit) delete[] posHit;
  if (depHit) delete[] depHit;
  if (timHit) delete[] timHit;
  if (xpe)    delete[] xpe;
  if (ype)    delete[] ype;
  if (zpe)    delete[] zpe;
  if (lpe)    delete[] lpe;
  if (tpe)    delete[] tpe;
  if (fibre)  delete   fibre;
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

#ifdef debug
  if (verbosity > 2)
    std::cout << "HFShowerLibrary: Interpolate " << partType
	      << " of energy " << pin/GeV << " GeV" << std::endl;
#endif
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
    if (posHit) delete[] posHit; posHit = new ptrThreeVector[npe];
    if (depHit) delete[] depHit; depHit = new int[npe];
    if (timHit) delete[] timHit; timHit = new double[npe];
  }
  for (int i = 0; i < npe; i++) {
#ifdef debug
    if (verbosity > 2)
      std::cout << "HFShowerLibrary: Hit " << i << " position " << xpe[i]
		<< ", " << ype[i] << ", " << zpe[i] << " Lambda " << lpe[i]
		<< " Time " << tpe[i] << std::endl;
#endif
    double zv = (zpe[i] >= 0 ? zpe[i] : -zpe[i]);
    if (zv <= gpar[1] && lpe[i] > 0 &&
	(zpe[i] >= 0 || zpe[i] <= -gpar[0])) {
      int depth = 1;
      if (zpe[i] < 0) depth = 2;
      double xx = xpe[i]*(ctheta + (1.-ctheta)*sphi*sphi) -
	ype[i]*sphi*cphi*(1.-ctheta) + zv*cphi*stheta;
      double yy = ype[i]*(ctheta + (1.-ctheta)*cphi*cphi) -
	xpe[i]*sphi*cphi*(1.-ctheta) + zv*sphi*stheta;
      double zz =-xpe[i]*cphi*stheta + ype[i]*sphi*stheta +zv*ctheta;
      G4ThreeVector* pos = new G4ThreeVector(xx,yy,zz);
      (*pos) += hitPoint;

      zv = gpar[1] - zv;
      double r  = (*pos).perp();
      double p  = fibre->attLength(lpe[i]);
      double fi = (*pos).phi();
      if (fi < 0) fi += twopi;
      int    isect = int(fi/dphi) + 1;
      isect        = (isect + 1) / 2;
      double dfi   = ((isect*2-1)*dphi - fi);
      if (dfi < 0) dfi = -dfi;
      double dfir  = r * sin(dfi);
#ifdef debug
      if (verbosity > 2)
	std::cout << "HFShowerLibrary: Position " << xx << ", " << yy << ", " 
		  << zz << ": " << (*pos) << " R " << r << " Phi " << fi 
		  << " Section " << isect << " R*Dfi " << dfir << std::endl;
#endif
      zz           = (pos->z() >= 0 ? pos->z() : -pos->z());
      double r1    = RandFlat::shoot(0.0,1.0);
      double r2    = RandFlat::shoot(0.0,1.0);
#ifdef debug
      if (verbosity > 2)
	std::cout << "                   rLimits " << rInside(r)
		  << " attenuation " << r1 <<":" << exp(-p*zv) << " r2 " << r2
		  << " rDfi " << gpar[5] << " zz " << zz << " zLim " 
		  << gpar[4] << ":" << gpar[4]+gpar[1] << std::endl;
#endif
      if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax && dfir > gpar[5] &&
	  zz >= gpar[4] && zz <= gpar[4]+gpar[1]) {
	posHit[nHit] = pos;
	depHit[nHit] = depth;
	timHit[nHit] = (tSlice + tpe[i]);
#ifdef debug
      if (verbosity > 2)
	std::cout << "HFShowerLibrary: Final Hit " << nHit <<" position "
		  << (*(posHit[nHit])) << " Depth " << depHit[nHit] 
		  << " Time " << timHit[nHit] << std::endl;
#endif
	nHit++;
      }
    }
  }

#ifdef debug
  if (verbosity > 1)
    std::cout << "HFShowerLibrary: Total Hits " << nHit << std::endl;
#endif
  if (nHit > npe && verbosity > 0) 
    std::cout << "HFShowerLibrary: Hit buffer " << npe << " smaller than "
	      << nHit << " Hits" << std::endl;
  return nHit;

}

G4ThreeVector HFShowerLibrary::getPosHit(int i) {

  G4ThreeVector pos;
  if (i < nHit) pos = (*(posHit[i]));
#ifdef debug
  if (verbosity > 2)
    std::cout << " HFShowerLibrary: PosHit (" << i << "/" << nHit << ") "
	      << pos << std::endl;
#endif
  return pos;
}

int HFShowerLibrary::getDepth(int i) {

  int depth = 0;
  if (i < nHit) depth = depHit[i];
#ifdef debug 
  if (verbosity > 2)
    std::cout << " HFShowerLibrary: Depth (" << i << "/" << nHit << ") " 
	      << depth << std::endl;
#endif
  return depth;
}

double HFShowerLibrary::getTSlice(int i) {
  
  double tim = 0.;
  if (i < nHit) tim = timHit[i];
#ifdef debug  
  if (verbosity > 2)
    std::cout << " HFShowerLibrary: Time (" << i << "/" << nHit << ") "  
	      << tim << std::endl; 
#endif
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
    tree->GetEntry(record);
  }
  return nph;
}

void HFShowerLibrary::getRecord(TTree* tree, int record) {

  nPhoton = getPhoton(tree, record);
  if (nPhoton > 0) {
    if (ixyz) delete[] ixyz;
    if (l)    delete[] l;
    if (it)   delete[] it;
    ixyz = new int[nPhoton];
    l    = new int[nPhoton];
    it   = new int[nPhoton];
#ifdef debug
    if (verbosity > 2)
      std::cout << "HFShowerLibrary: Record " << record << " with "
		<< nPhoton << " photons " << std::endl;
#endif
    int coor[9000], wl[9000], time[9000];
    tree->SetBranchAddress("XYZ", &coor);
    tree->SetBranchAddress("L",   &wl);
    tree->SetBranchAddress("T",   &time);
    tree->GetEntry(record);
    for (int j = 0; j < nPhoton; j++) {
      ixyz[j] = coor[j];
      l[j]    = wl[j];
      it[j]   = time[j];
#ifdef debug
      if (verbosity > 2)
        std::cout << "Photon " << j << " xyz " << ixyz[j] << " L " << l[j]
		  << " Time " << it[j] << std::endl;
#endif
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
  if (format) {
    tree->SetBranchAddress("LIBVERS",     &libVers);
    tree->SetBranchAddress("PHYLISTVERS", &listVersion);
  } else {
    libVers     = -1;
    listVersion = 0;
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

  int nevent = int(tree->GetEntries())/nMomBin;
#ifdef debug
  if (verbosity > 1)
    std::cout << "HFShowerLibrary:: Interpolate for Energy " << pin/GeV
	      << " GeV with " << nMomBin << " momentum bins and " << nevent 
	      << " entries/bin -- total " << tree->GetEntries() << std::endl;
#endif
  int irc[2], j;
  double w = 0.;
  double r = RandFlat::shoot(0.0,1.0);

  if (pin<pmom[0]) {
    w = pin/pmom[0];
    irc[1] = int(nevent*r) + 1;
    irc[0] = 0;
  } else {
    for (j=0; j<nMomBin-1; j++) {
      if (pin >= pmom[j] && pin < pmom[j+1]) {
	w = (pin-pmom[j])/(pmom[j+1]-pmom[j]);
	if (j == nMomBin-1) { 
	  irc[1] = int(nevent*0.5*r);
	} else {
	  irc[1] = int(nevent*r);
	}
	irc[1] += (j+1)*nevent + 1;
	r = RandFlat::shoot(0.0,1.0);
	irc[0] = int(nevent*r) + 1 + j*nevent;
      }
    }
  }

#ifdef debug
  if (verbosity > 2)
    std::cout << "HFShowerLibrary:: Select records " << irc[0] << " and "
	      << irc[1] << " with weights " << 1-w << " and " << w <<std::endl;
#endif
  int npold = getPhoton (tree, irc[1]);
  if (irc[0]>0) {
    getRecord (tree, irc[0]);
    npold += nPhoton;
  }
  if (npold <= 0) npold = 1;
  if (xpe) delete[] xpe; xpe = new double[npold];
  if (ype) delete[] ype; ype = new double[npold];
  if (zpe) delete[] zpe; zpe = new double[npold];
  if (lpe) delete[] lpe; lpe = new double[npold];
  if (tpe) delete[] tpe; tpe = new double[npold];

  npe = 0;
  if (irc[0]>0) {
    for (j=0; j<nPhoton; j++) {
      r = RandFlat::shoot(0.0,1.0);
      if (r > w) {
	storePhoton (j);
	npe++;
      }
    }
  }

  getRecord (tree, irc[1]);
  for (j=0; j<nPhoton; j++) {
    r = RandFlat::shoot(0.0,1.0);
    if (r < w) {
      storePhoton (j);
      npe++;
    }
  }

  if ((npe > npold || npold == 0) && verbosity > 0)
    std::cout << "HFShowerLibrary: Interpolation error == buffer " 
	      << npold << " filled " << npe << " *****" << std::endl;
#ifdef debug
  if (verbosity > 2) {
    std::cout << "HFShowerLibrary: Interpolation gives " << npe
              << " Photons == buffer " << npold << std::endl;
    for (j=0; j<npe; j++) {
      std::cout << "Photon " << j << " X " << xpe[j] << " Y " << ype[j]
	        << " Z " << zpe[j] << " Lam " << lpe[j] << " T " 
		<< tpe[j] << std::endl;
    }
  }
#endif
}

void HFShowerLibrary::extrapolate(TTree * tree, double pin) {

  int nevent = int((tree)->GetEntries())/nMomBin;
  int nrec   = int(pin/pmom[nMomBin-1]);
  double w   = (pin - pmom[nMomBin-1]*nrec)/pmom[nMomBin-1];
  nrec++;
#ifdef debug
  if (verbosity > 1)
    std::cout << "HFShowerLibrary:: Extrapolate for Energy " << pin 
	      << " GeV with " << nMomBin << " momentum bins and " << nevent 
	      << " entries/bin -- total " << tree->GetEntries() << " using " 
	      << nrec << " records"  << std::endl << " Records:";
#endif
  int * irc  = new int[nrec];
  int   j, ir;
  double r;

  npe = 0;
  int npold = 0;
  for (ir=0; ir<nrec; ir++) {
    r = RandFlat::shoot(0.0,1.0);
    irc[ir] = int(nevent*0.5*r) +(nMomBin-1)*nevent + 1;
#ifdef debug
    if (verbosity > 1) std::cout << " " << irc[ir];
#endif
    npold += getPhoton (tree, irc[ir]);
  }
#ifdef debug
  if (verbosity > 0) std::cout << " with " << npold << " photons" << std::endl;
#endif

  if (npold <= 0) npold = 1;
  if (xpe) delete[] xpe; xpe = new double[npold];
  if (ype) delete[] ype; ype = new double[npold];
  if (zpe) delete[] zpe; zpe = new double[npold];
  if (lpe) delete[] lpe; lpe = new double[npold];
  if (tpe) delete[] tpe; tpe = new double[npold];

  for (ir=0; ir<nrec; ir++) {
    getRecord (tree, irc[ir]);
    for (j=0; j<nPhoton; j++) {
      r = RandFlat::shoot(0.0,1.0);
      if (ir != nrec-1 || r < w) {
	storePhoton (j);
	npe++;
      }
    }
  }

  if ((npe > npold || npold == 0) && verbosity > 0)
    std::cout << "HFShowerLibrary: Extrapolation error == buffer " 
	      << npold << " filled " << npe << " *****" << std::endl;
#ifdef debug
  if (verbosity > 2) {
    std::cout << "HFShowerLibrary: Extrapolation gives " << npe
	      << " Photons == buffer "  << npold  << std::endl;
    for (j=0; j<npe; j++) {
      std::cout << "Photon " << j << " X " << xpe[j] << " Y " << ype[j]
		<< " Z " << zpe[j] << " Lam " << lpe[j] << " T " 
		<< tpe[j] << std::endl; 
    }
  }
#endif
  delete[] irc;
}

void HFShowerLibrary::storePhoton(int j) {

  int ix = ixyz[j]/xMultiplier;
  int iy = ixyz[j]/yMultiplier - ix*yMultiplier;
  int iz = ixyz[j]/zMultiplier - ix*xMultiplier - iy*yMultiplier;
  xpe[npe] = (ix/xScale - xOffset)*cm;
  ype[npe] = (iy/yScale - yOffset)*cm;
  zpe[npe] = (iz/zScale - zOffset)*cm;
  lpe[npe] = l[j];
  tpe[npe] = it[j]/100.;
}

std::vector<double> HFShowerLibrary::getDDDArray(const std::string & str, 
						 const DDsvalues_type & sv, 
						 int & nmin) {

#ifdef debug
  if (verbosity > 1) 
    std::cout << "HFShowerLibrary:getDDDArray called for " << str 
	      << " with nMin " << nmin << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef debug
    if (verbosity > 2) std::cout << value << " " << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	if (verbosity > 0) 
	  std::cout << "HFShowerLibrary : # of " << str << " bins " << nval
		    << " < " << nmin << " ==> illegal" << std::endl;
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < nmin for array " << str << "\n";
      }
    } else {
      if (nval < 2) {
	if (verbosity > 0) 
	  std::cout << "HFShowerLibrary : # of " << str << " bins " << nval
		    << " < 2 ==> illegal (nmin=" << nmin << ")" << std::endl;
	throw cms::Exception("Unknown", "HFShowerLibrary")
	  << "nval < 2 for array " << str << "\n";
      }
    }
    nmin = nval;

    return fvec;
  } else {
    if (verbosity > 0) 
      std::cout << "HFShowerLibrary : cannot get array " << str << std::endl;
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "cannot get array " << str << "\n";
  }
}
