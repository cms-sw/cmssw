///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4NavigationHistory.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

//#define DebugLog

HFShowerLibrary::HFShowerLibrary(const std::string & name, const DDCompactView & cpv,
                                 edm::ParameterSet const & p) : fibre(nullptr),hf(nullptr),
                                                                emBranch(nullptr),
                                                                hadBranch(nullptr),
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
  applyFidCut              = m_HS.getParameter<bool>("ApplyFiducialCut");

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  if (!hf->IsOpen()) { 
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree 
                              << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: opening " << nTree 
                             << " successfully"; 
  }

  newForm = (branchEvInfo.empty());
  TTree* event(nullptr);
  if (newForm) event = (TTree *) hf ->Get("HFSimHits");
  else         event = (TTree *) hf ->Get("Events");
  if (event) {
    TBranch *evtInfo(nullptr);
    if (!newForm) {
      std::string info = branchEvInfo + branchPost;
      evtInfo          = event->GetBranch(info.c_str());
    }
    if (evtInfo || newForm) {
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
  
  std::stringstream ss;
  ss << "HFShowerLibrary: Library " << libVers << " ListVersion " << listVersion 
     << " Events Total " << totEvents << " and " << evtPerBin << " per bin\n";
  ss << "HFShowerLibrary: Energies (GeV) with " << nMomBin << " bins\n";
  for (int i=0; i<nMomBin; ++i) {
    if(i/10*10 == i && i > 0) { ss << "\n"; }
    ss << "  " << pmom[i]/CLHEP::GeV;
  }
  edm::LogVerbatim("HFShower") << ss.str();

  std::string nameBr = branchPre + emName + branchPost;
  emBranch         = event->GetBranch(nameBr.c_str());
  if (verbose) emBranch->Print();
  nameBr           = branchPre + hadName + branchPost;
  hadBranch        = event->GetBranch(nameBr.c_str());
  if (verbose) hadBranch->Print();

  v3version=false;
  if ( emBranch->GetClassName() == std::string("vector<float>") ) {
    v3version=true;
  }
  
  edm::LogVerbatim("HFShower") << " HFShowerLibrary:Branch " << emName 
                           << " has " << emBranch->GetEntries() 
                           << " entries and Branch " << hadName 
                           << " has " << hadBranch->GetEntries() 
                           << " entries"
                           << "\n HFShowerLibrary::No packing information -"
                           << " Assume x, y, z are not in packed form"
                           << "\n Maximum probability cut off " 
                           << probMax << "  Back propagation of light prob. "
                           << backProb;
  
  fibre = new HFFibre(name, cpv, p);
  photo = new HFShowerPhotonCollection;
}

HFShowerLibrary::~HFShowerLibrary() {
  if (hf)     hf->Close();
  delete fibre;
  delete photo;
}

void HFShowerLibrary::initRun(G4ParticleTable*, const HcalDDDSimConstants* hcons) {

  if (fibre) fibre->initRun(hcons);
  
  //Radius (minimum and maximum)
  std::vector<double> rTable = hcons->getRTableHF();
  rMin = rTable[0];
  rMax = rTable[rTable.size()-1];

  //Delta phi
  std::vector<double> phibin   = hcons->getPhiTableHF();
  dphi       = phibin[0];
  edm::LogVerbatim("HFShower") << "HFShowerLibrary: rMIN " << rMin/cm 
                           << " cm and rMax " << rMax/cm
                           << " (Half) Phi Width of wedge " 
                           << dphi/deg;

  //Special Geometry parameters
  gpar = hcons->getGparHF();
}

std::vector<HFShowerLibrary::Hit> HFShowerLibrary::getHits(const G4Step * aStep,
                                                           bool & isKilled,
                                                           double weight,
                                                           bool onlyLong) {

  auto const preStepPoint  = aStep->GetPreStepPoint(); 
  auto const postStepPoint = aStep->GetPostStepPoint(); 
  auto const track = aStep->GetTrack();
  // Get Z-direction 
  auto const aParticle = track->GetDynamicParticle();
  const G4ThreeVector& momDir = aParticle->GetMomentumDirection();
  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();   
  int parCode   = track->GetDefinition()->GetPDGEncoding();

  // VI: for ions use internally pdg code of alpha in order to keep 
  // consistency with previous simulation
  if(track->GetDefinition()->IsGeneralIon()) { parCode = 1000020040; }

#ifdef DebugLog
  G4String      partType = track->GetDefinition()->GetParticleName();
  const G4ThreeVector localPos = 
    preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  double zoff   = localPos.z() + 0.5*gpar[1];

  edm::LogVerbatim("HFShower") << "HFShowerLibrary::getHits " << partType
                           << " of energy " << pin/GeV << " GeV"
                           << " weight= " << weight << " onlyLong: " << onlyLong
                           << "  dir.orts " << momDir.x() << ", " <<momDir.y()
                           << ", " << momDir.z() << "  Pos x,y,z = "
                           << hitPoint.x() << "," << hitPoint.y() << ","
                           << hitPoint.z() << " (" << zoff
                           << ")   sphi,cphi,stheta,ctheta  = " << sin(momDir.phi())
                           << ","  << cos(momDir.phi()) << ", " << sin(momDir.theta()) 
                           << "," << cos(momDir.theta());
#endif

  double tSlice = (postStepPoint->GetGlobalTime())/CLHEP::nanosecond;

  // use kinetic energy for protons and ions
  double pin = (track->GetDefinition()->GetBaryonNumber() > 0) 
    ? preStepPoint->GetKineticEnergy() : preStepPoint->GetTotalEnergy();

  return fillHits(hitPoint,momDir,parCode,pin,isKilled,weight,tSlice,onlyLong);
}

std::vector<HFShowerLibrary::Hit> HFShowerLibrary::fillHits(const G4ThreeVector & hitPoint,
                               const G4ThreeVector & momDir,
                               int parCode, double pin, bool & ok,
                               double weight, double tSlice,bool onlyLong) {

  std::vector<HFShowerLibrary::Hit> hit;
  ok = false;
  bool isEM = G4TrackToParticleID::isGammaElectronPositron(parCode);
  // shower is built only for gamma, e+- and stable hadrons
  if (!isEM && !G4TrackToParticleID::isStableHadron(parCode)) { 
    return hit;
  }
  ok = true;

  // remove low-energy component
  const double threshold = 50*MeV;
  if(pin < threshold) { return hit; }

  double pz     = momDir.z(); 
  double zint   = hitPoint.z(); 

  // if particle moves from interaction point or "backwards (halo)
  bool backward = (pz * zint < 0.) ? true : false;
  
  double sphi   = sin(momDir.phi());
  double cphi   = cos(momDir.phi());
  double ctheta = cos(momDir.theta());
  double stheta = sin(momDir.theta());

  if(isEM) {
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
    
  int nHit = 0;
  HFShowerLibrary::Hit oneHit;
  for (int i = 0; i < npe; ++i) {
    double zv = std::abs(pe[i].z()); // abs local z  
#ifdef DebugLog
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: Hit " << i << " " << pe[i] << " zv " << zv;
#endif
    if (zv <= gpar[1] && pe[i].lambda() > 0 && 
        (pe[i].z() >= 0 || (zv > gpar[0] && (!onlyLong)))) {
      int depth = 1;
      if (onlyLong) {
      } else if (!backward) {        // fully valid only for "front" particles
        if (pe[i].z() < 0) depth = 2;// with "front"-simulated shower lib.
      } else {                       // for "backward" particles - almost equal
        double r = G4UniformRand();  // share between L and S fibers
        if (r > 0.5) depth = 2;
      } 
      

      // Updated coordinate transformation from local
      //  back to global using two Euler angles: phi and theta
      double pex = pe[i].x();
      double pey = pe[i].y();

      double xx = pex*ctheta*cphi - pey*sphi + zv*stheta*cphi; 
      double yy = pex*ctheta*sphi + pey*cphi + zv*stheta*sphi;
      double zz = -pex*stheta + zv*ctheta;

      G4ThreeVector pos  = hitPoint + G4ThreeVector(xx,yy,zz);
      zv = std::abs(pos.z()) - gpar[4] - 0.5*gpar[1];
      G4ThreeVector lpos = G4ThreeVector(pos.x(),pos.y(),zv);

      zv = fibre->zShift(lpos,depth,0);     // distance to PMT !

      double r  = pos.perp();
      double p  = fibre->attLength(pe[i].lambda());
      double fi = pos.phi();
      if (fi < 0) fi += CLHEP::twopi;
      int    isect = int(fi/dphi) + 1;
      isect        = (isect + 1) / 2;
      double dfi   = ((isect*2-1)*dphi - fi);
      if (dfi < 0) dfi = -dfi;
      double dfir  = r * sin(dfi);
#ifdef DebugLog
      edm::LogVerbatim("HFShower") << "HFShowerLibrary: Position shift " << xx 
                               << ", " << yy << ", "  << zz << ": " << pos 
                               << " R " << r << " Phi " << fi << " Section " 
                               << isect << " R*Dfi " << dfir << " Dist " << zv;
#endif
      zz           = std::abs(pos.z());
      double r1    = G4UniformRand();
      double r2    = G4UniformRand();
      double r3    = backward ? G4UniformRand() : -9999.;
      if (!applyFidCut) dfir += gpar[5];

#ifdef DebugLog
      edm::LogVerbatim("HFShower") << "HFShowerLibrary: rLimits " << rInside(r)
                               << " attenuation " << r1 <<":" << exp(-p*zv) 
                               << " r2 " << r2 << " r3 " << r3 << " rDfi "  
                               << gpar[5] << " zz " 
                               << zz << " zLim " << gpar[4] << ":" 
                               << gpar[4]+gpar[1] << "\n"
                               << "  rInside(r) :" << rInside(r) 
                               << "  r1 <= exp(-p*zv) :" <<  (r1 <= exp(-p*zv))
                               << "  r2 <= probMax :"    <<  (r2 <= probMax*weight)
                               << "  r3 <= backProb :"   <<  (r3 <= backProb) 
                               << "  dfir > gpar[5] :"   <<  (dfir > gpar[5])
                               << "  zz >= gpar[4] :"    <<  (zz >= gpar[4])
                               << "  zz <= gpar[4]+gpar[1] :" 
                               << (zz <= gpar[4]+gpar[1]);   
#endif
      if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax*weight && 
          dfir > gpar[5] && zz >= gpar[4] && zz <= gpar[4]+gpar[1] && 
          r3 <= backProb && (depth != 2 || zz >= gpar[4]+gpar[0])) {
        oneHit.position = pos;
        oneHit.depth    = depth;
        oneHit.time     = (tSlice+(pe[i].t())+(fibre->tShift(lpos,depth,1)));
        hit.push_back(oneHit);
#ifdef DebugLog
        edm::LogVerbatim("HFShower") << "HFShowerLibrary: Final Hit " << nHit 
                                 <<" position " << (hit[nHit].position) 
                                 << " Depth " << (hit[nHit].depth) <<" Time " 
                                 << tSlice << ":" << pe[i].t() << ":" 
                                 << fibre->tShift(lpos,depth,1) << ":" 
                                 << (hit[nHit].time);
#endif
        ++nHit;
      }
#ifdef DebugLog
      else  LogDebug("HFShower") << "HFShowerLibrary: REJECTED !!!";
#endif
      if (onlyLong && zz >= gpar[4]+gpar[0] && zz <= gpar[4]+gpar[1]) {
        r1    = G4UniformRand();
        r2    = G4UniformRand();
        if (rInside(r) && r1 <= exp(-p*zv) && r2 <= probMax && dfir > gpar[5]){
          oneHit.position = pos;
          oneHit.depth    = 2;
          oneHit.time     = (tSlice+(pe[i].t())+(fibre->tShift(lpos,2,1)));
          hit.push_back(oneHit);
#ifdef DebugLog
          edm::LogVerbatim("HFShower") << "HFShowerLibrary: Final Hit " << nHit 
                                   << " position " << (hit[nHit].position) 
                                   << " Depth " << (hit[nHit].depth) <<" Time "
                                   << tSlice << ":" << pe[i].t() << ":" 
                                   << fibre->tShift(lpos,2,1) << ":" 
                                   << (hit[nHit].time);
#endif
          ++nHit;
        }
      }
    }
  }

#ifdef DebugLog
  edm::LogVerbatim("HFShower") << "HFShowerLibrary: Total Hits " << nHit
                           << " out of " << npe << " PE";
#endif
  if (nHit > npe && !onlyLong) {
    edm::LogWarning("HFShower") << "HFShowerLibrary: Hit buffer " << npe 
                                << " smaller than " << nHit << " Hits";
  }
  return hit;
}

bool HFShowerLibrary::rInside(double r) {

  return (r >= rMin && r <= rMax);
}

void HFShowerLibrary::getRecord(int type, int record) {

  int nrc     = record-1;
  photon.clear();
  photo->clear();
  if (type > 0) {
    if (newForm) {
      if ( !v3version ) {
        hadBranch->SetAddress(&photo);
        hadBranch->GetEntry(nrc+totEvents);
      }
      else{
        std::vector<float> t;
        std::vector<float> *tp=&t;
        hadBranch->SetAddress(&tp);
        hadBranch->GetEntry(nrc+totEvents);
        unsigned int tSize=t.size()/5;
        photo->reserve(tSize);
        for ( unsigned int i=0; i<tSize; i++ ) {
          photo->push_back( HFShowerPhoton( t[i], t[1*tSize+i], t[2*tSize+i], t[3*tSize+i], t[4*tSize+i] ) );
        }
      }
    } else {
      hadBranch->SetAddress(&photon);
      hadBranch->GetEntry(nrc);
    }
  } else {
    if (newForm) {
      if (!v3version) {
        emBranch->SetAddress(&photo);
        emBranch->GetEntry(nrc);
      }
      else{
        std::vector<float> t;
        std::vector<float> *tp=&t;
        emBranch->SetAddress(&tp);
        emBranch->GetEntry(nrc);
        unsigned int tSize=t.size()/5;
        photo->reserve(tSize);
        for ( unsigned int i=0; i<tSize; i++ ) {
          photo->push_back( HFShowerPhoton( t[i], t[1*tSize+i], t[2*tSize+i], t[3*tSize+i], t[4*tSize+i] ) );
        }
      }
    } else {
      emBranch->SetAddress(&photon);
      emBranch->GetEntry(nrc);
    }
  }
#ifdef DebugLog
  int nPhoton = (newForm) ? photo->size() : photon.size();
  LogDebug("HFShower") << "HFShowerLibrary::getRecord: Record " << record
                       << " of type " << type << " with " << nPhoton 
                       << " photons";
  for (int j = 0; j < nPhoton; j++) 
    if (newForm) LogDebug("HFShower") << "Photon " << j << " " << photo->at(j);
    else         LogDebug("HFShower") << "Photon " << j << " " << photon[j];
#endif
}

void HFShowerLibrary::loadEventInfo(TBranch* branch) {

  if (branch) {
    std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
    branch->SetAddress(&eventInfoCollection);
    branch->GetEntry(0);
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads "
                             << " EventInfo Collection of size "
                             << eventInfoCollection.size() << " records";
    totEvents   = eventInfoCollection[0].totalEvents();
    nMomBin     = eventInfoCollection[0].numberOfBins();
    evtPerBin   = eventInfoCollection[0].eventsPerBin();
    libVers     = eventInfoCollection[0].showerLibraryVersion();
    listVersion = eventInfoCollection[0].physListVersion();
    pmom        = eventInfoCollection[0].energyBins();
  } else {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads "
                             << " EventInfo from hardwired numbers";
    nMomBin     = 16;
    evtPerBin   = 5000;
    totEvents   = nMomBin*evtPerBin;
    libVers     = 1.1;
    listVersion = 3.6;
    pmom        = {2,3,5,7,10,15,20,30,50,75,100,150,250,350,500,1000};
  }
  for (int i=0; i<nMomBin; i++) 
    pmom[i] *= GeV;
}

void HFShowerLibrary::interpolate(int type, double pin) {

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary:: Interpolate for Energy " <<pin/GeV
                       << " GeV with " << nMomBin << " momentum bins and " 
                       << evtPerBin << " entries/bin -- total " << totEvents;
#endif
  int irc[2]={0,0};
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

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary:: Select records " << irc[0] 
                       << " and " << irc[1] << " with weights " << 1-w 
                       << " and " << w;
#endif
  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir < 2; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = (newForm) ? photo->size() : photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
        r = G4UniformRand();
        if ((ir==0 && r > w) || (ir > 0 && r < w)) {
          storePhoton (j);
        }
      }
    }
  }

  if ((npe > npold || (npold == 0 && irc[0] > 0)) && !(npe == 0 && npold == 0))
    edm::LogWarning("HFShower") << "HFShowerLibrary: Interpolation Warning =="
                                << " records " << irc[0] << " and " << irc[1]
                                << " gives a buffer of " << npold 
                                << " photons and fills " << npe << " *****";
#ifdef DebugLog
  else
    LogDebug("HFShower") << "HFShowerLibrary: Interpolation == records " 
                         << irc[0] << " and " << irc[1] << " gives a "
                         << "buffer of " << npold << " photons and fills "
                         << npe << " PE";
  for (int j=0; j<npe; j++)
    LogDebug("HFShower") << "Photon " << j << " " << pe[j];
#endif
}

void HFShowerLibrary::extrapolate(int type, double pin) {

  int nrec   = int(pin/pmom[nMomBin-1]);
  double w   = (pin - pmom[nMomBin-1]*nrec)/pmom[nMomBin-1];
  nrec++;
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary:: Extrapolate for Energy " << pin 
                       << " GeV with " << nMomBin << " momentum bins and " 
                       << evtPerBin << " entries/bin -- total " << totEvents 
                       << " using " << nrec << " records";
#endif
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
#ifdef DebugLog
    } else {
      LogDebug("HFShower") << "HFShowerLibrary::Extrapolation use irc[" 
                           << ir  << "] = " << irc[ir];
#endif
    }
  }

  pe.clear(); 
  npe       = 0;
  int npold = 0;
  for (int ir=0; ir<nrec; ir++) {
    if (irc[ir]>0) {
      getRecord (type, irc[ir]);
      int nPhoton = (newForm) ? photo->size() : photon.size();
      npold      += nPhoton;
      for (int j=0; j<nPhoton; j++) {
        double r = G4UniformRand();
        if (ir != nrec-1 || r < w) {
          storePhoton (j);
        }
      }
#ifdef DebugLog
      LogDebug("HFShower") << "HFShowerLibrary: Record [" << ir << "] = " 
                           << irc[ir] << " npold = " << npold;
#endif
    }
  }
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary:: uses " << npold << " photons";
#endif

  if (npe > npold || npold == 0)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Extrapolation Warning == "
                                << nrec << " records " << irc[0] << ", " 
                                << irc[1] << ", ... gives a buffer of " <<npold
                                << " photons and fills " << npe 
                                << " *****";
#ifdef DebugLog
  else
    LogDebug("HFShower") << "HFShowerLibrary: Extrapolation == " << nrec
                         << " records " << irc[0] << ", " << irc[1] 
                         << ", ... gives a buffer of " << npold 
                         << " photons and fills " << npe << " PE";
  for (int j=0; j<npe; j++)
    LogDebug("HFShower") << "Photon " << j << " " << pe[j];
#endif
}

void HFShowerLibrary::storePhoton(int j) {

  if (newForm) pe.push_back(photo->at(j));
  else         pe.push_back(photon[j]);
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary: storePhoton " << j << " npe " 
                       << npe << " " << pe[npe];
#endif
  npe++;
}

std::vector<double> HFShowerLibrary::getDDDArray(const std::string & str, 
                                                 const DDsvalues_type & sv, 
                                                 int & nmin) {

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerLibrary:getDDDArray called for " << str 
                       << " with nMin " << nmin;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
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
