
///////////////////////////////////////////////////////////////////////////////
// File: ZdcShowerLibrary.cc
// Description: Shower library for the Zero Degree Calorimeter
// E. Garcia June 2008
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"

ZdcShowerLibrary::ZdcShowerLibrary(std::string & name, const DDCompactView & cpv,
				 edm::ParameterSet const & p) {
  edm::ParameterSet m_HS   = p.getParameter<edm::ParameterSet>("ZdcShowerLibrary");
  verbose                  = m_HS.getUntrackedParameter<int>("Verbosity",0);

  ienergyBin  = 50000; // 20 GeV bins 50,000 -- 100 GeV 10,000 -- 500 GeV 2,000, -- 5000 GeV 200  
  ithetaBin   = 64;    // 3 degree theta binning                                                  
  iphiBin     = 64;    // 3 degree phi binning                                                    
  isideBin    = 1;     // side                                                                    
  isectionBin = 1;     // section                                                                 
  ichannelBin = 1;     // channel                                                                 
  ixBin       =  10;   // x 1 cm bin                                                              
  iyBin       =  10;   // y 1 cm bin                                                              
  izBin       =  40;   // z 4 cm bin                                                              
  iPIDBin     =  1;    // 1 bin part ID                                                           

  
  LogDebug("ZdcShower") <<"(##) Binning Information: "
                        <<" energy: "<<ienergyBin
                        <<" theta: "<<ithetaBin
                        <<" phi: "<<iphiBin
                        <<" side: "<<isideBin
                        <<" section: "<<isectionBin
                        <<" channel: "<<ichannelBin
                        <<" X: "<<ixBin
                        <<" Y: "<<iyBin
                        <<" Z: "<<izBin
                        <<" PID: "<<iPIDBin;
  

  maxBitsEnergy  = 512; // energy            
  maxBitsTheta   = 64;  // theta             
  maxBitsPhi     = 64;  // phi               
  maxBitsSide    = 2;   // detector side     
  maxBitsSection = 4;   // detector section  
  maxBitsChannel = 8;   // detector channel  
  maxBitsX       = 16;  // X                 
  maxBitsY       = 16;  // Y                 
  maxBitsZ       = 32;  // Z                 
  maxBitsPID     = 64;  // pid              

  LogDebug("ZdcShower") <<"(##) MaxBits Information: "
                        <<" energy: "<<maxBitsEnergy 
                        <<" theta: "<<maxBitsTheta
                        <<" phi: "<< maxBitsPhi
                        <<" side: "<< maxBitsSide
                        <<" section: "<<maxBitsSection
                        <<" channel: "<<maxBitsChannel
                        <<" X: "<<maxBitsX
                        <<" Y: "<<maxBitsY
                        <<" Z: "<<maxBitsZ
                        <<" PID: "<<maxBitsPID;
 
}

ZdcShowerLibrary::~ZdcShowerLibrary() {
}

void ZdcShowerLibrary::initRun(G4ParticleTable * theParticleTable) {
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
  LogDebug("ZdcShower") << "ZdcShowerLibrary: Particle codes for e- = " << emPDG
		       << ", e+ = " << epPDG << ", gamma = " << gammaPDG 
		       << ", pi0 = " << pi0PDG << ", eta = " << etaPDG
		       << ", geantino = " << geantinoPDG << "\n        nu_e = "
		       << nuePDG << ", nu_mu = " << numuPDG << ", nu_tau = "
		       << nutauPDG << ", anti_nu_e = " << anuePDG
		       << ", anti_nu_mu = " << anumuPDG << ", anti_nu_tau = "
		       << anutauPDG;
}

std::vector<ZdcShowerLibrary::Hit> ZdcShowerLibrary::getHits(G4Step * aStep, bool & ok) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track    = aStep->GetTrack();

  const G4DynamicParticle *aParticle = track->GetDynamicParticle();
  G4ThreeVector momDir = aParticle->GetMomentumDirection();
  double energy = preStepPoint->GetKineticEnergy();
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  int parCode  = track->GetDefinition()->GetPDGEncoding();
  
  std::vector<ZdcShowerLibrary::Hit> hits;
 
  ok = false;
  if (parCode == pi0PDG || parCode == etaPDG || parCode == nuePDG ||
      parCode == numuPDG || parCode == nutauPDG || parCode == anuePDG ||
      parCode == anumuPDG || parCode == anutauPDG || parCode == geantinoPDG) 
    return hits;
  ok = true;
  
  G4ThreeVector pos;
  G4ThreeVector posLocal;
  double tSlice = (postStepPoint->GetGlobalTime())/nanosecond;

  int nHit = 0;
  HcalZDCDetId::Section section;
  bool side = false;
  int channel = 0;
  double xx,yy,zz;
  double xxlocal, yylocal, zzlocal;
  
  ZdcShowerLibrary::Hit oneHit;
  side = (hitPoint.z() > 0.) ?  true : false;  
  
  int npe = 9; // number of channels o fibers where the energy will be deposited
  float xWidthEM = fabs(theXChannelBoundaries[0] - theXChannelBoundaries[1]);
  float zWidthEM = fabs(theZSectionBoundaries[0] - theZSectionBoundaries[1]); 
  float zWidthHAD = fabs(theZHadChannelBoundaries[0] -theZHadChannelBoundaries[1]); 

  for (int i = 0; i < npe; i++) {
    if(i < 5){
      section = HcalZDCDetId::EM;
      channel = i+1;
      xxlocal = theXChannelBoundaries[i]+(xWidthEM/2.);
      xx = xxlocal + X0; 
      yy = 0.0;
      yylocal = yy + Y0;
      zzlocal = theZSectionBoundaries[0]+(zWidthEM/2.);
      zz = (hitPoint.z() > 0.) ? zzlocal + Z0 : zzlocal - Z0;
      pos = G4ThreeVector(xx,yy,zz);
      posLocal = G4ThreeVector(xxlocal,yylocal,zzlocal);
	}
    if(i > 4){
      section = HcalZDCDetId::HAD;
      channel = i - 4;
      xxlocal = 0.0;
      xx = xxlocal + X0;  
      yylocal = 0;
      yy = yylocal + Y0;
      zzlocal = (hitPoint.z() > 0.) ? 
	theZHadChannelBoundaries[i-5] + (zWidthHAD/2.) : theZHadChannelBoundaries[i-5] - (zWidthHAD/2.);
      zz = (hitPoint.z() > 0.) ? zzlocal +  Z0 : zzlocal -  Z0; 
      pos = G4ThreeVector(xx,yy,zz);
      posLocal = G4ThreeVector(xxlocal,yylocal,zzlocal);
   }
    
    oneHit.position = pos;
    oneHit.entryLocal = posLocal;
    oneHit.depth    = channel;
    oneHit.time     = tSlice;
    oneHit.detID    = HcalZDCDetId(section,side,channel);


    // Note: coodinates of hit are relative to center of detector (X0,Y0,Z0)
    hitPoint.setX(hitPoint.x()-X0);
    hitPoint.setY(hitPoint.y()-Z0);
    double setZ= (hitPoint.z() > 0.) ? hitPoint.z()- Z0 : fabs(hitPoint.z()) - Z0;
    hitPoint.setZ(setZ);

    int dE = getEnergyFromLibrary(hitPoint,momDir,energy,parCode,section,side,channel);

    int iparCode = encodePartID(parCode);
    if ( iparCode == 0 ) {
      oneHit.DeEM  = dE; oneHit.DeHad = 0.;
    } else {
      oneHit.DeEM  = 0; oneHit.DeHad = dE;
    }
    
    hits.push_back(oneHit);
    
    LogDebug("ZdcShower")<< "ZdcShowerLibrary:  Generated Hits " << nHit 
                         <<" original hit position " << hitPoint
                         <<" position " << (hits[nHit].position) 
                         <<" Depth " <<(hits[nHit].depth) 
                         <<" side "<< side  
                         <<" Time " <<(hits[nHit].time)
                         <<" DetectorID " << (hits[nHit].detID)
                         <<" Had Energy " << (hits[nHit].DeHad)
                         <<" EM Energy  " << (hits[nHit].DeEM);    
    nHit++;
  }
   return hits;
}


int ZdcShowerLibrary::getEnergyFromLibrary(G4ThreeVector hitPoint, G4ThreeVector momDir, double energy,
					   int parCode,HcalZDCDetId::Section section, bool side, int channel){
  int nphotons = 0;
 
  LogDebug("ZdcShower") <<"GetEnergy variables: *---> "
                        <<" phi: "<<59.2956*momDir.phi()
                        <<" theta: "<<59.2956*momDir.theta()
                        <<" xin : "<<hitPoint.x()
                        <<" yin : "<<hitPoint.y()
                        <<" zin : "<<hitPoint.z()
                        <<" en: " <<energy
                        <<" section: "<<section
                        <<" side: "<<side
                        <<" partID: "<<parCode;

  int iphi   = int(59.2956*momDir.phi()/float(iphiBin));
  int itheta = int(59.2956*momDir.theta()/float(ithetaBin));
  int ixin = int(hitPoint.x()/float(ixBin)); 
  int iyin = int(hitPoint.y()/float(iyBin)); 
  int izin = int(hitPoint.z()/float(izBin)); 
  int ienergy = int(energy/float(ienergyBin));
  int isection = int(section);
  int iside = (side)? 1 : 2;     
  int iparCode  = encodePartID(parCode);

  LogDebug("ZdcShower") <<"Binned variables: #---> "
                        <<" iphi: "<<iphi
                        <<" itheta: "<<itheta
                        <<" ixin : "<<ixin
                        <<" iyin : "<<iyin
                        <<" izin : "<<izin
                        <<" ien: " <<ienergy
                        <<" isection: "<<isection
                        <<" iside: "<<iside
                        <<" iparcode "<<iparCode;
  
  double eav, esig, edis = 0.;

  if (/*int*/ iparCode==0){            // Jeff's fits here
    eav = ((((((-0.0002*ixin-2*10e-13)*ixin+0.0022)*ixin+10e-11)*ixin-0.0217)*ixin-3*10e-10)*ixin+1)*
      (((0.0001*iyin + 0.0056)*iyin + 0.0508)*iyin + 1)*44*pow(ienergy,0.99);       // EM
    esig = (eav*eav)*((((((0.0005*ixin - 10e-12)*ixin - 0.0052)*ixin + 5*10e-11)*ixin + 0.032)*ixin - 
                        2*10e-10)*ixin + 1)*(((0.0006*iyin + 0.0071)*iyin - 0.031)*iyin + 1)*26*pow(ienergy,0.54);  // EM
    edis = 1.0;                    // this is for EM Gaussian Distributions
  }else{                     
    eav = ((((((-0.0002*ixin-2*10e-13)*ixin+0.0022)*ixin+10e-11)*ixin-0.0217)*ixin-3*10e-10)*ixin+1)*
      (((0.0001*iyin + 0.0056)*iyin + 0.0508)*iyin + 1)*2.3*pow(ienergy,1.12);      // Hadronic
    esig = (eav*eav)*((((((0.0005*ixin - 10e-12)*ixin - 0.0052)*ixin + 5*10e-11)*ixin + 0.032)*ixin - 
                        2*10e-10)*ixin + 1)*(((0.0006*iyin + 0.0071)*iyin - 0.031)*iyin + 1)*1.2*pow(ienergy,0.93); // Hadronic
    edis = 3.0;                    // this is for hadronic Landau Distributions 
  }
  
  nphotons = photonFluctuation(eav,esig,edis);

  LogDebug("ZdcShower") <<" ########photons---->"<<nphotons;
  return nphotons;
}

int ZdcShowerLibrary::photonFluctuation(double eav, double esig,double edis){
  int nphot=0;
  double efluct = 0.;
  if(edis == 1.0)efluct = CLHEP::RandGaussQ::shoot(eav,esig);
  if(edis == 3.0)efluct = eav+esig*CLHEP::RandLandau::shoot(); // to be verified!!!!!!!!!!
  nphot = int(efluct);
  return nphot;
}

int ZdcShowerLibrary::encodePartID(int parCode){

  int iparCode = 1;

  if (parCode == emPDG ||
      parCode == epPDG ||
      parCode == gammaPDG ) {
    iparCode = 0;
  } else { return iparCode; }

  return iparCode;

}
