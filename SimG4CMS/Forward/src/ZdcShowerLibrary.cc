///////////////////////////////////////////////////////////////////////////////
// File: ZdcShowerLibrary.cc
// Description: Shower library for the Zero Degree Calorimeter
// E. Garcia June 2008
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

ZdcShowerLibrary::ZdcShowerLibrary(std::string & name, const DDCompactView & cpv,
				 edm::ParameterSet const & p) {
  edm::ParameterSet m_HS   = p.getParameter<edm::ParameterSet>("ZdcShowerLibrary");
  verbose                  = m_HS.getUntrackedParameter<int>("Verbosity",0);

  npe = 9; // number of channels or fibers where the energy will be deposited
  hits.reserve(npe);

}

ZdcShowerLibrary::~ZdcShowerLibrary() {
}

void ZdcShowerLibrary::initRun(G4ParticleTable *theParticleTable){
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
  LogDebug("ZdcShower")<< "ZdcShowerLibrary: Particle codes for e- = " << emPDG
		       << ", e+ = " << epPDG << ", gamma = " << gammaPDG 
		       << ", pi0 = " << pi0PDG << ", eta = " << etaPDG
		       << ", geantino = " << geantinoPDG << "\n        nu_e = "
		       << nuePDG << ", nu_mu = " << numuPDG << ", nu_tau = "
		       << nutauPDG << ", anti_nu_e = " << anuePDG
		       << ", anti_nu_mu = " << anumuPDG << ", anti_nu_tau = "
		       << anutauPDG;
}

std::vector<ZdcShowerLibrary::Hit> & ZdcShowerLibrary::getHits(G4Step * aStep, bool & ok) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track    = aStep->GetTrack();

  const G4DynamicParticle *aParticle = track->GetDynamicParticle();
  G4ThreeVector momDir = aParticle->GetMomentumDirection();
  double energy = preStepPoint->GetKineticEnergy();
  G4ThreeVector hitPoint = preStepPoint->GetPosition();  
  G4ThreeVector hitPointOrig = preStepPoint->GetPosition();
  G4int parCode  = track->GetDefinition()->GetPDGEncoding();

  hits.clear();
 
  ok = false;
  if (parCode == geantinoPDG)
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
  side = (hitPointOrig.z() > 0.) ?  true : false;  
  
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
      zz = (hitPointOrig.z() > 0.) ? zzlocal + Z0 : zzlocal - Z0;
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
      zzlocal = (hitPointOrig.z() > 0.) ? 
        theZHadChannelBoundaries[i-5] + (zWidthHAD/2.) : theZHadChannelBoundaries[i-5] - (zWidthHAD/2.);
      zz = (hitPointOrig.z() > 0.) ? zzlocal +  Z0 : zzlocal -  Z0; 
      pos = G4ThreeVector(xx,yy,zz);
      posLocal = G4ThreeVector(xxlocal,yylocal,zzlocal);
    }
    
    oneHit.position = pos;
    oneHit.entryLocal = posLocal;
    oneHit.depth    = channel;
    oneHit.time     = tSlice;
    oneHit.detID    = HcalZDCDetId(section,side,channel);


    // Note: coodinates of hit are relative to center of detector (X0,Y0,Z0) 
    hitPoint.setX(hitPointOrig.x()-X0);
    hitPoint.setY(hitPointOrig.y()-Y0);
    double setZ= (hitPointOrig.z()> 0.) ? hitPointOrig.z()- Z0 : fabs(hitPointOrig.z()) - Z0;
    hitPoint.setZ(setZ);

    int dE = getEnergyFromLibrary(hitPoint,momDir,energy,parCode,section,side,channel);

    int iparCode = encodePartID(parCode);
    if (iparCode == 0 ) {
      oneHit.DeEM  = dE; 
      oneHit.DeHad = 0.;
    } else {
      oneHit.DeEM  = 0; 
      oneHit.DeHad = dE;
    }
    
    hits.push_back(oneHit);
    
    LogDebug("ZdcShower")
      //std::cout
      << "\nZdcShowerLibrary:Generated Hit " << nHit 
      <<" orig hit pos " << hitPointOrig
      <<" orig hit pos local coord" << hitPoint
      <<" new position " << (hits[nHit].position) 
      <<" Channel " <<(hits[nHit].depth) 
      <<" side "<< side  
      <<" Time " <<(hits[nHit].time)
      <<" DetectorID " << (hits[nHit].detID)
      <<" Had Energy " << (hits[nHit].DeHad)
      <<" EM Energy  " << (hits[nHit].DeEM)
      <<"\n";
      nHit++;
  }
  return hits;
}


int ZdcShowerLibrary::getEnergyFromLibrary(G4ThreeVector hitPoint, G4ThreeVector momDir, double energy,
					   G4int parCode,HcalZDCDetId::Section section, bool side, int channel){
  int nphotons = -1;
  
  energy =energy/GeV;

  LogDebug("ZdcShower") 
  //std::cout
    <<"\n ZdcShowerLibrary::getEnergyFromLibrary input/output variables:"
    <<" phi: "<<59.2956*momDir.phi()
    <<" theta: "<<59.2956*momDir.theta()
    <<" xin : "<<hitPoint.x()
    <<" yin : "<<hitPoint.y()
    <<" zin : "<<hitPoint.z()
    <<" track en: " <<energy<< "(GeV)"
    <<" section: "<<section
    <<" side: "<<side
    <<" channel: "<< channel
    <<" partID: "<<parCode;
  //std::cout<<std::endl;
    
  // these varables are not used for now
  //float phi   = 59.2956*momDir.phi();
  //float theta = 59.2956*momDir.theta();
  //float zin = hitPoint.z();
  //int isection = int(section);
  //int iside = (side)? 1 : 2;
     
  int iparCode  = encodePartID(parCode);

  double eav = 0.;
  double esig = 0.;
  double edis = 0.;

  float xin = hitPoint.x(); 
  float yin = hitPoint.y(); 
  float fact = 0.;

  if(section == 1 && iparCode !=0){
    if(channel < 5 )
      if(((theXChannelBoundaries[channel-1])< (xin + X0)) && ((xin + X0)<= theXChannelBoundaries[channel]))fact= 0.18;
    if(channel ==5 )
      if(theXChannelBoundaries[channel-1]< xin + X0)fact = 0.18;
  }

  if(section == 2 && iparCode !=0){    
    if(channel == 1)fact = 0.34;
    if(channel == 2)fact = 0.24;
    if(channel == 3)fact = 0.17;
    if(channel == 4)fact = 0.07;
  }
  if(section == 1 && iparCode ==0){							
    if(channel < 5 )
      if(((theXChannelBoundaries[channel-1])< (xin + X0)) && ((xin + X0)<= theXChannelBoundaries[channel]))fact= 1.;
    if(channel ==5 )
      if(theXChannelBoundaries[channel-1]< xin + X0)fact = 1.0;
  }

  //change to cm for parametrization
  yin = yin/cm;
  xin = xin/cm;

  if (iparCode==0){          
    eav = ((((((-0.0002*xin-2.0e-13)*xin+0.0022)*xin+1.0e-11)*xin-0.0217)*xin-3.0e-10)*xin+1.0028)*
      (((0.0001*yin + 0.0056)*yin + 0.0508)*yin + 1.0)*300.0*pow((energy/300.0),0.99); // EM
    esig = ((((((0.0005*xin - 1.0e-12)*xin - 0.0052)*xin + 5.0e-11)*xin + 0.032)*xin - 
	     2.0e-10)*xin + 1.0)*(((0.0006*yin + 0.0071)*yin - 0.031)*yin + 1.0)*30.0*pow((energy/300.0),0.54);// EM
    edis = 1.0;
  }else{
  eav = ((((((-0.0002*xin-2.0e-13)*xin+0.0022)*xin+1.0e-11)*xin-0.0217)*xin-3.0e-10)*xin+1.0028)*
    (((0.0001*yin + 0.0056)*yin + 0.0508)*yin + 1.0)*300.0*pow((energy/300.0),1.12);// HD
  esig = ((((((0.0005*xin - 1.0e-12)*xin - 0.0052)*xin + 5.0e-11)*xin + 0.032)*xin - 
	   2.0e-10)*xin + 1.0)*(((0.0006*yin + 0.0071)*yin - 0.031)*yin + 1.0)*54.0*pow((energy/300.0),0.93); //HD
  edis = 3.0;
  }
  
  if(eav <0. || edis <0.){
    LogDebug("ZdcShower") 
      <<" Negative everage energy from parametrization \n" 
      <<" xin: "<<xin<< "(cm)"
      <<" yin: "<<yin<< "(cm)"
      <<" track en: " <<energy<< "(GeV)"
      <<" eaverage: "<<eav/GeV << " (GeV)"
      <<" esigma: "<<esig/GeV << "  (GeV)"
      <<" edist: "<<edis  << " (GeV)"
     <<" dE hit: "<<nphotons/GeV<<  " (GeV)";
    return 0;
  }
  
  // Convert from GeV to MeV for the code
  eav = eav*GeV;
  esig= esig*GeV;

  while(nphotons == -1 || nphotons > int(eav + 5.*esig))
    nphotons = (int)(fact*photonFluctuation(eav, esig, edis));
  
  LogDebug("ZdcShower") 
    //std::cout
    <<" track en: " <<energy<< "(GeV)"
    <<" eaverage: "<<eav/GeV << " (GeV)"
    <<" esigma: "<<esig/GeV << "  (GeV)"
    <<" edist: "<<edis  << " (GeV)"
    <<" dE hit: "<<nphotons/GeV<<  " (GeV)";
  //std::cout<<std::endl;
  
  return nphotons; 
}

int ZdcShowerLibrary::photonFluctuation(double eav, double esig,double edis){
  int nphot=0;
  double efluct = 0.;
  if(edis == 1.0) efluct = eav+esig*CLHEP::RandGaussQ::shoot();
  if(edis == 3.0) efluct = eav+esig*CLHEP::RandLandau::shoot();
  nphot = int(efluct);
  if(nphot<0)nphot = 0;
  return nphot;
}

int ZdcShowerLibrary::encodePartID(G4int parCode){
  G4int iparCode = 1;
  if (parCode == emPDG ||
      parCode == epPDG ||
      parCode == gammaPDG) {
    iparCode = 0;
  } else { return iparCode; }
  return iparCode;
}
