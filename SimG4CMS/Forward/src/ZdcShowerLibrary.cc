
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
				 edm::ParameterSet const & p) : zdc(0){
  edm::ParameterSet m_HS   = p.getParameter<edm::ParameterSet>("ZdcShowerLibrary");
  edm::FileInPath fp       = m_HS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  verbose                  = m_HS.getUntrackedParameter<bool>("Verbosity",false);

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nfile = pTreeName.c_str();
  zdc  = TFile::Open(nfile);

  if (!zdc->IsOpen()) { 
    edm::LogError("ZDCShower") << "ZDCShowerLibrary: opening " << nfile 
                               << " failed";
    throw cms::Exception("Unknown", "ZDCShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  }else {
    edm::LogInfo("ZDCShower") << "ZDCShowerLibrary: opening " << nfile
			      << " successfully"; 
  }

  binInfo = (TH1I*)zdc-> Get("binInfo");
  maxBitsInfo =(TH1I*)  zdc->Get("maxBitsInfo");
  lutPartIDLut =(TH1I*)  zdc->Get("lutPartIDLut");
  lutMatrixEAverage=(TH2I*)  zdc ->Get("lutMatrixEAverage");
  lutMatrixESigma =(TH2I*)  zdc ->Get("lutMatrixESigma");
  lutMatrixEDist  =(TH2I*)  zdc->Get("lutMatrixEDist");
  randomGen = new TRandom();

  if(!binInfo || !maxBitsInfo || !lutPartIDLut ||
     !lutMatrixEAverage || ! lutMatrixESigma || !lutMatrixEDist){
    edm::LogError("ZDCShowerLibrary") << "ZDCShowerLibrary: One of the LuT tablesis not "
			      << "present in shower LuT input file";
    throw cms::Exception("Unknown", "ZDCShowerLibrary")
      << "One of the LuTs is abscent\n";
  }

  ienergyBin =  binInfo->GetBinContent(1);
  ithetaBin =  binInfo->GetBinContent(2);
  iphiBin = binInfo->GetBinContent(3);
  isideBin =  binInfo->GetBinContent(4);
  isectionBin = binInfo->GetBinContent(5);
  ichannelBin = binInfo->GetBinContent(6);
  ixBin =  binInfo->GetBinContent(7);
  iyBin =  binInfo->GetBinContent(8);
  izBin = binInfo->GetBinContent(9);
  iPIDBin =  binInfo->GetBinContent(10);


  std::cout<<"(##) Binning Information: "
	   <<" energy: "<<ienergyBin
	   <<" theta: "<<ithetaBin
	   <<" phi: "<<iphiBin
	   <<" side: "<<isideBin
	   <<" section: "<<isectionBin
	   <<" channel: "<<ichannelBin
	   <<" X: "<<ixBin
	   <<" Y: "<<iyBin
	   <<" Z: "<<izBin
	   <<" PID: "<<iPIDBin<<std::endl;
  

  maxBitsEnergy = maxBitsInfo->GetBinContent(1);
  maxBitsTheta = maxBitsInfo->GetBinContent(2);
  maxBitsPhi = maxBitsInfo->GetBinContent(3);
  maxBitsSide = maxBitsInfo->GetBinContent(4);
  maxBitsSection = maxBitsInfo->GetBinContent(5);
  maxBitsChannel = maxBitsInfo->GetBinContent(6);
  maxBitsX = maxBitsInfo->GetBinContent(7);
  maxBitsY = maxBitsInfo->GetBinContent(8);
  maxBitsZ = maxBitsInfo->GetBinContent(9);
  maxBitsPID = maxBitsInfo->GetBinContent(10);

  std::cout<<"(##) MaxBits Information: "
	   <<" energy: "<<maxBitsEnergy 
	   <<" theta: "<<maxBitsTheta
	   <<" phi: "<< maxBitsPhi
	   <<" side: "<< maxBitsSide
	   <<" section: "<<maxBitsSection
	   <<" channel: "<<maxBitsChannel
	   <<" X: "<<maxBitsX
	   <<" Y: "<<maxBitsY
	   <<" Z: "<<maxBitsZ
	   <<" PID: "<<maxBitsPID
	   <<std::endl;
 
}

ZdcShowerLibrary::~ZdcShowerLibrary() {
  if (lutMatrixEAverage) delete lutMatrixEAverage;
  if (lutMatrixESigma) delete lutMatrixESigma;
  if (lutMatrixEDist) delete lutMatrixEDist;
  if (lutPartIDLut) delete lutPartIDLut;
  if (random) delete randomGen;
  if (zdc) zdc->Close();
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

    if (parCode == emPDG ||
	parCode == epPDG ||
	parCode == gammaPDG ) {
      oneHit.DeEM  = dE; oneHit.DeHad = 0.;
    } else {
      oneHit.DeEM  = 0; oneHit.DeHad = dE;
    }

    hits.push_back(oneHit);
    
    std::cout<< "Generated Hits " << nHit 
	     <<" original hit position " << hitPoint
	     <<" position " << (hits[nHit].position) 
	     <<" Depth " <<(hits[nHit].depth) 
	     <<" side "<< side  
	     <<" Time " <<(hits[nHit].time)
	     <<" DetectorID " << (hits[nHit].detID)
	     <<" Had Energy " << (hits[nHit].DeHad)
	     <<" EM Energy  " << (hits[nHit].DeEM)
	     <<std::endl;

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
 
  std::cout<<"GetEnergy variables: *---> "
	   <<" phi: "<<59.2956*momDir.phi()
	   <<" theta: "<<59.2956*momDir.theta()
	   <<" xin : "<<hitPoint.x()
	   <<" yin : "<<hitPoint.y()
	   <<" zin : "<<hitPoint.z()
	   <<" en: " <<energy
	   <<" section: "<<section
	   <<" side: "<<side
	   <<" partID: "<<parCode
	   <<std::endl;


  int iphi   = int(59.2956*momDir.phi()/float(iphiBin));
  int itheta = int(59.2956*momDir.theta()/float(ithetaBin));
  int ixin = int(hitPoint.x()/float(ixBin)); 
  int iyin = int(hitPoint.y()/float(iyBin)); 
  int izin = int(hitPoint.z()/float(izBin)); 
  int ienergy = int(energy/float(ienergyBin));
  int isection = int(section);
  int iside = (side)? 1 : 2;     
  int iparCode  = encodeParID(parCode);

  std::cout<<"Binned variables: #---> "
	   <<" iphi: "<<iphi
	   <<" itheta: "<<itheta
	   <<" ixin : "<<ixin
	   <<" iyin : "<<iyin
	   <<" izin : "<<izin
	   <<" ien: " <<ienergy
	   <<" isection: "<<isection
	   <<" iside: "<<iside
	   <<" iparcode "<<iparCode
	   <<std::endl;
    
  iLutIndex1 = encode1(iphi,itheta,ixin,iyin,izin);
  iLutIndex2 = encode2(ienergy,isection,iside,channel,iparCode);  

  double eav  = lutMatrixEAverage->GetBinContent(iLutIndex1,iLutIndex2);
  double esig = lutMatrixESigma ->GetBinContent(iLutIndex1,iLutIndex2);
  double edis = lutMatrixEDist  ->GetBinContent(iLutIndex1,iLutIndex2);
  
  nphotons = photonFluctuation(eav, esig,edis);

  std::cout<<" ########photons---->"<<nphotons<<std::endl;
  return nphotons;
}

int ZdcShowerLibrary::photonFluctuation(double eav, double esig,double edis){
  int nphot=0;
  double efluct = 0.;
  if(edis == 1.0)efluct = randomGen->Gaus(eav,esig);
  if(edis == 2.0)efluct = randomGen->Landau(eav,esig);
  nphot = int(efluct);
  return nphot;
}

int ZdcShowerLibrary::encodeParID(int parID){
  int partID = 0;
  for(int i = 1; i <= maxBitsPID; i++){
    partID = i;
    if(parID==lutPartIDLut->GetBinContent(i))break;
  }
  return partID;
}
      
void ZdcShowerLibrary::decode1(const unsigned long & lutidx,int& iphi, int& itheta, int& ix,int& iy, int& iz){
  // todo: make dependent on maxBits variables
  int iphisgn = (lutidx>>29)&1;
  int ithsgn  = (lutidx>>28)&1;
  int izsgn   = (lutidx>>27)&1;
  int iysgn   = (lutidx>>26)&1;
  int ixsgn   = (lutidx>>25)&1;
  itheta = (lutidx>>19)&63;
  iphi = (lutidx>>13)&63;
  iz = (lutidx>>8)&31;
  iy = (lutidx>>4)&15;
  ix = (lutidx)&15;

  if(ithsgn == 0)itheta*= -1;
  if(iphisgn == 0)iphi*= -1;
  if(izsgn == 0)iz*= -1;
  if(iysgn == 0)iy*= -1;
  if(ixsgn == 0)ix*= -1;

  std::cout<<"    %d1: "
	   <<iphi<<" "
	   <<itheta<<" "
	   <<ix<<" "
	   <<iy<<" "
	   <<iz<<" %"
	   <<lutidx;//<<std::endl;
  return;
}

void ZdcShowerLibrary::decode2(const unsigned long & lutidx,int& ien, int& isec, int& isid, int& icha, int& iparID){
  // todo: make dependent on maxBits variables
  ien = (lutidx>>12)&511;
  iparID = (lutidx>>6)&63;
  icha = (lutidx>>3)&7;
  isec = (lutidx>>1)&3;
  isid = 1 +(lutidx&1);

  std::cout<<"    *d2: "
           <<ien<<" "
           <<isec<<" "
	   <<isid<<" "
	   <<icha<<" "
	   <<iparID<<" *"
	   <<lutidx;//<<std::endl;
  return;
}

unsigned long ZdcShowerLibrary::encode1(int iphi, int itheta, int ix, int iy, int iz){
 // todo: make dependent on maxBits variables
 std::cout<<"    +e1: "
	  <<iphi<<" "
	  <<itheta<<" "
	  <<ix<<" "
	  <<iy<<" "
	  <<iz<<" +";
  int ixsgn = 1;
  if(ix<0){
    ix = -ix;
    ixsgn = 0;
  }
  int iysgn = 1;
  if(iy<0){
    iy = -iy;
    iysgn = 0;
  }
  int izsgn = 1;  
  if(iz<0){
    iz = -iz;
    izsgn = 0;
  }
  int ithsgn = 1;
  if(itheta<0){
    itheta = -itheta;
    ithsgn = 0;
  }
  int iphsgn = 1;
  if(iphi<0){
    iphi = -iphi;
    iphsgn = 0;
  }

  unsigned long lutindex = (iphsgn&1)<<29;
  lutindex += (ithsgn&1) <<28;
  lutindex += (izsgn&1)  <<27;
  lutindex += (iysgn&1)  <<26;
  lutindex += (ixsgn&1)  <<25;    //bits 25
  lutindex += (itheta&63)<<19;    //bits 19-24
  lutindex += (iphi&63)  <<13;    //bits 13-18
  lutindex += (iz&31)    <<8;     //bits  8-12
  lutindex += (iy&15)    <<4;     //bits  4- 7
  lutindex += (ix&15);            //bits  0- 3

  std::cout<<lutindex;//<<std::endl;
  int newiphi, newitheta, newix, newiy, newiz; 
  decode1(lutindex, newiphi, newitheta, newix, newiy, newiz);    
  return lutindex;

}

unsigned long ZdcShowerLibrary::encode2(int ien, int isec, int isid, int icha, int iparID){
 // todo: make dependent on maxBits variables
  unsigned long  lutindex = (ien&511)<<12;   //bits  12-20
  lutindex += (iparID&63)<<6;                //bits  6-11
  lutindex += (icha&7)   <<3;                //bits  3- 5
  lutindex += (isec&3)   <<1;                //bits  1- 2
  lutindex += ((isid-1)&1);                  //bits  0

  std::cout<<"    ^e2: "
           <<ien<<" "
           <<isec<<" "
	   <<isid<<" "
	   <<icha<<" "
	   <<iparID<<" ^"
	   <<lutindex;//<<std::endl;

  int newien, newisec, newisid, newicha, newipar; 
  decode2(lutindex, newien, newisec, newisid, newicha, newipar);    
  return lutindex;
}






