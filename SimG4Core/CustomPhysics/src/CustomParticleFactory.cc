#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <locale>     

#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <G4ParticleTable.hh>
#include "G4DecayTable.hh"
#include <G4PhaseSpaceDecayChannel.hh>
#include "G4ProcessManager.hh"

using namespace CLHEP;

bool CustomParticleFactory::loaded = false;
std::set<G4ParticleDefinition *> CustomParticleFactory::m_particles;

bool CustomParticleFactory::isCustomParticle(G4ParticleDefinition *particle)
{
  return (m_particles.find(particle)!=m_particles.end());
}

void CustomParticleFactory::loadCustomParticles(const std::string & filePath){
  if(loaded) return;
  loaded = true;

  std::ifstream configFile(filePath.c_str());

  std::string line;
  edm::LogInfo("CustomPhysics") << "Reading Custom Particle and G4DecayTable from " << filePath; 
  // This should be compatible IMO to SLHA 
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  while(getline(configFile,line)){
    line.erase(0, line.find_first_not_of(" \t"));         // Remove leading whitespace.
    if (line.length()==0 || line.at(0) == '#') continue;  // Skip blank lines and comments.  
    if (ToLower(line).find("block") < line.npos &&        // The mass table begins with a line containing "BLOCK MASS".  
	ToLower(line).find("mass")  < line.npos) {
      edm::LogInfo("CustomPhysics") << " Retrieving mass table."; 
      getMassTable(&configFile);
    }
    if(line.find("DECAY")<line.npos){
      int pdgId;
      double width; 
      std::string tmpString;
      std::stringstream lineStream(line);
      lineStream >> tmpString >> pdgId >> width; // assume SLHA format, e.g.: DECAY  1000021  5.50675438E+00   # gluino decays
      edm::LogInfo("CustomPhysics") << "G4DecayTable: pdgID, width " << pdgId << ",  " << width; 
      G4DecayTable* aDecayTable = getDecayTable(&configFile, pdgId);      
      G4ParticleDefinition *aParticle     = theParticleTable->FindParticle(pdgId);
      G4ParticleDefinition *aAntiParticle = theParticleTable->FindAntiParticle(pdgId);
      if (!aParticle) continue;    
      aParticle->SetDecayTable(aDecayTable); 
      aParticle->SetPDGStable(false);
      aParticle->SetPDGLifeTime(1.0/(width*GeV)*6.582122e-22*MeV*s);      
      if(aAntiParticle && aAntiParticle->GetPDGEncoding()!=pdgId){	
	aAntiParticle->SetDecayTable(getAntiDecayTable(pdgId,aDecayTable)); 
	aAntiParticle->SetPDGStable(false);
	aAntiParticle->SetPDGLifeTime(1.0/(width*GeV)*6.582122e-22*MeV*s);
      }         
    }
  }
}


void CustomParticleFactory::addCustomParticle(int pdgCode, double mass, const std::string & name){
  
  if(abs(pdgCode)%100 <14 && abs(pdgCode) / 1000000 == 0){
    edm::LogError("") << "Pdg code too low " << pdgCode << " "<<abs(pdgCode) / 1000000; 
    return;
  }
  
  
  /////////////////////// Check!!!!!!!!!!!!!
  G4String pType="custom";
  G4String pSubType="";
  G4double spectatormass;
  G4ParticleDefinition* spectator; 
  //////////////////////
  if(CustomPDGParser::s_isRHadron(pdgCode)) pType = "rhadron";
  if(CustomPDGParser::s_isSLepton(pdgCode)) pType = "sLepton";
  if(CustomPDGParser::s_isMesonino(pdgCode)) pType = "mesonino";
  if(CustomPDGParser::s_isSbaryon(pdgCode)) pType = "sbaryon";
 
  double massGeV =mass*GeV;
  double width = 0.0*MeV;
  double charge = eplus* CustomPDGParser::s_charge(pdgCode);
  if (name.compare(0,4,"~HIP") == 0)
    {

      if ((name.compare(0,7,"~HIPbar") == 0))  {std::string str = name.substr (7); charge=eplus*atoi(str.c_str())/3.;}
      else {std::string str = name.substr (4); charge=eplus*atoi(str.c_str())*-1./3.;  }
    }
  if (name.compare(0,9,"anti_~HIP") == 0)
    {

      if ((name.compare(0,12,"anti_~HIPbar") == 0))  {std::string str = name.substr (12); charge=eplus*atoi(str.c_str())*-1./3.;}
      else {std::string str = name.substr (9); charge=eplus*atoi(str.c_str())*1./3.;  }
    }
  int spin =  (int)CustomPDGParser::s_spin(pdgCode)-1;
  int parity = +1;
  int conjugation = 0;
  int isospin = 0;
  int isospinZ = 0;
  int gParity = 0;
  int lepton = 0;  //FIXME:
  int baryon = 1;  //FIXME: 
  bool stable = true;
  double lifetime = -1;

  if(CustomPDGParser::s_isDphoton(pdgCode)){
    pType = "darkpho";
    spin = 2;
    parity = -1;
    conjugation = -1;
    isospin = 0;
    isospinZ = 0;
    gParity = 0;
    lepton = 0;
    baryon =0;
    stable = true;
    lifetime = -1;
  }
 
  G4DecayTable *decaytable = NULL;
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();

  CustomParticle *particle  = new CustomParticle(name, massGeV, width, charge, spin, 
						 parity, conjugation, isospin, isospinZ,
						 gParity, pType, lepton, baryon, pdgCode,
						 stable, lifetime, decaytable);
 
  if(pType == "rhadron" && name!="~g"){  
    G4String cloudname = name+"cloud";
    G4String cloudtype = pType+"cloud";
    spectator = theParticleTable->FindParticle(1000021);
    spectatormass = spectator->GetPDGMass();
    G4double cloudmass = mass-spectatormass/GeV;
    CustomParticle *tmpParticle  = new CustomParticle(
						      cloudname,           cloudmass * GeV ,        0.0*MeV,  0 , 
						      0,              +1,             0,          
						      0,              0,             0,             
						      cloudtype,               0,            +1, 0,
						      true,            -1.0,          NULL );
    particle->SetCloud(tmpParticle);
    particle->SetSpectator(spectator);
    
    edm::LogInfo("CustomPhysics")<<name<<" being assigned "
				 <<particle->GetCloud()->GetParticleName()
				 <<" and "<<particle->GetSpectator()->GetParticleName(); 
    edm::LogInfo("CustomPhysics")<<"Masses: "
				 <<particle->GetPDGMass()/GeV<<" Gev, "
				 <<particle->GetCloud()->GetPDGMass()/GeV<<" GeV and "
				 <<particle->GetSpectator()->GetPDGMass()/GeV<<" GeV."; 
  } else if(pType == "mesonino" || pType == "sbaryon") {
    int sign=1;
    if(pdgCode < 0 ) sign=-1;

    G4String cloudname = name+"cloud";
    G4String cloudtype = pType+"cloud";
    if(CustomPDGParser::s_isstopHadron(pdgCode)) {
      spectator = theParticleTable->FindParticle(1000006*sign);
    }
    else { 
      if (CustomPDGParser::s_issbottomHadron(pdgCode)) {
	spectator = theParticleTable->FindParticle(1000005*sign);
      } else {
        spectator = 0;
        edm::LogError("CustomPhysics")<< " Cannot find spectator parton";
      }
    }
    spectatormass = spectator->GetPDGMass();
    G4double cloudmass = mass-spectatormass/GeV;
    CustomParticle *tmpParticle  = new CustomParticle(
                                                      cloudname,           cloudmass * GeV ,        0.0*MeV,  0 ,
                                                      0,              +1,             0,
                                                      0,              0,             0,
                                                      cloudtype,               0,            +1, 0,
                                                      true,            -1.0,          NULL );
    particle->SetCloud(tmpParticle);
    particle->SetSpectator(spectator);

    edm::LogInfo("CustomPhysics")<<name<<" being assigned "
				 <<particle->GetCloud()->GetParticleName()
				 <<" and "<<particle->GetSpectator()->GetParticleName(); 
    edm::LogInfo("CustomPhysics")<<"Masses: "
				 <<particle->GetPDGMass()/GeV<<" Gev, "
				 <<particle->GetCloud()->GetPDGMass()/GeV<<" GeV and "
				 <<particle->GetSpectator()->GetPDGMass()/GeV<<" GeV."; 
  }
  else{
    particle->SetCloud(0);
    particle->SetSpectator(0);
  } 
  m_particles.insert(particle);
}

void  CustomParticleFactory::getMassTable(std::ifstream *configFile) {

  int pdgId;
  double mass;
  std::string name, tmp;
  std::string line;
  // This should be compatible IMO to SLHA 
  while (getline(*configFile,line)) {
    line.erase(0, line.find_first_not_of(" \t"));         // remove leading whitespace
    if (line.length()==0 || line.at(0) == '#') continue;  // skip blank lines and comments
    if (ToLower(line).find("block") < line.npos) {
      edm::LogInfo("CustomPhysics") << " Finished the Mass Table "; 
      break;
    }
    std::stringstream sstr(line);
    sstr >> pdgId >> mass >> tmp >> name;  // Assume SLHA format, e.g.: 1000001 5.68441109E+02 # ~d_L 

    edm::LogInfo("CustomPhysics") << "Calling addCustomParticle for pdgId: " << pdgId 
				  << ", mass " << mass << ", name " << name; 
    addCustomParticle(pdgId, fabs(mass), name);
    ////Find SM particle partner and check for the antiparticle.
    int pdgIdPartner = pdgId%100;
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *aParticle = theParticleTable->FindParticle(pdgIdPartner);
    //Add antiparticles for SUSY particles only, not for rHadrons.
    edm::LogInfo("CustomPhysics") << "Found aParticle = " << aParticle
				  << ", pdgId = " << pdgId
				  << ", pdgIdPartner = " << pdgIdPartner  
				  << ", CustomPDGParser::s_isRHadron(pdgId) = " << CustomPDGParser::s_isRHadron(pdgId)    
				  << ", CustomPDGParser::s_isstopHadron(pdgId) = " << CustomPDGParser::s_isstopHadron(pdgId); 
    
    if (aParticle && 
	!CustomPDGParser::s_isRHadron(pdgId)    && 
	!CustomPDGParser::s_isstopHadron(pdgId) && 
	pdgId!=1000006 && 
	pdgId!=-1000006  && 
	pdgId!=25 && 
	pdgId!=35 && 
	pdgId!=36 && 
	pdgId!=37){ 
      int sign = aParticle->GetAntiPDGEncoding()/pdgIdPartner;   
      edm::LogInfo("CustomPhysics") << "Found sign = " << sign 
				    << ", aParticle->GetAntiPDGEncoding() " << aParticle->GetAntiPDGEncoding() 
				    << ", pdgIdPartner = " << pdgIdPartner;   
      if(abs(sign)!=1) {
	edm::LogInfo("CustomPhysics")<<"sgn: "<<sign<<" a "
				     <<aParticle->GetAntiPDGEncoding()
				     <<" b "<<pdgIdPartner; 
	aParticle->DumpTable();
      }
      if(sign==-1 && pdgId!=25 && pdgId!=35 && pdgId!=36 && pdgId!=37 && pdgId!=1000039){
	tmp = "anti_"+name;
	edm::LogInfo("CustomPhysics") << "Calling addCustomParticle for antiparticle with pdgId: " << -pdgId 
				      << ", mass " << mass << ", name " << tmp; 
	addCustomParticle(-pdgId, mass, tmp);
	theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(-pdgId);
      }
      else theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(pdgId);      
    }
    
    if(pdgId==1000039) theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(pdgId); // gravitino     
    if(pdgId==1000024 || pdgId==1000037 || pdgId==37) {   
      tmp = "anti_"+name;
      edm::LogInfo("CustomPhysics") << "Calling addCustomParticle for antiparticle (2) with pdgId: " << -pdgId 
				    << ", mass " << mass << ", name " << tmp; 
      addCustomParticle(-pdgId, mass, tmp);
      theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(-pdgId);
    }

  }
}

G4DecayTable*  CustomParticleFactory::getDecayTable(std::ifstream *configFile, int pdgId) {

  double br;
  int nDaughters;
  std::vector<int> pdg(4);
  std::string line;
  std::vector<std::string> name(4);

  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();

  std::string parentName = theParticleTable->FindParticle(pdgId)->GetParticleName();
  G4DecayTable *decaytable= new G4DecayTable();

  while(getline(*configFile,line)) {
    
    line.erase(0, line.find_first_not_of(" \t"));         // remove leading whitespace
    if (line.length()==0) continue;                       // skip blank lines 
    if (line.at(0) == '#' && 
	ToLower(line).find("br")  < line.npos &&
	ToLower(line).find("nda") < line.npos) continue;  // skip a comment of the form:  # BR  NDA  ID1  ID2
    if (line.at(0) == '#') {                              // other comments signal the end of the decay block  
      edm::LogInfo("CustomPhysics") << " Finished the Decay Table "; 
      break;
    }
    
    pdg.clear();
    name.clear();

    std::stringstream sstr(line);  
    sstr >> br >> nDaughters;  // assume SLHA format, e.g.:  1.49435135E-01  2  -15  16  # BR(H+ -> tau+ nu_tau)
    edm::LogInfo("CustomPhysics") << " Branching Ratio: " << br << ", Number of Daughters: " << nDaughters; 
    if (nDaughters > 4) {
      edm::LogError("CustomPhysics") << "Number of daughters is too large (max = 4): " << nDaughters << " for pdgId: " << pdgId; 
      break; 
    }
    for(int i=0; i<nDaughters; i++) {
      sstr >> pdg[i];
      edm::LogInfo("CustomPhysics") << " Daughter ID " << pdg[i]; 
    } 
    for (int i=0;i<nDaughters;i++) {
      if (!theParticleTable->FindParticle(pdg[i])) {
	edm::LogWarning("CustomPhysics")<<pdg[i]<<" CustomParticleFactory::getDecayTable():  not found in the table!"; 
	continue;
      }
      name[i] =  theParticleTable->FindParticle(pdg[i])->GetParticleName();
    }
    ////Set the G4 decay
    G4PhaseSpaceDecayChannel *aDecayChannel = new G4PhaseSpaceDecayChannel(parentName, br, nDaughters,
									   name[0],name[1],name[2],name[3]);    
    decaytable->Insert(aDecayChannel);
  
  }

  return decaytable;
}

G4DecayTable*  CustomParticleFactory::getAntiDecayTable(int pdgId,  G4DecayTable *theDecayTable) {

  std::vector<std::string> name(4);
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();

  std::string parentName = theParticleTable->FindParticle(-pdgId)->GetParticleName();
  G4DecayTable *decaytable= new G4DecayTable();

  for(int i=0;i<theDecayTable->entries();i++){
    //G4PhaseSpaceDecayChannel *theDecayChannel = theDecayTable->GetDecayChannel(i); 
    G4VDecayChannel *theDecayChannel = theDecayTable->GetDecayChannel(i); 
    for(int j=0;j<theDecayChannel->GetNumberOfDaughters();j++){
      int id = theDecayChannel->GetDaughter(j)->GetAntiPDGEncoding();
      std::string nameTmp = theParticleTable->FindParticle(id)->GetParticleName();
      name[j] = nameTmp;
    }
    G4PhaseSpaceDecayChannel *aDecayChannel = 
      new G4PhaseSpaceDecayChannel(parentName, 
				   theDecayChannel->GetBR(),
				   theDecayChannel->GetNumberOfDaughters(),
				   name[0],name[1],name[2],name[3]);  
    decaytable->Insert(aDecayChannel);
  }
  return decaytable;
}


std::string CustomParticleFactory::ToLower(std::string str) {
  std::locale loc;
  for (std::string::size_type i=0; i<str.length(); ++i)
    str.at(i) = std::tolower(str.at(i),loc);
  return str; 	
}

