#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <set>

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
  // This should be compatible IMO to SLHA 
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  while(getline(configFile,line)){
    if(line.find("PDG code")<line.npos) getMassTable(&configFile);
    if(line.find("DECAY")<line.npos){
    int pdgId;
    double width; 
    std::string tmpString;
    std::stringstream lineStream(line);
    lineStream>>tmpString>>pdgId>>width; 
    G4DecayTable* aDecayTable = getDecayTable(&configFile, pdgId);      
      G4ParticleDefinition *aParticle = theParticleTable->FindParticle(pdgId);
      G4ParticleDefinition *aAntiParticle = theParticleTable->FindAntiParticle(pdgId);
      if(!aParticle) continue;    
      aParticle->SetDecayTable(aDecayTable); 
      aParticle->SetPDGStable(false);
      aParticle->SetPDGLifeTime(1.0/(width*GeV)*6.582122e-22*MeV*s);      
      if(aAntiParticle && aAntiParticle->GetPDGEncoding()!=pdgId){	
	//aAntiParticle->SetDecayTable(getAntiDecayTable(pdgId,aDecayTable));
	aAntiParticle->SetPDGStable(false);
	aParticle->SetPDGLifeTime(1.0/(width*GeV)*6.582122e-22*MeV*s);
      }         
    }
  }
}


void CustomParticleFactory::addCustomParticle(int pdgCode, double mass, const std::string & name){
  
  
  if(pdgCode%100 <25 && abs(pdgCode) / 1000000 == 0){
    edm::LogError("") << "Pdg code too low " << pdgCode << " "<<abs(pdgCode) / 1000000  << std::endl;
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
		    <<" and "<<particle->GetSpectator()->GetParticleName()<<std::endl;
    edm::LogInfo("CustomPhysics")<<"Masses: "
		    <<particle->GetPDGMass()/GeV<<" Gev, "
		    <<particle->GetCloud()->GetPDGMass()/GeV<<" GeV and "
		    <<particle->GetSpectator()->GetPDGMass()/GeV<<" GeV."
		    <<std::endl;
  }else if(pType == "mesonino" || pType == "sbaryon")
  {
      int sign=1;
      if(pdgCode < 0 ) sign=-1;

    G4String cloudname = name+"cloud";
    G4String cloudtype = pType+"cloud";
    spectator = theParticleTable->FindParticle(1000006*sign);
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
                    <<" and "<<particle->GetSpectator()->GetParticleName()<<std::endl;
    edm::LogInfo("CustomPhysics")<<"Masses: "
                    <<particle->GetPDGMass()/GeV<<" Gev, "
                    <<particle->GetCloud()->GetPDGMass()/GeV<<" GeV and "
                    <<particle->GetSpectator()->GetPDGMass()/GeV<<" GeV."
                    <<std::endl;
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
  while(getline(*configFile,line))
    {
    if(tmp.find("Blo")<tmp.npos) break;
    std::stringstream sstr(line);
    sstr >>pdgId>>mass>>tmp>>name;

     addCustomParticle(pdgId, fabs(mass), name);
    ////Find SM particle partner and check for the antiparticle.
    int pdgIdPartner = pdgId%100;
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *aParticle = theParticleTable->FindParticle(pdgIdPartner);
    //Add antiparticles for SUSY particles only, not for rHadrons.
    if(aParticle && !CustomPDGParser::s_isRHadron(pdgId) && !CustomPDGParser::s_isstopHadron(pdgId)&& pdgId!=1000006 && pdgId!=-1000006  && pdgId!=25 && pdgId!=35 && pdgId!=36 && pdgId!=37){ 
    int sign = aParticle->GetAntiPDGEncoding()/pdgIdPartner;   
      if(abs(sign)!=1) {
	std::cout<<"sgn: "<<sign<<" a "
		 <<aParticle->GetAntiPDGEncoding()
		 <<" b "<<pdgIdPartner
		 <<std::endl;
	aParticle->DumpTable();
      }
      if(sign==-1 && pdgId!=25 && pdgId!=35 && pdgId!=36 && pdgId!=37 && pdgId!=1000039){
	  tmp = "anti_"+name;
	  addCustomParticle(-pdgId, mass, tmp);
	  theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(-pdgId);
	}
  	else theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(pdgId);      
    }

    if(pdgId==1000039) theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(pdgId);      
    if(pdgId==1000024 || pdgId==1000037 || pdgId==37) {
      tmp = "anti_"+name;
      addCustomParticle(-pdgId, mass, tmp);
      theParticleTable->FindParticle(pdgId)->SetAntiPDGEncoding(-pdgId);
    }

/*    getline(*configFile,tmp);
    char text[100];
    configFile->get(text,3);
    tmp.clear();
    tmp.append(text);
    if(tmp.find("Bl")<tmp.npos) break;*/
  }
}

G4DecayTable*  CustomParticleFactory::getDecayTable(std::ifstream *configFile, int pdgId) {

  double br;
  int nDaughters;
  std::vector<int> pdg(4);
  std::string tmp;
  std::vector<std::string> name(4);

  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();

  std::string parentName = theParticleTable->FindParticle(pdgId)->GetParticleName();
  G4DecayTable *decaytable= new G4DecayTable();

  getline(*configFile,tmp);

  while(!configFile->eof()){
    pdg.clear();
    name.clear();
    (*configFile)>>br>>nDaughters;
    for(int i=0;i<nDaughters;i++) (*configFile)>>pdg[i];
    getline(*configFile,tmp);
    for(int i=0;i<nDaughters;i++){
      if(!theParticleTable->FindParticle(pdg[i])){
	//std::cout<<pdg[i]<<" CustomParticleFactory::getDecayTable():  not found in the table!"<<std::endl;
        continue;
      }
      name[i] =  theParticleTable->FindParticle(pdg[i])->GetParticleName();
    }
     ////Set the G4 decay
    G4PhaseSpaceDecayChannel *aDecayChannel = new G4PhaseSpaceDecayChannel(parentName, br, nDaughters,
									   name[0],name[1],name[2],name[3]);    
    decaytable->Insert(aDecayChannel);
  
    /////////////////////////

    char text[200];
    configFile->get(text,2);
    tmp.clear();
    tmp.append(text);
    if(tmp.find("#")<tmp.npos) break;  
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
