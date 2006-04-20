#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

bool CustomParticleFactory::loaded = false;
std::set<G4ParticleDefinition *> CustomParticleFactory::m_particles;

bool CustomParticleFactory::isCustomParticle(G4ParticleDefinition * particle)
{
    return (m_particles.find(particle)!=m_particles.end());
}

void CustomParticleFactory::loadCustomParticles()
{
    if(loaded) return;
    loaded = true;
    std::ifstream configFile("customparticles");
    std::string pType="custom";
    std::string pSubType="";
    double mass;
    int pdgCode;
    std::string name,line;
    // This should be compatible IMO to SLHA 
    while (getline(configFile,line))
    {
	std::string::size_type beg_idx,end_idx;
     
	beg_idx = line.find_first_not_of("\t #");
	if (beg_idx > 0 && line[beg_idx-1] == '#') continue; 
	end_idx = line.find_first_of( "\t ", beg_idx);
	if (end_idx == std::string::npos) continue;
	pdgCode = atoi(line.substr( beg_idx, end_idx - beg_idx ).c_str());
     
	beg_idx = line.find_first_not_of("\t ",end_idx);
	end_idx = line.find_first_of( "\t #", beg_idx);
	if (end_idx == std::string::npos) continue;
	mass  = atof(line.substr( beg_idx, end_idx - beg_idx ).c_str());

	beg_idx = line.find_first_not_of("\t# ",end_idx);
	end_idx = line.length();
	name = line.substr( beg_idx, end_idx - beg_idx );
	while (name.c_str()[0] == ' ') name.erase(0,1);
	while (name[name.size()-1] == ' ') name.erase(name.size()-1,1);
	int pos;
	while ((pos = name.find(" "))>0)
	{
	    name=name.replace(pos,1,"_");
	    std::cout << name << std::endl;
	}
     
	if (abs(pdgCode) / 1000000 == 0)
	{
	    std::cout << "Pdg code too low " << pdgCode 
		      << " " <<abs(pdgCode) / 1000000  << std::endl;
	    continue;
	}
	std::cout << name << std::endl;
    
	if (CustomPDGParser::s_isRHadron(pdgCode)) pType = "rhadron";
	if (CustomPDGParser::s_isSLepton(pdgCode)) pType = "sLepton";
      
	CustomParticle * particle = 
	    new CustomParticle(name,mass*GeV,0.0*MeV,
			       eplus * CustomPDGParser::s_charge(pdgCode),
			       (int)CustomPDGParser::s_spin(pdgCode)-1,
			       +1,0,0,0,0,pType,0,+1,pdgCode,true,-1.0,0);
	m_particles.insert(particle);
    }
}

