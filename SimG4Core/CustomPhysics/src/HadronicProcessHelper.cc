#include "Randomize.hh"
#include "G4ParticleTable.hh" 

#include <iostream>
#include <fstream>
#include <string>

#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"
#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace CLHEP;

HadronicProcessHelper::HadronicProcessHelper(const std::string & fileName){

  m_particleTable = G4ParticleTable::GetParticleTable();
  m_proton = m_particleTable->FindParticle("proton");
  m_neutron = m_particleTable->FindParticle("neutron");

  G4String line;

  std::ifstream processListStream (fileName.c_str());

  while(getline(processListStream,line)){
    std::vector<G4String> tokens;

    //Getting a line
    m_readAndParse(line,tokens,"#");
    
    //Important info
    G4String incident = tokens[0];
    G4ParticleDefinition* incidentDef = m_particleTable->FindParticle(incident);
    G4int incidentPDG = incidentDef->GetPDGEncoding();
    m_knownParticles[incidentDef]=true;
    //    G4cout<<"Incident: "<<incident<<G4endl;
    G4String target = tokens[1];
    //    G4cout<<"Target: "<<target<<G4endl;
    
    // Making a ReactionProduct
    ReactionProduct prod;
    for (size_t i = 2; i != tokens.size();i++){
      G4String part = tokens[i];
      if (m_particleTable->contains(part))
	{
	  prod.push_back(m_particleTable->FindParticle(part)->GetPDGEncoding());
	} else {
	  G4Exception("HadronicProcessHelper", "UnkownParticle", FatalException,
		      "Initialization: The reaction product list contained an unknown particle");
	}
    }
    if (target == "proton")
      {
	m_protonReactionMap[incidentPDG].push_back(prod);
      } else if (target == "neutron") {
	m_neutronReactionMap[incidentPDG].push_back(prod);
      } else {
	G4Exception("HadronicProcessHelper", "IllegalTarget", FatalException,
		    "Initialization: The reaction product list contained an illegal target particle");
      }
   }

  processListStream.close();

  m_checkFraction = 0;
  m_n22 = 0;
  m_n23 = 0;
}

G4bool HadronicProcessHelper::applicabilityTester(const G4ParticleDefinition& aPart){
  const G4ParticleDefinition* aP = &aPart; 
  if (m_knownParticles[aP]) return true;
  return false;
}

G4double HadronicProcessHelper::inclusiveCrossSection(const G4DynamicParticle *particle,
						   const G4Element *element){

  //We really do need a dedicated class to handle the cross sections. They might not always be constant

  G4int pdgCode = particle->GetDefinition()->GetPDGEncoding();

  //24mb for gluino-balls
  if(CustomPDGParser::s_isRGlueball(pdgCode)) return 24 * millibarn * element->GetN();
  
  //get quark vector
  std::vector<G4int> quarks=CustomPDGParser::s_containedQuarks(pdgCode);
  
  G4double totalNucleonCrossSection = 0;

  for (std::vector<G4int>::iterator it = quarks.begin();
       it != quarks.end();
       it++)
    {
      // 12mb for each 'up' or 'down'
      if (*it == 1 || *it == 2) totalNucleonCrossSection += 12 * millibarn;
      //  6mb for each 'strange'
      if (*it == 3) totalNucleonCrossSection += 6 * millibarn;
    }
  
  //Convert to xsec per nucleus
  //  return totalNucleonCrossSection * element->GetN();
  return totalNucleonCrossSection * pow(element->GetN(),0.7)*1.25;// * 0.523598775598299;
}

HadronicProcessHelper::ReactionProduct HadronicProcessHelper::finalState(const G4DynamicParticle* incidentDynamicParticle,
   const G4Material *material, G4ParticleDefinition*& target){

//  const G4DynamicParticle* incidentDynamicParticle = track.GetDynamicParticle();

  //-----------------------------------------------
  // Choose n / p as target
  // and get ReactionProductList pointer
  //-----------------------------------------------
  ReactionMap* m_reactionMap;
  //G4Material* material = track.GetMaterial();
  const G4ElementVector* elementVector = material->GetElementVector() ;
  const G4double* numberOfAtomsPerVolume = material->GetVecNbOfAtomsPerVolume();

  G4double numberOfProtons=0;
  G4double numberOfNucleons=0;

  //Loop on elements 
  for ( size_t elm=0 ; elm < material->GetNumberOfElements() ; elm++ )
    {
      //Summing number of protons per unit volume
      numberOfProtons += numberOfAtomsPerVolume[elm]*(*elementVector)[elm]->GetZ();
      //Summing nucleons (not neutrons)
      numberOfNucleons += numberOfAtomsPerVolume[elm]*(*elementVector)[elm]->GetN();
    }
  
  //random decision of the target
  if(G4UniformRand()<numberOfProtons/numberOfNucleons)
    { //target is a proton
      m_reactionMap = &m_protonReactionMap;
      target = m_proton;
    } else { //target is a neutron
      m_reactionMap = &m_neutronReactionMap;
      target = m_neutron;
    }
  
  const G4int incidentPDG = incidentDynamicParticle->GetDefinition()->GetPDGEncoding();

  //Making a pointer directly to the ReactionProductList we are looking at. Makes life easier :-)
  ReactionProductList*  reactionProductList = &((*m_reactionMap)[incidentPDG]);

  //-----------------------------------------------
  // Count processes
  // kinematic check
  // compute number of 2 -> 2 and 2 -> 3 processes
  //-----------------------------------------------

  G4int good22 = 0; //Number of 2 -> 2 processes that are possible
  G4int good23 = 0; //Number of 2 -> 3 processes that are possible
  
  //This is the list to be populated
  ReactionProductList goodReactionProductList;

  for (ReactionProductList::iterator prod_it = reactionProductList->begin();
       prod_it != reactionProductList->end();
       prod_it++){
    G4int secondaries = prod_it->size();
    // If the reaction is not possible we will not consider it
    if(m_reactionIsPossible(*prod_it,incidentDynamicParticle,target)){
      // The reaction is possible. Let's store and count it
      goodReactionProductList.push_back(*prod_it);
      if (secondaries == 2){
	good22++;
      } else if (secondaries ==3) {
	good23++;
      } else {
	G4cerr << "ReactionProduct has unsupported number of secondaries: "<<secondaries<<G4endl;
      }
    } /*else {
      G4cout<<"There was an impossible process"<<G4endl;
      }*/
  }
  //  G4cout<<"The size of the ReactionProductList is: "<<theReactionProductList.size()<<G4endl;

  if (goodReactionProductList.size()==0) G4Exception("HadronicProcessHelper", "NoProcessPossible", FatalException,
						    "GetFinalState: No process could be selected from the given list.");
  // Fill a probability map. Remember total probability
  // 2->2 is 0.15*1/n_22 2->3 uses phase space
  G4double prob22 = 0.15;
  G4double prob23 = 1-prob22; // :-)

  std::vector<G4double> probabilities;
  std::vector<G4bool> twoToThreeFlag;
  
  G4double cumulatedProbability = 0;

  // To each ReactionProduct we assign a cumulated probability and a flag
  // discerning between 2 -> 2 and 2 -> 3
  size_t numberOfReactions = goodReactionProductList.size();
  for (size_t i = 0; i != numberOfReactions; i++){
    if (goodReactionProductList[i].size() == 2) {
      cumulatedProbability += prob22/good22;
      twoToThreeFlag.push_back(false);
    } else {
      cumulatedProbability += prob23/good23;
      twoToThreeFlag.push_back(true);
    }
    probabilities.push_back(cumulatedProbability);
  }

  //Normalising probabilities to 1
  for (std::vector<G4double>::iterator it = probabilities.begin();
       it != probabilities.end();
       it++)
    {
      *it /= cumulatedProbability;
    }

  // Choosing ReactionProduct
  G4bool selected = false;
  G4int tries = 0;
  //  ReactionProductList::iterator prod_it;

  //Keep looping over the list until we have a choice, or until we have tried 100 times  
  size_t i;
  while(!selected && tries < 100){
    i=0;
    G4double dice = G4UniformRand();
    //select the process using the dice
    while(dice>probabilities[i] && i<numberOfReactions)  i++;

    if(twoToThreeFlag[i]) {
      // 2 -> 3 processes require a phase space lookup
      if (m_phaseSpace(goodReactionProductList[i],incidentDynamicParticle,target)>G4UniformRand()) selected = true;
    } else {
      // 2 -> 2 processes are chosen immediately
      selected = true;
    }
    tries++;
  }
  if(tries>=100) G4cerr<<"Could not select process!!!!"<<G4endl;

  
  //Debugging stuff
  //Updating checkfraction:
  if (goodReactionProductList[i].size()==2) {
    m_n22++;
  } else {
    m_n23++;
  }
  m_checkFraction = (1.0*m_n22)/(m_n22+m_n23);
  
  //return the selected productList
  return goodReactionProductList[i];
}

G4double HadronicProcessHelper::m_reactionProductMass(const ReactionProduct& reactionProd,
  const G4DynamicParticle* incidentDynamicParticle,G4ParticleDefinition* target){
  // Incident energy:
  G4double incidentEnergy = incidentDynamicParticle->GetTotalEnergy();
  G4double m_1 = incidentDynamicParticle->GetDefinition()->GetPDGMass();
  G4double m_2 = target->GetPDGMass();
  //square root of "s"
  G4double sqrtS = sqrt(m_1*m_1 + m_2*(m_2 + 2 * incidentEnergy));
  // Sum of rest masses after reaction:
  G4double productsMass = 0;
  //Loop on reaction producs
  for (ReactionProduct::const_iterator r_it = reactionProd.begin(); r_it !=reactionProd.end(); r_it++){
    //Sum the masses of the products
    productsMass += m_particleTable->FindParticle(*r_it)->GetPDGMass();
  }
  //the result is square root of "s" minus the masses of the products
  return sqrtS - productsMass;
}

G4bool HadronicProcessHelper::m_reactionIsPossible(const ReactionProduct& aReaction,
     const G4DynamicParticle* aDynamicParticle,G4ParticleDefinition* target){
  if (m_reactionProductMass(aReaction,aDynamicParticle,target)>0) return true;
  return false;
}

G4double HadronicProcessHelper::m_phaseSpace(const ReactionProduct& aReaction,
     const G4DynamicParticle* aDynamicParticle,G4ParticleDefinition* target){
  G4double qValue = m_reactionProductMass(aReaction,aDynamicParticle,target);
  G4double phi = sqrt(1+qValue/(2*0.139*GeV))*pow(qValue/(1.1*GeV),3./2.);
  return (phi/(1+phi));
}

void HadronicProcessHelper::m_readAndParse(const G4String& str,
				   std::vector<G4String>& tokens,
				   const G4String& delimiters)
{
  // Skip delimiters at beginning.
  G4String::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  G4String::size_type pos     = str.find_first_of(delimiters, lastPos);
  
  while (G4String::npos != pos || G4String::npos != lastPos)
    {
      //Skipping leading / trailing whitespaces
      G4String temp = str.substr(lastPos, pos - lastPos);
      while(temp.c_str()[0] == ' ') temp.erase(0,1);
      while(temp[temp.size()-1] == ' ') temp.erase(temp.size()-1,1);
      // Found a token, add it to the vector.
      tokens.push_back(temp);
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
    }
}
