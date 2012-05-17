#include"G4ParticleTable.hh" 
#include "Randomize.hh"

#include<iostream>
#include<fstream>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include"SimG4Core/CustomPhysics/interface/G4ProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"

G4ProcessHelper::G4ProcessHelper(const edm::ParameterSet & p){

  particleTable = G4ParticleTable::GetParticleTable();

  theProton = particleTable->FindParticle("proton");
  theNeutron = particleTable->FindParticle("neutron");
  
  G4String line;

  edm::FileInPath fp = p.getParameter<edm::FileInPath>("processesDef");
  std::string processDefFilePath = fp.fullPath();
  std::ifstream process_stream (processDefFilePath.c_str());

  resonant = p.getParameter<bool>("resonant");
  ek_0 = p.getParameter<double>("resonanceEnergy")*GeV;
  gamma = p.getParameter<double>("gamma")*GeV;
  amplitude = p.getParameter<double>("amplitude")*millibarn;
  suppressionfactor = p.getParameter<double>("reggeSuppression");
  
  edm::LogInfo("CustomPhysics")<<"Read in physics parameters:"<<G4endl;
  edm::LogInfo("CustomPhysics")<<"Resonant = "<< resonant <<G4endl;
  edm::LogInfo("CustomPhysics")<<"ResonanceEnergy = "<<ek_0/GeV<<" GeV"<<G4endl;
  edm::LogInfo("CustomPhysics")<<"Gamma = "<<gamma/GeV<<" GeV"<<G4endl;
  edm::LogInfo("CustomPhysics")<<"Amplitude = "<<amplitude/millibarn<<" millibarn"<<G4endl;
  edm::LogInfo("CustomPhysics")<<"ReggeSuppression = "<<100*suppressionfactor<<" %"<<G4endl;

  checkfraction = 0;
  n_22 = 0;
  n_23 = 0;


  while(getline(process_stream,line)){
    std::vector<G4String> tokens;
    //Getting a line
    ReadAndParse(line,tokens,"#");
    //Important info
    G4String incident = tokens[0];
    
    G4ParticleDefinition* incidentDef = particleTable->FindParticle(incident);
    //particleTable->DumpTable();
    G4int incidentPDG = incidentDef->GetPDGEncoding();
    known_particles[incidentDef]=true;

    G4String target = tokens[1];
    edm::LogInfo("CustomPhysics")<<"Incident: "<<incident
		    <<" Target: "<<target<<G4endl;
    
    // Making a ReactionProduct
    ReactionProduct prod;
    for (size_t i = 2; i != tokens.size();i++){
      G4String part = tokens[i];
      if (particleTable->contains(part))
	{
	  prod.push_back(particleTable->FindParticle(part)->GetPDGEncoding());
	} else {
	  edm::LogInfo("CustomPhysics")<<"Particle: "<<part<<" is unknown."<<G4endl;
	  G4Exception("G4ProcessHelper", "UnkownParticle", FatalException,
		      "Initialization: The reaction product list contained an unknown particle");
	}
    }
    if (target == "proton")
      {
	pReactionMap[incidentPDG].push_back(prod);
      } else if (target == "neutron") {
	nReactionMap[incidentPDG].push_back(prod);
      } else {
	G4Exception("G4ProcessHelper", "IllegalTarget", FatalException,
		    "Initialization: The reaction product list contained an illegal target particle");
      }
   }

  process_stream.close();

}

G4bool G4ProcessHelper::ApplicabilityTester(const G4ParticleDefinition& aPart){
  const G4ParticleDefinition* aP = &aPart; 
  if (known_particles[aP]) return true;
  return false;
}

G4double G4ProcessHelper::GetInclusiveCrossSection(const G4DynamicParticle *aParticle,
						   const G4Element *anElement){

  //We really do need a dedicated class to handle the cross sections. They might not always be constant


  //Disassemble the PDG-code

  G4int thePDGCode = aParticle->GetDefinition()->GetPDGEncoding();

  G4double theXsec = 0;

  //Flat cross section
  if(CustomPDGParser::s_isRGlueball(thePDGCode)) {
    theXsec = 24 * millibarn;
  } else {
    std::vector<G4int> nq=CustomPDGParser::s_containedQuarks(thePDGCode);
    //    edm::LogInfo("CustomPhysics")<<"Number of quarks: "<<nq.size()<<G4endl;
    for (std::vector<G4int>::iterator it = nq.begin();
	 it != nq.end();
	 it++)
      {
	//	  edm::LogInfo("CustomPhysics")<<"Quarkvector: "<<*it<<G4endl;
	if (*it == 1 || *it == 2) theXsec += 12 * millibarn;
	if (*it == 3) theXsec += 6 * millibarn;
      }
  }

  //Adding resonance

  if(resonant)
    {
      double e_0 = ek_0 + aParticle->GetDefinition()->GetPDGMass(); //Now total energy

      e_0 = sqrt(aParticle->GetDefinition()->GetPDGMass()*aParticle->GetDefinition()->GetPDGMass()
		 + theProton->GetPDGMass()*theProton->GetPDGMass()
		 + 2.*e_0*theProton->GetPDGMass());
      //      edm::LogInfo("CustomPhysics")<<e_0/GeV<<G4endl;
      //      edm::LogInfo("CustomPhysics")<<ek_0/GeV<<" "<<aParticle->GetDefinition()->GetPDGMass()/GeV<<" "<<theProton->GetPDGMass()/GeV<<G4endl;
      double sqrts=sqrt(aParticle->GetDefinition()->GetPDGMass()*aParticle->GetDefinition()->GetPDGMass()
			+ theProton->GetPDGMass()*theProton->GetPDGMass() + 2*aParticle->GetTotalEnergy()*theProton->GetPDGMass());

      double res_result = amplitude*(gamma*gamma/4.)/((sqrts-e_0)*(sqrts-e_0)+(gamma*gamma/4.));//Non-relativistic Breit Wigner

      theXsec += res_result;
      //      if(fabs(aParticle->GetKineticEnergy()/GeV-200)<10)  std::cout<<sqrts/GeV<<" "<<theXsec/millibarn<<std::endl;
    }


  //  std::cout<<"Xsec/nucleon: "<<theXsec/millibarn<<"millibarn, Total Xsec: "<<theXsec * anElement->GetN()/millibarn<<" millibarn"<<std::endl;
  //  return theXsec * anElement->GetN();// * 0.523598775598299;
  return theXsec * pow(anElement->GetN(),0.7)*1.25;// * 0.523598775598299;

}

ReactionProduct G4ProcessHelper::GetFinalState(const G4Track& aTrack, G4ParticleDefinition*& aTarget){

  const G4DynamicParticle* aDynamicParticle = aTrack.GetDynamicParticle();

  //-----------------------------------------------
  // Choose n / p as target
  // and get ReactionProductList pointer
  //-----------------------------------------------

  G4Material* aMaterial = aTrack.GetMaterial();
  const G4ElementVector* theElementVector = aMaterial->GetElementVector() ;
  const G4double* NbOfAtomsPerVolume = aMaterial->GetVecNbOfAtomsPerVolume();

  G4double NumberOfProtons=0;
  G4double NumberOfNucleons=0;

  for ( size_t elm=0 ; elm < aMaterial->GetNumberOfElements() ; elm++ )
    {
      //Summing number of protons per unit volume
      NumberOfProtons += NbOfAtomsPerVolume[elm]*(*theElementVector)[elm]->GetZ();
      //Summing nucleons (not neutrons)
      NumberOfNucleons += NbOfAtomsPerVolume[elm]*(*theElementVector)[elm]->GetN();
    }

  if(G4UniformRand()<NumberOfProtons/NumberOfNucleons)
    {
      theReactionMap = &pReactionMap;
      theTarget = theProton;
    } else {
      theReactionMap = &nReactionMap;
      theTarget = theNeutron;
    }
  aTarget = theTarget;
  const G4int theIncidentPDG = aDynamicParticle->GetDefinition()->GetPDGEncoding();

  //Making a pointer directly to the ReactionProductList we are looking at. Makes life easier :-)
  ReactionProductList*  aReactionProductList = &((*theReactionMap)[theIncidentPDG]);

  //-----------------------------------------------
  // Count processes
  // kinematic check
  // compute number of 2 -> 2 and 2 -> 3 processes
  //-----------------------------------------------

  G4int N22 = 0; //Number of 2 -> 2 processes
  G4int N23 = 0; //Number of 2 -> 3 processes. Couldn't think of more informative names
  
  //This is the list to be populated
  ReactionProductList theReactionProductList;
  std::vector<bool> theChargeChangeList;

  for (ReactionProductList::iterator prod_it = aReactionProductList->begin();
       prod_it != aReactionProductList->end();
       prod_it++){
    G4int secondaries = prod_it->size();
    // If the reaction is not possible we will not consider it
    if(ReactionIsPossible(*prod_it,aDynamicParticle)){
      // The reaction is possible. Let's store and count it
      theReactionProductList.push_back(*prod_it);
      if (secondaries == 2){
	N22++;
      } else if (secondaries ==3) {
	N23++;
      } else {
	G4cerr << "ReactionProduct has unsupported number of secondaries: "<<secondaries<<G4endl;
      }
    } /*else {
      edm::LogInfo("CustomPhysics")<<"There was an impossible process"<<G4endl;
      }*/
  }
  //  edm::LogInfo("CustomPhysics")<<"The size of the ReactionProductList is: "<<theReactionProductList.size()<<G4endl;

  if (theReactionProductList.size()==0) G4Exception("G4ProcessHelper", "NoProcessPossible", FatalException,
						    "GetFinalState: No process could be selected from the given list.");
  // Fill a probability map. Remember total probability
  // 2->2 is 0.15*1/n_22 2->3 uses phase space
  G4double p22 = 0.15;
  G4double p23 = 1-p22; // :-)

  std::vector<G4double> Probabilities;
  std::vector<G4bool> TwotoThreeFlag;
  
  G4double CumulatedProbability = 0;

  // To each ReactionProduct we assign a cumulated probability and a flag
  // discerning between 2 -> 2 and 2 -> 3
  for (unsigned int i = 0; i != theReactionProductList.size(); i++){
    if (theReactionProductList[i].size() == 2) {
      CumulatedProbability += p22/N22;
      TwotoThreeFlag.push_back(false);
    } else {
      CumulatedProbability += p23/N23;
      TwotoThreeFlag.push_back(true);
    }
    Probabilities.push_back(CumulatedProbability);
    //    edm::LogInfo("CustomPhysics")<<"Pushing back cumulated probability: "<<CumulatedProbability<<G4endl;
  }

  //Renormalising probabilities
  //  edm::LogInfo("CustomPhysics")<<"Probs: ";
  for (std::vector<G4double>::iterator it = Probabilities.begin();
       it != Probabilities.end();
       it++)
    {
      *it /= CumulatedProbability;
      //      edm::LogInfo("CustomPhysics")<<*it<<" ";
    }
  //  edm::LogInfo("CustomPhysics")<<G4endl;

  // Choosing ReactionProduct

  G4bool selected = false;
  G4int tries = 0;
  //  ReactionProductList::iterator prod_it;

  //Keep looping over the list until we have a choice, or until we have tried 100 times  
  unsigned int i;
  while(!selected && tries < 100){
    i=0;
    G4double dice = G4UniformRand();
    // edm::LogInfo("CustomPhysics")<<"What's the dice?"<<dice<<G4endl;
    while(dice>Probabilities[i] && i<theReactionProductList.size()){
      //      edm::LogInfo("CustomPhysics")<<"i: "<<i<<G4endl;
      i++;
    }

    //    edm::LogInfo("CustomPhysics")<<"Chosen i: "<<i<<G4endl;

    if(!TwotoThreeFlag[i]) {
      // 2 -> 2 processes are chosen immediately
      selected = true;
    } else {
      // 2 -> 3 processes require a phase space lookup
      if (PhaseSpace(theReactionProductList[i],aDynamicParticle)>G4UniformRand()) selected = true;
      //selected = true;
    }
    //    double suppressionfactor=0.5;
    if(selected&&particleTable->FindParticle(theReactionProductList[i][0])->GetPDGCharge()!=aDynamicParticle->GetDefinition()->GetPDGCharge())
      {
	/*
	edm::LogInfo("CustomPhysics")<<"Incoming particle "<<aDynamicParticle->GetDefinition()->GetParticleName()
	      <<" has charge "<<aDynamicParticle->GetDefinition()->GetPDGCharge()<<G4endl;
	edm::LogInfo("CustomPhysics")<<"Suggested particle "<<particleTable->FindParticle(theReactionProductList[i][0])->GetParticleName()
	      <<" has charge "<<particleTable->FindParticle(theReactionProductList[i][0])->GetPDGCharge()<<G4endl;
	*/
	if(G4UniformRand()<suppressionfactor) selected = false;
      }
    tries++;
    //    edm::LogInfo("CustomPhysics")<<"Tries: "<<tries<<G4endl;
  }
  if(tries>=100) G4cerr<<"Could not select process!!!!"<<G4endl;

  //  edm::LogInfo("CustomPhysics")<<"So far so good"<<G4endl;
  //  edm::LogInfo("CustomPhysics")<<"Sec's: "<<theReactionProductList[i].size()<<G4endl;
  
  //Updating checkfraction:
  if (theReactionProductList[i].size()==2) {
    n_22++;
  } else {
    n_23++;
  }

  checkfraction = (1.0*n_22)/(n_22+n_23);
  //  edm::LogInfo("CustomPhysics")<<"n_22: "<<n_22<<" n_23: "<<n_23<<" Checkfraction: "<<checkfraction<<G4endl;
  //  edm::LogInfo("CustomPhysics") <<"Biig number: "<<n_22+n_23<<G4endl;
  //Return the chosen ReactionProduct
  return theReactionProductList[i];
}

G4double G4ProcessHelper::ReactionProductMass(const ReactionProduct& aReaction,const G4DynamicParticle* aDynamicParticle){
  // Incident energy:
  G4double E_incident = aDynamicParticle->GetTotalEnergy();
  //edm::LogInfo("CustomPhysics")<<"Total energy: "<<E_incident<<" Kinetic: "<<aDynamicParticle->GetKineticEnergy()<<G4endl;
  // sqrt(s)= sqrt(m_1^2 + m_2^2 + 2 E_1 m_2)
  G4double m_1 = aDynamicParticle->GetDefinition()->GetPDGMass();
  G4double m_2 = theTarget->GetPDGMass();
  //edm::LogInfo("CustomPhysics")<<"M_R: "<<m_1/GeV<<" GeV, M_np: "<<m_2/GeV<<" GeV"<<G4endl;
  G4double sqrts = sqrt(m_1*m_1 + m_2*(m_2 + 2 * E_incident));
  //edm::LogInfo("CustomPhysics")<<"sqrt(s) = "<<sqrts/GeV<<" GeV"<<G4endl;
  // Sum of rest masses after reaction:
  G4double M_after = 0;
  for (ReactionProduct::const_iterator r_it = aReaction.begin(); r_it !=aReaction.end(); r_it++){
    //edm::LogInfo("CustomPhysics")<<"Mass contrib: "<<(particleTable->FindParticle(*r_it)->GetPDGMass())/MeV<<" MeV"<<G4endl;
    M_after += particleTable->FindParticle(*r_it)->GetPDGMass();
  }
  //edm::LogInfo("CustomPhysics")<<"Intending to return this ReactionProductMass: "<<(sqrts - M_after)/MeV<<" MeV"<<G4endl;
  return sqrts - M_after;
}

G4bool G4ProcessHelper::ReactionIsPossible(const ReactionProduct& aReaction,const G4DynamicParticle* aDynamicParticle){
  if (ReactionProductMass(aReaction,aDynamicParticle)>0) return true;
  return false;
}

G4double G4ProcessHelper::PhaseSpace(const ReactionProduct& aReaction,const G4DynamicParticle* aDynamicParticle){
  G4double qValue = ReactionProductMass(aReaction,aDynamicParticle);
  G4double phi = sqrt(1+qValue/(2*0.139*GeV))*pow(qValue/(1.1*GeV),3./2.);
  return (phi/(1+phi));
}

void G4ProcessHelper::ReadAndParse(const G4String& str,
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
