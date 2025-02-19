#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"
#include "SimG4Core/Physics/interface/ProcessTypeEnumerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "G4VProcess.hh"

#include <iostream>

//#define MYDEB

G4ProcessTypeEnumerator::G4ProcessTypeEnumerator(){
  mapProcesses["Undefined"] = "Undefined";
  mapProcesses["Unknown"] = "Unknown";
  //
  mapProcesses["Primary"] = "Primary";
  // nuclear stuff
  mapProcesses["HadronCapture"]  = "Hadronic";
  mapProcesses["AntiNeutronInelastic"] = "Hadronic";
  mapProcesses["PositronNuclear"] = "Hadronic";
  mapProcesses["ElectroNuclear"] = "Hadronic";
  mapProcesses["AntiProtonAnnihilationAtRest"] = "Hadronic";
  mapProcesses["AntiProtonInelastic"] = "Hadronic";
  mapProcesses["ProtonInelastic"] = "Hadronic";
  mapProcesses["PhotonInelastic"] = "Hadronic";
  mapProcesses["DeuteronInelastic"] = "Hadronic";
  mapProcesses["KaonMinusAbsorption"] = "Hadronic";
  mapProcesses["KaonMinusInelastic"] = "Hadronic";
  mapProcesses["KaonPlusInelastic"] = "Hadronic";
  mapProcesses["KaonZeroLInelastic"] = "Hadronic";
  mapProcesses["KaonZeroSInelastic"] = "Hadronic";
  mapProcesses["LCapture"] = "Hadronic";
  mapProcesses["LElastic"] = "Hadronic";
  mapProcesses["hElastic"] = "Hadronic";
  mapProcesses["LambdaInelastic"] = "Hadronic";
  mapProcesses["NeutronInelastic"] = "Hadronic";
  mapProcesses["CHIPSNuclearAbsorptionAtRest"] = "Hadronic";
  mapProcesses["PionMinusAbsorptionAtRest"] = "Hadronic";
  mapProcesses["PionMinusInelastic"] = "Hadronic";
  mapProcesses["PionPlusInelastic"] = "Hadronic";
  mapProcesses["SigmaMinusInelastic"] = "Hadronic";
  mapProcesses["AntiSigmaMinusInelastic"] = "Hadronic";
  mapProcesses["AntiSigmaPlusInelastic"] = "Hadronic";
  mapProcesses["AntiLambdaInelastic"] = "Hadronic";
  mapProcesses["TritonInelastic"] = "Hadronic"; 
  mapProcesses["XiMinusInelastic"] = "Hadronic";
  mapProcesses["XiPlusInelastic"] = "Hadronic";
  mapProcesses["AntiXiZeroInelastic"] = "Hadronic";
  mapProcesses["SigmaPlusInelastic"] = "Hadronic";
  mapProcesses["XiZeroInelastic"] = "Hadronic";
  mapProcesses["AntiXiMinusInelastic"] = "Hadronic";
  mapProcesses["AlphaInelastic"] = "Hadronic";
  mapProcesses["FullModelHadronicProcess"] = "Hadronic";
  mapProcesses["hInelastic"] = "Hadronic";
  mapProcesses["dInelastic"] = "Hadronic";
  mapProcesses["tInelastic"] = "Hadronic";
  mapProcesses["nCapture"] = "Hadronic";
  mapProcesses["alphaInelastic"] = "Hadronic";
  mapProcesses["CHIPSElasticScattering"] = "Hadronic";
  mapProcesses["MixedProtonInelasticProcess"] = "Hadronic";

  // for GFlash Hadron process
  mapProcesses["WrappedPionMinusInelastic"] = "Hadronic";
  mapProcesses["WrappedPionPlusInelastic"] = "Hadronic";

  // ionizations
  mapProcesses["eIoni"] = "EIoni";
  mapProcesses["hIoni"] = "HIoni";
  mapProcesses["ionIoni"] = "HIoni";
  mapProcesses["muIoni"] = "MuIoni";
  // Annihilation
  mapProcesses["annihil"] = "Annihilation";
  // MuBrem
  mapProcesses["muBrems"] = "MuBrem";
  // MuNucl
  mapProcesses["muMinusCaptureAtRest"] = "MuNucl";
  mapProcesses["MuonMinusCaptureAtRest"] = "MuNucl";
  mapProcesses["MuonPlusCaptureAtRest"] = "MuNucl";
  // Conversions
  mapProcesses["conv"] = "Conversions";
  // Brems
  mapProcesses["eBrem"] = "EBrem";
  // Decay
  mapProcesses["Decay"] = "Decay";
  // PairProd
  mapProcesses["muPairProd"] = "MuPairProd";
  // Photon
  mapProcesses["phot"] = "Photon";
  // Sync
  mapProcesses["SynchrotronRadiation"] = "SynchrotronRadiation";
  // Compton
  mapProcesses["compt"] = "Compton";
  // hbrem etc;
  mapProcesses["hBrems"] = "hBrems";
  mapProcesses["hPairProd"] = "hPairProd";
  //
  map2Process["Undefined"] = -1;
  map2Process["Unknown"] = 0;
  map2Process["Primary"] = 100;
  //Hadronic
  map2Process["HadronCapture"]  = 1;
  map2Process["AntiNeutronInelastic"] = 2;
  map2Process["PositronNuclear"] = 3;
  map2Process["ElectroNuclear"] = 4;
  map2Process["AntiProtonAnnihilationAtRest"] = 5;
  map2Process["AntiProtonInelastic"] = 6;
  map2Process["ProtonInelastic"] = 7;
  map2Process["PhotonInelastic"] = 8;
  map2Process["DeuteronInelastic"] = 9;
  map2Process["KaonMinusAbsorption"] = 10;
  map2Process["KaonMinusInelastic"] = 11;
  map2Process["KaonPlusInelastic"] = 12;
  map2Process["KaonZeroLInelastic"] = 13;
  map2Process["KaonZeroSInelastic"] = 14;
  map2Process["LCapture"] = 15;
  map2Process["LElastic"] = 16;
  map2Process["hElastic"] = 17;
  map2Process["LambdaInelastic"] = 18;
  map2Process["NeutronInelastic"] = 19;
  map2Process["CHIPSNuclearAbsorptionAtRest"] = 20;
  map2Process["PionMinusAbsorptionAtRest"] = 21;
  map2Process["PionMinusInelastic"] = 22;
  map2Process["PionPlusInelastic"] = 23;
  map2Process["SigmaMinusInelastic"] = 24;
  map2Process["AntiSigmaMinusInelastic"] = 25;
  map2Process["AntiSigmaPlusInelastic"] = 26;
  map2Process["AntiLambdaInelastic"] = 27;
  map2Process["TritonInelastic"] = 28;
  map2Process["XiMinusInelastic"] = 29;
  map2Process["XiPlusInelastic"] = 30;
  map2Process["AntiXiZeroInelastic"] = 31;
  map2Process["SigmaPlusInelastic"] = 32;
  map2Process["XiZeroInelastic"] = 33;
  map2Process["AntiXiMinusInelastic"] = 34;
  map2Process["FullModelHadronicProcess"] = 35;
  map2Process["hInelastic"] = 36;
  map2Process["dInelastic"] = 37;
  map2Process["tInelastic"] = 38;
  map2Process["alphaInelastic"] = 39;
  map2Process["nCapture"] = 40;
  map2Process["CHIPSElasticScattering"] = 17;
  map2Process["MixedProtonInelasticProcess"] = 7;

  // for GFlash hadron process
  map2Process["WrappedPionMinusInelastic"] = 68;
  map2Process["WrappedPionPlusInelastic"] = 69;

  // Decay
  map2Process["Decay"] = 50;
  // EM
  map2Process["eIoni"] = 51;
  map2Process["hIoni"] = 52;
  map2Process["ionIoni"] = 53;
  map2Process["muIoni"] = 54;
  map2Process["annihil"] = 55;
  map2Process["muBrems"] = 56;
  map2Process["muMinusCaptureAtRest"] = 57;
  map2Process["MuonMinusCaptureAtRest"] = 58;
  map2Process["MuonPlusCaptureAtRest"] = 59;
  map2Process["conv"] = 60;
  map2Process["eBrem"] = 61;
  map2Process["muPairProd"] = 62;
  map2Process["phot"] = 63;
  map2Process["SynchrotronRadiation"] = 64;
  map2Process["compt"] = 65;
  map2Process["hBrems"] = 66;
  map2Process["hPairProd"] = 67;
  //
  buildReverseMap();
  //
  theProcessTypeEnumerator = new ProcessTypeEnumerator;
}

G4ProcessTypeEnumerator::~G4ProcessTypeEnumerator(){
  delete theProcessTypeEnumerator;
  mapProcesses.clear();
  reverseMapProcesses.clear();
  map2Process.clear();
  reverseMap2Process.clear();
}


unsigned int G4ProcessTypeEnumerator::processId(const G4VProcess* process){
  if (process == 0) {
    //
    // it is primary!
    //
    std::string temp = "Primary";
#ifdef MYDEB
    LogDebug("Physics") <<"G4ProcessTypeEnumerator : Primary process, returning "
			<< theProcessTypeEnumerator->processId(temp);
#endif
    return theProcessTypeEnumerator->processId(temp);
  } else {
    std::string temp = process->GetProcessName();
#ifdef MYDEB
    LogDebug("Physics") <<"G4ProcessTypeEnumerator : G4Process "<<temp
			<<" mapped to "<< processCMSName(temp)<<"; returning "
			<<theProcessTypeEnumerator->processId(processCMSName(temp));
#endif
    return theProcessTypeEnumerator->processId(processCMSName(temp));
  }
}

int G4ProcessTypeEnumerator::processIdLong(const G4VProcess* process) {
  std::string temp;
  if (process == 0) temp = "Primary";
  else              temp = process->GetProcessName();
  std::map<std::string,int>::const_iterator it = map2Process.find(temp);
  if (it != map2Process.end()) return it->second;
  else                         return map2Process["Undefined"];
}

std::string G4ProcessTypeEnumerator::processCMSName(std::string in){
  if (mapProcesses[in] == ""){
    //    throw MantisException("G4ProcessTypeEnumerator: unknown G4 process "+in);
    LogDebug("Physics")<<" NOT FOUND G4ProcessTypeEnumerator: "<<in;
    return "Unknown";
  }
  return mapProcesses[in];
}

std::vector<std::string> G4ProcessTypeEnumerator::processG4Name(std::string in){
  if (reverseMapProcesses[in].size() ==0)
    //    throw MantisException("G4ProcessTypeEnumerator: unknown CMS process "+in);
    ;
  return reverseMapProcesses[in];
}

std::string G4ProcessTypeEnumerator::processG4Name(int in) {
  std::map<int,std::string>::const_iterator it = reverseMap2Process.find(in);
  if (it != reverseMap2Process.end()) return it->second;
  else                                return "Undefined";
}

unsigned int G4ProcessTypeEnumerator::numberOfKnownCMSProcesses(){
  return reverseMapProcesses.size();
}

unsigned int G4ProcessTypeEnumerator::numberOfKnownG4Processes(){
  return mapProcesses.size();
}

void G4ProcessTypeEnumerator::buildReverseMap(){
  for (MapType::const_iterator it = mapProcesses.begin();  
       it != mapProcesses.end(); it++)
    (reverseMapProcesses[(*it).second]).push_back((*it).first);
  for (std::map<std::string,int>::const_iterator it = map2Process.begin();  
       it != map2Process.end(); it++) 
    reverseMap2Process[(*it).second] = (*it).first;
}
