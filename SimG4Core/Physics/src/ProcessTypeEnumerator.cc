#include "SimG4Core/Physics/interface/ProcessTypeEnumerator.h"


ProcessTypeEnumerator::ProcessTypeEnumerator(){
  mapProcesses["Undefined"] = 0;
  mapProcesses["Unknown"] = 1;
  mapProcesses["Primary"] = 2;
  mapProcesses["Hadronic"] = 3;
  mapProcesses["Decay"] = 4;
  mapProcesses["Compton"] = 5;
  mapProcesses["Annihilation"] = 6;
  mapProcesses["EIoni"] = 7;
  mapProcesses["HIoni"] = 8;
  mapProcesses["MuIoni"] = 9;
  mapProcesses["Photon"] = 10;
  mapProcesses["MuPairProd"] = 11;
  mapProcesses["Conversions"] = 12;
  mapProcesses["EBrem"] = 13;
  mapProcesses["SynchrotronRadiation"] = 14;
  mapProcesses["MuBrem"] = 15;
  mapProcesses["MuNucl"] = 16;

  //
  //
  buildReverseMap();
}

unsigned int ProcessTypeEnumerator::processId(std::string in){
  if (in == "Undefined") return 0;
  if (mapProcesses[in] == 0)
    //    throw Genexception("ProcessTypeEnumerator: unknown process "+in);
    ;
  return mapProcesses[in];
}

std::string ProcessTypeEnumerator::processName(unsigned int in){
  if (reverseMapProcesses[in] == "")
    //    throw Genexception("ProcessTypeEnumerator: unknown process id "+in);
    ;
  return reverseMapProcesses[in];
}

unsigned int ProcessTypeEnumerator::numberOfKnownProcesses(){
  return mapProcesses.size();
}

void ProcessTypeEnumerator::buildReverseMap(){
  for (MapType::const_iterator it = mapProcesses.begin();  it != mapProcesses.end(); it++)
    reverseMapProcesses[(*it).second] = (*it).first;
}
