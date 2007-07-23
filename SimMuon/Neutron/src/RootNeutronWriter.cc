#include "SimMuon/Neutron/src/RootNeutronWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
#include <iostream>
RootNeutronWriter::RootNeutronWriter(const string & fileName) 
{
  theFile = new TFile(fileName.c_str(),"update");
}


RootNeutronWriter::~RootNeutronWriter() 
{
  for(std::map<int, RootChamberWriter>::iterator mapItr = theChamberWriters.begin();
      mapItr != theChamberWriters.end(); ++mapItr)
  {
    mapItr->second.tree()->Print();
    // the tree will remember which file it's from
    theFile = mapItr->second.tree()->GetCurrentFile();
  }
  theFile->Write();
//  theFile->Close();
}


RootChamberWriter & RootNeutronWriter::chamberWriter(int chamberType)
{
  std::map<int, RootChamberWriter>::iterator mapItr = theChamberWriters.find(chamberType);
  if(mapItr != theChamberWriters.end())
  {
    return mapItr->second;
  }
  else
  {
    // make a new one
    ostringstream treeName;
    treeName << "ChamberType" << chamberType;
    theChamberWriters[chamberType] = RootChamberWriter(treeName.str());
    return theChamberWriters[chamberType];
  }
}


void RootNeutronWriter::writeEvent(int chamberType, const edm::PSimHitContainer & hits) 
{
std::cout << "ROOTNEUTRONWRITER " << chamberType << " HITS SIZE " << hits.size() <<std::endl;
  chamberWriter(chamberType).write(hits);
}

