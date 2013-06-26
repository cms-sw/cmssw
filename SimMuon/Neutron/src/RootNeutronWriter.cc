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


void RootNeutronWriter::initialize(int chamberType)
{
    ostringstream treeName;
    treeName << "ChamberType" << chamberType;
    theChamberWriters[chamberType] = RootChamberWriter(treeName.str());
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
    throw cms::Exception("NeutronWriter") << "It's dangerous to create ROOT chamber "
      << "writers during processing.  The global file may change";
      
    // make a new one
    initialize(chamberType);
    return theChamberWriters[chamberType];
  }
}


void RootNeutronWriter::writeCluster(int chamberType, const edm::PSimHitContainer & hits) 
{
std::cout << "ROOTNEUTRONWRITER " << chamberType << " HITS SIZE " << hits.size() <<std::endl;
  chamberWriter(chamberType).write(hits);
}

