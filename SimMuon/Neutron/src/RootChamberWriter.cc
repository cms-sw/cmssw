#include "SimMuon/Neutron/src/RootChamberWriter.h"
#include "SimMuon/Neutron/src/RootSimHit.h"
using namespace std;

RootChamberWriter::RootChamberWriter(const std::string & treeName)
{
  theHits = new TClonesArray("RootSimHit");
  theTree = new TTree(treeName.c_str(), "Neutron Background");
  theTree->Bronch("Hits", "TClonesArray", &theHits);
}


RootChamberWriter::~RootChamberWriter()
{
//std::cout << "WRITING " << theTree->GetEntries() << std::endl;
//  theTree->Write();
  //delete theHits;
  //delete theTree;
}


void RootChamberWriter::write(const edm::PSimHitContainer & hits)
{
std::cout << "ENTRIES BEFORE " << theTree->GetEntries() << std::endl;
  theHits->Delete();
  theHits->Expand(hits.size());
  for(unsigned int i = 0; i < hits.size(); ++i)
  {
    new((*theHits)[i]) RootSimHit(hits[i]);
  }
  theTree->Fill();
std::cout << "ENTRIES AFTER " << theTree->GetEntries() << std::endl;
}

