#include "SimMuon/Neutron/src/RootChamberReader.h"
#include "SimMuon/Neutron/src/RootSimHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TClonesArray.h"
using namespace std;

RootChamberReader::RootChamberReader()
: theTree(0),
  theHits(0),
  thePosition(0),
  theSize(0)
{
}


RootChamberReader::RootChamberReader(TFile * file, const std::string & treeName)
: theTree( (TTree *) file->Get(treeName.c_str()) ),
  theHits(new TClonesArray("RootSimHit")),
  thePosition(-1),
  theSize(0)
{
  if(theTree != 0) 
  {
    theTree->SetBranchAddress("Hits", &theHits);
    theSize = theTree->GetEntries();
  }
}


RootChamberReader::~RootChamberReader()
{
//  delete theHits;
//  delete theTree;
}


void RootChamberReader::read(edm::PSimHitContainer & hits)
{
  // if there's no tree, make no events
  if(theTree != 0 && theSize != 0)
  {
    ++thePosition;
    // start again from the beginning, if needed
    if(thePosition >= theSize) thePosition = 0;
    theTree->GetEntry(thePosition);

    TIter next(theHits);
    RootSimHit * rootHit;
    while( (rootHit = (RootSimHit *) next()) )
    {
      hits.push_back(rootHit->get());
    }
    LogTrace("Neutrons") << "Event " << thePosition << " OF " << theSize 
         << " has " << hits.size() << " hits ";
  }
}


