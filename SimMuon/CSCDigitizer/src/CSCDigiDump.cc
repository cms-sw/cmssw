#include "SimMuon/CSCDigitizer/src/CSCDigiDump.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include <iostream>
using std::endl;
using std::cout;


void CSCDigiDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCComparatorDigiCollection> comparators;


  try {
    e.getByLabel("cscDigis", "MuonCSCWireDigi", wires);
      for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
        std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
           digiItr->print();
        }
      }

  } catch (...) {
    cout << "Cannot get wires by label cscDigis MuonCSCWireDigi" << endl;
  }


  try {
    e.getByLabel("cscDigis", "MuonCSCStripDigi", strips);

      for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
        std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
           digiItr->print();
        }
      }

  } catch (...) {
    cout << "Cannot get by label cscDigis MuonCSCStripDigi" << endl;
  }


  try {
    e.getByLabel("cscDigis", "MuonCSCComparatorDigi", comparators);

      for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); 
           j!=comparators->end(); j++) 
      {
        std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
           digiItr->print();
        }
      }

  } catch (...) {
    cout << "Cannot get by label cscDigis MuonCSCComparatorDigi" << endl;
  }

}


