#include "SimMuon/CSCDigitizer/src/CSCDigiDump.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include <iostream>
using std::endl;
using std::cout;
using std::string;

CSCDigiDump::CSCDigiDump(edm::ParameterSet const& conf) {
  label_ = conf.getParameter<string>("label");
}


void CSCDigiDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCComparatorDigiCollection> comparators;


  try {
    e.getByLabel(label_, "MuonCSCWireDigi", wires);
      for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
        std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
           digiItr->print();
        }
      }

  } catch (...) {
    edm::LogError("CSCDigiDump") << "Cannot get wires by label " << label_;
  }


  try {
    e.getByLabel(label_, "MuonCSCStripDigi", strips);

      for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
        std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
           digiItr->print();
        }
      }

  } catch (...) {
     edm::LogError("CSCDigiDump") << "Cannot get strips by label " << label_;
  }


  try {
    e.getByLabel(label_, "MuonCSCComparatorDigi", comparators);

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
    edm::LogError("CSCDigiDump") << "Cannot get comparators  by label " << label_;
  }

}


