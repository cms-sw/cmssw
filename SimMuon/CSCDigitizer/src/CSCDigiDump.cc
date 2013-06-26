#include "SimMuon/CSCDigitizer/src/CSCDigiDump.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include <iostream>
using std::endl;
using std::cout;
using std::string;

CSCDigiDump::CSCDigiDump(edm::ParameterSet const& conf)
: wireDigiTag_(conf.getParameter<edm::InputTag>("wireDigiTag")),
  stripDigiTag_(conf.getParameter<edm::InputTag>("stripDigiTag")),
  comparatorDigiTag_(conf.getParameter<edm::InputTag>("comparatorDigiTag"))
{
}


void CSCDigiDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCComparatorDigiCollection> comparators;

  std::cout << "Event " << e.id() << std::endl;

  e.getByLabel(wireDigiTag_, wires);
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    std::cout << "Wire digis from "<< CSCDetId((*j).first) << std::endl;
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }

  e.getByLabel(stripDigiTag_, strips);

  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    std::cout << "Strip digis from "<< CSCDetId((*j).first) << std::endl;
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }

  e.getByLabel(comparatorDigiTag_, comparators);

  for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); 
       j!=comparators->end(); j++) 
  {
    std::cout << "Comparator digis from "<< CSCDetId((*j).first) << std::endl;
    std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }
}


