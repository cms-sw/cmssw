// Mar-2015: Changed cout to LogVerbatim - and uses Digi::print() which also used cout
// until I switched to LogVerbatim on 03-Mar-2015 for 75X

#include "SimMuon/CSCDigitizer/src/CSCDigiDump.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <iostream>

CSCDigiDump::CSCDigiDump(edm::ParameterSet const& conf)
{  
  wd_token = consumes<CSCWireDigiCollection>(conf.getParameter<edm::InputTag>("wireDigiTag"));
  sd_token = consumes<CSCStripDigiCollection>(conf.getParameter<edm::InputTag>("stripDigiTag"));
  cd_token = consumes<CSCComparatorDigiCollection>(conf.getParameter<edm::InputTag>("comparatorDigiTag"));
}


void CSCDigiDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCComparatorDigiCollection> comparators;

  edm::LogVerbatim("CSCDigi") << "Event " << e.id();

  e.getByToken(wd_token, wires);

  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    edm::LogVerbatim("CSCDigi") << "Wire digis from " << CSCDetId((*j).first);
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }

  e.getByToken(sd_token, strips);

  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    edm::LogVerbatim("CSCDigi") << "Strip digis from " << CSCDetId((*j).first);
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }

  e.getByToken(cd_token, comparators);

  for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); 
       j!=comparators->end(); j++) 
  {
    edm::LogVerbatim("CSCDigi") << "Comparator digis from " << CSCDetId((*j).first);
    std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
       digiItr->print();
    }
  }
}


