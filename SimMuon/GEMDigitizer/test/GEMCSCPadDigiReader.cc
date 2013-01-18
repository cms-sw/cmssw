#ifndef SimMuon_GEMCSCPadDigiReader_h
#define SimMuon_GEMCSCPadDigiReader_h

/** \class GEMDigiReader
 *  Dumps GEM-CSC trigger pad digis 
 *  
 *  $Id: GEMCSCPadDigiReader.cc,v 1.1 2013/01/18 04:42:32 khotilov Exp $
 *  \authors: Vadim Khotilovich
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <map>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "DataFormats/Common/interface/DetSet.h"

using namespace std;


class GEMCSCPadDigiReader: public edm::EDAnalyzer
{
public:

  explicit GEMCSCPadDigiReader(const edm::ParameterSet& pset);
  
  virtual ~GEMCSCPadDigiReader(){}
  
  void analyze(const edm::Event &, const edm::EventSetup&); 
  
private:

  string label_pads_;
  string label_digis_;
};



GEMCSCPadDigiReader::GEMCSCPadDigiReader(const edm::ParameterSet& pset)
{
  label_pads_ = pset.getUntrackedParameter<string>("labelPads", "simMuonGEMCSCPadDigis");
  label_digis_ = pset.getUntrackedParameter<string>("labelDigis", "simMuonGEMDigis");
}


void GEMCSCPadDigiReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::Handle<GEMCSCPadDigiCollection> pads;
  event.getByLabel(label_pads_, pads);

  edm::Handle<GEMDigiCollection> digis;
  event.getByLabel(label_digis_, digis);

  edm::ESHandle<GEMGeometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get( pDD );
 
  /* 
  size_t digi_dets_size = std::distance(digis->begin(), digis->end());
  size_t pads_dets_size = std::distance(pads->begin(), pads->end());
  cout<<"#dets with digis = "<<digi_dets_size<<"  with pad digis = "<<pads_dets_size<<"   "
      <<( (digi_dets_size == pads_dets_size) ? "GOOD" : "BAD")<<endl;
  */

  GEMCSCPadDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = pads->begin();	detUnitIt != pads->end(); ++detUnitIt)
  {
    const GEMDetId& id = (*detUnitIt).first;
    const GEMEtaPartition* roll = pDD->etaPartition(id);

    //if(id.rawId() != 637567293) continue;

    // GEMDetId print-out
    cout<<"--------------"<<endl;
    //cout<<"id: "<<id.rawId()<<" #strips "<<roll->nstrips()<<"  #pads "<<roll->npads()<<endl;

    // retrieve this DetUnit's digis
    std::map< std::pair<unsigned int, int>, // #pad, BX
              std::vector<unsigned int>              // digi strip numbers
      > digi_map;
    auto digis_in_det = digis->get(id);
    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d)
    {
      unsigned int pad_num = 1 + static_cast<unsigned int>( roll->padOfStrip(d->strip()) );
      digi_map[std::make_pair(pad_num, d->bx())].push_back(d->strip());
    }

    // loop over pads of this DetUnit and print stuff
    auto pads_range = (*detUnitIt).second;
    for (auto padIt = pads_range.first; padIt != pads_range.second; ++padIt)
    {
      cout<<id <<" paddigi(pad,bx) "<<*padIt<<endl;
      if (padIt->pad() < 1 || padIt->pad() > roll->npads() )
      {
        cout <<" XXXXXXXXXXXXX Problem! "<<id<<" has pad digi with too large pad# = "<<padIt->pad()<<endl;
      }
      unsigned int first_strip = roll->firstStripInPad(padIt->pad() - 1);
      unsigned int last_strip = roll->lastStripInPad(padIt->pad() - 1);

      auto strips = digi_map[std::make_pair(padIt->pad(), padIt->bx())];
      std::vector<unsigned int> pads_strips;
      remove_copy_if(strips.begin(), strips.end(), 
                     inserter(pads_strips, pads_strips.end()), 
                     [first_strip, last_strip](unsigned int s) { return s < first_strip || s > last_strip; }
                    );
      cout<<" has "<<pads_strips.size()<<" strip digis at strips ";
      copy(pads_strips.begin(), pads_strips.end(), ostream_iterator<unsigned int>(cout, " "));
      cout<<endl;
    }


  }// for layers

  cout<<"--------------"<<endl;
}



#endif
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(GEMCSCPadDigiReader);

