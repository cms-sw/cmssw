#ifndef SimMuon_GEMCSCPadDigiReader_h
#define SimMuon_GEMCSCPadDigiReader_h

/** \class GEMDigiReader
 *  Dumps GEM-CSC trigger pad digis 
 *  
 *  $Id: GEMCSCPadDigiReader.cc,v 1.4 2013/01/30 12:12:45 khotilov Exp $
 *  \authors: Vadim Khotilovich
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
  //cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::Handle<GEMCSCPadDigiCollection> pads;
  event.getByLabel(label_pads_, pads);

  if (pads->begin() == pads->end()) return; // no pads in event

  edm::Handle<GEMCSCPadDigiCollection> co_pads;
  event.getByLabel(edm::InputTag(label_pads_, "Coincidence"), co_pads);

  edm::Handle<GEMDigiCollection> digis;
  event.getByLabel(label_digis_, digis);

  edm::ESHandle<GEMGeometry> geometry;
  eventSetup.get<MuonGeometryRecord>().get( geometry );
 
  for (auto pad_range_it = pads->begin(); pad_range_it != pads->end(); ++pad_range_it)
  {
    auto id = (*pad_range_it).first;
    auto roll = geometry->etaPartition(id);

    // GEMDetId print-out
    cout<<"--------------"<<endl;
    //cout<<"id: "<<id.rawId()<<" #strips "<<roll->nstrips()<<"  #pads "<<roll->npads()<<endl;

    // retrieve this DetUnit's digis
    std::map< std::pair<int, int>, // #pad (starting from 1), BX
              std::vector<int>     // digi strip numbers (starting from 1)
            > digi_map;
    auto digis_in_det = digis->get(id);
    cout<<"strip digis in detid: ";
    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d)
    {
      int pad_num = 1 + static_cast<int>( roll->padOfStrip(d->strip()) ); // d->strip() is int
      digi_map[ make_pair(pad_num, d->bx()) ].push_back( d->strip() );
      cout<<"  ("<<d->strip()<<","<<d->bx()<<") -> "<<pad_num;
    }
    cout<<endl;

    // loop over pads of this DetUnit and print stuff
    auto pads_range = (*pad_range_it).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p)
    {
      int first_strip = roll->firstStripInPad(p->pad()); // p->pad() is int, firstStripInPad returns int
      int last_strip = roll->lastStripInPad(p->pad());

      if (p->pad() < 1 || p->pad() > roll->npads() )
      {
        cout <<" XXXXXXXXXXXXX Problem! "<<id<<" has pad digi with too large pad# = "<<p->pad()<<endl;
      }

      auto& strips = digi_map[ make_pair(p->pad(), p->bx()) ];
      std::vector<int> pads_strips;
      remove_copy_if(strips.begin(), strips.end(), inserter(pads_strips, pads_strips.end()),
                     [first_strip, last_strip](int s)
                     {
                       return s < first_strip || s > last_strip;
                     }
                    );
      cout<<id <<" paddigi(pad,bx) "<<*p<<"   has "<<pads_strips.size()
          <<" strip digis strips in range ["<<first_strip<<","<<last_strip<<"]: ";
      copy(pads_strips.begin(), pads_strips.end(), ostream_iterator<int>(cout, " "));
      cout<<endl;
    }

  }// for (detids with pads)

  cout<<"--------------"<<endl;
  cout<<" Coincidence pads:"<<endl;

  for (auto pad_range_it = co_pads->begin(); pad_range_it != co_pads->end(); ++pad_range_it)
  {
    auto id = (*pad_range_it).first;

    // loop over copads of this DetUnit and print stuff
    auto pads_range = (*pad_range_it).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p)
    {
      cout<< id <<" copad(pad,bx) "<<*p<<endl;
    }
    cout<<"----- end event -----"<<endl;
  }
}


#endif
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(GEMCSCPadDigiReader);
