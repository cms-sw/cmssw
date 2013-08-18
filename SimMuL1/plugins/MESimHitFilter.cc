// -*- C++ -*-
//
// Package:    MESimHitFilter
// Class:      MESimHitFilter
// 
/**\class MESimHitFilter

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Vadim Khotilovich
//         Created:  Wed Aug 14 13:32:41 CDT 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "GEMCode/SimMuL1/interface/PSimHitMapCSC.h"

#include <vector>
#include <set>

//
// class declaration
//

class MESimHitFilter : public edm::EDFilter
{
public:
  explicit MESimHitFilter(const edm::ParameterSet&);
  ~MESimHitFilter();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  SimHitAnalysis::PSimHitMapCSC simhit_map_csc_;

  std::set<int> me_types_;
};



MESimHitFilter::MESimHitFilter(const edm::ParameterSet& cfg)
{
  const std::vector<int> def_types {1,4,5}; // ME1/a ME1/b ME2/1
  std::vector<int> types_cfg = cfg.getUntrackedParameter<std::vector<int> >("me_types", def_types);
  std::copy(types_cfg.begin(), types_cfg.end(), inserter(me_types_, me_types_.begin()));
  
  edm::InputTag def_input("g4SimHits","MuonCSCHits");
  simhit_map_csc_.setInputTag(def_input);

}


MESimHitFilter::~MESimHitFilter()
{
}


bool MESimHitFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  simhit_map_csc_.fill(iEvent);

  vector<int> ch_ids = simhit_map_csc_.chambersWithHits();
  if (ch_ids.empty()) return false;

  for(auto d: ch_ids)
  {
    CSCDetId ch_id(d);

    // is it a chamber type of interest?
    if (me_types_.count(ch_id.iChamberType()) == 0) continue;

    // count number of layers with hits
    vector<int> layer_ids = simhit_map_csc_.chamberLayersWithHits(d);
    //cout<<ch_id<<" #L "<<layer_ids.size()<<endl;
    if (layer_ids.size() >= 4) return true;
  }

  return false;
}


void MESimHitFilter::beginJob()
{
}


void MESimHitFilter::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MESimHitFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MESimHitFilter);

