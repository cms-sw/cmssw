// -*- C++ -*-
//
// Package:    SkimSummary
// Class:      SkimSummary
// 
/**\class SkimSummary SkimSummary.cc UserCode/SkimSummary/src/SkimSummary.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Adam Everett
//         Created:  Wed May 18 13:10:45 CDT 2011
// $Id$
//
//


// system include files
#include <memory>
#include <stdio.h>
#include <iomanip>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/TriggerResults.h"

//
// class declaration
//

class SkimSummary : public edm::EDAnalyzer {
   public:
      explicit SkimSummary(const edm::ParameterSet&);
      ~SkimSummary();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
  bool firstEvent;
  int nHltPaths;
  edm::InputTag hltLabel;
  int maxPaths;
  std::vector<std::string>  hlNames_;  

  int nSelectedEvents;

  TH1D* hHltPaths;
  TH1D* hTotalEvents;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SkimSummary::SkimSummary(const edm::ParameterSet& iConfig):
  hltLabel(iConfig.getParameter<edm::InputTag>("HltLabel")),
  maxPaths(iConfig.getUntrackedParameter<int>("maxPaths",25))
{
   //now do what ever initialization is needed

}


SkimSummary::~SkimSummary()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
SkimSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
 
   edm::Handle<edm::TriggerResults> trhv;
   iEvent.getByLabel(hltLabel,trhv);
   if(firstEvent) {
     nHltPaths = trhv->size();
     const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*trhv);
     hlNames_=triggerNames_.triggerNames();
     hlNames_.push_back("Total");
     //hHltPaths = fs-><TH1D>("filterSelected", "Selected Filters", nHltPaths+1, 0, nHltPaths+1);
     int max = (int(hlNames_.size()) < int(maxPaths)) ? hlNames_.size() : maxPaths;
     for (int i=0; i<max; ++i) {
       hHltPaths->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
     }    
   }
   
  std::vector<std::string> firedFilters;
  bool somethingAccepted = false;
  for(unsigned int i=0; i< trhv->size(); i++) {
    if(trhv->at(i).accept()) {
      hHltPaths->Fill(i);
      somethingAccepted = true;
      //firedFilters.push_back(hHltPaths->GetXaxis()->GetBinLabel(i+1));
    }
  }



  if(somethingAccepted) hHltPaths->Fill(trhv->size());
  hTotalEvents->Fill(0.5);
  nSelectedEvents++;
  firstEvent=false;
}


// ------------ method called once each job just before starting event loop  ------------
void 
SkimSummary::beginJob()
{
  firstEvent = true;
  edm::Service<TFileService> fs;
  nSelectedEvents = 0;

  //hHltPaths = fs-><TH1D>("filterSelected", "Selected Filters", nHltPaths+1, 0, nHltPaths+1);
  hHltPaths = fs->make<TH1D>("filterSelected", "Selected Filters", maxPaths+1, 0, maxPaths+1);
  hTotalEvents = fs->make<TH1D>("totalEvents","Total Events in File",1,0,1);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SkimSummary::endJob() 
{
  using namespace std;
  if(nSelectedEvents) {
    std::string theSummaryFileName = "EventsPerFilter.stat";
    ofstream statFile1(theSummaryFileName.c_str(),ios::out);
    for(int i=0; i<nHltPaths; i++) {
      if(hHltPaths->GetBinContent(i+1)) {
	statFile1 << "-----------------------------" << endl;
	statFile1 << "Filter: " <<  hHltPaths->GetXaxis()->GetBinLabel(i+1) << endl;
	statFile1 << "Number of selected events = " << hHltPaths->GetBinContent(i+1) << endl;
	statFile1 << endl;
      }
    }
  }
}

// ------------ method called when starting to processes a run  ------------
void 
SkimSummary::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
SkimSummary::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
SkimSummary::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
SkimSummary::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SkimSummary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SkimSummary);
