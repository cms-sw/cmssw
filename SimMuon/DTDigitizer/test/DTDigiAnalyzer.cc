/** \class DTDigiAnalyzer
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2006/03/09 16:56:34 $
 *  $Revision: 1.1 $
 *  \authors: R. Bellan
 */

#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include "SimMuon/DTDigitizer/test/DTDigiAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "SimMuon/DTDigitizer/test/Histograms.h"

// #include "SimMuon/DTDigitizer/test/analysis/DTMCStatistics.h"     
// #include "SimMuon/DTDigitizer/test/analysis/DTMuonDigiStatistics.h" 
// #include "SimMuon/DTDigitizer/test/analysis/DTHitsAnalysis.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <iostream>

#include "TH1F.h"  //FIXME
#include "TFile.h"

using namespace edm;
using namespace std;

DTDigiAnalyzer:: DTDigiAnalyzer(const ParameterSet& pset){
//   MCStatistics = new DTMCStatistics();
//   MuonDigiStatistics = new DTMuonDigiStatistics();
//   HitsAnalysis = new DTHitsAnalysis();
  
  file = new TFile("DTDigiPlots.root","RECREATE");
  file->cd();
  DigiTimeBox = new TH1F("DigiTimeBox","Digi Time Box",100,500,1000);
  if(file->IsOpen()) cout<<"file open!"<<endl;
  else cout<<"*** Error in opening file ***"<<endl;
}

DTDigiAnalyzer::~DTDigiAnalyzer(){
  //  cout<<"Number of analyzed event: "<<nevts<<endl;
  //HitsAnalysis->Report();
  
  file->cd();
  DigiTimeBox->Write();
  file->Close();

  //    delete file;
  // delete DigiTimeBox;
}

void  DTDigiAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  cout << "--- Run: " << event.id().run()
       << " Event: " << event.id().event() << endl;
  
  Handle<DTDigiCollection> dtDigis;
  event.getByLabel("dtdigitizer", dtDigis);
  // event.getByLabel("MuonDTDigis", dtDigis);
  
  Handle<PSimHitContainer> simHits; 
  event.getByLabel("r","MuonDTHits",simHits);


  ESHandle<DTGeometry> muonGeom;
  eventSetup.get<MuonGeometryRecord>().get(muonGeom);


  float theta = 0;
  
  for(vector<PSimHit>::const_iterator simHit = simHits->begin();
      simHit != simHits->end(); simHit++){
    if(abs((*simHit).particleType()) == 13)
      theta = atan(simHit->momentumAtEntry().x()/-simHit->momentumAtEntry().z())*180/M_PI;
  }
  
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtDigis->begin();
       detUnitIt!=dtDigis->end();
       ++detUnitIt){
    
    const DTLayerId& id = (*detUnitIt).first;
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    
    // DTLayerId print-out
    cout<<"--------------"<<endl;
    cout<<"id: "<<id;
    
    // Loop over the digis of this DetUnit
    for (DTDigiCollection::const_iterator digiIt = range.first;
	 digiIt!=range.second;
	 ++digiIt){
      cout<<" Wire: "<<(*digiIt).wire()<<endl
	  <<" digi time (ns): "<<(*digiIt).time()<<endl;
      DigiTimeBox->Fill((*digiIt).time());

      hDigis_global.Fill((*digiIt).time(),theta,id.superlayer());
      
      //filling digi histos for wheel and for RZ and RPhi
      WheelHistos(id.wheel())->Fill((*digiIt).time(),theta,id.superlayer());
      
    }// for digis in layer
  }// for layers
  cout<<"--------------"<<endl;
}

hDigis* DTDigiAnalyzer::WheelHistos(int wheel){
  switch(abs(wheel)){

  case 0: return  &hDigis_W0;
  
  case 1: return  &hDigis_W1;
    
  case 2: return  &hDigis_W2;
     
  default: return NULL;
  }
}
