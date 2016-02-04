/** \class DTDigiAnalyzer
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2007/09/05 07:29:06 $
 *  $Revision: 1.10 $
 *  \authors: R. Bellan
 */

#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include "SimMuon/DTDigitizer/test/DTDigiAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

// #include "SimMuon/DTDigitizer/test/analysis/DTMCStatistics.h"     
// #include "SimMuon/DTDigitizer/test/analysis/DTMuonDigiStatistics.h" 
// #include "SimMuon/DTDigitizer/test/analysis/DTHitsAnalysis.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <iostream>
#include <string>

#include "TFile.h"

#include "SimMuon/DTDigitizer/test/Histograms.h"

hDigis hDigis_global("Global");
hDigis hDigis_W0("Wheel0");
hDigis hDigis_W1("Wheel1");
hDigis hDigis_W2("Wheel2");

hHits hAllHits("AllHits");


using namespace edm;
using namespace std;

DTDigiAnalyzer:: DTDigiAnalyzer(const ParameterSet& pset){
//   MCStatistics = new DTMCStatistics();
//   MuonDigiStatistics = new DTMuonDigiStatistics();
//   HitsAnalysis = new DTHitsAnalysis();
  label = pset.getUntrackedParameter<string>("label");  
  file = new TFile("DTDigiPlots.root","RECREATE");
  file->cd();
  DigiTimeBox = new TH1F("DigiTimeBox","Digi Time Box",2048,0,1600);
  if(file->IsOpen()) cout<<"file open!"<<endl;
  else cout<<"*** Error in opening file ***"<<endl;
}

DTDigiAnalyzer::~DTDigiAnalyzer(){
}

void DTDigiAnalyzer::endJob(){
  //cout<<"Number of analyzed event: "<<nevts<<endl;
  //HitsAnalysis->Report();
  file->cd();
  DigiTimeBox->Write();
  hDigis_global.Write();
  hDigis_W0.Write();
  hDigis_W1.Write();
  hDigis_W2.Write();
  hAllHits.Write();
  file->Close();
  //    delete file;
  // delete DigiTimeBox;
}

void  DTDigiAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  cout << "--- Run: " << event.id().run()
       << " Event: " << event.id().event() << endl;
  
  Handle<DTDigiCollection> dtDigis;
  event.getByLabel(label, dtDigis);
  
  Handle<PSimHitContainer> simHits; 
  event.getByLabel("g4SimHits","MuonDTHits",simHits);    

  ESHandle<DTGeometry> muonGeom;
  eventSetup.get<MuonGeometryRecord>().get(muonGeom);


  DTWireIdMap wireMap;     
  
  for(vector<PSimHit>::const_iterator hit = simHits->begin();
      hit != simHits->end(); hit++){    
    // Create the id of the wire, the simHits in the DT known also the wireId
     DTWireId wireId(hit->detUnitId());
    // Fill the map
    wireMap[wireId].push_back(&(*hit));

    LocalPoint entryP = hit->entryPoint();
    LocalPoint exitP = hit->exitPoint();
    int partType = hit->particleType();
    
    float path = (exitP-entryP).mag();
    float path_x = fabs((exitP-entryP).x());
    
    hAllHits.Fill(entryP.x(),exitP.x(),
		   entryP.y(),exitP.y(),
		   entryP.z(),exitP.z(),
		   path , path_x, 
		   partType, hit->processType(),
		  hit->pabs());

    if( hit->timeOfFlight() > 1e4){
      cout<<"PID: "<<hit->particleType()
	  <<" TOF: "<<hit->timeOfFlight()
	  <<" Proc Type: "<<hit->processType() 
	  <<" p: " << hit->pabs() <<endl;
      hAllHits.FillTOF(hit->timeOfFlight());
    }
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
      
      DTWireId wireId(id,(*digiIt).wire());
      int mu=0;
      float theta = 0;
      
      for(vector<const PSimHit*>::iterator hit = wireMap[wireId].begin();
	  hit != wireMap[wireId].end(); hit++){
	cout<<"momentum x: "<<(*hit)->momentumAtEntry().x()<<endl
	    <<"momentum z: "<<(*hit)->momentumAtEntry().z()<<endl;
	if( abs((*hit)->particleType()) == 13){
	  theta = atan( (*hit)->momentumAtEntry().x()/ (-(*hit)->momentumAtEntry().z()) )*180/M_PI;
	  cout<<"atan: "<<theta<<endl;
	  mu++;
	}
	else{
	  // 	  cout<<"PID: "<<(*hit)->particleType()
	  // 	      <<" TOF: "<<(*hit)->timeOfFlight()
	  // 	      <<" Proc Type: "<<(*hit)->processType() 
	  // 	      <<" p: " << (*hit)->pabs() <<endl;
	}
      }
      if(mu && theta){
	hDigis_global.Fill((*digiIt).time(),theta,id.superlayer());
	//filling digi histos for wheel and for RZ and RPhi
	WheelHistos(id.wheel())->Fill((*digiIt).time(),theta,id.superlayer());
      }
	  
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


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DTDigiAnalyzer);
