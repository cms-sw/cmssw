#ifndef SimMuon_DTDigiReader_h
#define SimMuon_DTDigiReader_h

/** \class DTDigiReader
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2006/01/25 10:40:00 $
 *  $Revision: 1.2 $
 *  \authors: R. Bellan
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

#include <iostream>

#include "TH1F.h"  //FIXME
#include "TFile.h"

using namespace edm;
using namespace std;

class DTDigiReader: public EDAnalyzer{
  
public:
  explicit DTDigiReader(const ParameterSet& pset){
    file = new TFile("DTDigiPlots.root","RECREATE");
    file->cd();
    DigiTimeBox = new TH1F("DigiTimeBox","Digi Time Box",100,500,1000);
    if(file->IsOpen()) cout<<"file open!"<<endl;
    else cout<<"*** Error in opening file ***"<<endl;
  }
  
  virtual ~DTDigiReader(){
    file->cd();
    DigiTimeBox->Write();
    file->Close();
    //    delete file;
    // delete DigiTimeBox;
  }
  
  void analyze(const Event & event, const EventSetup& eventSetup){
    cout << "--- Run: " << event.id().run()
	 << " Event: " << event.id().event() << endl;
    
    Handle<DTDigiCollection> dtDigis;
    event.getByLabel("dtdigitizer", dtDigis);
    // event.getByLabel("MuonDTDigis", dtDigis);
    
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

      }// for digis in layer
    }// for layers
    cout<<"--------------"<<endl;
  }
  
private:
  TH1F *DigiTimeBox;
  TFile *file;
  
};

#endif    
