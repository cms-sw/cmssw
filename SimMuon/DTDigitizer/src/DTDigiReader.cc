#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>


#include <iostream>

using namespace edm;
using namespace std;

class DTDigiReader: public EDAnalyzer{
  
  public:
    explicit DTDigiReader(const ParameterSet& pset){}
 
    void analyze(const Event & event, const EventSetup& eventSetup){
      cout << "--- Run: " << event.id().run()
	   << " Event: " << event.id().event() << endl;

      Handle<DTDigiCollection> dtDigis;
      event.getByLabel("dtdigitizer", dtDigis);

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
	  cout<<" Wi: "<<(*digiIt).wire()<<endl
	      <<"digi time: "<<(*digiIt).time()<<endl;
	  
	}// for digis in layer
      }// for layers
    }
  
};
    
