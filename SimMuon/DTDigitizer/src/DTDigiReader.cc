#ifndef SimMuon_DTDigiReader_h
#define SimMuon_DTDigiReader_h

/** \class DTDigiReader
 *  Analyse the the muon-drift-tubes digitizer.
 *
 *  \authors: R. Bellan
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/one/EDAnalyzer.h>

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

#include <iostream>

#include "TFile.h"
#include "TH1F.h"  //FIXME

using namespace edm;
using namespace std;

class DTDigiReader : public edm::one::EDAnalyzer<> {
public:
  explicit DTDigiReader(const ParameterSet &pset) {
    file = new TFile("DTDigiPlots.root", "RECREATE");
    file->cd();
    DigiTimeBox = new TH1F("DigiTimeBox", "Digi Time Box", 2048, 0, 1600);
    DigiTimeBoxW0 = new TH1F("DigiTimeBoxW0", "Digi Time Box W0", 2000, 0, 1600);
    DigiTimeBoxW1 = new TH1F("DigiTimeBoxW1", "Digi Time Box W1", 2000, 0, 1600);
    DigiTimeBoxW2 = new TH1F("DigiTimeBoxW2", "Digi Time Box W2", 2000, 0, 1600);
    if (file->IsOpen())
      cout << "file open!" << endl;
    else
      cout << "*** Error in opening file ***" << endl;
    label = pset.getUntrackedParameter<string>("label");
    psim_token = consumes<PSimHitContainer>(edm::InputTag("g4SimHits", "MuonDTHits"));
    DTd_token = consumes<DTDigiCollection>(edm::InputTag(label));
  }

  ~DTDigiReader() override {
    file->cd();
    DigiTimeBox->Write();
    DigiTimeBoxW0->Write();
    DigiTimeBoxW1->Write();
    DigiTimeBoxW2->Write();
    file->Close();
    //    delete file;
    // delete DigiTimeBox;
  }

  void analyze(const Event &event, const EventSetup &eventSetup) override {
    cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

    Handle<DTDigiCollection> dtDigis;
    event.getByToken(DTd_token, dtDigis);
    // event.getByLabel("MuonDTDigis", dtDigis);
    Handle<PSimHitContainer> simHits;
    event.getByToken(psim_token, simHits);

    DTDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = dtDigis->begin(); detUnitIt != dtDigis->end(); ++detUnitIt) {
      const DTLayerId &id = (*detUnitIt).first;
      const DTDigiCollection::Range &range = (*detUnitIt).second;

      // DTLayerId print-out
      cout << "--------------" << endl;
      cout << "id: " << id;

      // Loop over the digis of this DetUnit
      for (DTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        //	if((*digiIt).time()<703 &&(*digiIt).time()>699) {
        cout << " Wire: " << (*digiIt).wire() << endl << " digi time (ns): " << (*digiIt).time() << endl;

        for (vector<PSimHit>::const_iterator simHit = simHits->begin(); simHit != simHits->end(); simHit++) {
          DTWireId wireId((*simHit).detUnitId());
          if (wireId.layerId() == id && abs((*simHit).particleType()) == 13) {
            cout << "entry: " << (*simHit).entryPoint() << endl
                 << "exit: " << (*simHit).exitPoint() << endl
                 << "TOF: " << (*simHit).timeOfFlight() << endl;
          }
        }

        //	}

        if (id.layer() == 3)
          DigiTimeBoxW0->Fill((*digiIt).time());
        else if (abs(id.superlayer()) == 1)
          DigiTimeBoxW1->Fill((*digiIt).time());
        else if (abs(id.superlayer()) == 2)
          DigiTimeBoxW2->Fill((*digiIt).time());
        else
          cout << "Error" << endl;
        DigiTimeBox->Fill((*digiIt).time());

      }  // for digis in layer
    }    // for layers
    cout << "--------------" << endl;
  }

private:
  string label;
  TH1F *DigiTimeBox;
  TH1F *DigiTimeBoxW0;
  TH1F *DigiTimeBoxW1;
  TH1F *DigiTimeBoxW2;
  TFile *file;

  edm::EDGetTokenT<PSimHitContainer> psim_token;
  edm::EDGetTokenT<DTDigiCollection> DTd_token;
};

#endif
