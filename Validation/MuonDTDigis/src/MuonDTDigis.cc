/** \class MuonDTDigis
 *  Analyse the the muon-drift-tubes digitizer.
 *
 *  \authors: R. Bellan
 */

#include "MuonDTDigis.h"

#include <iostream>
#include <string>

#include "TFile.h"

#include "SimMuon/DTDigitizer/test/Histograms.h"

using namespace edm;
using namespace std;

MuonDTDigis::MuonDTDigis(const ParameterSet& pset){

  // ----------------------
  // Get the debug parameter for verbose output
  verbose_ = pset.getUntrackedParameter<bool>("verbose",false);

  // the name of the Digi collection
  SimHitToken_ = consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("SimHitLabel"));

  // the name of the Digi collection
  DigiToken_ = consumes<DTDigiCollection> (pset.getParameter<edm::InputTag>(("DigiLabel")));

  // ----------------------
  // DQM ROOT output
  outputFile_ =  pset.getUntrackedParameter<std::string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " DT Muon Digis Task histograms will be saved to '"
                          << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " DT Muon Digis Task histograms will NOT be saved";
  }

  string::size_type loc = outputFile_.find( ".root", 0 );
  std::string outputFile_more_plots_;
  if( loc != string::npos ) {
    outputFile_more_plots_ = outputFile_.substr(0,loc)+"_more_plots.root";
  } else {
    outputFile_more_plots_ = " DTDigis_more_plots.root";
  }

  //    Please, uncomment next lines if you want a secong root file with additional histos

  //  file_more_plots = new TFile(outputFile_more_plots_.c_str(),"RECREATE");
  //  file_more_plots->cd();
  //  if(file_more_plots->IsOpen()) cout<<"File for additional plots: " << outputFile_more_plots_ << "  open!"<<endl;
  //  else cout<<"*** Error in opening file for additional plots ***"<<endl;

  hDigis_global = new hDigis("Global");
  hDigis_W0 = new hDigis("Wheel0");
  hDigis_W1 = new hDigis("Wheel1");
  hDigis_W2 = new hDigis("Wheel2");
  hAllHits = new hHits("AllHits");

  // End of comment . See more in Destructor (~MuonDTDigis) method



  // ----------------------
  // get hold of back-end interface
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }

  // ----------------------

  meDigiTimeBox_          = 0;
  meDigiTimeBox_wheel2m_  = 0;
  meDigiTimeBox_wheel1m_  = 0;
  meDigiTimeBox_wheel0_   = 0;
  meDigiTimeBox_wheel1p_  = 0;
  meDigiTimeBox_wheel2p_  = 0;
  meDigiEfficiency_       = 0;
  meDigiEfficiencyMu_     = 0;
  meDoubleDigi_           = 0;
  meSimvsDigi_            = 0;
  meWire_DoubleDigi_      = 0;

  meMB1_sim_occup_        = 0;
  meMB1_digi_occup_       = 0;
  meMB2_sim_occup_        = 0;
  meMB2_digi_occup_       = 0;
  meMB3_sim_occup_        = 0;
  meMB3_digi_occup_       = 0;
  meMB4_sim_occup_        = 0;
  meMB4_digi_occup_       = 0;

  //meDigiTimeBox_SL_       = 0;
  meDigiHisto_            = 0;


  // ----------------------
  // We go
  // ----------------------

  Char_t histo_n[100];
  Char_t histo_t[100];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("MuonDTDigisV/DTDigiValidationTask");

    sprintf (histo_n, "DigiTimeBox" );
    sprintf (histo_t, "Digi Time Box" );
    meDigiTimeBox_ = dbe_->book1D(histo_n, histo_t, 1536,0,1200);

    sprintf (histo_n, "DigiTimeBox_wheel2m" );
    sprintf (histo_t, "Digi Time Box wheel -2" );
    meDigiTimeBox_wheel2m_ = dbe_->book1D(histo_n, histo_t, 384,0,1200);

    sprintf (histo_n, "DigiTimeBox_wheel1m" );
    sprintf (histo_t, "Digi Time Box wheel -1" );
    meDigiTimeBox_wheel1m_ = dbe_->book1D(histo_n, histo_t, 384,0,1200);

    sprintf (histo_n, "DigiTimeBox_wheel0" );
    sprintf (histo_t, "Digi Time Box wheel 0" );
    meDigiTimeBox_wheel0_ = dbe_->book1D(histo_n, histo_t, 384,0,1200);

    sprintf (histo_n, "DigiTimeBox_wheel1p" );
    sprintf (histo_t, "Digi Time Box wheel 1" );
    meDigiTimeBox_wheel1p_ = dbe_->book1D(histo_n, histo_t, 384,0,1200);

    sprintf (histo_n, "DigiTimeBox_wheel2p" );
    sprintf (histo_t, "Digi Time Box wheel 2" );
    meDigiTimeBox_wheel2p_ = dbe_->book1D(histo_n, histo_t, 384,0,1200);

    sprintf (histo_n, "DigiEfficiencyMu" );
    sprintf (histo_t, "Ratio (#Digis Mu)/(#SimHits Mu)" );
    meDigiEfficiencyMu_ = dbe_->book1D(histo_n, histo_t, 100, 0., 5.);

    sprintf (histo_n, "DigiEfficiency" );
    sprintf (histo_t, "Ratio (#Digis)/(#SimHits)" );
    meDigiEfficiency_ = dbe_->book1D(histo_n, histo_t, 100, 0., 5.);

    sprintf (histo_n, "Number_Digi_per_layer" );
    sprintf (histo_t, "Number_Digi_per_layer" );
    meDoubleDigi_ = dbe_->book1D(histo_n, histo_t, 10,0.,10.);

    sprintf (histo_n, "Number_simhit_vs_digi" );
    sprintf (histo_t, "Number_simhit_vs_digi" );
    meSimvsDigi_ = dbe_->book2D(histo_n, histo_t, 100, 0., 140., 100, 0., 140.);

    sprintf (histo_n, "Wire_Number_with_double_Digi" );
    sprintf (histo_t, "Wire_Number_with_double_Digi" );
    meWire_DoubleDigi_ = dbe_->book1D(histo_n, histo_t, 100,0.,100.);

    sprintf (histo_n, "Simhit_occupancy_MB1" );
    sprintf (histo_t, "Simhit_occupancy_MB1" );
    meMB1_sim_occup_ = dbe_->book1D(histo_n, histo_t, 55, 0., 55. );

    sprintf (histo_n, "Digi_occupancy_MB1" );
    sprintf (histo_t, "Digi_occupancy_MB1" );
    meMB1_digi_occup_ = dbe_->book1D(histo_n, histo_t, 55, 0., 55. );

    sprintf (histo_n, "Simhit_occupancy_MB2" );
    sprintf (histo_t, "Simhit_occupancy_MB2" );
    meMB2_sim_occup_ = dbe_->book1D(histo_n, histo_t, 63, 0., 63. );

    sprintf (histo_n, "Digi_occupancy_MB2" );
    sprintf (histo_t, "Digi_occupancy_MB2" );
    meMB2_digi_occup_ = dbe_->book1D(histo_n, histo_t, 63, 0., 63. );

    sprintf (histo_n, "Simhit_occupancy_MB3" );
    sprintf (histo_t, "Simhit_occupancy_MB3" );
    meMB3_sim_occup_ = dbe_->book1D(histo_n, histo_t, 75, 0., 75. );

    sprintf (histo_n, "Digi_occupancy_MB3" );
    sprintf (histo_t, "Digi_occupancy_MB3" );
    meMB3_digi_occup_ = dbe_->book1D(histo_n, histo_t, 75, 0., 75. );

    sprintf (histo_n, "Simhit_occupancy_MB4" );
    sprintf (histo_t, "Simhit_occupancy_MB4" );
    meMB4_sim_occup_ = dbe_->book1D(histo_n, histo_t, 99, 0., 99. );

    sprintf (histo_n, "Digi_occupancy_MB4" );
    sprintf (histo_t, "Digi_occupancy_MB4" );
    meMB4_digi_occup_ = dbe_->book1D(histo_n, histo_t, 99, 0., 99. );

    //    sprintf (histo_n, "" );
    //    sprintf (histo_t, "" );

    /*
    //  Other option
    string histoTag = "DigiTimeBox_slid_";
    for ( int wheel = -2; wheel <= +2; ++wheel ) {
    for ( int station = 1; station <= 4; ++station )  {
    for ( int superLayer = 1; superLayer <= 3; ++superLayer ) {
    string histoName = histoTag
    + "_W" + wheel.str()
    + "_St" + station.str()
    + "_SL" + superLayer.str();
    // the booking is not yet done
    }
    }
    }
    */
    // Begona's option
    char stringcham[40];
    for ( int slnum = 1; slnum < 62; ++slnum ) {
      sprintf(stringcham, "DigiTimeBox_slid_%d", slnum) ;
      meDigiHisto_ =  dbe_->book1D(stringcham, stringcham, 100,0,1200);
      meDigiTimeBox_SL_.push_back(meDigiHisto_);
    }
  }

}

MuonDTDigis::~MuonDTDigis(){

  // Uncomment these next lines if you want additional plots in a second .root file

  //  file_more_plots->cd();

  // hDigis_global->Write();

  //  hDigis_W0->Write();
  //  hDigis_W1->Write();
  //  hDigis_W2->Write();
  //  hAllHits->Write();

  //  file_more_plots->Close();

  // End of comment.

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

  if(verbose_)
    cout << "[MuonDTDigis] Destructor called" << endl;

}

//void  MuonDTDigis::endJob(void){

//}

void  MuonDTDigis::analyze(const Event & event, const EventSetup& eventSetup){

  if(verbose_)
    cout << "--- [MuonDTDigis] Analysing Event: #Run: " << event.id().run()
         << " #Event: " << event.id().event() << endl;

  // Get the DT Geometry
  ESHandle<DTGeometry> muonGeom;
  eventSetup.get<MuonGeometryRecord>().get(muonGeom);

  // Get the Digi collection from the event
  Handle<DTDigiCollection> dtDigis;
  event.getByToken(DigiToken_, dtDigis);

  // Get simhits
  Handle<PSimHitContainer> simHits;
  event.getByToken(SimHitToken_, simHits);


  int num_mudigis;
  int num_musimhits;
  int num_digis;
  int num_digis_layer;
  int cham_num ;
  int wire_touched;
  num_digis = 0;
  num_mudigis = 0;
  num_musimhits = 0;
  DTWireIdMap wireMap;

  for(vector<PSimHit>::const_iterator hit = simHits->begin();
      hit != simHits->end(); hit++){
    // Create the id of the wire, the simHits in the DT known also the wireId
    DTWireId wireId(hit->detUnitId());
    //   cout << " PSimHits wire id " << wireId << " part type " << hit->particleType() << endl;

    // Fill the map
    wireMap[wireId].push_back(&(*hit));

    LocalPoint entryP = hit->entryPoint();
    LocalPoint exitP = hit->exitPoint();
    int partType = hit->particleType();
    if ( abs(partType) == 13 ) num_musimhits++;

    if ( wireId.station() == 1 && abs(partType) == 13 ) meMB1_sim_occup_->Fill(wireId.wire());
    if ( wireId.station() == 2 && abs(partType) == 13 ) meMB2_sim_occup_->Fill(wireId.wire());
    if ( wireId.station() == 3 && abs(partType) == 13 ) meMB3_sim_occup_->Fill(wireId.wire());
    if ( wireId.station() == 4 && abs(partType) == 13 ) meMB4_sim_occup_->Fill(wireId.wire());


    float path = (exitP-entryP).mag();
    float path_x = fabs((exitP-entryP).x());

    hAllHits->Fill(entryP.x(),exitP.x(),
		   entryP.y(),exitP.y(),
		   entryP.z(),exitP.z(),
		   path , path_x,
		   partType, hit->processType(),
                   hit->pabs());

  }

  //  cout << "num muon simhits " << num_musimhits << endl;

  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtDigis->begin();
       detUnitIt!=dtDigis->end();
       ++detUnitIt){

    const DTLayerId& id = (*detUnitIt).first;
    const DTDigiCollection::Range& range = (*detUnitIt).second;

    num_digis_layer = 0 ;
    cham_num = 0 ;
    wire_touched = 0;

    // Loop over the digis of this DetUnit
    for (DTDigiCollection::const_iterator digiIt = range.first;
	 digiIt!=range.second;
	 ++digiIt){
      //   cout<<" Wire: "<<(*digiIt).wire()<<endl
      //  <<" digi time (ns): "<<(*digiIt).time()<<endl;

      num_digis++;
      num_digis_layer++;
      if (num_digis_layer > 1 )
      {
        if ( (*digiIt).wire() == wire_touched )
        {
          meWire_DoubleDigi_->Fill((*digiIt).wire());
          //      cout << "old & new wire " << wire_touched << " " << (*digiIt).wire() << endl;
        }
      }
      wire_touched = (*digiIt).wire();

      meDigiTimeBox_->Fill((*digiIt).time());
      if (id.wheel() == -2 ) meDigiTimeBox_wheel2m_->Fill((*digiIt).time());
      if (id.wheel() == -1 ) meDigiTimeBox_wheel1m_->Fill((*digiIt).time());
      if (id.wheel() == 0 )  meDigiTimeBox_wheel0_ ->Fill((*digiIt).time());
      if (id.wheel() == 1 )  meDigiTimeBox_wheel1p_->Fill((*digiIt).time());
      if (id.wheel() == 2 )  meDigiTimeBox_wheel2p_->Fill((*digiIt).time());

      //   Superlayer number and fill histo with digi timebox

      cham_num = (id.wheel() +2)*12 + (id.station() -1)*3 + id.superlayer();
      //   cout << " Histo number " << cham_num << endl;

      meDigiTimeBox_SL_[cham_num]->Fill((*digiIt).time());

      //    cout << " size de digis " << (*digiIt).size() << endl;

      DTWireId wireId(id,(*digiIt).wire());
      if (wireId.station() == 1 ) meMB1_digi_occup_->Fill((*digiIt).wire());
      if (wireId.station() == 2 ) meMB2_digi_occup_->Fill((*digiIt).wire());
      if (wireId.station() == 3 ) meMB3_digi_occup_->Fill((*digiIt).wire());
      if (wireId.station() == 4 ) meMB4_digi_occup_->Fill((*digiIt).wire());

      int mu=0;
      float theta = 0;

      for(vector<const PSimHit*>::iterator hit = wireMap[wireId].begin();
	  hit != wireMap[wireId].end(); hit++)
	if( abs((*hit)->particleType()) == 13){
	  theta = atan( (*hit)->momentumAtEntry().x()/ (-(*hit)->momentumAtEntry().z()) )*180/M_PI;
          //	  cout<<"momentum x: "<<(*hit)->momentumAtEntry().x()<<endl
          //	      <<"momentum z: "<<(*hit)->momentumAtEntry().z()<<endl
          //	      <<"atan: "<<theta<<endl;
	  mu++;
	}

      if( mu ) num_mudigis++;

      if(mu && theta){
	hDigis_global->Fill((*digiIt).time(),theta,id.superlayer());
	//filling digi histos for wheel and for RZ and RPhi
	WheelHistos(id.wheel())->Fill((*digiIt).time(),theta,id.superlayer());
      }

    }// for digis in layer

    meDoubleDigi_->Fill( (float)num_digis_layer );

  }// for layers

  //cout << "num_digis " << num_digis << "mu digis " << num_mudigis << endl;

  if (num_musimhits != 0) {
    meDigiEfficiencyMu_->Fill( (float)num_mudigis/(float)num_musimhits );
    meDigiEfficiency_->Fill( (float)num_digis/(float)num_musimhits );
  }

  meSimvsDigi_->Fill( (float)num_musimhits, (float)num_digis ) ;
  //  cout<<"--------------"<<endl;

}

hDigis* MuonDTDigis::WheelHistos(int wheel){
  switch(abs(wheel)){

    case 0: return  hDigis_W0;

    case 1: return  hDigis_W1;

    case 2: return  hDigis_W2;

    default: return NULL;
  }
}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_FWK_MODULE(MuonDTDigis);
