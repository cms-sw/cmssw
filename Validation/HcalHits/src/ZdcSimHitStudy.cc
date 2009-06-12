#include "Validation/HcalHits/interface/ZdcSimHitStudy.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

ZdcSimHitStudy::ZdcSimHitStudy(const edm::ParameterSet& ps) {

  g4Label  = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  zdcHits = ps.getUntrackedParameter<std::string>("HitCollection","ZdcHits");
  outFile_ = ps.getUntrackedParameter<std::string>("outputFile", "zdcHitStudy.root");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  checkHit_= true;

  edm::LogInfo("ZdcSimHitStudy") 
    //std::cout
    << "Module Label: " << g4Label << "   Hits: "
    << zdcHits << " / "<< checkHit_ 
    << "   Output: " << outFile_;

  dbe_ = edm::Service<DQMStore>().operator->();
  if (dbe_) {
    if (verbose_) {
      dbe_->setVerbose(1);
      sleep (3);
      dbe_->showDirStructure();
    } else {
      dbe_->setVerbose(0);
    }
  }
}

ZdcSimHitStudy::~ZdcSimHitStudy() {
  if (dbe_ && outFile_.size() > 0) dbe_->save(outFile_);
}

void ZdcSimHitStudy::beginJob() {

  if (dbe_) {
    dbe_->setCurrentFolder("ZdcHitsV/ZdcSimHitsTask");
    //Histograms for Hits
    if (checkHit_) {
      meAllZdcNHit_ = dbe_->book1D("Hit01","Number of All Hits in Zdc",100,0.,100.);
      meBadZdcDetHit_= dbe_->book1D("Hit02","Hits with wrong Det in Zdc",100,0.,100.);
      meBadZdcSecHit_= dbe_->book1D("Hit03","Hits with wrong Section in Zdc",100,0.,100.);
      meBadZdcIdHit_ = dbe_->book1D("Hit04","Hits with wrong ID in Zdc",100,0.,100.);
      meZdcNHitEM_   = dbe_->book1D("Hit05","Number of Hits in Zdc EM",100,0.,100.);
      meZdcNHitHad_   = dbe_->book1D("Hit06","Number of Hits in Zdc Had",100,0.,100.);
      meZdcNHitLum_   = dbe_->book1D("Hit07","Number of Hits in Zdc Lum",100,0.,100.);
      meZdcDetectHit_= dbe_->book1D("Hit08","Calo Detector ID",50,0.,50.);
      meZdcSideHit_ = dbe_->book1D("Hit09","Side in Zdc",4,-2,2.);
      meZdcSectionHit_   = dbe_->book1D("Hit10","Section in Zdc",4,0.,4.);
      meZdcChannelHit_   = dbe_->book1D("Hit11","Channel in Zdc",10,0.,10.);
      meZdcEnergyHit_= dbe_->book1D("Hit12","Hits Energy",4000,0.,8000.);
      meZdcHadEnergyHit_= dbe_->book1D("Hit13","Hits Energy in Had Section",4000,0.,8000.);
      meZdcEMEnergyHit_ = dbe_->book1D("Hit14","Hits Energy in EM Section",4000,0.,8000.);
      meZdcTimeHit_  = dbe_->book1D("Hit15","Time in Zdc",300,0.,600.);
      meZdcTimeWHit_ = dbe_->book1D("Hit16","Time in Zdc (E wtd)", 300,0.,600.);
      meZdc10Ene_ = dbe_->book1D("Hit17","Log10Energy in Zdc", 140, -20., 20. );
      meZdcHadL10EneP_ = dbe_->bookProfile("Hit18","Log10Energy in Had Zdc vs Hit contribution", 140, -1., 20., 100, 0., 1. );
      meZdcEML10EneP_ = dbe_->bookProfile("Hit19","Log10Energy in EM Zdc vs Hit contribution", 140, -1., 20., 100, 0., 1. );
      meZdcEHadCh_ = dbe_->book2D("Hit20","Zdc Had Section Energy vs Channel", 4000, 0., 8000., 6, 0., 6. );
      meZdcEEMCh_ = dbe_->book2D("Hit21","Zdc EM Section Energy vs Channel", 4000, 0., 8000., 6, 0., 6. );
      meZdcETime_ = dbe_->book2D("Hit22","Hits Zdc Energy vs Time", 4000, 0., 8000., 300, 0., 600. );
    }
  }
}

void ZdcSimHitStudy::endJob() {}

void ZdcSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("ZdcSimHitStudy") 
    //std::cout
    << "Run = " << e.id().run() << " Event = " 
    << e.id().event();
    //<<std::endl;
  
  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsZdc;

  bool getHits = false;
  if (checkHit_) {
    e.getByLabel(g4Label,zdcHits,hitsZdc); 
    if (hitsZdc.isValid()) getHits = true;
  }

  LogDebug("ZdcSim") << "ZdcValidation: Input flags Hits " << getHits;

  if (getHits) {
    caloHits.insert(caloHits.end(),hitsZdc->begin(),hitsZdc->end());
    LogDebug("ZdcSimHitStudy") 
      //std::cout
      << "ZdcValidation: Hit buffer " 
      << caloHits.size();
      //<< std::endl;
    analyzeHits (caloHits);
  }
}

void ZdcSimHitStudy::analyzeHits (std::vector<PCaloHit>& hits){

  int nHit = hits.size();
  int nZdcEM = 0, nZdcHad = 0, nZdcLum = 0; 
  int nBad1=0, nBad2=0, nBad=0;
  std::vector<double> encontZdcEM(140, 0.);
  std::vector<double> encontZdcHad(140, 0.);
  double entotZdcEM = 0;
  double entotZdcHad = 0;
  
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double log10en   = log10(energy);
    int log10i       = int( (log10en+10.)*10. );
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    HcalZDCDetId id  = HcalZDCDetId(id_);
    int det          = id.det();
    int side         = id.zside();
    int section      = id.section();
    int channel      = id.channel();
    
    LogDebug("ZdcSimHitStudy") 
      //std::cout
      << "Hit[" << i << "] ID " << std::hex << id_ 
      << std::dec <<" DetID "<<id
      << " Det "<< det << " side "<< side 
      << " Section " << section
      << " channel "<< channel
      << " E " << energy 
      << " time \n" << time;
      //<<std::endl;

    if(det == 5) { // Check DetId.h
      if(section == HcalZDCDetId::EM)nZdcEM++;
      else if(section == HcalZDCDetId::HAD)nZdcHad++;
      else if(section == HcalZDCDetId::LUM)nZdcLum++;
      else    { nBad++;  nBad2++;}
    } else    { nBad++;  nBad1++;}
    if (dbe_) {
      meZdcDetectHit_->Fill(double(det));
      if (det ==  5) {
	meZdcSideHit_->Fill(double(side));
	meZdcSectionHit_->Fill(double(section));
	meZdcChannelHit_->Fill(double(channel));
	meZdcEnergyHit_->Fill(energy);
      if(section == HcalZDCDetId::EM){
	meZdcEMEnergyHit_->Fill(energy);
	meZdcEEMCh_->Fill(energy,channel);
	if( log10i >=0 && log10i < 140 )encontZdcEM[log10i] += energy;
	entotZdcEM += energy;
      }
      if(section == HcalZDCDetId::HAD){
	meZdcHadEnergyHit_->Fill(energy);
	meZdcEHadCh_->Fill(energy,channel);
	if( log10i >=0 && log10i < 140 )encontZdcHad[log10i] += energy;
	entotZdcHad += energy; 
      }	
      meZdcTimeHit_->Fill(time);
      meZdcTimeWHit_->Fill(double(time),energy);
      meZdc10Ene_->Fill(log10en);
      meZdcETime_->Fill(energy, double(time));
      
      }
    }
  }
  
  if( entotZdcEM  != 0 ) for( int i=0; i<140; i++ ) meZdcEML10EneP_->Fill( -10.+(float(i)+0.5)/10., encontZdcEM[i]/entotZdcEM);
  if( entotZdcHad != 0 ) for( int i=0; i<140; i++ ) meZdcHadL10EneP_->Fill( -10.+(float(i)+0.5)/10.,encontZdcHad[i]/entotZdcHad);
  
  if (dbe_) {
    meAllZdcNHit_->Fill(double(nHit));
    meBadZdcDetHit_->Fill(double(nBad1));
    meBadZdcSecHit_->Fill(double(nBad2));
    meBadZdcIdHit_->Fill(double(nBad));
    meZdcNHitEM_->Fill(double(nZdcEM));
    meZdcNHitHad_->Fill(double(nZdcHad));
    meZdcNHitLum_->Fill(double(nZdcLum));
  }
  LogDebug("HcalSimHitStudy") 
  //std::cout
    <<"HcalSimHitStudy::analyzeHits: Had " << nZdcHad 
    << " EM "<< nZdcEM
    << " Bad " << nBad << " All " << nHit;
    //<<std::endl;
}
