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

ZdcSimHitStudy::~ZdcSimHitStudy() {}

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
      meZdcEneEmN1_  = dbe_->book1D("HitVal01","Energy EM module N1",4000,0.,8000.);
      meZdcEneEmN2_  = dbe_->book1D("HitVal02","Energy EM module N2",4000,0.,8000.);
      meZdcEneEmN3_  = dbe_->book1D("HitVal03","Energy EM module N3",4000,0.,8000.);
      meZdcEneEmN4_  = dbe_->book1D("HitVal04","Energy EM module N4",4000,0.,8000.);
      meZdcEneEmN5_  = dbe_->book1D("HitVal05","Energy EM module N5",4000,0.,8000.);
      meZdcEneHadN1_ = dbe_->book1D("HitVal06","Energy HAD module N1",4000,0.,8000.);
      meZdcEneHadN2_ = dbe_->book1D("HitVal07","Energy HAD module N2",4000,0.,8000.);
      meZdcEneHadN3_ = dbe_->book1D("HitVal08","Energy HAD module N3",4000,0.,8000.);
      meZdcEneHadN4_ = dbe_->book1D("HitVal09","Energy HAD module N4",4000,0.,8000.);
      meZdcEneTEmN1_ = dbe_->book2D("HitVal11","Energy EM mod N1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN2_ = dbe_->book2D("HitVal12","Energy EM mod N2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTEmN3_ = dbe_->book2D("HitVal13","Energy EM mod N3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN4_ = dbe_->book2D("HitVal14","Energy EM mod N4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN5_ = dbe_->book2D("HitVal15","Energy EM mod N5 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN1_ = dbe_->book2D("HitVal16","Energy HAD mod N1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN2_ = dbe_->book2D("HitVal17","Energy HAD mod N2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTHadN3_ = dbe_->book2D("HitVal18","Energy HAD mod N3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN4_ = dbe_->book2D("HitVal19","Energy HAD mod N4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneHadNTot_ = dbe_->book1D("HitVal20","Total HAD Energy N",4000,0.,8000.);
      meZdcEneEmNTot_  = dbe_->book1D("HitVal21","Total EM Energy N",4000,0.,8000.);
      meZdcEneNTot_    = dbe_->book1D("HitVal22","Total Energy N",4000,0.,8000.);
      meZdcEneEmP1_  = dbe_->book1D("HitVal23","Energy EM module P1",4000,0.,8000.);
      meZdcEneEmP2_  = dbe_->book1D("HitVal24","Energy EM module P2",4000,0.,8000.);
      meZdcEneEmP3_  = dbe_->book1D("HitVal25","Energy EM module P3",4000,0.,8000.);
      meZdcEneEmP4_  = dbe_->book1D("HitVal26","Energy EM module P4",4000,0.,8000.);
      meZdcEneEmP5_  = dbe_->book1D("HitVal27","Energy EM module P5",4000,0.,8000.);
      meZdcEneHadP1_ = dbe_->book1D("HitVal29","Energy HAD module P1",4000,0.,8000.);
      meZdcEneHadP2_ = dbe_->book1D("HitVal29","Energy HAD module P2",4000,0.,8000.);
      meZdcEneHadP3_ = dbe_->book1D("HitVal30","Energy HAD module P3",4000,0.,8000.);
      meZdcEneHadP4_ = dbe_->book1D("HitVal31","Energy HAD module P4",4000,0.,8000.);
      meZdcEneTEmP1_ = dbe_->book2D("HitVal32","Energy EM mod P1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP2_ = dbe_->book2D("HitVal33","Energy EM mod P2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTEmP3_ = dbe_->book2D("HitVal34","Energy EM mod P3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP4_ = dbe_->book2D("HitVal35","Energy EM mod P4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP5_ = dbe_->book2D("HitVal36","Energy EM mod P5 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP1_ = dbe_->book2D("HitVal37","Energy HAD mod P1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP2_ = dbe_->book2D("HitVal38","Energy HAD mod P2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTHadP3_ = dbe_->book2D("HitVal39","Energy HAD mod P3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP4_ = dbe_->book2D("HitVal40","Energy HAD mod P4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneHadPTot_ = dbe_->book1D("HitVal41","Total HAD Energy P",4000,0.,8000.);
      meZdcEneEmPTot_  = dbe_->book1D("HitVal42","Total EM Energy P",4000,0.,8000.);
      meZdcEnePTot_    = dbe_->book1D("HitVal43","Total Energy P",4000,0.,8000.);
      meZdcCorEEmNEHadN_= dbe_->book2D("HitVal47","Energy EMN vs HADN", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcCorEEmPEHadP_= dbe_->book2D("HitVal44","Energy EMP vs HADP", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcCorEtotNEtotP_ = dbe_->book2D("HitVal45","Energy N vs P", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcEneTot_ = dbe_->book1D("HitVal46","Total Energy ZDCs",4000,0.,8000.);
    }
  }
}

void ZdcSimHitStudy::endJob() {
  if (dbe_ && outFile_.size() > 0) dbe_->save(outFile_);
}

void ZdcSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("ZdcSimHitStudy") 
    //std::cout
    << "Run = " << e.id().run() << " Event = " 
    << e.id().event();
  //std::cout<<std::endl;
  
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

void ZdcSimHitStudy::analyzeHits(std::vector<PCaloHit>& hits){
  int nHit = hits.size();
  int nZdcEM = 0, nZdcHad = 0, nZdcLum = 0; 
  int nBad1=0, nBad2=0, nBad=0;
  std::vector<double> encontZdcEM(140, 0.);
  std::vector<double> encontZdcHad(140, 0.);
  double entotZdcEM = 0;
  double entotZdcHad = 0;
 
  enetotEmN = 0;
  enetotHadN = 0.;
  enetotN = 0;  
  enetotEmP = 0;
  enetotHadP = 0;
  enetotP = 0;
  enetot = 0;
  
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

    FillHitValHist(side,section,channel,energy,time);
  
    
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
  
  if (dbe_ && nHit>0) {
    meAllZdcNHit_->Fill(double(nHit));
    meBadZdcDetHit_->Fill(double(nBad1));
    meBadZdcSecHit_->Fill(double(nBad2));
    meBadZdcIdHit_->Fill(double(nBad));
    meZdcNHitEM_->Fill(double(nZdcEM));
    meZdcNHitHad_->Fill(double(nZdcHad));
    meZdcNHitLum_->Fill(double(nZdcLum)); 
    meZdcEnePTot_->Fill(enetotP);
    meZdcEneHadNTot_->Fill(enetotHadN);
    meZdcEneHadPTot_->Fill(enetotHadP);
    meZdcEneEmNTot_->Fill(enetotEmN);
    meZdcEneEmPTot_->Fill(enetotEmP);
    meZdcCorEEmNEHadN_->Fill(enetotEmN,enetotHadN);
    meZdcCorEEmPEHadP_->Fill(enetotEmP,enetotHadP);
    meZdcCorEtotNEtotP_->Fill(enetotN,enetotP);
    meZdcEneTot_->Fill(enetot);
  }
  LogDebug("HcalSimHitStudy") 
  //std::cout
    <<"HcalSimHitStudy::analyzeHits: Had " << nZdcHad 
    << " EM "<< nZdcEM
    << " Bad " << nBad << " All " << nHit;
    //<<std::endl;
}

int ZdcSimHitStudy::FillHitValHist(int side,int section,int channel,double energy,double time){  
  enetot += enetot;
  if(side == -1){
    enetotN += energy;
    if(section == HcalZDCDetId::EM){
      enetotEmN += energy;
      switch(channel){
      case 1 :
	meZdcEneEmN1_->Fill(energy);
	meZdcEneTEmN1_->Fill(energy,time);
	break;
      case 2 :
       meZdcEneEmN2_->Fill(energy);
       meZdcEneTEmN2_->Fill(energy,time);
       	break;
      case 3 :
	meZdcEneEmN3_->Fill(energy);
       meZdcEneTEmN3_->Fill(energy,time);
       	break;
      case 4 :
	meZdcEneEmN4_->Fill(energy);
	meZdcEneTEmN4_->Fill(energy,time);
	break; 
     case 5 :
	meZdcEneEmN4_->Fill(energy);
	meZdcEneTEmN4_->Fill(energy,time);
	break;
      }
    }
    if(section == HcalZDCDetId::HAD){
      enetotHadN += energy;
      switch(channel){
      case 1 :
	meZdcEneHadN1_->Fill(energy);
	meZdcEneTHadN1_->Fill(energy,time);
	break;
      case 2 :
	meZdcEneHadN2_->Fill(energy);
	meZdcEneTHadN2_->Fill(energy,time);
	break;
      case 3 :
	meZdcEneHadN3_->Fill(energy);
	meZdcEneTHadN3_->Fill(energy,time);
	break;
      case 4 :
	meZdcEneHadN4_->Fill(energy);
	meZdcEneTHadN4_->Fill(energy,time);
	break;
      }
    }
  }
  if(side == 1){
    enetotP += energy;
    if(section == HcalZDCDetId::EM){
      enetotEmP += energy;
      switch(channel){
      case 1 :
	meZdcEneEmP1_->Fill(energy);
	meZdcEneTEmP1_->Fill(energy,time);
	break;
      case 2 :
	meZdcEneEmP2_->Fill(energy);
	meZdcEneTEmP2_->Fill(energy,time);
	break;
      case 3 :
	meZdcEneEmP3_->Fill(energy);
	meZdcEneTEmP3_->Fill(energy,time);
	break;
      case 4 :
	meZdcEneEmP4_->Fill(energy);
	meZdcEneTEmP4_->Fill(energy,time);
	break; 
      case 5 :
	meZdcEneEmP4_->Fill(energy);
	meZdcEneTEmP4_->Fill(energy,time);
	break;
      }
    }
    if(section == HcalZDCDetId::HAD){
      enetotHadP += energy;
      switch(channel){
      case 1 :
	meZdcEneHadP1_->Fill(energy);
	meZdcEneTHadP1_->Fill(energy,time);
	break;
      case 2 :
	meZdcEneHadP2_->Fill(energy);
	meZdcEneTHadP2_->Fill(energy,time);
	break;
      case 3 :
	meZdcEneHadP3_->Fill(energy);
	meZdcEneTHadP3_->Fill(energy,time);
	break;
      case 4 :
	meZdcEneHadP4_->Fill(energy);
	meZdcEneTHadP4_->Fill(energy,time);
	break;
      }
    }
  }       
  return 0;
}
  
 














