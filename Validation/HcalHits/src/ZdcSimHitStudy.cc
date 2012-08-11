////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Package:    ZdcSimHitStudy
// Class:      ZdcSimHitStudy
//
/*
 Description: 
              This code has been developed to be a check for the ZDC sim. In 2009, it was found that the ZDC Simulation was unrealistic and needed repair. The aim of this code is to show the user the input and output of a ZDC MinBias simulation.

 Implementation:
      First a MinBias simulation should be run, it could be pythia,hijin,or hydjet. This will output a .root file which should have information about recoGenParticles, hcalunsuppresseddigis, and g4SimHits_ZDCHits. Use this .root file as the input into the cfg.py which is found in the main directory of this package. This output will be another .root file which is meant to be viewed in a TBrowser.

*/
//
// Original Author: Jaime Gomez (U. of Maryland) with SIGNIFICANT assistance of Dr. Jefferey Temple (U. of Maryland) 
// Adapted from: E. Garcia-Solis' (CSU) original code    
//
//         Created:  Summer 2012
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





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
    dbe_->setCurrentFolder("ZDCValidation");
    //Histograms for Hits
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//# Below we are filling the histograms made in the .h file. The syntax is as follows:                                      #
//# plot_code_name = dbe_->TypeofPlot[(1,2,3)-D,(F,I,D)]("Name as it will appear","Title",axis options);                    #
//# They will be stored in the TFile subdirectory set by :    dbe_->setCurrentFolder("FolderIwant")                         #
//# axis options are like (#ofbins,min,max)                                                                                 #
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    if (checkHit_) {
/////////////////////////1///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits");
      meAllZdcNHit_ = dbe_->book1D("ZDC Hits","Number of All Hits in ZDC",100,0.,100.);
      meAllZdcNHit_->setAxisTitle("Total Hits",1);
      meAllZdcNHit_->setAxisTitle("Counts",2);
/////////////////////////2///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/Debug_Helper");
      meBadZdcDetHit_= dbe_->book1D("Hiits with the wrong Det","Hits with wrong Det in ZDC",100,0.,100.);
      meBadZdcDetHit_->setAxisTitle("Wrong Hits",1);
      meBadZdcDetHit_->setAxisTitle("Counts",2);
/////////////////////////3///////////////////////////
      meBadZdcSecHit_= dbe_->book1D("Wrong Section Hits","Hits with wrong Section in ZDC",100,0.,100.);
      meBadZdcSecHit_->setAxisTitle("Hits in wrong section",1);
      meBadZdcSecHit_->setAxisTitle("Counts",2);
/////////////////////////4///////////////////////////      
      meBadZdcIdHit_ = dbe_->book1D("Wrong_ID_Hits","Hits with wrong ID in ZDC",100,0.,100.);
      meBadZdcIdHit_->setAxisTitle("Hits with wrong ID",1);
      meBadZdcIdHit_->setAxisTitle("Counts",2);
/////////////////////////5///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/BasicHitInfo");      
      meZdcNHitEM_   = dbe_->book1D("Hits in EM","Number of Hits in ZDC EM",100,0.,100.);
      meZdcNHitEM_->setAxisTitle("EM Hits",1);
      meZdcNHitEM_->setAxisTitle("Counts",2);
/////////////////////////6///////////////////////////      
      meZdcNHitHad_   = dbe_->book1D("Hits in HAD","Number of Hits in ZDC Had",100,0.,100.);
      meZdcNHitHad_->setAxisTitle("HAD Hits",1);
      meZdcNHitHad_->setAxisTitle("Counts",2);
/////////////////////////7///////////////////////////      
      meZdcNHitLum_   = dbe_->book1D("Hits in LUM","Number of Hits in ZDC Lum",100,0.,100.);
      meZdcNHitLum_->setAxisTitle("LUM Hits",1);
      meZdcNHitLum_->setAxisTitle("Counts",2);
/////////////////////////8///////////////////////////      
      meZdcDetectHit_= dbe_->book1D("Calo Detector ID","Calo Detector ID",50,0.,50.);
      meZdcDetectHit_->setAxisTitle("Detector Hits",1);
      meZdcDetectHit_->setAxisTitle("Counts",2);
      /////////////////////////9///////////////////////////      
      meZdcSideHit_ = dbe_->book1D("ZDC Side","Side in ZDC",4,-2,2.);
      meZdcSideHit_->setAxisTitle("ZDC Side",1);
      meZdcSideHit_->setAxisTitle("Counts",2);
/////////////////////////10///////////////////////////      
      meZdcSectionHit_   = dbe_->book1D("ZDC Section","Section in ZDC",4,0.,4.);
      meZdcSectionHit_->setAxisTitle("ZDC Section",1);
      meZdcSectionHit_->setAxisTitle("Counts",2);
/////////////////////////11///////////////////////////      
      meZdcChannelHit_   = dbe_->book1D("ZDC Channel","Channel in ZDC",10,0.,10.);
      meZdcChannelHit_->setAxisTitle("ZDC Channel",1);
      meZdcChannelHit_->setAxisTitle("Counts",2);
/////////////////////////12///////////////////////////      
      meZdcEnergyHit_= dbe_->book1D("Hit Energy","Hits Energy",4000,0.,8000.);
      meZdcEnergyHit_->setAxisTitle("Counts",2);
      meZdcEnergyHit_->setAxisTitle("Energy (GeV)",1);
/////////////////////////13///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits");      
      meZdcHadEnergyHit_= dbe_->book1D("Hit Energy HAD","Hits Energy in Had Section",4000,0.,8000.);
      meZdcHadEnergyHit_->setAxisTitle("Counts",2);
      meZdcHadEnergyHit_->setAxisTitle("Energy (GeV)",1);
/////////////////////////14///////////////////////////      
      meZdcEMEnergyHit_ = dbe_->book1D("Hit Energy EM","Hits Energy in EM Section",4000,0.,8000.);
      meZdcEMEnergyHit_->setAxisTitle("Counts",2);
      meZdcEMEnergyHit_->setAxisTitle("Energy (GeV)",1);
/////////////////////////15///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/BasicHitInfo");      
      meZdcTimeHit_  = dbe_->book1D("Time in ZDC","Time in ZDC",300,0.,600.);
      meZdcTimeHit_->setAxisTitle("Time (ns)",1);
      meZdcTimeHit_->setAxisTitle("Counts",2);
/////////////////////////16///////////////////////////      
      meZdcTimeWHit_ = dbe_->book1D("Energy Weighted Time in ZDC","Time in ZDC (E wtd)", 300,0.,600.);
      meZdcTimeWHit_->setAxisTitle("Time (ns)",1);
      meZdcTimeWHit_->setAxisTitle("Counts",2);
/////////////////////////17///////////////////////////      
      meZdc10Ene_ = dbe_->book1D("ZDC Log(E)","Log10Energy in ZDC", 140, -20., 20. );
      meZdc10Ene_->setAxisTitle("Log(E) (GeV)",1);
      meZdc10Ene_->setAxisTitle("Counts",2);
/////////////////////////18///////////////////////////      
      meZdcHadL10EneP_ = dbe_->bookProfile("Log(EHAD) vs Contribution","Log10Energy in Had ZDC vs Hit contribution", 140, -1., 20., 100, 0., 1. );
      meZdcHadL10EneP_->setAxisTitle("Log(EHAD) (GeV)",1);
      meZdcHadL10EneP_->setAxisTitle("Counts",2);
/////////////////////////19///////////////////////////      
      meZdcEML10EneP_ = dbe_->bookProfile("Log(EEM) vs Contribution","Log10Energy in EM ZDC vs Hit contribution", 140, -1., 20., 100, 0., 1. );
      meZdcEML10EneP_->setAxisTitle("Log(EEM) (GeV)",1);
      meZdcEML10EneP_->setAxisTitle("Counts",2);
/////////////////////////20///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits");      
      meZdcEHadCh_ = dbe_->book2D("ZDC EHAD vs Channel","ZDC Had Section Energy vs Channel", 4000, 0., 8000., 6, 0., 6. );
      meZdcEHadCh_->setAxisTitle("Hadronic Channel Number",2);
      meZdcEHadCh_->setAxisTitle("Energy (GeV)",1);
/////////////////////////21///////////////////////////      
      meZdcEEMCh_ = dbe_->book2D("ZDC EEM vs Channel","ZDC EM Section Energy vs Channel", 4000, 0., 8000., 6, 0., 6. );
      meZdcEEMCh_->setAxisTitle("EM Channel Number",2);
      meZdcEEMCh_->setAxisTitle("Energy (GeV)",1);
/////////////////////////22///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/BasicHitInfo");
      meZdcETime_ = dbe_->book2D("E vs T","Hits ZDC Energy vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcETime_->setAxisTitle("Energy (GeV)",1);
      meZdcETime_->setAxisTitle("Time (ns)",2);
/////////////////////////1///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/Individual_Channels/NZDC");
      meZdcEneEmN1_  = dbe_->book1D("NZDC EM1 Energy","Energy EM module N1",4000,0.,8000.);
      meZdcEneEmN1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmN1_->setAxisTitle("Counts",2);
/////////////////////////2///////////////////////////
      meZdcEneEmN2_  = dbe_->book1D("NZDC EM2 Energy","Energy EM module N2",4000,0.,8000.);
      meZdcEneEmN2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmN2_->setAxisTitle("Counts",2);
/////////////////////////3///////////////////////////
      meZdcEneEmN3_  = dbe_->book1D("NZDC EM3 Energy","Energy EM module N3",4000,0.,8000.);
      meZdcEneEmN3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmN3_->setAxisTitle("Counts",2);
/////////////////////////4///////////////////////////
      meZdcEneEmN4_  = dbe_->book1D("NZDC EM4 Energy","Energy EM module N4",4000,0.,8000.);
      meZdcEneEmN4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmN4_->setAxisTitle("Counts",2);
/////////////////////////5///////////////////////////
      meZdcEneEmN5_  = dbe_->book1D("NZDC EM5 Energy","Energy EM module N5",4000,0.,8000.);
      meZdcEneEmN5_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmN5_->setAxisTitle("Counts",2);
/////////////////////////6///////////////////////////
      meZdcEneHadN1_ = dbe_->book1D("NZDC HAD1 Energy","Energy HAD module N1",4000,0.,8000.);
      meZdcEneHadN1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadN1_->setAxisTitle("Counts",2);
/////////////////////////7///////////////////////////
      meZdcEneHadN2_ = dbe_->book1D("NZDC HAD2 Energy","Energy HAD module N2",4000,0.,8000.);
      meZdcEneHadN2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadN2_->setAxisTitle("Counts",2);
/////////////////////////8///////////////////////////
      meZdcEneHadN3_ = dbe_->book1D("NZDC HAD3 Energy","Energy HAD module N3",4000,0.,8000.);
      meZdcEneHadN3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadN3_->setAxisTitle("Counts",2);
/////////////////////////9///////////////////////////
      meZdcEneHadN4_ = dbe_->book1D("NZDC HAD4 Energy","Energy HAD module N4",4000,0.,8000.);
      meZdcEneHadN4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadN4_->setAxisTitle("Counts",2);
/////////////////////////11///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/Individual_ChannelvsTime/NZDC");
      meZdcEneTEmN1_ = dbe_->book2D("NZDC EM1 Energy vs Time","Energy EM mod N1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmN1_->setAxisTitle("Time (ns)",2);
/////////////////////////12///////////////////////////
      meZdcEneTEmN2_ = dbe_->book2D("NZDC EM2 Energy vs Time","Energy EM mod N2 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmN2_->setAxisTitle("Time (ns)",2); 
/////////////////////////13///////////////////////////
      meZdcEneTEmN3_ = dbe_->book2D("NZDC EM3 Energy vs Time","Energy EM mod N3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmN3_->setAxisTitle("Time (ns)",2);
/////////////////////////14///////////////////////////
      meZdcEneTEmN4_ = dbe_->book2D("NZDC EM4 Energy vs Time","Energy EM mod N4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmN4_->setAxisTitle("Time (ns)",2);
/////////////////////////15///////////////////////////
      meZdcEneTEmN5_ = dbe_->book2D("NZDC EM5 Energy vs Time","Energy EM mod N5 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmN5_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmN5_->setAxisTitle("Time (ns)",2);
/////////////////////////16///////////////////////////
      meZdcEneTHadN1_ = dbe_->book2D("NZDC HAD1 Energy vs Time","Energy HAD mod N1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadN1_->setAxisTitle("Time (ns)",2);
/////////////////////////17///////////////////////////
      meZdcEneTHadN2_ = dbe_->book2D("NZDC HAD2 Energy vs Time","Energy HAD mod N2 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadN2_->setAxisTitle("Time (ns)",2); 
/////////////////////////18///////////////////////////
      meZdcEneTHadN3_ = dbe_->book2D("NZDC HAD3 Energy vs Time","Energy HAD mod N3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadN3_->setAxisTitle("Time (ns)",2);
/////////////////////////19///////////////////////////
      meZdcEneTHadN4_ = dbe_->book2D("NZDC HAD4 Energy vs Time","Energy HAD mod N4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadN4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadN4_->setAxisTitle("Time (ns)",2);
/////////////////////////20///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/ENERGY_SUMS/NZDC");
      meZdcEneHadNTot_ = dbe_->book1D("NZDC EHAD","Total N-ZDC HAD Energy",4000,0.,8000.);
      meZdcEneHadNTot_->setAxisTitle("Counts",2);
      meZdcEneHadNTot_->setAxisTitle("Energy (GeV)",1);
/////////////////////////21///////////////////////////
      meZdcEneEmNTot_  = dbe_->book1D("NZDC EEM","Total N-ZDC EM Energy",4000,0.,8000.);
      meZdcEneEmNTot_->setAxisTitle("Counts",2);
      meZdcEneEmNTot_->setAxisTitle("Energy (GeV)",1);
/////////////////////////22///////////////////////////
      meZdcEneNTot_    = dbe_->book1D("NZDC ETOT","Total N-ZDC Energy ",4000,0.,8000.);
      meZdcEneNTot_->setAxisTitle("Counts",2);
      meZdcEneNTot_->setAxisTitle("Energy (GeV)",1);
/////////////////////////23///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/Individual_Channels/PZDC");
      meZdcEneEmP1_  = dbe_->book1D("PZDC EM1 Energy","Energy EM module P1",4000,0.,8000.);
      meZdcEneEmP1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmP1_->setAxisTitle("Counts",2);
/////////////////////////24///////////////////////////
      meZdcEneEmP2_  = dbe_->book1D("PZDC EM2 Energy","Energy EM module P2",4000,0.,8000.);
      meZdcEneEmP2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmP2_->setAxisTitle("Counts",2);
/////////////////////////25///////////////////////////
      meZdcEneEmP3_  = dbe_->book1D("PZDC EM3 Energy","Energy EM module P3",4000,0.,8000.);
      meZdcEneEmP3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmP3_->setAxisTitle("Counts",2);
/////////////////////////26///////////////////////////
      meZdcEneEmP4_  = dbe_->book1D("PZDC EM4 Energy","Energy EM module P4",4000,0.,8000.);
      meZdcEneEmP4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmP4_->setAxisTitle("Counts",2);
/////////////////////////27///////////////////////////
      meZdcEneEmP5_  = dbe_->book1D("PZDC EM5 Energy","Energy EM module P5",4000,0.,8000.);
      meZdcEneEmP5_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmP5_->setAxisTitle("Counts",2);
/////////////////////////28///////////////////////////
      meZdcEneHadP1_ = dbe_->book1D("PZDC HAD1 Energy","Energy HAD module P1",4000,0.,8000.);
      meZdcEneHadP1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadP1_->setAxisTitle("Counts",2);
/////////////////////////29///////////////////////////
      meZdcEneHadP2_ = dbe_->book1D("PZDC HAD2 Energy","Energy HAD module P2",4000,0.,8000.);
      meZdcEneHadP2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadP2_->setAxisTitle("Counts",2);
/////////////////////////30///////////////////////////
      meZdcEneHadP3_ = dbe_->book1D("PZDC HAD3 Energy","Energy HAD module P3",4000,0.,8000.);
      meZdcEneHadP3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadP3_->setAxisTitle("Counts",2);
/////////////////////////31///////////////////////////
      meZdcEneHadP4_ = dbe_->book1D("PZDC HAD4 Energy","Energy HAD module P4",4000,0.,8000.);
      meZdcEneHadP4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadP4_->setAxisTitle("Counts",2);
/////////////////////////32///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/Excess_Info/Individual_ChannelvsTime/PZDC");
      meZdcEneTEmP1_ = dbe_->book2D("PZDC EM1 Energy vs Time","Energy EM mod P1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmP1_->setAxisTitle("Time (ns)",2);
/////////////////////////33///////////////////////////
      meZdcEneTEmP2_ = dbe_->book2D("PZDC EM2 Energy vs Time","Energy EM mod P2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTEmP2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmP2_->setAxisTitle("Time (ns)",2);
/////////////////////////34///////////////////////////
      meZdcEneTEmP3_ = dbe_->book2D("PZDC EM3 Energy vs Time","Energy EM mod P3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmP3_->setAxisTitle("Time (ns)",2);
/////////////////////////35///////////////////////////
      meZdcEneTEmP4_ = dbe_->book2D("PZDC EM4 Energy vs Time","Energy EM mod P4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmP4_->setAxisTitle("Time (ns)",2);
/////////////////////////36///////////////////////////
      meZdcEneTEmP5_ = dbe_->book2D("PZDC EM5 Energy vs Time","Energy EM mod P5 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTEmP5_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTEmP5_->setAxisTitle("Time (ns)",2);
/////////////////////////37///////////////////////////
      meZdcEneTHadP1_ = dbe_->book2D("PZDC HAD1 Energy vs Time","Energy HAD mod P1 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP1_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadP1_->setAxisTitle("Time (ns)",2);
/////////////////////////38///////////////////////////
      meZdcEneTHadP2_ = dbe_->book2D("PZDC HAD2 Energy vs Time","Energy HAD mod P2 vs Time", 4000, 0., 8000., 300, 0., 600. ); 
      meZdcEneTHadP2_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadP2_->setAxisTitle("Time (ns)",2);
/////////////////////////39///////////////////////////
      meZdcEneTHadP3_ = dbe_->book2D("PZDC HAD3 Energy vs Time","Energy HAD mod P3 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP3_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadP3_->setAxisTitle("Time (ns)",2);
/////////////////////////40///////////////////////////
      meZdcEneTHadP4_ = dbe_->book2D("PZDC HAD4 Energy vs Time","Energy HAD mod P4 vs Time", 4000, 0., 8000., 300, 0., 600. );
      meZdcEneTHadP4_->setAxisTitle("Energy (GeV)",1);
      meZdcEneTHadP4_->setAxisTitle("Time (ns)",2);
/////////////////////////41/////////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/ENERGY_SUMS/PZDC");
      meZdcEneHadPTot_ = dbe_->book1D("PZDC EHAD","Total P-ZDC HAD Energy",4000,0.,8000.);
      meZdcEneHadPTot_->setAxisTitle("Energy (GeV)",1);
      meZdcEneHadPTot_->setAxisTitle("Counts",2);
/////////////////////////42///////////////////////////
      meZdcEneEmPTot_  = dbe_->book1D("PZDC EEM","Total P-ZDC EM Energy",4000,0.,8000.);
      meZdcEneEmPTot_->setAxisTitle("Energy (GeV)",1);
      meZdcEneEmPTot_->setAxisTitle("Counts",2);
/////////////////////////43///////////////////////////
      meZdcEnePTot_    = dbe_->book1D("PZDC ETOT","Total P-ZDC Energy",4000,0.,8000.);
      meZdcEnePTot_->setAxisTitle("Energy (GeV)",1);
      meZdcEnePTot_->setAxisTitle("Counts",2);
/////////////////////////47///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/ENERGY_SUMS/NZDC");
      meZdcCorEEmNEHadN_= dbe_->book2D("NZDC EMvHAD","N-ZDC Energy EM vs HAD", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcCorEEmNEHadN_->setAxisTitle("EM Energy (GeV)",1);
      meZdcCorEEmNEHadN_->setAxisTitle("HAD Energy (GeV)",2);
/////////////////////////44///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/ENERGY_SUMS/PZDC");
      meZdcCorEEmPEHadP_= dbe_->book2D("PZDC EMvHAD","P-ZDC Energy EM vs HAD", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcCorEEmPEHadP_->setAxisTitle("EM Energy (GeV)",1);
      meZdcCorEEmPEHadP_->setAxisTitle("HAD Energy (GeV)",2);
/////////////////////////45///////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZdcSimHits/ENERGY_SUMS");
      meZdcCorEtotNEtotP_ = dbe_->book2D("PZDC vs NZDC","Energy N-ZDC vs P-ZDC", 4000, 0., 8000.,4000, 0., 8000.);
      meZdcCorEtotNEtotP_->setAxisTitle("N-ZDC Total Energy (GeV)",1);
      meZdcCorEtotNEtotP_->setAxisTitle("P-ZDC Total Energy (GeV)",2);
/////////////////////////46///////////////////////////
      meZdcEneTot_ = dbe_->book1D("ETOT ZDCs","Total Energy ZDCs",4000,0.,8000.);
      meZdcEneTot_->setAxisTitle("Counts",2);
      meZdcEneTot_->setAxisTitle("Energy (GeV)",1);
///////////////////////////////////////////////////////////




//////////////////New Plots////////////////////////////////

////////////////////////// 1-D TotalfC per Side ///////////////////////

///////////////////////////////// 47 ////////////////////////////////////////////
      dbe_->setCurrentFolder("ZDCValidation/ZDC_Digis/1D_fC");
      meZdcfCPHAD = dbe_->book1D("PHAD_TotalfC","PZDC_HAD_TotalfC",1000,-50,950);
      meZdcfCPHAD->setAxisTitle("Counts",2);
      meZdcfCPHAD->setAxisTitle("fC",1);
/////////////////////////////////48////////////////////////////     
      meZdcfCPTOT = dbe_->book1D("PZDC_TotalfC","PZDC_TotalfC",1000,-50,950);
      meZdcfCPTOT->setAxisTitle("Counts",2);
      meZdcfCPTOT->setAxisTitle("fC",1);
/////////////////////////////////49/////////////////////////////////
      meZdcfCNHAD = dbe_->book1D("NHAD_TotalfC","NZDC_HAD_TotalfC",1000,-50,950);
      meZdcfCNHAD->setAxisTitle("Counts",2);
      meZdcfCNHAD->setAxisTitle("fC",1);
////////////////////////////////50/////////////////////////////////////////
      meZdcfCNTOT = dbe_->book1D("NZDC_TotalfC","NZDC_TotalfC",1000,-50,950);
      meZdcfCNTOT->setAxisTitle("Counts",2);
      meZdcfCNTOT->setAxisTitle("fC",1);
/////////////////////////////////////////////////////////////////////////

//////////////////////// 1-D fC vs TS ///////////////////////////////////////
     dbe_->setCurrentFolder("ZDCValidation/ZDC_Digis/fCvsTS/PZDC");
     
/////////////////////////////////51/////////////////////////////////////////
     meZdcPEM1fCvsTS = dbe_->book1D("PEM1_fCvsTS","P-EM1_AveragefC_vsTS",10,0,9);
     meZdcPEM1fCvsTS->setAxisTitle("fC",2);
     meZdcPEM1fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////52/////////////////////////////////////////
     meZdcPEM2fCvsTS = dbe_->book1D("PEM2_fCvsTS","P-EM2_AveragefC_vsTS",10,0,9);
     meZdcPEM2fCvsTS->setAxisTitle("fC",2);
     meZdcPEM2fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////53/////////////////////////////////////////
     meZdcPEM3fCvsTS = dbe_->book1D("PEM3_fCvsTS","P-EM3_AveragefC_vsTS",10,0,9);
     meZdcPEM3fCvsTS->setAxisTitle("fC",2);
     meZdcPEM3fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////54/////////////////////////////////////////
     meZdcPEM4fCvsTS = dbe_->book1D("PEM4_fCvsTS","P-EM4_AveragefC_vsTS",10,0,9);
     meZdcPEM4fCvsTS->setAxisTitle("fC",2);
     meZdcPEM4fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////55/////////////////////////////////////////
     meZdcPEM5fCvsTS = dbe_->book1D("PEM5_fCvsTS","P-EM5_AveragefC_vsTS",10,0,9);
     meZdcPEM5fCvsTS->setAxisTitle("fC",2);
     meZdcPEM5fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////56/////////////////////////////////////////
     meZdcPHAD1fCvsTS = dbe_->book1D("PHAD1_fCvsTS","P-HAD1_AveragefC_vsTS",10,0,9);
     meZdcPHAD1fCvsTS->setAxisTitle("fC",2);
     meZdcPHAD1fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////57/////////////////////////////////////////
     meZdcPHAD2fCvsTS = dbe_->book1D("PHAD2_fCvsTS","P-HAD2_AveragefC_vsTS",10,0,9);
     meZdcPHAD2fCvsTS->setAxisTitle("fC",2);
     meZdcPHAD2fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////58/////////////////////////////////////////
     meZdcPHAD3fCvsTS = dbe_->book1D("PHAD3_fCvsTS","P-HAD3_AveragefC_vsTS",10,0,9);
     meZdcPHAD3fCvsTS->setAxisTitle("fC",2);
     meZdcPHAD3fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////59/////////////////////////////////////////
     meZdcPHAD4fCvsTS = dbe_->book1D("PHAD4_fCvsTS","P-HAD4_AveragefC_vsTS",10,0,9);
     meZdcPHAD4fCvsTS->setAxisTitle("fC",2);
     meZdcPHAD4fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
     dbe_->setCurrentFolder("ZDCValidation/ZDC_Digis/fCvsTS/NZDC");
     
/////////////////////////////////60/////////////////////////////////////////
     meZdcNEM1fCvsTS = dbe_->book1D("NEM1_fCvsTS","N-EM1_AveragefC_vsTS",10,0,9);
     meZdcNEM1fCvsTS->setAxisTitle("fC",2);
     meZdcNEM1fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////61/////////////////////////////////////////
     meZdcNEM2fCvsTS = dbe_->book1D("NEM2_fCvsTS","N-EM2_AveragefC_vsTS",10,0,9);
     meZdcNEM2fCvsTS->setAxisTitle("fC",2);
     meZdcNEM2fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////62/////////////////////////////////////////
     meZdcNEM3fCvsTS = dbe_->book1D("NEM3_fCvsTS","N-EM3_AveragefC_vsTS",10,0,9);
     meZdcNEM3fCvsTS->setAxisTitle("fC",2);
     meZdcNEM3fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////63/////////////////////////////////////////
     meZdcNEM4fCvsTS = dbe_->book1D("NEM4_fCvsTS","N-EM4_AveragefC_vsTS",10,0,9);
     meZdcNEM4fCvsTS->setAxisTitle("fC",2);
     meZdcNEM4fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////64/////////////////////////////////////////
     meZdcNEM5fCvsTS = dbe_->book1D("NEM5_fCvsTS","N-EM5_AveragefC_vsTS",10,0,9);
     meZdcNEM5fCvsTS->setAxisTitle("fC",2);
     meZdcNEM5fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////65/////////////////////////////////////////
     meZdcNHAD1fCvsTS = dbe_->book1D("NHAD1_fCvsTS","N-HAD1_AveragefC_vsTS",10,0,9);
     meZdcNHAD1fCvsTS->setAxisTitle("fC",2);
     meZdcNHAD1fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////66/////////////////////////////////////////
     meZdcNHAD2fCvsTS = dbe_->book1D("NHAD2_fCvsTS","N-HAD2_AveragefC_vsTS",10,0,9);
     meZdcNHAD2fCvsTS->setAxisTitle("fC",2);
     meZdcNHAD2fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////67/////////////////////////////////////////
     meZdcNHAD3fCvsTS = dbe_->book1D("NHAD3_fCvsTS","N-HAD3_AveragefC_vsTS",10,0,9);
     meZdcNHAD3fCvsTS->setAxisTitle("fC",2);
     meZdcNHAD3fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////68/////////////////////////////////////////
     meZdcNHAD4fCvsTS = dbe_->book1D("NHAD4_fCvsTS","N-HAD4_AveragefC_vsTS",10,0,9);
     meZdcNHAD4fCvsTS->setAxisTitle("fC",2);
     meZdcNHAD4fCvsTS->setAxisTitle("TS",1);
////////////////////////////////////////////////////////////////////////////

//////////////////// 2-D EMvHAD plots/////////////////////////////////////////
    dbe_->setCurrentFolder("ZDCValidation/ZDC_Digis/2D_EMvHAD");
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////69//////////////////////////////////////////
    meZdcfCPEMvHAD = dbe_->book2D("PEMvPHAD","PZDC_EMvHAD",1000,-25,1000,1000,-25,1000);
    meZdcfCPEMvHAD->setAxisTitle("SumEM_fC",2);
    meZdcfCPEMvHAD->setAxisTitle("SumHAD_fC",1);
    meZdcfCPEMvHAD->getTH2F()->SetOption("colz");
////////////////////////////////70///////////////////////////////////////////
    meZdcfCNEMvHAD = dbe_->book2D("NEMvNHAD","NZDC_EMvHAD",1000,-25,1000,1000,-25,1000);
    meZdcfCNEMvHAD->setAxisTitle("SumEM_fC",2);
    meZdcfCNEMvHAD->setAxisTitle("SumHAD_fC",1);
    meZdcfCNEMvHAD->getTH2F()->SetOption("colz");
///////////////////////////////////////////////////////////////////////////////


////////////////////// GenParticle Plots///////////////////////////////////////
   dbe_->setCurrentFolder("ZDCValidation/GenParticles/Forward");
//////////////////////////////////////////////////////////////////////////////
///////////////////////////////71/////////////////////////////////////////////
    genpart_Pi0F = dbe_->book2D("Pi0_Forward","Forward Generated Pi0s",200,4.5,7,100,-3.15,3.15);
   //   genpart_Pi0F = dbe_->bookProfile2D("blah","balh",200,4.5,7,100,-3.15,3.15,2000,0,3000,"s");
   genpart_Pi0F->setAxisTitle("Eta",1);
   genpart_Pi0F->setAxisTitle("Phi (radians)",2);
   genpart_Pi0F->setAxisTitle("Energy (GeV)",3);
   genpart_Pi0F->getTH2F()->SetOption("lego2z,prof");
   genpart_Pi0F->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_Pi0F->getTH2F()->SetTitleOffset(1.4,"y");
/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////72/////////////////////////////////////////////
   genpart_NeutF = dbe_->book2D("Neutron_Forward","Forward Generated Neutrons",200,4.5,7,100,-3.15,3.15);
   genpart_NeutF->setAxisTitle("Eta",1);
   genpart_NeutF->setAxisTitle("Phi (radians)",2);
   genpart_NeutF->setAxisTitle("Energy (GeV)",3);
   genpart_NeutF->getTH2F()->SetOption("lego2z,prof");
   genpart_NeutF->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_NeutF->getTH2F()->SetTitleOffset(1.4,"y");
/////////////////////////////////////////////////////////////////////////////////
   ///////////////////////////////73/////////////////////////////////////////////
   genpart_GammaF = dbe_->book2D("Gamma_Forward","Forward Generated Gammas",200,4.5,7,100,-3.15,3.15);
   genpart_GammaF->setAxisTitle("Eta",1);
   genpart_GammaF->setAxisTitle("Phi (radians)",2);
   genpart_GammaF->setAxisTitle("Energy (GeV)",3);
   genpart_GammaF->getTH2F()->SetOption("lego2z,prof");
   genpart_GammaF->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_GammaF->getTH2F()->SetTitleOffset(1.4,"y");
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
genpart_Pi0F_energydist = dbe_->book1D("Pi0_Forward_EDistribution","Gen-Level Forward Pi0 Energy",1500,0,1500);
   genpart_Pi0F_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_Pi0F_energydist->setAxisTitle("Counts",2);
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
   genpart_NeutF_energydist = dbe_->book1D("N_Forward_EDistribution","Gen-Level Forward Neutron Energy",1500,0,1500);
   genpart_NeutF_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_NeutF_energydist->setAxisTitle("Counts",2);
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
   genpart_GammaF_energydist = dbe_->book1D("Gamma_Forward_EDistribution","Gen-Level Fowarad Gamma Energy",1500,0,1500);
   genpart_GammaF_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_GammaF_energydist->setAxisTitle("Counts",2);
/////////////////////////////////////////////////////////////////////////////////
  dbe_->setCurrentFolder("ZDCValidation/GenParticles/Backward");
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////74/////////////////////////////////////////////
   genpart_Pi0B = dbe_->book2D("Pi0_Backward","Backward Generated Pi0s",200,-7,-4.5,100,-3.15,3.15);
   genpart_Pi0B->setAxisTitle("Eta",1);
   genpart_Pi0B->setAxisTitle("Phi (radians)",2);
   genpart_Pi0B->setAxisTitle("Energy (GeV)",3);
   genpart_Pi0B->getTH2F()->SetOption("lego2z,prof");
   genpart_Pi0B->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_Pi0B->getTH2F()->SetTitleOffset(1.4,"y");
/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////75/////////////////////////////////////////////
   genpart_NeutB = dbe_->book2D("Neutron_Backward","Backward Generated Neutrons",200,-7,-4.5,100,-3.15,3.15);
   genpart_NeutB->setAxisTitle("Eta",1);
   genpart_NeutB->setAxisTitle("Phi (radians)",2);
   genpart_NeutB->setAxisTitle("Energy (GeV)",3);
   genpart_NeutB->getTH2F()->SetOption("lego2z,prof");
   genpart_NeutB->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_NeutB->getTH2F()->SetTitleOffset(1.4,"y");
/////////////////////////////////////////////////////////////////////////////////
   ///////////////////////////////76/////////////////////////////////////////////
   genpart_GammaB = dbe_->book2D("Gamma_Backward","Backward Generated Gammas",200,-7,-4.5,100,-3.15,3.15);
   genpart_GammaB->setAxisTitle("Eta",1);
   genpart_GammaB->setAxisTitle("Phi (radians)",2);
   genpart_GammaB->setAxisTitle("Energy (GeV)",3);
   genpart_GammaB->getTH2F()->SetOption("lego2z,prof");
   genpart_GammaB->getTH2F()->SetTitleOffset(1.4,"x");
   genpart_GammaB->getTH2F()->SetTitleOffset(1.4,"y");
/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////GEN Particle Energy Distributions/////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////
   genpart_Pi0B_energydist = dbe_->book1D("Pi0_Backward_EDistribution","Gen-Level Backward Pi0 Energy",1500,0,1500);
   genpart_Pi0B_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_Pi0B_energydist->setAxisTitle("Counts",2);
   //////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////
   genpart_NeutB_energydist = dbe_->book1D("N_Backward_EDistribution","Gen-Level Foward Neutron Energy",1500,0,1500);
   genpart_NeutB_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_NeutB_energydist->setAxisTitle("Counts",2);
   ///////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   genpart_GammaB_energydist = dbe_->book1D("Gamma_Backward_EDistribution","Gen-Level Backward Gamma Energy",1500,0,1500);
   genpart_GammaB_energydist->setAxisTitle("Energy (GeV)",1);
   genpart_GammaB_energydist->setAxisTitle("Counts",2);
   /////////////////////////////////////////////////////////////////////////////////////////





    }
  }
}

void ZdcSimHitStudy::endJob() {
  if (dbe_ && outFile_.size() > 0) dbe_->save(outFile_);
}

//void ZdcSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {
void ZdcSimHitStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
//////////NEW STUFF//////////////////////

   using namespace edm;
   bool gotZDCDigis=true;
   bool gotGenParticles=true;
   

   Handle<ZDCDigiCollection> zdchandle; 
   if (!(iEvent.getByLabel("simHcalUnsuppressedDigis",zdchandle)))
   {
    gotZDCDigis=false; //this is a boolean set up to check if there are ZDCDigis in the input root file
   }
   if (!(zdchandle.isValid()))
   {
   gotZDCDigis=false; //if it is not there, leave it false
   }
   

    
   Handle<reco::GenParticleCollection> genhandle;
   
   if (!(iEvent.getByLabel("genParticles",genhandle)))
   {
    gotGenParticles=false; //this is the same kind of boolean except for the genparticles collection
   }
   if (!(genhandle.isValid()))
   {
    gotGenParticles=false;
   }
   

   //Handle<edm::PCaloHitContainer> zdcsimhandle;
   //iEvent.getByLabel("g4SimHits_ZDCHITS",zdcsimhandle);
 
   double totalPHADCharge=0;
   double totalNHADCharge=0;
   double totalPEMCharge=0;
   double totalNEMCharge=0;
   double totalPCharge=0;
   double totalNCharge=0;
   

   //////////////////////////////////////////////////DIGIS///////////////////////////////////
      if (gotZDCDigis==true){
    for (ZDCDigiCollection::const_iterator zdc = zdchandle->begin();
	zdc!=zdchandle->end();
	++zdc)
     {
       const ZDCDataFrame digi = (const ZDCDataFrame)(*zdc);
       //std::cout <<"CHANNEL = "<<zdc->id().channel()<<std::endl;
       


       /////////////////////////////HAD SECTIONS///////////////

       if (digi.id().section()==2){  // require HAD
       if (digi.id().zside()==1)
           { // require POS
         for (int i=0;i<digi.size();++i) // loop over all 10 TS because each digi has 10 entries
	 {
	   if (digi.id().channel()==1){ //here i specify PHAD1
	     meZdcPHAD1fCvsTS->Fill(i,digi.sample(i).nominal_fC()); //filling the plot name with the nominal fC value for each TS
	     if (i==0) meZdcPHAD1fCvsTS->Fill(-1,1);  // on first iteration of loop, increment underflow bin
	   }//NEW AVERAGE Thingy
           if (digi.id().channel()==2){
	     meZdcPHAD2fCvsTS->Fill(i,digi.sample(i).nominal_fC());
             if (i==0) meZdcPHAD2fCvsTS->Fill(-1,1);
	                              }
           if (digi.id().channel()==3){
             meZdcPHAD3fCvsTS->Fill(i,digi.sample(i).nominal_fC());
	     if (i==0) meZdcPHAD3fCvsTS->Fill(-1,1);
	                               }
           if (digi.id().channel()==4){
             meZdcPHAD4fCvsTS->Fill(i,digi.sample(i).nominal_fC());
             if (i==0) meZdcPHAD4fCvsTS->Fill(-1,1);
	                              }
            totalPHADCharge+=digi.sample(i).nominal_fC(); //here i am looking for the total charge in PHAD so i sum over every TS and channel and add it up
	 } // loop over all (10) TS for the given digi
	   }
       else {
	 for (int i=0; i<digi.size();++i)
	   {
	     if (digi.id().channel()==1){
	       meZdcNHAD1fCvsTS->Fill(i,digi.sample(i).nominal_fC());
	       if (i==0) meZdcNHAD1fCvsTS->Fill(-1,1);
	                                }
             if (digi.id().channel()==2){
               meZdcNHAD2fCvsTS->Fill(i,digi.sample(i).nominal_fC());
	       if (i==0) meZdcNHAD2fCvsTS->Fill(-1,1);
	                                 }
             if (digi.id().channel()==3){
               meZdcNHAD3fCvsTS->Fill(i,digi.sample(i).nominal_fC());
               if (i==0) meZdcNHAD3fCvsTS->Fill(-1,1);
	                                }
             if (digi.id().channel()==4){
               meZdcNHAD4fCvsTS->Fill(i,digi.sample(i).nominal_fC());
               if (i==0) meZdcNHAD4fCvsTS->Fill(-1,1);
	                                 }
            totalNHADCharge+=digi.sample(i).nominal_fC();
           }
       }
       }
       ///////////////////////////////EM SECTIONS////////////////////////////
       if (digi.id().section()==1){//require EM....here i do the smae thing that i did above but now for P/N EM sections
	 if (digi.id().zside()==1)
	   {//require pos
             for (int i=0;i<digi.size();++i)
	       {
		 if (digi.id().channel()==1){
                   meZdcPEM1fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		   if (i==0) meZdcPEM1fCvsTS->Fill(-1,1);
		                             }
		 if (digi.id().channel()==2){
                   meZdcPEM2fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		   if (i==0) meZdcPEM2fCvsTS->Fill(-1,1);
		                             }
		 if (digi.id().channel()==3){
                   meZdcPEM3fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		   if (i==0) meZdcPEM3fCvsTS->Fill(-1,1);
		                             }
		 if (digi.id().channel()==4){
                   meZdcPEM4fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		   if (i==0) meZdcPEM4fCvsTS->Fill(-1,1);
		                            }
		 if (digi.id().channel()==5){
                   meZdcPEM5fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		   if (i==0) meZdcPEM5fCvsTS->Fill(-1,1);
		                             }
                totalPEMCharge+=digi.sample(i).nominal_fC();
               }
	   }
	 else {
           for (int i=0;i<digi.size();++i)
	     {
	       if (digi.id().channel()==1){
		 meZdcNEM1fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		 if (i==0) meZdcNEM1fCvsTS->Fill(-1,1);
	                                  }
	       if (digi.id().channel()==2){
                 meZdcNEM2fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		 if (i==0) meZdcNEM2fCvsTS->Fill(-1,1);
	                                  }
	       if (digi.id().channel()==3){
                 meZdcNEM3fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		 if (i==0) meZdcNEM3fCvsTS->Fill(-1,1);
	                                  }
               if (digi.id().channel()==4){
                 meZdcNEM4fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		 if (i==0) meZdcNEM4fCvsTS->Fill(-1,1);
	                                  }
	       if (digi.id().channel()==5){
                 meZdcNEM5fCvsTS->Fill(i,digi.sample(i).nominal_fC());
		 if (i==0) meZdcNEM5fCvsTS->Fill(-1,1);
	                                  }
	       totalNEMCharge+=digi.sample(i).nominal_fC();
	     }
	 }
       }     

       totalPCharge=totalPHADCharge+totalPEMCharge;
       totalNCharge=totalNHADCharge+totalNEMCharge;

       /*       std::cout <<"CHANNEL = "<<digi.id().channel()<<std::endl;
       for (int i=0;i<digi.size();++i)
	 std::cout <<"SAMPLE = "<<i<<"  ADC = "<<digi.sample(i).adc()<<" fC =  "<<digi.sample(i).nominal_fC()<<std::endl;
       */
       //  digi[i] should be the sample as digi.sample(i), I think
     } // loop on all (22) ZDC digis
     }
   ////////////////////////////////////////////////////////////////////////////////////////////

   ////////////////////////////////////GEN PARTICLE HISTOS///////////////////////////////////


      if (gotGenParticles==true){ //if the boolean was able to find the leaf "genparticles" then do this
   for (reco::GenParticleCollection::const_iterator gen = genhandle->begin();
	  gen!=genhandle->end();
	  ++gen) //here we iterate over all generated particles
       {
	 //         double energy=gen->energy();
	 reco::GenParticle thisParticle = (reco::GenParticle)(*gen); //get the particle "gen" points to
         double energy_2= thisParticle.energy(); //here I grab some of the attributes of the generated particle....like its energy, its phi and its eta and what kind of particle it is
         double gen_phi = thisParticle.phi();
         double gen_eta = thisParticle.eta();
         int gen_id = thisParticle.pdgId();	
   
	 if (gen_id==111){ //here i require a pi0
	   if (gen_eta>3){ //eta requirement

//////////////////////////////////////////////////////////////////////////////////////
//# IMPORTANT     IMPORTANT         IMPORTANT            IMPORTANT                  #
//# The real eta of the ZDC is |eta| > 8.3, I have only changed it here to 3 because#
//# in the PG simulation the ZDC is at an eta of about 4.5-7, in the real GEANT the #
//# ZDC is in its appropriate place at the very foward region...please edit this if #
//# looking at MinBias data or the like                                             #
//#                                                                                 #
//# IMPORTANT     IMPORTANT      IMPORTANT       IMPORTANT                          #
/////////////////////////////////////////////////////////////////////////////////////



	   genpart_Pi0F->Fill(gen_eta,gen_phi,energy_2); //fill the lego plot
           genpart_Pi0F_energydist->Fill(energy_2); //fill the 1D distribution
	                 }
           if (gen_eta<-8){ //neg eta requirement
	     genpart_Pi0B->Fill(gen_eta,gen_phi,energy_2);
             genpart_Pi0B_energydist->Fill(energy_2);
	                   }
                                       }
	 if (gen_id==2112){ //require neutron
	   if (gen_eta>3){
         genpart_NeutF->Fill(gen_eta,gen_phi,energy_2);
         genpart_NeutF_energydist->Fill(energy_2);
	                  }
	   if (gen_eta<-8){
	     genpart_NeutB->Fill(gen_eta,gen_phi,energy_2);
             genpart_NeutB_energydist->Fill(energy_2);
	                   }
                          }

	 if (gen_id==22){ //require gamma
	   if (gen_eta>3){
           genpart_GammaF->Fill(gen_eta,gen_phi,energy_2);
           genpart_GammaF_energydist->Fill(energy_2);
	                 }
           if (gen_eta<-8){
	     genpart_GammaB->Fill(gen_eta,gen_phi,energy_2);
             genpart_GammaB_energydist->Fill(energy_2);
	                   }
                         }

       } //end of GEN loop
      }
 
   // Now fill total charge histogram
   meZdcfCPEMvHAD->Fill(totalPHADCharge,totalPEMCharge);
   meZdcfCNEMvHAD->Fill(totalNHADCharge,totalNEMCharge);
   meZdcfCPHAD->Fill(totalPHADCharge);
   meZdcfCNHAD->Fill(totalNHADCharge);
   meZdcfCNTOT->Fill(totalNCharge);
   meZdcfCPTOT->Fill(totalPCharge);



/////////////////////////////////////////////////////////////////////

//Below is the old script which I will comment later



  LogDebug("ZdcSimHitStudy") 
    //std::cout
    //std::cout
    << "Run = " << iEvent.id().run() << " Event = " 
    << iEvent.id().event();
/*    << "Run = " << e.id().run() << " Event = " 
    << e.id().event();*/
  //std::cout<<std::endl;
  
  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsZdc;

  bool getHits = false;
  if (checkHit_) {
    iEvent.getByLabel(g4Label,zdcHits,hitsZdc);
//    e.getByLabel(g4Label,zdcHits,hitsZdc); 
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
  
 

void ZdcSimHitStudy::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  int nevents=(meZdcPHAD1fCvsTS->getTH1F())->GetBinContent(0); //grab the number of digis that were read in and stored in the underflow bin, and call them Nevents
  (meZdcPHAD1fCvsTS->getTH1F())->Scale(1./nevents);  // divide histogram by nevents thereby creating an average..it was done this way so that in DQM when everything is done in parallel and added at the end then the average will add appropriately

  int nevents1=(meZdcPHAD2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD2fCvsTS->getTH1F())->Scale(1./nevents1);

  int nevents2=(meZdcPHAD3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD3fCvsTS->getTH1F())->Scale(1./nevents2);

  int nevents3=(meZdcPHAD4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD4fCvsTS->getTH1F())->Scale(1./nevents3);

  int nevents4=(meZdcNHAD1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD1fCvsTS->getTH1F())->Scale(1./nevents4);

  int nevents5=(meZdcNHAD2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD2fCvsTS->getTH1F())->Scale(1./nevents5);

  int nevents6=(meZdcNHAD3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD3fCvsTS->getTH1F())->Scale(1./nevents6);

  int nevents7=(meZdcNHAD4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD4fCvsTS->getTH1F())->Scale(1./nevents7);

  int nevents8=(meZdcPEM1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM1fCvsTS->getTH1F())->Scale(1./nevents8);

  int nevents9=(meZdcPEM2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM2fCvsTS->getTH1F())->Scale(1./nevents9);

  int nevents10=(meZdcPEM3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM3fCvsTS->getTH1F())->Scale(1./nevents10);

  int nevents11=(meZdcPEM4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM4fCvsTS->getTH1F())->Scale(1./nevents11);

  int nevents12=(meZdcPEM5fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM5fCvsTS->getTH1F())->Scale(1./nevents12);

  int nevents13=(meZdcNEM1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM1fCvsTS->getTH1F())->Scale(1./nevents13);

  int nevents14=(meZdcNEM2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM2fCvsTS->getTH1F())->Scale(1./nevents14);

  int nevents15=(meZdcNEM3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM3fCvsTS->getTH1F())->Scale(1./nevents15);

  int nevents16=(meZdcNEM4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM4fCvsTS->getTH1F())->Scale(1./nevents16);

  int nevents17=(meZdcNEM5fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM5fCvsTS->getTH1F())->Scale(1./nevents17);



}












