////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Package:    ZDCDigiStudy
// Class:      ZDCDigiStudy
//
/*
  Description:
  This code has been developed to be a check for the ZDC sim. In 2009, it was found that the ZDC Simulation was unrealistic and needed repair. The aim of this code is to show the user the input and output of a ZDC MinBias simulation.

  Implementation:
  First a MinBias simulation should be run, it could be pythia,hijin,or hydjet. This will output a .root file which should have information about recoGenParticles, hcalunsuppresseddigis. Use this .root file as the input into the cfg.py which is found in the main directory of this package. This output will be another .root file which is meant to be viewed in a TBrowser

*/
//
// Original Author: Jaime Gomez (U. of Maryland) with SIGNIFICANT assistance of Dr. Jefferey Temple (U. of Maryland)
//
//
//         Created:  Summer 2012
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Validation/HcalDigis/interface/ZDCDigiStudy.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

ZDCDigiStudy::ZDCDigiStudy(const edm::ParameterSet& ps) {
  zdcHits = ps.getUntrackedParameter<std::string>("HitCollection", "ZdcHits");
  outFile_ = ps.getUntrackedParameter<std::string>("outputFile", "zdcHitStudy.root");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  checkHit_ = true;

  tok_zdc_ = consumes<ZDCDigiCollection>(edm::InputTag("simHcalUnsuppressedDigis"));

  edm::LogInfo("ZDCDigiStudy")
      //std::cout
      << "   Hits: " << zdcHits << " / " << checkHit_ << "   Output: " << outFile_;
}

ZDCDigiStudy::~ZDCDigiStudy() {}

void ZDCDigiStudy::bookHistograms(DQMStore::IBooker& ib, edm::Run const& run, edm::EventSetup const& es) {
  ib.setCurrentFolder("ZDCDigiValidation");
  // run histos only since there is dqmEndRun processing.
  ib.setScope(MonitorElementData::Scope::RUN);

  //Histograms for Hits
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //# Below we are filling the histograms made in the .h file. The syntax is as follows:                                      #
  //# plot_code_name = dbe_->TypeofPlot[(1,2,3)-D,(F,I,D)]("Name as it will appear","Title",axis options);                    #
  //# They will be stored in the TFile subdirectory set by :    dbe_->setCurrentFolder("FolderIwant")                         #
  //# axis options are like (#ofbins,min,max)                                                                                 #
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (checkHit_) {
    ////////////////////////// 1-D TotalfC per Side ///////////////////////

    ///////////////////////////////// 1 ////////////////////////////////////////////
    ib.setCurrentFolder("ZDCDigiValidation/ZDC_Digis/1D_fC");
    meZdcfCPHAD = ib.book1D("PHAD_TotalfC", "PZDC_HAD_TotalfC", 1000, -50, 10000);
    meZdcfCPHAD->setAxisTitle("Counts", 2);
    meZdcfCPHAD->setAxisTitle("fC", 1);
    /////////////////////////////////2////////////////////////////
    meZdcfCPTOT = ib.book1D("PZDC_TotalfC", "PZDC_TotalfC", 1000, -50, 20000);
    meZdcfCPTOT->setAxisTitle("Counts", 2);
    meZdcfCPTOT->setAxisTitle("fC", 1);
    /////////////////////////////////3/////////////////////////////////
    meZdcfCNHAD = ib.book1D("NHAD_TotalfC", "NZDC_HAD_TotalfC", 1000, -50, 10000);
    meZdcfCNHAD->setAxisTitle("Counts", 2);
    meZdcfCNHAD->setAxisTitle("fC", 1);
    ////////////////////////////////4/////////////////////////////////////////
    meZdcfCNTOT = ib.book1D("NZDC_TotalfC", "NZDC_TotalfC", 1000, -50, 20000);
    meZdcfCNTOT->setAxisTitle("Counts", 2);
    meZdcfCNTOT->setAxisTitle("fC", 1);
    /////////////////////////////////////////////////////////////////////////

    //////////////////////// 1-D fC vs TS ///////////////////////////////////////
    ib.setCurrentFolder("ZDCDigiValidation/ZDC_Digis/fCvsTS/PZDC");

    /////////////////////////////////5/////////////////////////////////////////
    meZdcPEM1fCvsTS = ib.book1D("PEM1_fCvsTS", "P-EM1_AveragefC_vsTS", 10, 0, 9);
    meZdcPEM1fCvsTS->setAxisTitle("fC", 2);
    meZdcPEM1fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////6/////////////////////////////////////////
    meZdcPEM2fCvsTS = ib.book1D("PEM2_fCvsTS", "P-EM2_AveragefC_vsTS", 10, 0, 9);
    meZdcPEM2fCvsTS->setAxisTitle("fC", 2);
    meZdcPEM2fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////7/////////////////////////////////////////
    meZdcPEM3fCvsTS = ib.book1D("PEM3_fCvsTS", "P-EM3_AveragefC_vsTS", 10, 0, 9);
    meZdcPEM3fCvsTS->setAxisTitle("fC", 2);
    meZdcPEM3fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////8/////////////////////////////////////////
    meZdcPEM4fCvsTS = ib.book1D("PEM4_fCvsTS", "P-EM4_AveragefC_vsTS", 10, 0, 9);
    meZdcPEM4fCvsTS->setAxisTitle("fC", 2);
    meZdcPEM4fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////9/////////////////////////////////////////
    meZdcPEM5fCvsTS = ib.book1D("PEM5_fCvsTS", "P-EM5_AveragefC_vsTS", 10, 0, 9);
    meZdcPEM5fCvsTS->setAxisTitle("fC", 2);
    meZdcPEM5fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////10/////////////////////////////////////////
    meZdcPHAD1fCvsTS = ib.book1D("PHAD1_fCvsTS", "P-HAD1_AveragefC_vsTS", 10, 0, 9);
    meZdcPHAD1fCvsTS->setAxisTitle("fC", 2);
    meZdcPHAD1fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////11/////////////////////////////////////////
    meZdcPHAD2fCvsTS = ib.book1D("PHAD2_fCvsTS", "P-HAD2_AveragefC_vsTS", 10, 0, 9);
    meZdcPHAD2fCvsTS->setAxisTitle("fC", 2);
    meZdcPHAD2fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////12/////////////////////////////////////////
    meZdcPHAD3fCvsTS = ib.book1D("PHAD3_fCvsTS", "P-HAD3_AveragefC_vsTS", 10, 0, 9);
    meZdcPHAD3fCvsTS->setAxisTitle("fC", 2);
    meZdcPHAD3fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////13/////////////////////////////////////////
    meZdcPHAD4fCvsTS = ib.book1D("PHAD4_fCvsTS", "P-HAD4_AveragefC_vsTS", 10, 0, 9);
    meZdcPHAD4fCvsTS->setAxisTitle("fC", 2);
    meZdcPHAD4fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    ib.setCurrentFolder("ZDCDigiValidation/ZDC_Digis/fCvsTS/NZDC");

    /////////////////////////////////14/////////////////////////////////////////
    meZdcNEM1fCvsTS = ib.book1D("NEM1_fCvsTS", "N-EM1_AveragefC_vsTS", 10, 0, 9);
    meZdcNEM1fCvsTS->setAxisTitle("fC", 2);
    meZdcNEM1fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////15/////////////////////////////////////////
    meZdcNEM2fCvsTS = ib.book1D("NEM2_fCvsTS", "N-EM2_AveragefC_vsTS", 10, 0, 9);
    meZdcNEM2fCvsTS->setAxisTitle("fC", 2);
    meZdcNEM2fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////16/////////////////////////////////////////
    meZdcNEM3fCvsTS = ib.book1D("NEM3_fCvsTS", "N-EM3_AveragefC_vsTS", 10, 0, 9);
    meZdcNEM3fCvsTS->setAxisTitle("fC", 2);
    meZdcNEM3fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////17/////////////////////////////////////////
    meZdcNEM4fCvsTS = ib.book1D("NEM4_fCvsTS", "N-EM4_AveragefC_vsTS", 10, 0, 9);
    meZdcNEM4fCvsTS->setAxisTitle("fC", 2);
    meZdcNEM4fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////18/////////////////////////////////////////
    meZdcNEM5fCvsTS = ib.book1D("NEM5_fCvsTS", "N-EM5_AveragefC_vsTS", 10, 0, 9);
    meZdcNEM5fCvsTS->setAxisTitle("fC", 2);
    meZdcNEM5fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////19/////////////////////////////////////////
    meZdcNHAD1fCvsTS = ib.book1D("NHAD1_fCvsTS", "N-HAD1_AveragefC_vsTS", 10, 0, 9);
    meZdcNHAD1fCvsTS->setAxisTitle("fC", 2);
    meZdcNHAD1fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////20/////////////////////////////////////////
    meZdcNHAD2fCvsTS = ib.book1D("NHAD2_fCvsTS", "N-HAD2_AveragefC_vsTS", 10, 0, 9);
    meZdcNHAD2fCvsTS->setAxisTitle("fC", 2);
    meZdcNHAD2fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////21/////////////////////////////////////////
    meZdcNHAD3fCvsTS = ib.book1D("NHAD3_fCvsTS", "N-HAD3_AveragefC_vsTS", 10, 0, 9);
    meZdcNHAD3fCvsTS->setAxisTitle("fC", 2);
    meZdcNHAD3fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////22/////////////////////////////////////////
    meZdcNHAD4fCvsTS = ib.book1D("NHAD4_fCvsTS", "N-HAD4_AveragefC_vsTS", 10, 0, 9);
    meZdcNHAD4fCvsTS->setAxisTitle("fC", 2);
    meZdcNHAD4fCvsTS->setAxisTitle("TS", 1);
    ////////////////////////////////////////////////////////////////////////////

    //////////////////// 2-D EMvHAD plots/////////////////////////////////////////
    ib.setCurrentFolder("ZDCDigiValidation/ZDC_Digis/2D_EMvHAD");
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////23//////////////////////////////////////////
    meZdcfCPEMvHAD = ib.book2D("PEMvPHAD", "PZDC_EMvHAD", 200, -25, 12000, 200, -25, 15000);
    meZdcfCPEMvHAD->setAxisTitle("SumEM_fC", 2);
    meZdcfCPEMvHAD->setAxisTitle("SumHAD_fC", 1);
    meZdcfCPEMvHAD->setOption("colz");
    ////////////////////////////////24///////////////////////////////////////////
    meZdcfCNEMvHAD = ib.book2D("NEMvNHAD", "NZDC_EMvHAD", 1000, -25, 12000, 1000, -25, 15000);
    meZdcfCNEMvHAD->setAxisTitle("SumEM_fC", 2);
    meZdcfCNEMvHAD->setAxisTitle("SumHAD_fC", 1);
    meZdcfCNEMvHAD->setOption("colz");
    ///////////////////////////////////////////////////////////////////////////////
  }
}

/*void ZDCDigiStudy::endJob() {
  if (dbe_ && outFile_.size() > 0) dbe_->save(outFile_);
  }*/

//void ZDCDigiStudy::analyze(const edm::Event& e, const edm::EventSetup& ) {
void ZDCDigiStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //////////NEW STUFF//////////////////////

  using namespace edm;
  bool gotZDCDigis = true;

  Handle<ZDCDigiCollection> zdchandle;
  if (!(iEvent.getByToken(tok_zdc_, zdchandle))) {
    gotZDCDigis = false;  //this is a boolean set up to check if there are ZDCDigis in the input root file
  }
  if (!(zdchandle.isValid())) {
    gotZDCDigis = false;  //if it is not there, leave it false
  }

  double totalPHADCharge = 0;
  double totalNHADCharge = 0;
  double totalPEMCharge = 0;
  double totalNEMCharge = 0;
  double totalPCharge = 0;
  double totalNCharge = 0;

  //////////////////////////////////////////////////DIGIS///////////////////////////////////
  if (gotZDCDigis == true) {
    for (ZDCDigiCollection::const_iterator zdc = zdchandle->begin(); zdc != zdchandle->end(); ++zdc) {
      const ZDCDataFrame digi = (const ZDCDataFrame)(*zdc);
      //std::cout <<"CHANNEL = "<<zdc->id().channel()<<std::endl;

      /////////////////////////////HAD SECTIONS///////////////

      if (digi.id().section() == 2) {            // require HAD
        if (digi.id().zside() == 1) {            // require POS
          for (int i = 0; i < digi.size(); ++i)  // loop over all 10 TS because each digi has 10 entries
          {
            if (digi.id().channel() == 1) {  //here i specify PHAD1
              meZdcPHAD1fCvsTS->Fill(
                  i, digi.sample(i).nominal_fC());  //filling the plot name with the nominal fC value for each TS
              if (i == 0)
                meZdcPHAD1fCvsTS->Fill(-1, 1);  // on first iteration of loop, increment underflow bin
            }                                   //NEW AVERAGE Thingy
            if (digi.id().channel() == 2) {
              meZdcPHAD2fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPHAD2fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 3) {
              meZdcPHAD3fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPHAD3fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 4) {
              meZdcPHAD4fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPHAD4fCvsTS->Fill(-1, 1);
            }
            if (i == 4 || i == 5 || i == 6)
              totalPHADCharge += digi.sample(i).nominal_fC();
          }  // loop over all (10) TS for the given digi
        } else {
          for (int i = 0; i < digi.size(); ++i) {
            if (digi.id().channel() == 1) {
              meZdcNHAD1fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNHAD1fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 2) {
              meZdcNHAD2fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNHAD2fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 3) {
              meZdcNHAD3fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNHAD3fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 4) {
              meZdcNHAD4fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNHAD4fCvsTS->Fill(-1, 1);
            }
            if (i == 4 || i == 5 || i == 6)
              totalNHADCharge += digi.sample(i).nominal_fC();
          }  //loop over all 10 TS
        }    //Requires NHAd
      }      //Requires HAD sections
      ///////////////////////////////EM SECTIONS////////////////////////////
      if (digi.id().section() ==
          1) {  //require EM....here i do the smae thing that i did above but now for P/N EM sections
        if (digi.id().zside() == 1) {  //require pos
          for (int i = 0; i < digi.size(); ++i) {
            if (digi.id().channel() == 1) {
              meZdcPEM1fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPEM1fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 2) {
              meZdcPEM2fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPEM2fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 3) {
              meZdcPEM3fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPEM3fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 4) {
              meZdcPEM4fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPEM4fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 5) {
              meZdcPEM5fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcPEM5fCvsTS->Fill(-1, 1);
            }
            if (i == 4 || i == 5 || i == 6)
              totalPEMCharge += digi.sample(i).nominal_fC();
          }
        } else {
          for (int i = 0; i < digi.size(); ++i) {
            if (digi.id().channel() == 1) {
              meZdcNEM1fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNEM1fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 2) {
              meZdcNEM2fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNEM2fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 3) {
              meZdcNEM3fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNEM3fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 4) {
              meZdcNEM4fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNEM4fCvsTS->Fill(-1, 1);
            }
            if (digi.id().channel() == 5) {
              meZdcNEM5fCvsTS->Fill(i, digi.sample(i).nominal_fC());
              if (i == 0)
                meZdcNEM5fCvsTS->Fill(-1, 1);
            }
            if (i == 4 || i == 5 || i == 6)
              totalNEMCharge += digi.sample(i).nominal_fC();
          }
        }
      }

      totalPCharge = totalPHADCharge + (0.1) * totalPEMCharge;
      totalNCharge = totalNHADCharge + (0.1) * totalNEMCharge;

      /*       std::cout <<"CHANNEL = "<<digi.id().channel()<<std::endl;
                 for (int i=0;i<digi.size();++i)
                 std::cout <<"SAMPLE = "<<i<<"  ADC = "<<digi.sample(i).adc()<<" fC =  "<<digi.sample(i).nominal_fC()<<std::endl;
        */
      //  digi[i] should be the sample as digi.sample(i), I think
    }  // loop on all (22) ZDC digis
  }
  ////////////////////////////////////////////////////////////////////////////////////////////

  // Now fill total charge histogram
  meZdcfCPEMvHAD->Fill(totalPCharge, totalPEMCharge);
  meZdcfCNEMvHAD->Fill(totalNCharge, totalNEMCharge);
  meZdcfCPHAD->Fill(totalPHADCharge);
  meZdcfCNHAD->Fill(totalNHADCharge);
  meZdcfCNTOT->Fill(totalNCharge);
  meZdcfCPTOT->Fill(totalPCharge);
}

////////////////////////////////////////////////////////////////////

void ZDCDigiStudy::dqmEndRun(const edm::Run& run, const edm::EventSetup& c) {
  int nevents =
      (meZdcPHAD1fCvsTS->getTH1F())
          ->GetBinContent(
              0);  //grab the number of digis that were read in and stored in the underflow bin, and call them Nevents
  (meZdcPHAD1fCvsTS->getTH1F())
      ->Scale(
          1. /
          nevents);  // divide histogram by nevents thereby creating an average..it was done this way so that in DQM when everything is done in parallel and added at the end then the average will add appropriately

  int nevents1 = (meZdcPHAD2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD2fCvsTS->getTH1F())->Scale(1. / nevents1);

  int nevents2 = (meZdcPHAD3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD3fCvsTS->getTH1F())->Scale(1. / nevents2);

  int nevents3 = (meZdcPHAD4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPHAD4fCvsTS->getTH1F())->Scale(1. / nevents3);

  int nevents4 = (meZdcNHAD1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD1fCvsTS->getTH1F())->Scale(1. / nevents4);

  int nevents5 = (meZdcNHAD2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD2fCvsTS->getTH1F())->Scale(1. / nevents5);

  int nevents6 = (meZdcNHAD3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD3fCvsTS->getTH1F())->Scale(1. / nevents6);

  int nevents7 = (meZdcNHAD4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNHAD4fCvsTS->getTH1F())->Scale(1. / nevents7);

  int nevents8 = (meZdcPEM1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM1fCvsTS->getTH1F())->Scale(1. / nevents8);

  int nevents9 = (meZdcPEM2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM2fCvsTS->getTH1F())->Scale(1. / nevents9);

  int nevents10 = (meZdcPEM3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM3fCvsTS->getTH1F())->Scale(1. / nevents10);

  int nevents11 = (meZdcPEM4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM4fCvsTS->getTH1F())->Scale(1. / nevents11);

  int nevents12 = (meZdcPEM5fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcPEM5fCvsTS->getTH1F())->Scale(1. / nevents12);

  int nevents13 = (meZdcNEM1fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM1fCvsTS->getTH1F())->Scale(1. / nevents13);

  int nevents14 = (meZdcNEM2fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM2fCvsTS->getTH1F())->Scale(1. / nevents14);

  int nevents15 = (meZdcNEM3fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM3fCvsTS->getTH1F())->Scale(1. / nevents15);

  int nevents16 = (meZdcNEM4fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM4fCvsTS->getTH1F())->Scale(1. / nevents16);

  int nevents17 = (meZdcNEM5fCvsTS->getTH1F())->GetBinContent(0);
  (meZdcNEM5fCvsTS->getTH1F())->Scale(1. / nevents17);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZDCDigiStudy);
