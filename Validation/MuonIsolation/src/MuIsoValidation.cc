// -*- C++ -*-
// MuIsoValidation.cc
// Package:    Muon Isolation Validation
// Class:      MuIsoValidation
//
/*


 Description: Muon Isolation Validation class

 Implementation: This code will accept a data set and generate plots of
	various quantities relevent to the Muon Isolation module. We will 
	be using the IsoDeposit class, *not* the MuonIsolation struct.
	 
	The sequence of events is... 
 		* initalize statics (which variables to plot, axis limtis, etc.)
 		* run over events
 			* for each event, run over the muons in that event
 				*record IsoDeposit data
 		* transfer data to histograms, profile plots
 		* save histograms to a root file
 		
 	Easy-peasy.
	
*/
//
// Original Author:  "C. Jess Riedel", UC Santa Barbara
//         Created:  Tue Jul 17 15:58:24 CDT 2007
//

//Class header file
#include "Validation/MuonIsolation/interface/MuIsoValidation.h"

//System included files
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

//Root included files
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

//Event framework included files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Other included files
#include "DataFormats/TrackReco/interface/Track.h"

//Using declarations
using std::pair;
using std::string;
using std::vector;

//
//-----------------Constructors---------------------
//
MuIsoValidation::MuIsoValidation(const edm::ParameterSet& ps) {
  iConfig = ps;

  //  rootfilename = iConfig.getUntrackedParameter<string>("rootfilename"); // comment out for inclusion
  requireCombinedMuon = iConfig.getUntrackedParameter<bool>("requireCombinedMuon");
  dirName = iConfig.getUntrackedParameter<std::string>("directory");
  //subDirName = iConfig.getParameter<std::string>("@module_label");

  //dirName += subDirName;

  //--------Initialize tags-------
  Muon_Tag = iConfig.getUntrackedParameter<edm::InputTag>("Global_Muon_Label");
  Muon_Token = consumes<edm::View<reco::Muon> >(Muon_Tag);

  //-------Initialize counters----------------
  nEvents = 0;
  nIncMuons = 0;
  //  nCombinedMuons = 0;

  InitStatics();

  //Set up DAQ
  subsystemname_ = iConfig.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem");

  //------"allocate" space for the data vectors-------
  h_1D.resize(NUM_VARS);
  cd_plots.resize(NUM_VARS);
  p_2D.resize(NUM_VARS, vector<MonitorElement*>(NUM_VARS));
}

//
//----------Destructor-----------------
//
MuIsoValidation::~MuIsoValidation() {
  //Deallocate memory
}

//
//------------Methods-----------------------
//
void MuIsoValidation::InitStatics() {
  //-----------Initialize primatives-----------
  S_BIN_WIDTH = 1.0;  //in GeV
  L_BIN_WIDTH = 2.0;  //in GeV
  LOG_BINNING_ENABLED = 1;
  NUM_LOG_BINS = 15;
  LOG_BINNING_RATIO = 1.1;  //ratio by which each bin is wider than the last for log binning
  //i.e.  bin widths are (x), (r*x), (r^2*x), ..., (r^(nbins)*x)

  //-------Initialize Titles---------
  title_sam = "";   //"[Sample b-jet events] ";
  title_cone = "";  //" [in R=0.3 IsoDeposit Cone]";
  //The above two pieces of info will be printed on the title of the whole page,
  //not for each individual histogram
  title_cd = "C.D. of ";

  //-------"Allocate" memory for vectors
  main_titles.resize(NUM_VARS);
  axis_titles.resize(NUM_VARS);
  names.resize(NUM_VARS);
  param.resize(NUM_VARS, vector<double>(3));
  isContinuous.resize(NUM_VARS);
  cdCompNeeded.resize(NUM_VARS);

  //-----Titles of the plots-----------
  main_titles[0] = "Total Tracker Momentum";
  main_titles[1] = "Total EM Cal Energy";
  main_titles[2] = "Total Had Cal Energy";
  main_titles[3] = "Total HO Cal Energy";
  main_titles[4] = "Number of Tracker Tracks";
  main_titles[5] = "Number of Jets around Muon";
  main_titles[6] = "Tracker p_{T} within veto cone";
  main_titles[7] = "EM E_{T} within veto cone";
  main_titles[8] = "Had E_{T} within veto cone";
  main_titles[9] = "HO E_{T} within veto cone";
  main_titles[10] = "Muon p_{T}";
  main_titles[11] = "Muon #eta";
  main_titles[12] = "Muon #phi";
  main_titles[13] = "Average Momentum per Track ";
  main_titles[14] = "Weighted Energy";
  main_titles[15] = "PF Sum of Charged Hadron Pt";
  main_titles[16] = "PF Sum of Total Hadron Pt";
  main_titles[17] = "PF Sum of E,Mu Pt";
  main_titles[18] = "PF Sum of Neutral Hadron Et";
  main_titles[19] = "PF Sum of Photon Et";
  main_titles[20] = "PF Sum of Pt from non-PV";

  //------Titles on the X or Y axis------------
  axis_titles[0] = "#Sigma p_{T}   (GeV)";
  axis_titles[1] = "#Sigma E_{T}^{EM}   (GeV)";
  axis_titles[2] = "#Sigma E_{T}^{Had}   (GeV)";
  axis_titles[3] = "#Sigma E_{T}^{HO}   (GeV)";
  axis_titles[4] = "N_{Tracks}";
  axis_titles[5] = "N_{Jets}";
  axis_titles[6] = "#Sigma p_{T,veto} (GeV)";
  axis_titles[7] = "#Sigma E_{T,veto}^{EM}   (GeV)";
  axis_titles[8] = "#Sigma E_{T,veto}^{Had}   (GeV)";
  axis_titles[9] = "#Sigma E_{T,veto}^{HO}   (GeV)";
  axis_titles[10] = "p_{T,#mu} (GeV)";
  axis_titles[11] = "#eta_{#mu}";
  axis_titles[12] = "#phi_{#mu}";
  axis_titles[13] = "#Sigma p_{T} / N_{Tracks} (GeV)";
  axis_titles[14] = "(1.5) X #Sigma E_{T}^{EM} + #Sigma E_{T}^{Had}";
  axis_titles[15] = "#Sigma p_{T}^{PFHadCha}   (GeV)";
  axis_titles[16] = "#Sigma p_{T}^{PFTotCha}   (GeV)";
  axis_titles[17] = "#Sigma p_{T}^{PFEMu}   (GeV)";
  axis_titles[18] = "#Sigma E_{T}^{PFHadNeu}   (GeV)";
  axis_titles[19] = "#Sigma E_{T}^{PFPhot}   (GeV)";
  axis_titles[20] = "#Sigma p_{T}^{PFPU}   (GeV)";

  //-----------Names given for the root file----------
  names[0] = "sumPt";
  names[1] = "emEt";
  names[2] = "hadEt";
  names[3] = "hoEt";
  names[4] = "nTracks";
  names[5] = "nJets";
  names[6] = "trackerVetoPt";
  names[7] = "emVetoEt";
  names[8] = "hadVetoEt";
  names[9] = "hoVetoEt";
  names[10] = "muonPt";
  names[11] = "muonEta";
  names[12] = "muonPhi";
  names[13] = "avgPt";
  names[14] = "weightedEt";
  names[15] = "PFsumChargedHadronPt";
  names[16] = "PFsumChargedTotalPt";
  names[17] = "PFsumEMuPt";
  names[18] = "PFsumNeutralHadronEt";
  names[19] = "PFsumPhotonEt";
  names[20] = "PFsumPUPt";

  //----------Parameters for binning of histograms---------
  //param[var][0] is the number of bins
  //param[var][1] is the low edge of the low bin
  //param[var][2] is the high edge of the high bin
  //
  // maximum value------,
  //                    |
  //                    V
  param[0][0] = (int)(20.0 / S_BIN_WIDTH);
  param[0][1] = 0.0;
  param[0][2] = param[0][0] * S_BIN_WIDTH;
  param[1][0] = (int)(20.0 / S_BIN_WIDTH);
  param[1][1] = 0.0;
  param[1][2] = param[1][0] * S_BIN_WIDTH;
  param[2][0] = (int)(20.0 / S_BIN_WIDTH);
  param[2][1] = 0.0;
  param[2][2] = param[2][0] * S_BIN_WIDTH;
  param[3][0] = 20;
  param[3][1] = 0.0;
  param[3][2] = 2.0;
  param[4][0] = 16;
  param[4][1] = -0.5;
  param[4][2] = param[4][0] - 0.5;
  param[5][0] = 4;
  param[5][1] = -0.5;
  param[5][2] = param[5][0] - 0.5;
  param[6][0] = (int)(40.0 / S_BIN_WIDTH);
  param[6][1] = 0.0;
  param[6][2] = param[6][0] * S_BIN_WIDTH;
  param[7][0] = 20;
  param[7][1] = 0.0;
  param[7][2] = 10.0;
  param[8][0] = (int)(20.0 / S_BIN_WIDTH);
  param[8][1] = 0.0;
  param[8][2] = param[8][0] * S_BIN_WIDTH;
  param[9][0] = 20;
  param[9][1] = 0.0;
  param[9][2] = 5.0;
  param[10][0] = (int)(40.0 / S_BIN_WIDTH);
  param[10][1] = 0.0;
  param[10][2] = param[10][0] * S_BIN_WIDTH;
  param[11][0] = 24;
  param[11][1] = -2.4;
  param[11][2] = 2.4;
  param[12][0] = 32;
  param[12][1] = -3.2;
  param[12][2] = 3.2;
  param[13][0] = (int)(15.0 / S_BIN_WIDTH);
  param[13][1] = 0.0;
  param[13][2] = param[13][0] * S_BIN_WIDTH;
  param[14][0] = (int)(20.0 / S_BIN_WIDTH);
  param[14][1] = 0.0;
  param[14][2] = param[14][0] * S_BIN_WIDTH;
  param[15][0] = (int)(20.0 / S_BIN_WIDTH);
  param[15][1] = 0.0;
  param[15][2] = param[15][0] * S_BIN_WIDTH;
  param[16][0] = (int)(20.0 / S_BIN_WIDTH);
  param[15][1] = 0.0;
  param[16][2] = param[16][0] * S_BIN_WIDTH;
  param[17][0] = (int)(20.0 / S_BIN_WIDTH) + 1;
  param[17][1] = -S_BIN_WIDTH;
  param[17][2] = param[17][0] * S_BIN_WIDTH;
  param[18][0] = (int)(20.0 / S_BIN_WIDTH);
  param[18][1] = 0.0;
  param[18][2] = param[18][0] * S_BIN_WIDTH;
  param[19][0] = (int)(20.0 / S_BIN_WIDTH);
  param[19][1] = 0.0;
  param[19][2] = param[19][0] * S_BIN_WIDTH;
  param[20][0] = (int)(20.0 / S_BIN_WIDTH);
  param[20][1] = 0.0;
  param[20][2] = param[20][0] * S_BIN_WIDTH;

  //--------------Is the variable continuous (i.e. non-integer)?-------------
  //---------(Log binning will only be used for continuous variables)--------
  isContinuous[0] = 1;
  isContinuous[1] = 1;
  isContinuous[2] = 1;
  isContinuous[3] = 1;
  isContinuous[4] = 0;
  isContinuous[5] = 0;
  isContinuous[6] = 1;
  isContinuous[7] = 1;
  isContinuous[8] = 1;
  isContinuous[9] = 1;
  isContinuous[10] = 1;
  isContinuous[11] = 1;
  isContinuous[12] = 1;
  isContinuous[13] = 1;
  isContinuous[14] = 1;
  isContinuous[15] = 1;
  isContinuous[16] = 1;
  isContinuous[17] = 1;
  isContinuous[18] = 1;
  isContinuous[19] = 1;
  isContinuous[20] = 1;

  //----Should the cumulative distribution be calculated for this variable?-----
  cdCompNeeded[0] = 1;
  cdCompNeeded[1] = 1;
  cdCompNeeded[2] = 1;
  cdCompNeeded[3] = 1;
  cdCompNeeded[4] = 1;
  cdCompNeeded[5] = 1;
  cdCompNeeded[6] = 1;
  cdCompNeeded[7] = 1;
  cdCompNeeded[8] = 1;
  cdCompNeeded[9] = 1;
  cdCompNeeded[10] = 0;
  cdCompNeeded[11] = 0;
  cdCompNeeded[12] = 0;
  cdCompNeeded[13] = 1;
  cdCompNeeded[14] = 1;
  cdCompNeeded[15] = 1;
  cdCompNeeded[16] = 1;
  cdCompNeeded[17] = 1;
  cdCompNeeded[18] = 1;
  cdCompNeeded[19] = 1;
  cdCompNeeded[20] = 1;
}

// ------------ method called for each event  ------------
void MuIsoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++nEvents;
  edm::LogInfo("Tutorial") << "\nInvestigating event #" << nEvents << "\n";

  // Get Muon Collection
  edm::Handle<edm::View<reco::Muon> > muonsHandle;  //
  iEvent.getByToken(Muon_Token, muonsHandle);

  //Fill event entry in histogram of number of muons
  edm::LogInfo("Tutorial") << "Number of Muons: " << muonsHandle->size();
  theMuonData = muonsHandle->size();
  h_nMuons->Fill(theMuonData);

  //Fill historgams concerning muon isolation
  uint iMuon = 0;
  for (MuonIterator muon = muonsHandle->begin(); muon != muonsHandle->end(); ++muon, ++iMuon) {
    ++nIncMuons;
    if (requireCombinedMuon) {
      if (muon->combinedMuon().isNull())
        continue;
    }
    RecordData(muon);
    FillHistos();
  }
}

//---------------Record data for a signle muon's data---------------------
void MuIsoValidation::RecordData(MuonIterator muon) {
  theData[0] = muon->isolationR03().sumPt;
  theData[1] = muon->isolationR03().emEt;
  theData[2] = muon->isolationR03().hadEt;
  theData[3] = muon->isolationR03().hoEt;

  theData[4] = muon->isolationR03().nTracks;
  theData[5] = muon->isolationR03().nJets;
  theData[6] = muon->isolationR03().trackerVetoPt;
  theData[7] = muon->isolationR03().emVetoEt;
  theData[8] = muon->isolationR03().hadVetoEt;
  theData[9] = muon->isolationR03().hoVetoEt;

  theData[10] = muon->pt();
  theData[11] = muon->eta();
  theData[12] = muon->phi();

  // make sure nTracks != 0 before filling this one
  if (theData[4] != 0)
    theData[13] = (double)theData[0] / (double)theData[4];
  else
    theData[13] = -99;

  theData[14] = 1.5 * theData[1] + theData[2];

  // Now PF isolation
  theData[15] = -99.;
  theData[16] = -99.;
  theData[17] = -99.;
  theData[18] = -99.;
  theData[19] = -99.;
  theData[20] = -99.;
  if (muon->isPFMuon() && muon->isPFIsolationValid()) {
    theData[15] = muon->pfIsolationR03().sumChargedHadronPt;
    theData[16] = muon->pfIsolationR03().sumChargedParticlePt;
    theData[17] = muon->pfIsolationR03().sumChargedParticlePt - muon->pfIsolationR03().sumChargedHadronPt;
    theData[18] = muon->pfIsolationR03().sumNeutralHadronEt;
    theData[19] = muon->pfIsolationR03().sumPhotonEt;
    theData[20] = muon->pfIsolationR03().sumPUPt;
  }
}

void MuIsoValidation::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  ibooker.setCurrentFolder(dirName);
  //---initialize number of muons histogram---
  h_nMuons = ibooker.book1D("nMuons", title_sam + "Number of Muons", 20, 0., 20.);
  h_nMuons->setAxisTitle("Number of Muons", XAXIS);
  h_nMuons->setAxisTitle("Fraction of Events", YAXIS);

  //---Initialize 1D Histograms---
  for (int var = 0; var < NUM_VARS; var++) {
    h_1D[var] = ibooker.book1D(names[var],
                               title_sam + main_titles[var] + title_cone,
                               (int)param[var][0],
                               param[var][1],
                               param[var][2],
                               [](TH1* th1) { th1->Sumw2(); });
    h_1D[var]->setAxisTitle(axis_titles[var], XAXIS);
    h_1D[var]->setAxisTitle("Fraction of Muons", YAXIS);

    if (cdCompNeeded[var]) {
      cd_plots[var] = ibooker.book1D(names[var] + "_cd",
                                     title_sam + title_cd + main_titles[var] + title_cone,
                                     (int)param[var][0],
                                     param[var][1],
                                     param[var][2],
                                     [](TH1* th1) { th1->Sumw2(); });
      cd_plots[var]->setAxisTitle(axis_titles[var], XAXIS);
      cd_plots[var]->setAxisTitle("Fraction of Muons", YAXIS);
    }
  }  //Finish 1D

  //---Initialize 2D Histograms---
  for (int var1 = 0; var1 < NUM_VARS; var1++) {
    for (int var2 = 0; var2 < NUM_VARS; var2++) {
      if (var1 == var2)
        continue;

      //Monitor elements is weird and takes y axis parameters as well
      //as x axis parameters for a 1D profile plot
      p_2D[var1][var2] = ibooker.bookProfile(names[var1] + "_" + names[var2],
                                             title_sam + main_titles[var2] + " <vs> " + main_titles[var1] + title_cone,
                                             (int)param[var1][0],
                                             param[var1][1],
                                             param[var1][2],
                                             (int)param[var2][0],  //documentation says this is disregarded
                                             param[var2][1],       //does this do anything?
                                             param[var2][2],       //does this do anything?
                                             " ",                  //profile errors = spread/sqrt(num_datums)
                                             [&](TProfile* tprof) {
                                               if (LOG_BINNING_ENABLED && isContinuous[var1]) {
                                                 Double_t* bin_edges = new Double_t[NUM_LOG_BINS + 1];
                                                 // nbins+1 because there is one more edge than there are bins
                                                 MakeLogBinsForProfile(bin_edges, param[var1][1], param[var1][2]);
                                                 tprof->SetBins(NUM_LOG_BINS, bin_edges);
                                                 delete[] bin_edges;
                                               }
                                             });

      p_2D[var1][var2]->setAxisTitle(axis_titles[var1], XAXIS);
      p_2D[var1][var2]->setAxisTitle(axis_titles[var2], YAXIS);
    }
  }  //Finish 2D

  //avg pT not defined for zero tracks.
  //MonitorElement is inflxible and won't let me change the
  //number of bins!  I guess all I'm doing here is changing
  //range of the x axis when it is printed, not the actual
  //bins that are filled
  p_2D[4][9]->setAxisRange(0.5, 15.5, XAXIS);
}

void MuIsoValidation::MakeLogBinsForProfile(Double_t* bin_edges, const double min, const double max) {
  const double& r = LOG_BINNING_RATIO;
  const int& nbins = NUM_LOG_BINS;

  const double first_bin_width = (r > 1.0) ?  //so we don't divide by zero
                                     (max - min) * (1 - r) / (1 - pow(r, nbins))
                                           : (max - min) / nbins;

  bin_edges[0] = min;
  bin_edges[1] = min + first_bin_width;
  for (int n = 2; n < nbins; ++n) {
    bin_edges[n] = bin_edges[n - 1] + (bin_edges[n - 1] - bin_edges[n - 2]) * r;
  }
  bin_edges[nbins] = max;
}

void MuIsoValidation::FillHistos() {
  //----------Fill 1D histograms---------------
  for (int var = 0; var < NUM_VARS; var++) {
    h_1D[var]->Fill(theData[var]);
    if (cdCompNeeded[var])
      cd_plots[var]->Fill(theData[var]);  //right now, this is a regular PDF (just like h_1D)
  }                                       //Finish 1D

  //----------Fill 2D histograms---------------
  for (int var1 = 0; var1 < NUM_VARS; ++var1) {
    for (int var2 = 0; var2 < NUM_VARS; ++var2) {
      if (var1 == var2)
        continue;
      //change below to regular int interating!
      //      h_2D[var1][var2]->Fill(theData[var1], theData[var2]);
      p_2D[var1][var2]->Fill(theData[var1], theData[var2]);
    }
  }  //Finish 2D
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuIsoValidation);
