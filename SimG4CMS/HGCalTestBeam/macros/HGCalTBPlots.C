#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TF1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Compare plots with and without beamline counters
void PlotHistTBCompare(std::string infile1, std::string infile2, 
		       std::string text, std::string prefix, 
		       int maxMod=16, bool save=false);
// Plot energy distribution in a given layer
void PlotHistTBHitEn(std::string infile, std::string text, std::string prefix, 
		     int type, bool separate, int rebin, int maxMod=16, 
		     bool save=false);
// Plot general TB plots at SIM/Digi/Reco labels
void PlotHistSimDigRec(std::string fname="TBGenSimDigiReco.root", 
		       std::string dirnm="HGCalTBAnalyzer", int type=0,
		       std::string prefix="EL32", int maxMod=16,
		       bool save=false);
// Class to manipulate the Tree produced by the analysis code
// HGCTB l1(std::string infile, std::string outfile);
// l1.Loop(std::string calFile, std::string wtFile, unsigned int nLayers,
//         float beamE, float cutETotByEBeam, int ntotEvent, int clusterEn, 
//         int nPrintEvent)

class HexTopology {

public :
  HexTopology(bool fine);
  virtual ~HexTopology() {}

  double             cluster(const std::vector<unsigned int>& ids,
			     const std::vector<float>& energy, 
			     unsigned int id, int extent);
  unsigned int       localMax(const std::vector<unsigned int>& ids,
			      const std::vector<float>& energy, 
			      unsigned int layer);
  std::vector<int>   neighbours(int cell, int extent);
  std::pair<int,int> findRowCol(int cell);
private :
  std::vector<int> leftCell, cells, addCell, subCell;
  unsigned int     nrows;
};

HexTopology::HexTopology(bool fine) {
  int cellCoarse[15] = {2,5,8,11,12,11,12,11,12,11,12,11,8,5,2};
  int addCoarse[15] =  {1,0,1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1,0,1};
  int cellFine[20] = {3,6,9,12,15,16,15,16,15,16,15,16,15,16,15,14,11,8,5,2};
  int addFine[20]  = {0,1,0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1,0,1};
  int fullCoarse(11), fullFine(15);
  
  int full(0);
  if (fine) {
    for (int k=0; k<20; ++k) {
      cells.push_back(cellFine[k]); addCell.push_back(addFine[k]);
    }
    full = fullFine;
  } else {
    for (int k=0; k<15; ++k) {
      cells.push_back(cellCoarse[k]); addCell.push_back(addCoarse[k]);
    }
    full = fullCoarse;
  }
  leftCell.push_back(0);
  nrows = cells.size();
  for (unsigned int k=0; k<nrows; ++k) {
    leftCell.push_back((leftCell[k]+cells[k]));
    int ii = (cells[k]-full -1)/2;
    subCell.push_back(ii);
  }
  /*
  std::cout << "Initialize Wafer of type " << fine << " with " << nrows
	    << " rows and " << leftCell[nrows] << " cells" << std::endl;
  for (unsigned int k=0; k<nrows; ++k) 
    std::cout << "Row[" << k << "] with " << cells[k] << ":" << addCell[k]
	      << " cells: leftmost " << leftCell[k] << " offset "
	      << subCell[k] << std::endl;
  */
}

std::pair<int,int> HexTopology::findRowCol(int cell) {
  int row(-1), col(-1);
  if (cell >= leftCell[0] && cell < leftCell[nrows]) {
    for (unsigned int i=0; i<=nrows; ++i) {
      if (cell < leftCell[i]) {
	row = i-1; col = cell-leftCell[i-1]-subCell[i-1]-addCell[i-1]; break;
      }
    }
  }
// std::cout << "Cell " << cell << " row " << row << " column " << col << std::endl;
  return std::pair<int,int>(row,col);
}

std::vector<int> HexTopology::neighbours(int cell, int extent) {
  int cols[11] = {0,-1,1,-2,2,-3,3,-4,4,-5,5};
  std::vector<int>   listCell;
  std::pair<int,int> rowCol = findRowCol(cell);
  if (rowCol.first >= 0) {
    int nadd = 2*extent + 1;
    int addc = addCell[rowCol.first];
    for (int i = 0; i <= extent; ++i) {
      for (int j = 0; j < nadd; ++j) {
	int row = rowCol.first + i;
	if (row >= 0 && row <= (int)(nrows)) {
	  int adc = (addc > addCell[row]) ? addc : addCell[row];
	  int col = rowCol.second+cols[j]+leftCell[row]+subCell[row]+adc;
	  if (col >= leftCell[row] && col < leftCell[row+1]) 
	    listCell.push_back(col);
	}
	if (i > 0) {
	  int row = rowCol.first - i;
	  if (row >= 0 && row <= (int)(nrows)) {
	    int adc = (addc > addCell[row]) ? addc : addCell[row];
	    int col = rowCol.second+cols[j]+leftCell[row]+subCell[row]+adc;
	    if (col >= leftCell[row] && col < leftCell[row+1]) 
	      listCell.push_back(col);
	  }
	}
      }
      nadd--;
    }
  }
  return listCell;
}

double HexTopology::cluster(const std::vector<unsigned int>& ids,
			    const std::vector<float>& energies, 
			    unsigned int id, int extent) {
  double energy(0);
  int cell = id&0xFF;
  std::vector<int> listCell = neighbours(cell, extent);
  for (unsigned int k=0; k<ids.size(); ++k) {
    int idcell = ids[k]&0xFF;
    if ((std::find(listCell.begin(),listCell.end(),idcell) != listCell.end()) &&
	((id&0xFFFFFF00) == (ids[k]&0xFFFFFF00))) energy += energies[k];
  }
  return energy;
}

unsigned int HexTopology::localMax(const std::vector<unsigned int>& ids,
				   const std::vector<float>& energy, 
				   unsigned int layer) {
  float        emax(0);
  unsigned int idmax(0);
  for (unsigned int k=0; k<ids.size(); ++k) {
    if (layer == ((ids[k]>>19)&0x7F)) {
      if (energy[k] > emax) {
	emax  = energy[k];
	idmax = ids[k];
      }
    }
  }
  return idmax;
}
  
class HexGeometry {

public :
  HexGeometry(bool fine);
  virtual ~HexGeometry() {}

  std::pair<double,double> position(const int cell);

private :
  std::vector<std::pair<double,double> > xypos;

};

HexGeometry::HexGeometry(bool fine) {
  const int nC(15), nF(20);
  int nCoarse(11), nyCoarse(-42), nFine(15), nyFine(-56);
  int cellCoarse[nC] = {2,5,8,11,12,11,12,11,12,11,12,11,8,5,2};
  int cellFine[nF] = {3,6,9,12,15,16,15,16,15,16,15,16,15,16,15,14,11,8,5,2};
  double wafer(123.7);

  int    rows = (fine) ? nF : nC;
  double cell = (fine) ? wafer/nFine : wafer/nCoarse;
  double dx   = 0.5*cell;
  double dy   = 0.5*dx*tan(30.0*M_PI/180.0);
  int    ny   = (fine) ? nyFine : nyCoarse;
  for (int ir = 0; ir < rows; ++ir) {
    int    column = (fine) ? cellFine[ir] : cellCoarse[ir];
    int    nx     = 1 - column;
    double ypos   = dy*ny;
    for (int ic = 0; ic<column; ++ic) {
      double xpos = dx*nx;
      nx += 2;
      xypos.push_back(std::pair<double,double>(xpos,ypos));
    }
    ny += 6;
  }
  std::cout << "Initialize HexGeometry for " << xypos.size() << " cells"
	    << std::endl;
}

std::pair<double,double> HexGeometry::position(const int cell) {
  std::pair<double,double> xy;
  if (cell >= 0 && cell < (int)(xypos.size())) {
    xy = xypos[cell];
  } else {
    xy = std::pair<double,double>(0,0);
  }
  return xy;
}

class HGCTB {
public :
  TTree                *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t                 fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  std::vector<float>        *simHitLayEn1E;
  std::vector<float>        *simHitLayEn2E;
  std::vector<float>        *simHitLayEn1H;
  std::vector<float>        *simHitLayEn2H;
  Double_t                   xBeam, yBeam, zBeam, pBeam;
  std::vector<unsigned int> *simHitCellIdE;
  std::vector<float>        *simHitCellEnE;
  std::vector<unsigned int> *simHitCellIdH;
  std::vector<float>        *simHitCellEnH;

  // List of branches
  TBranch                   *b_simHitLayEn1E;   //!
  TBranch                   *b_simHitLayEn2E;   //!
  TBranch                   *b_simHitLayEn1H;   //!
  TBranch                   *b_simHitLayEn2H;   //!
  TBranch                   *b_xBeam;   //!
  TBranch                   *b_yBeam;   //!
  TBranch                   *b_zBeam;   //!
  TBranch                   *b_pBeam;   //!
  TBranch                   *b_simHitCellIdE;   //!
  TBranch                   *b_simHitCellEnE;   //!
  TBranch                   *b_simHitCellIdH;   //!
  TBranch                   *b_simHitCellEnH;   //!

  HGCTB(std::string inName, std::string outName);
  virtual ~HGCTB();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop(std::string calFile, std::string wtFile, 
			unsigned int nLayers, float beamE, float cutETotByEBeam,
			int ntotEvent, int clusterEn=1, int nPrintEvent=100);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  std::string      outName_;
};

HGCTB::HGCTB(std::string inName, std::string outName) : fChain(0),
							outName_(outName) {
  TFile      *file = new TFile(inName.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny("HGCalTBAnalyzer");
  TTree      *tree = (TTree*)dir->Get("HGCTB");
  Init(tree);
}

HGCTB::~HGCTB() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t HGCTB::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t HGCTB::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (!fChain->InheritsFrom(TChain::Class()))  return centry;
  TChain *chain = (TChain*)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void HGCTB::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  
  // Set object pointer
  simHitLayEn1E = 0;
  simHitLayEn2E = 0;
  simHitLayEn1H = 0;
  simHitLayEn2H = 0;
  simHitCellIdE = 0;
  simHitCellEnE = 0;
  simHitCellIdH = 0;
  simHitCellEnH = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("simHitLayEn1E", &simHitLayEn1E, &b_simHitLayEn1E);
  fChain->SetBranchAddress("simHitLayEn2E", &simHitLayEn2E, &b_simHitLayEn2E);
  fChain->SetBranchAddress("simHitLayEn1H", &simHitLayEn1H, &b_simHitLayEn1H);
  fChain->SetBranchAddress("simHitLayEn2H", &simHitLayEn2H, &b_simHitLayEn2H);
  fChain->SetBranchAddress("xBeam",         &xBeam,         &b_xBeam);
  fChain->SetBranchAddress("yBeam",         &yBeam,         &b_yBeam);
  fChain->SetBranchAddress("zBeam",         &zBeam,         &b_zBeam);
  fChain->SetBranchAddress("pBeam",         &pBeam,         &b_pBeam);
  fChain->SetBranchAddress("simHitCellIdE", &simHitCellIdE, &b_simHitCellIdE);
  fChain->SetBranchAddress("simHitCellEnE", &simHitCellEnE, &b_simHitCellEnE);
  fChain->SetBranchAddress("simHitCellIdH", &simHitCellIdH, &b_simHitCellIdH);
  fChain->SetBranchAddress("simHitCellEnH", &simHitCellEnH, &b_simHitCellEnH);
  Notify();
}

Bool_t HGCTB::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void HGCTB::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t HGCTB::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void HGCTB::Loop(std::string calFile, std::string wtFile, unsigned int nLayers,
		 float beamE, float cutETotByEBeam, int ntotEvent, 
		 int clusterEn, int nPrintEvent) {
  //   In a ROOT session, you can do:
  //      Root > .L HGCTB.C
  //      Root > HGCTB t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch

  if (fChain == 0) return;

  // Read in calibration and layerweights
  std::vector<double> calFactor(nLayers,1.0), layerWeight(nLayers,1.0);
  ifstream infil1, infil2;
  unsigned int line1(0), line2(0);
  infil1.open(calFile.c_str());
  if (infil1.is_open()) {
    while (!infil1.eof()) {
      float MPV;      
      infil1 >> MPV;
      calFactor.push_back(MPV/1.3);
      std::cout << line1 << " MPV: calFactor(MIP) = " << MPV << " : " 
		<< calFactor[line1] << std::endl;
      line1++;
      if (line1 >= nLayers) break;
    }
    infil1.close();
  }
  infil2.open(wtFile.c_str());
  if (infil2.is_open()) {
    while (!infil2.eof()) {
      float weight;      
      infil2 >> weight;
      layerWeight.push_back(weight);
      std::cout << "layerWeight[" << line2 << "] = " << layerWeight[line2] 
		<< std::endl;
      line2++;
      if (line2 >= nLayers) break;
    }
    infil2.close();
  }

  //Create histograms
  TFile *fout = new TFile(outName_.c_str(), "RECREATE");
  int   nbins(40000);
  float rangeGeV(0.05);
  float eLayerMax= (nLayers > 5) ? rangeGeV/calFactor[5] : rangeGeV/calFactor[nLayers-1];
  TH1F* h_SimHitTotEGeV      = new TH1F("h_SimHitTotEdepGeV","",nbins,0.,rangeGeV*nLayers/2);
  TH1F* h_SimHitTotEMIP      = new TH1F("h_SimHitTotEdepMIP","",nbins,0.,eLayerMax*nLayers/2);
  TH1F* h_SimHitTotEtoBE     = new TH1F("h_SimHitTotEdeptoBeamE","",nbins,0.,rangeGeV*nLayers/(4*beamE));
  TH1F* h_SimHitTotEtoBEWt   = new TH1F("h_SimHitTotEdeptoBeamEWt","",nbins,0.,rangeGeV*nLayers/(4*beamE));
  TH1F* h_SimHitTotEMIPCut   = new TH1F("h_SimHitCutTEtoBETotEdepMIP","",nbins,0.,eLayerMax*nLayers/2);
  TH1F* h_SimHitTotEMIPCutWt = new TH1F("h_SimHitCutWtTEtoBETotEdepMIP","",nbins,0.,eLayerMax*nLayers/2);
  std::vector<TH1F*> h_SimHitLayerE, h_SimHitLayerECut, h_SimHitLayerECutWt;
  for (unsigned int i=0; i<nLayers; i++) {
    char str1[20];
    sprintf(str1, "%d", i+1);
    TString hstr1 = str1;
    h_SimHitLayerE.push_back(new TH1F("h_SimHitLayer"+hstr1+"E","",nbins,0.,eLayerMax));
    h_SimHitLayerECut.push_back(new TH1F("h_SimHitCutTEtoBELayer"+hstr1+"E","",nbins,0.,eLayerMax));
    h_SimHitLayerECutWt.push_back(new TH1F("h_SimHitCutWtTEtoBELayer"+hstr1+"E","",nbins,0.,eLayerMax));
  }

  HexTopology ht1(false);

  Long64_t nentries = fChain->GetEntriesFast();
  if (ntotEvent >= 0 && ntotEvent < nentries) nentries = ntotEvent;

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    // Fill histograms
    if (jentry % nPrintEvent == 0) std::cout << "  " << jentry  << "  Events Processed... " << std::endl;

    float TotEdepGeV(0.), TotEdepMIP(0.), TotEdepGeVWt(0.), TotEdepMIPWt(0.);
    std::vector<double> clusterEn2E;
    unsigned int nLayMax = (simHitLayEn2E->size() > nLayers) ? nLayers : simHitLayEn2E->size();
    for (unsigned int ilayer=0; ilayer<nLayMax; ilayer++) {
      unsigned int locMaxId = ht1.localMax((*simHitCellIdE),(*simHitCellEnE),ilayer+1);
      double clusterE2 = ht1.cluster((*simHitCellIdE),(*simHitCellEnE),locMaxId,2);
      clusterEn2E.push_back(clusterE2);
      float edep  = (clusterEn == 0) ? simHitLayEn2E->at(ilayer) : clusterE2;
      float CaliE = edep/calFactor[ilayer];
      TotEdepGeV   += edep;
      TotEdepGeVWt += (edep*layerWeight[ilayer]);
      TotEdepMIP   += CaliE;
      TotEdepMIPWt += CaliE*layerWeight[ilayer];
      h_SimHitLayerE[ilayer]->Fill(CaliE);
      if (jentry % nPrintEvent == 0) 
	std::cout << ilayer+1 << " local Max Id : cluster Energy 2 " 
		  << locMaxId << " : " << clusterE2 << " : " 
		  << (*simHitLayEn2E)[ilayer] << std::endl;
    }
    h_SimHitTotEGeV->Fill(TotEdepGeV);
    h_SimHitTotEMIP->Fill(TotEdepMIP);
    h_SimHitTotEtoBE->Fill(TotEdepGeV/beamE);
    h_SimHitTotEtoBEWt->Fill(TotEdepGeVWt/beamE);
    if ((TotEdepGeV/beamE) > cutETotByEBeam) {
      h_SimHitTotEMIPCut->Fill(TotEdepMIP);
      h_SimHitTotEMIPCutWt->Fill(TotEdepMIPWt);
    }
    for (unsigned int ilayer=0; ilayer<nLayMax; ++ilayer) {
      float CaliE = (clusterEn == 0) ? (simHitLayEn2E->at(ilayer)/calFactor[ilayer]) : (clusterEn2E[ilayer]/calFactor[ilayer]);
      if ((TotEdepGeV/beamE) > cutETotByEBeam) {
	h_SimHitLayerECut[ilayer]->Fill(CaliE);
	h_SimHitLayerECutWt[ilayer]->Fill(CaliE*layerWeight[ilayer]);
      }
    }
  }
  fout->cd(); 
  h_SimHitTotEGeV->Write();     h_SimHitTotEMIP->Write();
  h_SimHitTotEtoBE->Write();    h_SimHitTotEtoBEWt->Write();
  h_SimHitTotEMIPCut->Write();  h_SimHitTotEMIPCutWt->Write();
  for (unsigned int i=0; i<nLayers; i++) {
    h_SimHitLayerE[i]->Write();
    h_SimHitLayerECut[i]->Write();
    h_SimHitLayerECutWt[i]->Write();
  }
  fout->Write(); fout->Close();
}

void PlotHistTBCompare(std::string infile1, std::string infile2, 
		       std::string text, std::string prefix, int maxMod,
		       bool save) {

  std::string name1[6] = {"BeamP","SimHitEn","SimHitTm","DigiADC","DigiLng",
			  "RecHitEn"};
  std::string titl1[6] = {"","SIM","SIM","DIGI","DIGI","RECO"};
  std::string xttl1[6] = {"Beam Momentum (GeV/c)", "Energy (GeV)", "Time (ns)",
			  "ADC", "Layer #", "Energy (GeV)"};
  std::string yttl1[6] = {"Event","Hit","Hit","Hit","Hit"};
  int         type1[6] = {0,1,1,0,0,1};
  int         rebin[6] = {1,1,5,1,1,1};
  double      xmin1[6] = {0,0,0,0,0,0};
  double      xmax1[6] = {150,0.01,100.,100,15,1.0};
  std::string name2[2] = {"DigiOcc","RecHitOcc"};
  std::string xttl2[2] = {"x (cm)", "x (cm)"};
  std::string yttl2[2] = {"y (cm)", "y (cm)"};
  std::string name3[5] = {"SimHitLng","SimHitLng1", "SimHitLng2","RecHitLng",
			  "RecHitLng1"};
  std::string titl3[5] = {"SIM","SIM","SIM","RECO","RECO"};
  std::string xttl3[5] = {"z (cm)","Layer #","Layer #","z (cm)","Layer #"};
  std::string yttl3[5] = {"Mean Energy (GeV)","Mean Energy (Gev)",
			  "Mean Energy (GeV)","Mean Energy (GeV)",
			  "Mean Energy (GeV)"};
  double      xmin3[5] = {10,0,0,0,0};
  double      xmax30[5]= {25,15,15,15,15};
  double      xmax31[5]= {25,50,50,50,50};
  std::string name4[2] = {"SimHitLat","RecHitLat"};
  std::string titl4[2] = {"SIM","RECO"};
  std::string xttl4[2] = {"x (mm)", "x (cm)"};
  std::string yttl4[2] = {"y (mm)", "y (cm)"};
  std::string detect[2]= {"HGCalEESensitive","HGCalHEFSensitive"};
  std::string label[2] = {"Without","With"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(110);  gStyle->SetOptFit(0);
  TFile      *file1 = new TFile(infile1.c_str());
  TFile      *file2 = new TFile(infile2.c_str());
  bool isopen = file2->IsOpen();
  char name[100], namep[100];
  int  color[2] = {2,4};
  int  marker[2]= {20,21};
  for (int k=0; k<6; ++k) {
    if (k == 0) {
      sprintf (name, "%s", name1[k].c_str());
    } else {
      sprintf (name, "%s%s", name1[k].c_str(), detect[0].c_str());
    }
    TH1D* hist1 = (TH1D*)file1->FindObjectAny(name);
    if (hist1 != 0) {
      TLegend *legend = new TLegend(0.50, 0.70, 0.90, 0.79);
      legend->SetFillColor(kWhite);
      sprintf (namep, "c_%s%s", name1[k].c_str(), prefix.c_str());
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if (type1[k] != 0) pad->SetLogy();
      if (rebin[k] > 1) hist1->Rebin(rebin[k]);
      hist1->GetXaxis()->SetTitle(xttl1[k].c_str());
      hist1->GetYaxis()->SetTitle(yttl1[k].c_str());
      hist1->GetXaxis()->SetLabelOffset(0.005);
      hist1->GetXaxis()->SetTitleOffset(1.40);
      hist1->GetYaxis()->SetLabelOffset(0.005);
      hist1->GetYaxis()->SetTitleOffset(1.40);
      hist1->GetXaxis()->SetRangeUser(xmin1[k],xmax1[k]);
      hist1->SetLineColor(color[0]);
      hist1->SetMarkerStyle(marker[0]);
      hist1->SetMarkerColor(color[0]);
      sprintf (namep, "%s (without beam line)", text.c_str());
      legend->AddEntry(hist1,namep,"lp");
      hist1->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(color[0]);
	st1->SetTextColor(color[0]);
	st1->SetY1NDC(0.85); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
      }
      TPaveText *txt1 = new TPaveText(0.80,0.64,0.90,0.69,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      sprintf (txt, "%s", titl1[k].c_str());
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (isopen) {
	hist1 = (TH1D*)file2->FindObjectAny(name);
	if (hist1 != 0) {
	  if (rebin[k] > 1) hist1->Rebin(rebin[k]);
	  hist1->GetXaxis()->SetTitle(xttl1[k].c_str());
	  hist1->GetYaxis()->SetTitle(yttl1[k].c_str());
	  hist1->GetXaxis()->SetLabelOffset(0.005);
	  hist1->GetXaxis()->SetTitleOffset(1.40);
	  hist1->GetYaxis()->SetLabelOffset(0.005);
	  hist1->GetYaxis()->SetTitleOffset(1.40);
	  hist1->GetXaxis()->SetRangeUser(xmin1[k],xmax1[k]);
	  hist1->SetLineColor(color[1]);
	  hist1->SetMarkerStyle(marker[1]);
	  hist1->SetMarkerColor(color[1]);
	  hist1->Draw("sames");
	  sprintf (namep, "%s (with beam line)", text.c_str());
	  legend->AddEntry(hist1,namep,"lp");
	  pad->Update();
	  st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
	  if (st1 != NULL) {
	    st1->SetLineColor(color[1]);
	    st1->SetTextColor(color[1]);
	    st1->SetY1NDC(0.80); st1->SetY2NDC(0.85);
	    st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	  }
	  pad->Modified();
	  pad->Update();
	}
	legend->Draw("same");
	pad->Update();
      }
      if (save) {
	sprintf (name, "%s.jpg", pad->GetName());
	pad->Print(name);
      }	
    }
  }

  for (int k=0; k<5; ++k) {
    sprintf (name, "%s%s", name3[k].c_str(), detect[0].c_str());
    TProfile* hist1 = (TProfile*)file1->FindObjectAny(name);
    if (hist1 != 0) {
      TLegend *legend = new TLegend(0.50, 0.65, 0.90, 0.73);
      legend->SetFillColor(kWhite);
      sprintf (namep, "c_%s%s", name3[k].c_str(), prefix.c_str());
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      hist1->GetXaxis()->SetTitle(xttl3[k].c_str());
      hist1->GetYaxis()->SetTitle(yttl3[k].c_str());
      hist1->GetXaxis()->SetLabelOffset(0.005);
      hist1->GetXaxis()->SetTitleOffset(1.40);
      hist1->GetYaxis()->SetLabelOffset(0.005);
      hist1->GetYaxis()->SetTitleOffset(1.40);
      double xmax3 = (maxMod == 4) ? xmax30[k] : xmax31[k];
      hist1->GetXaxis()->SetRangeUser(xmin3[k],xmax3);
      hist1->SetLineColor(color[0]);
      hist1->SetMarkerStyle(marker[0]);
      hist1->SetMarkerColor(color[0]);
      sprintf (namep, "%s (without beam line)", text.c_str());
      legend->AddEntry(hist1,namep,"lp");
      hist1->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(color[0]);
	st1->SetTextColor(color[0]);
	st1->SetY1NDC(0.82); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
      }
      TPaveText *txt1 = new TPaveText(0.80,0.59,0.90,0.64,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      sprintf (txt, "%s", titl3[k].c_str());
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (isopen) {
	hist1 = (TProfile*)file2->FindObjectAny(name);
	if (hist1 != 0) {
	  hist1->GetXaxis()->SetTitle(xttl3[k].c_str());
	  hist1->GetYaxis()->SetTitle(yttl3[k].c_str());
	  hist1->GetXaxis()->SetLabelOffset(0.005);
	  hist1->GetXaxis()->SetTitleOffset(1.40);
	  hist1->GetYaxis()->SetLabelOffset(0.005);
	  hist1->GetYaxis()->SetTitleOffset(1.40);
	  double xmax3 = (maxMod == 4) ? xmax30[k] : xmax31[k];
	  hist1->GetXaxis()->SetRangeUser(xmin3[k],xmax3);
	  hist1->SetLineColor(color[1]);
	  hist1->SetMarkerStyle(marker[1]);
	  hist1->SetMarkerColor(color[1]);
	  hist1->Draw("sames");
	  sprintf (namep, "%s (with beam line)", text.c_str());
	  legend->AddEntry(hist1,namep,"lp");
	  pad->Update();
	  st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
	  if (st1 != NULL) {
	    st1->SetLineColor(color[1]);
	    st1->SetTextColor(color[1]);
	    st1->SetY1NDC(0.74); st1->SetY2NDC(0.82);
	    st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	  }
	  pad->Modified();
	  pad->Update();
	}
	legend->Draw("same");
	pad->Update();
      }
      if (save) {
	sprintf (name, "%s.jpg", pad->GetName());
	pad->Print(name);
      }	
    }
  }

  for (int j=0; j<2; ++j) {
    for (int k=0; k<2; ++k) {
      sprintf (name, "%s%s", name2[k].c_str(), detect[0].c_str());
      TH2D* hist1(0);
      if (j==0) hist1 = (TH2D*)file1->FindObjectAny(name);
      else      hist1 = (TH2D*)file2->FindObjectAny(name);
      if (hist1 != 0) {
	TLegend *legend = new TLegend(0.10, 0.86, 0.50, 0.90);
	legend->SetFillColor(kWhite);
	sprintf (namep, "c_%s%s%s", name2[k].c_str(), prefix.c_str(),
		 label[j].c_str());
	TCanvas *pad = new TCanvas(namep, namep, 500, 500);
	pad->SetRightMargin(0.10);
	pad->SetTopMargin(0.10);
	hist1->GetXaxis()->SetTitle(xttl2[k].c_str());
	hist1->GetYaxis()->SetTitle(yttl2[k].c_str());
	hist1->GetXaxis()->SetLabelOffset(0.005);
	hist1->GetXaxis()->SetTitleOffset(1.40);
	hist1->GetYaxis()->SetLabelOffset(0.005);
	hist1->GetYaxis()->SetTitleOffset(1.40);
	hist1->SetLineColor(color[j]);
	hist1->SetMarkerStyle(marker[j]);
	hist1->SetMarkerColor(color[j]);
	sprintf (namep, "%s (%s beam line)", text.c_str(), label[j].c_str());
	legend->AddEntry(hist1,namep,"lp");
	hist1->Draw("lego");
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetLineColor(color[j]);
	  st1->SetTextColor(color[j]);
	  st1->SetY1NDC(0.82); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	TPaveText *txt1 = new TPaveText(0.80,0.76,0.90,0.81,"blNDC");
	txt1->SetFillColor(0);
	char txt[100];
	sprintf (txt, "%s", titl3[k].c_str());
	txt1->AddText(txt);
	txt1->Draw("same");
	pad->Modified();
	pad->Update();
	legend->Draw("same");
	pad->Update();
	if (save) {
	  sprintf (name, "%s.jpg", pad->GetName());
	  pad->Print(name);
	}
      }
    }
  }

  for (int j=0; j<2; ++j) {
    for (int k=0; k<2; ++k) {
      sprintf (name, "%s%s", name4[k].c_str(), detect[0].c_str());
      TProfile2D* hist1(0);
      if (j==0) hist1 = (TProfile2D*)file1->FindObjectAny(name);
      else      hist1 = (TProfile2D*)file2->FindObjectAny(name);
      if (hist1 != 0) {
	TLegend *legend = new TLegend(0.10, 0.86, 0.50, 0.90);
	legend->SetFillColor(kWhite);
	sprintf (namep, "c_%s%s%s", name4[k].c_str(), prefix.c_str(),
		 label[j].c_str());
	TCanvas *pad = new TCanvas(namep, namep, 500, 500);
	pad->SetRightMargin(0.10);
	pad->SetTopMargin(0.10);
	hist1->GetXaxis()->SetTitle(xttl4[k].c_str());
	hist1->GetYaxis()->SetTitle(yttl4[k].c_str());
	hist1->GetXaxis()->SetLabelOffset(0.001);
	hist1->GetXaxis()->SetLabelSize(0.028);
	hist1->GetXaxis()->SetTitleOffset(1.40);
	hist1->GetYaxis()->SetLabelOffset(0.001);
	hist1->GetYaxis()->SetLabelSize(0.028);
	hist1->GetYaxis()->SetTitleOffset(1.40);
	hist1->SetLineColor(color[j]);
	hist1->SetMarkerStyle(marker[j]);
	hist1->SetMarkerColor(color[j]);
	sprintf (namep, "%s (%s beam line)", text.c_str(), label[j].c_str());
	legend->AddEntry(hist1,namep,"lp");
	hist1->Draw("lego");
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetLineColor(color[j]);
	  st1->SetTextColor(color[j]);
	  st1->SetY1NDC(0.82); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	TPaveText *txt1 = new TPaveText(0.80,0.76,0.90,0.81,"blNDC");
	txt1->SetFillColor(0);
	char txt[100];
	sprintf (txt, "%s", titl4[k].c_str());
	txt1->AddText(txt);
	txt1->Draw("same");
	pad->Modified();
	pad->Update();
	legend->Draw("same");
	pad->Update();
	if (save) {
	  sprintf (name, "%s.jpg", pad->GetName());
	  pad->Print(name);
	}
      }
    }
  }
}

void PlotHistTBHitEn(std::string infile, std::string text, std::string prefix, 
		     int type, bool separate, int rebin, int maxMod, bool save){

  std::string name1[2] = {"SimHitEnA","SimHitEnB"};
  std::string title[2] = {"SIM Layer","Layer"};
  std::string xtitl[2] = {"Energy (GeV)", "Energy (GeV)"};
  std::string ytitl[2] = {"Tracks","Tracks"};
  std::string detect[2]= {"HGCalEESensitive","HGCalHEFSensitive"};
  std::string label[2] = {"Without","With"};
  int         start[2] = {0, 1};
  int         nhmax0[2]= {12,4};
  int         nhmax1[2]= {48,16};
  int         color[6] = {1,2,4,7,6,9};
  int         ltype[6] = {1,1,1,1,1,1};
  int         mtype[6] = {20,21,22,23,24,33};
 
  if (type != 1) type = 0;
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);       gStyle->SetOptFit(0);
  TFile      *file = new TFile(infile.c_str());

  for (int k=0; k<2; ++k) {
    int nhmax = (maxMod == 4) ? nhmax0[k] : nhmax1[k];
    double dxs = 0.80/nhmax;
    if ((dxs > 0.15) || separate) dxs = 0.15;
    double dyl = (separate) ? 0.025 : 0.025*nhmax;
    double xhi(0.90), yhi(0.81);
    TCanvas* pad(0);
    TLegend* legend(0);
    char     name[100], namep[100];
    if (!separate) {
      sprintf (namep,"%s%s%s",name1[k].c_str(),prefix.c_str(),label[type].c_str());
      pad    = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10); pad->SetTopMargin(0.10); pad->SetLogy();
      legend = new TLegend(0.70, yhi-dyl, 0.90, yhi);
      legend->SetFillColor(kWhite);
    }
    TPaveText *text1 = new TPaveText(0.60,yhi-dyl-0.04,0.90,yhi-dyl-0.01,"blNDC");
    text1->SetFillColor(0);
    sprintf (namep, "%s (%s beam line)",text.c_str(),label[type].c_str());
    text1->AddText(namep);
    bool first(true);
    for (int j=0; j<nhmax; ++j) {
      sprintf(name,"%s%d%s",name1[k].c_str(),(start[k]+j),detect[0].c_str());
      TH1D* hist = (TH1D*)file->FindObjectAny(name);
      if (hist != 0) {
	if (separate) {
	  sprintf (namep,"%s%s%s%d",name1[k].c_str(),prefix.c_str(),label[type].c_str(),j);
	  pad    = new TCanvas(namep, namep, 700, 500);
	  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10); pad->SetLogy();
	  legend = new TLegend(0.70, 0.81-dyl, 0.90, 0.81);
	  legend->SetFillColor(kWhite);
	}
	int j1 = j%6;
	int j2 = (j-j1)/6;
	hist->Rebin(rebin);
	hist->GetXaxis()->SetTitle(xtitl[k].c_str());
	hist->GetYaxis()->SetTitle(ytitl[k].c_str());
	hist->GetXaxis()->SetLabelOffset(0.005);
	hist->GetXaxis()->SetTitleOffset(1.40);
	hist->GetYaxis()->SetLabelOffset(0.005);
	hist->GetYaxis()->SetTitleOffset(1.40);
	hist->SetLineColor(color[j1]);
	hist->SetLineStyle(ltype[j2]);
	hist->SetMarkerStyle(mtype[j1]);
	hist->SetMarkerColor(color[j1]);
	double xmax = (maxMod == 4) ? 0.025 : 0.040;
	hist->GetXaxis()->SetRangeUser(0.0,xmax);
	sprintf (namep, "%s %d",title[k].c_str(),(j+1));
	legend->AddEntry(hist,namep,"lp");
	if (separate || first) hist->Draw();
	else                   hist->Draw("sames");
	first = false;
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetLineColor(color[j1]);
	  st1->SetTextColor(color[j1]);
	  st1->SetY1NDC(0.82); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(xhi-dxs); st1->SetX2NDC(xhi);
	  if (!separate) xhi -= dxs;
	}
	pad->Update();
	if (separate) {
	  legend->Draw("same");
	  text1->Draw("same");
	  pad->Update();
	  if (save) {
	    sprintf (name, "%s.jpg", pad->GetName());
	    pad->Print(name);
	  }
	}
      }
    }
    if (!separate) {
      legend->Draw("same");
      text1->Draw("same");
      pad->Update();
      if (save) {
	sprintf (name, "%s.jpg", pad->GetName());
	pad->Print(name);
      }
    }
  }
}

void PlotHistSimDigRec(std::string fname, std::string text, int type,
		       std::string prefix, int maxMod, bool save) {

  std::string name1[6] = {"BeamP", "SimHitEnHGCalEESensitive", 
			  "SimHitTmHGCalEESensitive",
			  "DigiADCHGCalEESensitive",
			  "DigiLngHGCalEESensitive",
			  "RecHitEnHGCalEESensitive"};
  std::string name2[1] = {"DigiOccHGCalEESensitive"};
  std::string name3[5] = {"SimHitLngHGCalEESensitive",
			  "SimHitLng1HGCalEESensitive",
			  "SimHitLng2HGCalEESensitive",
			  "RecHitLngHGCalEESensitive",
			  "RecHitLng1HGCalEESensitive"};
  std::string name4[2] = {"SimHitLatHGCalEESensitive",
			  "RecHitLatHGCalEESensitive"};
  double xrnglo1[6] = {  0.0,0.0,  0.0, 0.0, 0.0,0.0};
  double xrnghi1[6] = {150.0,0.1,200.0,20.0,50.0,0.1};
  double xrnglo2[1] = {-10.0};
  double xrnghi2[1] = { 10.0};
  double xrnglo3[5] = {10.0, 0.0, 0.0,10.0, 0.0};
  double xrnghi30[5]= {50.0,20.0,20.0,50.0,20.0};
  double xrnghi31[5]= {50.0,50.0,50.0,50.0,20.0};
  double xrnglo4[2] = {-40.0,-10.0};
  double xrnghi4[2] = { 40.0, 10.0};
  int    type1[6]   = {0,1, 1,1,0,1};
  int    type2[6]   = {1,1,10,1,1,1};
  std::string xtitl1[6] = {"Beam momentum", "SimHit energy (GeV)", 
			   "Hit time (ns)", "ADC (Digi)", 
			   "Longitudinal profile", "RecHit energy (GeV)"};
  std::string ytitl1[6] = {"Events", "Hits", "Hits", "Hits", " ", "Hits"};
  std::string title1[6] = {"", "SIM", "SIM", "DIGI", "DIGI", "RECO"};
  std::string xtitl3[5] = {"z (mm)", "Layer #", "Layer #", "z (cm)", "Layer #"};
  std::string ytitl3[5] = {"Mean Energy (GeV)", "Mean Energy (GeV)",
			   "Mean Energy (GeV)", "Mean Energy (GeV)",
			   "Mean Energy (GeV)"};
  std::string title3[5] = {"SIM", "SIM", "SIM", "RECO", "RECO"};
  std::string xtitl2[1] = {"x (cm)"};
  std::string ytitl2[1] = {"y (cm)"};
  std::string title2[1] = {"Digis"};
  std::string xtitl4[2] = {"x (mm)", "x (cm)"};
  std::string ytitl4[2] = {"y (mm)", "y (cm)"};
  std::string title4[2] = {"SimHits","RecHits"};
  std::string label[2] = {"Without","With"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(1110);
  TFile      *file = new TFile(fname.c_str());
  if (file) {
    char name[100];
    TPaveText *text1 = new TPaveText(0.15,0.91,0.70,0.95,"blNDC");
    text1->SetFillColor(0);
    sprintf (name, "%s (%s beam line)",text.c_str(),label[type].c_str());
    text1->AddText(name);
    for (int k=0; k<6; ++k) {
      TH1D* hist = (TH1D*)file->FindObjectAny(name1[k].c_str());
      //	std::cout << name1[k] << " read out at " << hist << std::endl;
      if (hist != 0) {
	sprintf (name,"%s%s",prefix.c_str(),name1[k].c_str());
	TCanvas *pad = new TCanvas(name,name,500,500);
	pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	hist->GetYaxis()->SetTitle(ytitl1[k].c_str());
	hist->GetXaxis()->SetTitle(xtitl1[k].c_str());
	hist->SetTitle(title1[k].c_str()); hist->Rebin(type2[k]);
	hist->GetXaxis()->SetRangeUser(xrnglo1[k],xrnghi1[k]);
	hist->GetYaxis()->SetTitleOffset(1.4);
	if (type1[k] == 1) pad->SetLogy();
	hist->Draw();
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetFillColor(0);
	  st1->SetY1NDC(0.81); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	pad->Modified();
	pad->Update();
	if (save) {
	  sprintf (name, "c_%s%s.gif", prefix.c_str(), name1[k].c_str());
	  pad->Print(name);
	}
      }
    }
    for (int k=0; k<5; ++k) {
      TProfile* hist = (TProfile*)file->FindObjectAny(name3[k].c_str());
      //	std::cout << name3[k] << " read out at " << hist << std::endl;
      if (hist != 0) {
	sprintf (name,"%s%s",prefix.c_str(),name3[k].c_str());
	TCanvas *pad = new TCanvas(name,name,500,500);
	pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	hist->GetYaxis()->SetTitle(ytitl3[k].c_str());
	hist->GetXaxis()->SetTitle(xtitl3[k].c_str());
	hist->SetMarkerStyle(20); hist->SetMarkerSize(0.8);
	hist->SetTitle(title3[k].c_str()); 
	double xrnghi3 = (maxMod == 4) ? xrnghi30[k] : xrnghi31[k];
	hist->GetXaxis()->SetRangeUser(xrnglo3[k],xrnghi3);
	hist->GetYaxis()->SetTitleOffset(1.4);
	hist->Draw();
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetFillColor(0);
	  st1->SetY1NDC(0.75); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	text1->Draw("same");
	pad->Modified();
	pad->Update();
	if (save) {
	  sprintf (name, "c_%s%s.gif", prefix.c_str(), name3[k].c_str());
	  pad->Print(name);
	}
      }
    }
    for (int k=0; k<1; ++k) {
      TH2D* hist = (TH2D*)file->FindObjectAny(name2[k].c_str());
      //	std::cout << name2[k] << " read out at " << hist << std::endl;
      if (hist != 0) {
	sprintf (name,"%s%s",prefix.c_str(),name2[k].c_str());
	TCanvas *pad = new TCanvas(name,name,500,500);
	pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	hist->GetYaxis()->SetTitle(ytitl2[k].c_str());
	hist->GetXaxis()->SetTitle(xtitl2[k].c_str());
	hist->GetXaxis()->SetRangeUser(xrnglo2[k],xrnghi2[k]);
	hist->GetYaxis()->SetRangeUser(xrnglo2[k],xrnghi2[k]);
	hist->SetMarkerStyle(20); hist->SetMarkerSize(0.2);
	hist->SetTitle(title2[k].c_str());
	hist->GetXaxis()->SetTitleOffset(1.4);
	hist->GetYaxis()->SetTitleOffset(1.4);
	hist->Draw();
	text1->Draw("same");
	if (save) {
	  sprintf (name, "c_%s%s.gif", prefix.c_str(), name2[k].c_str());
	  pad->Print(name);
	}
      }
    }
    for (int k=0; k<2; ++k) {
      TProfile2D* hist = (TProfile2D*)file->FindObjectAny(name4[k].c_str());
      //	std::cout << name4[k] << " read out at " << hist << std::endl;
      if (hist != 0) {
	sprintf (name,"%s%s",prefix.c_str(),name4[k].c_str());
	TCanvas *pad = new TCanvas(name,name,500,500);
	pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	hist->GetYaxis()->SetTitle(ytitl4[k].c_str());
	hist->GetXaxis()->SetTitle(xtitl4[k].c_str());
	hist->GetXaxis()->SetRangeUser(xrnglo4[k],xrnghi4[k]);
	hist->GetYaxis()->SetRangeUser(xrnglo4[k],xrnghi4[k]);
	hist->SetMarkerStyle(20); hist->SetMarkerSize(0.2);
	hist->SetTitle(title4[k].c_str());
	hist->GetXaxis()->SetTitleOffset(1.4);
	hist->GetYaxis()->SetTitleOffset(1.4);
	hist->Draw();
	text1->Draw("same");
	if (save) {
	  sprintf (name, "c_%s%s.gif", prefix.c_str(), name2[k].c_str());
	  pad->Print(name);
	}
      }
    }
  }
}

void testTopology(bool fine) {
  HexTopology ht(fine);
  std::string ans;
  int cellmax = (fine) ? 239 : 132;
  int cell(-1), cellmin(0);
  std::cout << "Input Cell: ";
  std::cin >> cell;
  if (cell >= 0) cellmin = cellmax = cell;
  for (cell = cellmin; cell <= cellmax; ++cell) {
    for (int ext = 0; ext < 3; ++ext) {
      std::vector<int> listCell = ht.neighbours(cell,ext);
      std::cout << "Extent " << ext << " for Cell " << cell << " has "
		<< listCell.size() << " cells" << std::endl;
      for (unsigned int k=0; k<listCell.size(); ++k) 
	std::cout << " " << listCell[k];
      std::cout << std::endl;
      std::cout << "Want to continue:";
      std::cin  >> ans;
      if (ans == "q" || ans == "Q" || ans == "s" || ans == "S") break;
    }
    if (ans == "q" || ans == "Q") break;
  }
}

bool inside(double x, double y) {

  const double size(6.185);
  const double tan30deg(0.5773502693);
  double absx = std::abs(x);
  double absy = std::abs(y);
  bool   flag(false);

  if (absx <= size && absy <= (2.*size*tan30deg)) {
    if (absy <= size*tan30deg || absx <= (2.*size - absy/tan30deg)) flag = true;
  }
  return flag;
}

void testGeometry() {

  HexGeometry geomc(false);
  for (int k = 0; k < 133; ++k) {
    std::pair<double,double> xy = geomc.position(k);
    std::cout << "Coarse Cell[" << k << "] " << xy.first << ":" << xy.second
	      << std::endl;
  }

  HexGeometry geomf(true);
  for (int k = 0; k < 240; ++k) {
    std::pair<double,double> xy = geomf.position(k);
    std::cout << "Fine Cell[" << k << "] " << xy.first << ":" << xy.second
	      << std::endl;
  }
}
