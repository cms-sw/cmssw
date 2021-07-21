#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <cmath>
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "FWCore/Utilities/interface/Exception.h"

const double ptRanges[RPCConst::m_PT_CODE_MAX + 2] = {0.0, 0.01, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,  4.5,  5.,   6.,
                                                      7.,  8.,   10., 12., 14., 16., 18., 20.,  25.,  30.,  35.,
                                                      40., 45.,  50., 60., 70., 80., 90., 100., 120., 140., 160.};

class RPCHistos {
public:
  RPCHistos();
  ~RPCHistos();
  void go();
  TH2D *eff();
  void drawActCurv(int towerFrom, int towerTo);
  TH1D *drawActCurv(int ptCut, int towerFrom, int towerTo);
  void rateVsPt();
  double frate(double p);
  double rate(int pt, int tower);
  double rate(int pt);
  void phiEff();
  void etaCorr();

  void setFormat(std::string fm) { m_ext = fm; };

private:
  std::ifstream m_ifile;
  std::string m_ext;
  std::map<int, TH2D *> m_tower2pt;

  TH1D *m_phiEffRec;  // efficienct vs eta
  TH1D *m_phiEffAll;  // efficienct vs eta
  TH2F *m_etaCorr;

  RPCConst m_const;
};

RPCHistos::RPCHistos() {
  m_ext = "png";
  m_ifile.open("muons.txt");
  if (!m_ifile) {
    throw cms::Exception("IOError") << "Bkah\n";
  }

  for (int i = -RPCConst::m_TOWER_COUNT + 1; i < RPCConst::m_TOWER_COUNT; ++i) {
    std::stringstream ss;
    ss << i;
    m_tower2pt[i] = new TH2D(ss.str().c_str(),
                             "",
                             RPCConst::m_PT_CODE_MAX + 1,
                             -0.5,
                             RPCConst::m_PT_CODE_MAX + 0.5,  // gen
                             RPCConst::m_PT_CODE_MAX + 1,
                             -0.5,
                             RPCConst::m_PT_CODE_MAX + 0.5);  // rec
  }

  m_etaCorr = new TH2F("etaCorr", "", 35, -17, 17, 35, -17, 17);

  //int bins = 440;
  int bins = 110;
  m_phiEffRec = new TH1D("phiEffRec", "efficiency vs eta", bins, -2.2, 2.2);
  m_phiEffAll = new TH1D("phiEffAll", "eff vs eta", bins, -2.2, 2.2);
}

/****************************************************************
*
* dtor
*
*****************************************************************/
RPCHistos::~RPCHistos() {
  // std::for_each(m_tower2pt.begin(), m_tower2pt.end(), std::with_data(delete));

  std::map<int, TH2D *>::iterator it = m_tower2pt.begin();
  for (; it != m_tower2pt.end(); ++it) {
    delete it->second;
  }

  delete m_phiEffRec;
  delete m_phiEffAll;
}
/****************************************************************
*
* `Do the work` function
*
*****************************************************************/
void RPCHistos::go() {
  float etaGen = 0, ptGen = 0, phiGen = 0;
  int ptCodeRec = 0, towerRec = 0, phiRec = 0, ghost = 0, qual = 0;

  std::cout << "Readout start" << std::endl;
  int count = 0;
  while (1) {
    if ((++count) % 20000 == 0)
      std::cout << " Read: " << count << std::endl;

    if (count == 1000000)
      break;
    // -1.01966 -1.67928 99.9912 7 34 31 2

    /*      
                m_outfileR << etaGen << " " << phiGen << " " << ptGen << " "
                            << towerRec << " " << phiRec << " " << ptCodeRec << " " << qual<< " "
                            << ghost
  
   */
    m_ifile >> etaGen >> phiGen >> ptGen >> towerRec >> phiRec >> ptCodeRec >> qual >> ghost;

    if (!m_ifile.good())
      break;
    int ptCodeGen = m_const.iptFromPt(ptGen);
    int towerGen = m_const.towerNumFromEta(etaGen);
    m_tower2pt[towerGen]->Fill(ptCodeGen, ptCodeRec);

    // PhiEff plot, rate plot
    //    float etaGenAbs = etaGen > 0 ? etaGen : -etaGen;
    //m_phiEffAll->Fill(etaGenAbs);
    m_phiEffAll->Fill(etaGen);
    if (ghost == 0 && ptCodeRec != 0) {
      // m_phiEffRec->Fill(etaGenAbs);
      m_phiEffRec->Fill(etaGen);
      m_etaCorr->Fill(towerGen, towerRec);
    }
  }

  std::cout << "Readout finished. Read " << count << std::endl;
}
// integrated rate for pt > p, whole detector (|eta|<2.1)
double RPCHistos::frate(double p) {
  double a = -0.235802;  //        +/- 0.01494      (6.335%)
  double b = -2.82345;   //         +/- 0.09143      (3.238%)
  double c = 17.162;     //           +/- 0.1342       (0.7822%)

  float ret = pow(p, (a * log(p))) * pow(p, b) * exp(c);

  if (ret <= 0)
    throw cms::Exception("float") << "Lower then zero \n";

  return ret;
}

double RPCHistos::rate(int ptCode, int tower) {
  if (tower > 16 || tower < -16)
    throw cms::Exception("data") << " Bad tower\n ";

  double etaNorm = 2.1 * 2;

  std::vector<float> towerEtaSize;
  towerEtaSize.push_back((0.07 - 0.00) * 2);  //T0
  towerEtaSize.push_back(0.27 - 0.07);        //1
  towerEtaSize.push_back(0.44 - 0.27);        //2
  towerEtaSize.push_back(0.58 - 0.44);        //3
  towerEtaSize.push_back(0.72 - 0.58);        //4
  towerEtaSize.push_back(0.83 - 0.72);        //5
  towerEtaSize.push_back(0.93 - 0.83);        //6
  towerEtaSize.push_back(1.04 - 0.93);        //7
  towerEtaSize.push_back(1.14 - 1.04);        //8
  towerEtaSize.push_back(1.24 - 1.14);        //9
  towerEtaSize.push_back(1.36 - 1.24);        //10
  towerEtaSize.push_back(1.48 - 1.36);        //11
  towerEtaSize.push_back(1.61 - 1.48);        //12
  towerEtaSize.push_back(1.73 - 1.61);        //13
  towerEtaSize.push_back(1.85 - 1.73);        //14
  towerEtaSize.push_back(1.97 - 1.85);        //15
  towerEtaSize.push_back(2.10 - 1.97);        //16

  return rate(ptCode) * towerEtaSize.at(std::abs(tower)) / etaNorm;
}
double RPCHistos::rate(int ptCode) {
  // Is it ok??
  if (ptCode == 0)
    return 0;

  if (ptCode > 31 || ptCode < 0)
    throw cms::Exception("data") << " Bad ptCode " << ptCode << "\n";

  double pts[] = {
      0.0, 0.01, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.,  6.,  7.,  8.,   10.,  12.,  14.,  16.,
      18., 20.,  25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140., 1000.};  // Make 1 TeV highest (inf ;)

  //std::cout<< pts[32] << std::endl;
  double r2 = frate(pts[ptCode + 1]);
  double r1 = frate(pts[ptCode]);

  return r1 - r2;
}
/****************************************************************
*
* rate vs ptCut histo
*
****************************************************************/
void RPCHistos::rateVsPt() {
  //  m_rateVsptCut->Divide(m_rateVsptCutAll);

  TH2D *th2eff = eff();

  for (int x = 1; x <= 16 + 1; ++x)  // tower
  {
    // 1 means ptCode 0
    for (int y = 1; y <= 31 + 1; ++y) {  // ptCode
      double bincont = th2eff->GetBinContent(x, y);
      //bincont*=m_const.vxIntegMuRate(y-1,x-1);
      bincont *= rate(y - 1, x - 1);
      th2eff->SetBinContent(x, y, bincont);
    }
  }

  // Sum rate in all towers;
  TH1D *th1rate = th2eff->ProjectionY("ptRate", 1, 18);
  TH1D *genRate = new TH1D("genRate", "", RPCConst::m_PT_CODE_MAX + 1, -0.5, RPCConst::m_PT_CODE_MAX + 0.5);

  double gratesum = 0;
  for (int y = 1; y <= 31 + 1; ++y) {
    gratesum += rate(y - 1);
  }

  for (int y = 1; y <= 31 + 1; ++y) {
    //     genRate->SetBinContent(y, m_const.vxMuRate(y-1));
    gratesum -= rate(y - 1);
    genRate->SetBinContent(y, gratesum);
  }

  //th1rate->Multiply(genRate);
  // Make integrated rate
  double records = 0, sum = 0;

  //*
  for (int i = 1; i <= 31 + 1; ++i) {
    records += th1rate->GetBinContent(i);
  }

  for (int i = 1; i <= 31 + 1; ++i) {
    double sumDelta = th1rate->GetBinContent(i);
    double cont = (records - sum);
    sum += sumDelta;
    th1rate->SetBinContent(i, cont);
  }  //*/
     //////////////////
     ///////////////

  for (int ptCut = 0; ptCut <= 31; ++ptCut) {
    double arate = 0;
    //for (int tower = 0; tower < 17; ++tower){
    for (int tower = 0; tower < 13; ++tower) {
      TH1D *cur = drawActCurv(ptCut, tower, tower);
      for (int i = 2; i <= RPCConst::m_PT_CODE_MAX + 1; ++i) {
        double cont = cur->GetBinContent(i);
        arate += cont * rate(i - 1, tower);
      }
      delete cur;
    }
    std::cout << "TT " << ptCut << " " << arate << std::endl;
    th1rate->SetBinContent(ptCut + 1, arate);
  }

  TCanvas c1;
  c1.SetLogy();
  c1.SetLogx();

  th1rate->SetBins(RPCConst::m_PT_CODE_MAX + 1, ptRanges);
  genRate->SetBins(RPCConst::m_PT_CODE_MAX + 1, ptRanges);
  th1rate->SetLineColor(1);
  genRate->SetLineColor(2);

  th1rate->SetStats(false);
  genRate->SetStats(false);

  //th1rate->GetXaxis()->SetRange(1,100);
  //genRate->GetXaxis()->SetRange(1,100);
  th1rate->Draw("");
  genRate->Draw("SAME");
  th1rate->GetXaxis()->SetRangeUser(2, 100);

  th1rate->SetMaximum(6000000);
  th1rate->SetMinimum(0.1);
  //genRate->GetXaxis()->SetRangeUser(1,100);

  th1rate->GetYaxis()->SetTitle("rate [Hz]");
  th1rate->GetXaxis()->SetTitle("ptCodeCut");

  TLegend *legend = new TLegend(0.66, 0.15 + 0.5, 0.86, 0.37 + 0.5);
  legend->AddEntry(th1rate, "RPC", "l");
  legend->AddEntry(genRate, "Generated rate", "l");
  legend->Draw();
  std::string fname = "rate." + m_ext;
  c1.Print(fname.c_str());
}
/****************************************************************
*
* Reco vs eta histo 
*
****************************************************************/
void RPCHistos::phiEff() {
  m_phiEffRec->Divide(m_phiEffAll);
  TCanvas c1;
  m_phiEffRec->SetStats(false);
  m_phiEffRec->Draw("");
  m_phiEffRec->SetMaximum(1);
  m_phiEffRec->SetMinimum(0);
  m_phiEffRec->GetYaxis()->SetTitle("rpc trg efficiency");
  m_phiEffRec->GetXaxis()->SetTitle("eta generated");
  std::string fname = "phiEff." + m_ext;
  c1.Print(fname.c_str());
}
/****************************************************************
*
* Eta corelation
*
****************************************************************/
void RPCHistos::etaCorr() {
  TCanvas c1;
  m_etaCorr->SetFillColor(2);
  m_etaCorr->GetXaxis()->SetTitle("tower generated");
  m_etaCorr->GetYaxis()->SetTitle("tower reconstructed");
  m_etaCorr->SetStats(false);
  m_etaCorr->Draw("BOX");

  std::string filename = "etacorr." + m_ext;
  c1.Print(filename.c_str());
}
/****************************************************************
*
* Draw efficiency histo (tower vs ptCode). Split draw and return.
*
****************************************************************/
TH2D *RPCHistos::eff() {
  TH2D *reco2d = new TH2D("reco2d",
                          "",
                          //RPCConst::m_TOWER_COUNT, -0.5, RPCConst::m_TOWER_COUNT-0.5, //Towers
                          RPCConst::m_TOWER_COUNT * 2 - 1,
                          -RPCConst::m_TOWER_COUNT + 0.5,
                          RPCConst::m_TOWER_COUNT - 0.5,  //Towers
                          RPCConst::m_PT_CODE_MAX + 1,
                          -0.5,
                          RPCConst::m_PT_CODE_MAX + 0.5);  // Pt's
  TH2D *lost2d = new TH2D("lost2d",
                          "",
                          //RPCConst::m_TOWER_COUNT, -0.5, RPCConst::m_TOWER_COUNT-0.5, //Towers
                          RPCConst::m_TOWER_COUNT * 2 - 1,
                          -RPCConst::m_TOWER_COUNT + 0.5,
                          RPCConst::m_TOWER_COUNT - 0.5,  //Towers
                          RPCConst::m_PT_CODE_MAX + 1,
                          -0.5,
                          RPCConst::m_PT_CODE_MAX + 0.5);  // Pt's

  std::map<int, TH2D *>::iterator it = m_tower2pt.begin();
  for (; it != m_tower2pt.end(); ++it) {
    TH1D *reco = it->second->ProjectionX("dd", 2, RPCConst::m_PT_CODE_MAX + 1);
    TH1D *lost = it->second->ProjectionX("lost", 1, 1);  // ptCode 0 is in first bin
    int tower = it->first;
    for (int i = 1; i <= RPCConst::m_PT_CODE_MAX + 1; ++i) {
      //double rcur=reco2d->GetBinContent(std::abs(tower)+1,i);
      double rcur = reco2d->GetBinContent(tower + RPCConst::m_TOWER_COUNT, i);
      double radd = reco->GetBinContent(i);
      //reco2d->SetBinContent(std::abs(tower)+1,i,rcur+radd);
      reco2d->SetBinContent(tower + RPCConst::m_TOWER_COUNT, i, rcur + radd);

      //double lcur=lost2d->GetBinContent(std::abs(tower)+1,i);
      double lcur = lost2d->GetBinContent(tower + RPCConst::m_TOWER_COUNT, i);
      double ladd = lost->GetBinContent(i);
      //lost2d->SetBinContent(std::abs(tower)+1,i,lcur+ladd);
      lost2d->SetBinContent(tower + RPCConst::m_TOWER_COUNT, i, lcur + ladd);
    }

    delete reco;
    delete lost;
  }

  TH2D *all2d = (TH2D *)reco2d->Clone();
  all2d->Add(lost2d);

  reco2d->Divide(all2d);
  TCanvas c1;
  gStyle->SetPalette(1, 0);
  reco2d->SetStats(false);
  reco2d->Draw("COLZ");
  reco2d->SetMaximum(1);
  reco2d->SetMinimum(0);
  reco2d->GetXaxis()->SetTitle("tower generated");
  reco2d->GetYaxis()->SetTitle("ptCode generated");
  std::string fname = "effkar." + m_ext;
  c1.Print(fname.c_str());

  return reco2d;
}

/****************************************************************
*
* Returns act. curve for single ptCut for given towers range
*  (inspiered by Karols implementation)
*
****************************************************************/
TH1D *RPCHistos::drawActCurv(int ptCut, int towerFrom, int towerTo) {
  TH2D *sum = new TH2D("sums",
                       "",
                       RPCConst::m_PT_CODE_MAX + 1,
                       -0.5,
                       RPCConst::m_PT_CODE_MAX + 0.5,  // gen
                       RPCConst::m_PT_CODE_MAX + 1,
                       -0.5,
                       RPCConst::m_PT_CODE_MAX + 0.5);  // rec

  std::map<int, TH2D *>::iterator it = m_tower2pt.begin();
  for (; it != m_tower2pt.end(); ++it) {
    int absTower = std::abs(it->first);
    if (absTower < towerFrom || absTower > towerTo)
      continue;
    sum->Add(it->second);
  }

  std::stringstream name;
  name << ptCut << "_" << towerFrom << "_" << towerTo;
  TH1D *passed = sum->ProjectionX(name.str().c_str(), ptCut + 1, RPCConst::m_PT_CODE_MAX + 1);
  TH1D *all = sum->ProjectionX("lost", 1, RPCConst::m_PT_CODE_MAX + 1);

  std::stringstream title;
  title << "towers " << towerFrom << " - " << towerTo;
  passed->SetTitle(title.str().c_str());

  passed->Divide(all);

  passed->SetBins(RPCConst::m_PT_CODE_MAX + 1, ptRanges);
  delete sum;
  return passed;
}
/****************************************************************
*
* Draws act. curves for given tower range
*
****************************************************************/
void RPCHistos::drawActCurv(int twFrom, int twTo) {
  std::vector<int> ptTresh;
  ptTresh.push_back(9);
  ptTresh.push_back(15);
  ptTresh.push_back(18);
  ptTresh.push_back(22);

  TCanvas c1;
  c1.SetTicks(1, 1);
  c1.SetLogx();
  c1.SetGrid();

  gStyle->SetPalette(1, 0);

  Color_t color = kRed;
  TLegend *legend = new TLegend(0.66, 0.15, 0.86, 0.37);

  for (unsigned int i = 0; i < ptTresh.size(); ++i) {
    ++color;
    TH1D *a1 = drawActCurv(ptTresh.at(i), twFrom, twTo);
    a1->SetLineColor(color);
    if (color == kYellow)
      a1->SetLineColor(46);

    a1->SetStats(false);
    a1->GetXaxis()->SetRangeUser(2, 200);
    a1->Draw("SAME");
    a1->SetMaximum(1);
    a1->SetMinimum(0);
    a1->GetXaxis()->SetTitle("pt generated [GeV]");
    a1->GetYaxis()->SetTitle("efficiency");

    // Legend
    std::stringstream leg;
    leg << "pt >= " << m_const.ptFromIpt(ptTresh.at(i)) << " GeV";
    //leg << "ptCode >= " <<ptTresh.at(i) << " GeV";
    legend->AddEntry(a1, leg.str().c_str(), "l");
  }

  legend->Draw();
  c1.Update();

  std::stringstream fname;
  fname << "akt" << twFrom << "_" << twTo << "." << m_ext;
  c1.Print(fname.str().c_str());
}
/****************************************************************
*
*
*
****************************************************************/
int main() {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(0);
  gStyle->SetOptDate(0);
  gStyle->SetPalette(1);
  gStyle->SetOptFit(0111);
  gStyle->SetOptStat(1000000);

  RPCHistos rh;
  rh.setFormat("png");
  rh.go();
  rh.eff();
  rh.phiEff();
  rh.etaCorr();
  //rh.rateVsPt();
  rh.drawActCurv(0, 7);
  rh.drawActCurv(8, 12);
  return 0;
}
