/*
*
*    Tomasz Fruboes,  IPJ Warsaw 2007
*
*/
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>


#include "Validation/MuonRPCGeometry/interface/PTStatistics.h"
#include "Validation/MuonRPCGeometry/test/style.h"

#include "TH2D.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TFile.h"

/*************************************************
*
* Ugly global data definitions
*
**************************************************/
//                     0     1     2     3     4      5
double etas[] = {-0.07, 0.07, 0.27, 0.44, 0.58, 0.72,
   //                  6     7     8     9     10     11 
                  0.83, 0.93, 1.04, 1.14, 1.24, 1.36,
                  //   12    13    14    15    16                
                 1.48,  1.61, 1.73, 1.85, 1.97, 2.10};

double pts[] = {
   0.0,  0.01,    
//1    2     3    4     5      6     7    8
   1.5,  2.0, 2.5,  3.0,  3.5,  4.0,  4.5, 
//    9     10    11    12
   5.,   6.,   7.,   8.,  
//
   10.,  12., 14.,  16.,  18.,  
   20.,  25.,  30., 35.,  40.,  45., 
   50.,  60.,  70., 80.,  90.,  100., 120., 140.};
/*************************************************
*
* Data structures filled when reading muons.txt
*
**************************************************/
struct SMuonBinPos{
   SMuonBinPos():  m_binNo(-1), m_tower(0) {};
   SMuonBinPos(int tower, int bin):  m_binNo(bin), m_tower(tower) {};
   int m_binNo;
   int m_tower;
   
   bool operator<(const SMuonBinPos & mp1) const{
   
      if (this->m_binNo != mp1.m_binNo){
         return (this->m_binNo < mp1.m_binNo);
      } 
      
      return this->m_tower < mp1.m_tower;
   };
};

struct SMuonBin {
   SMuonBin(): m_rate(-1) {};
   PTStatistics m_stats;
   std::map<int, PTStatistics> m_qual2stats; // m_stats = sum of m_qual2stats->seconds 
   long double m_rate;
};

typedef std::map<SMuonBinPos,SMuonBin> TEffMap;
/*************************************************
*
* Fwd definitions
*
**************************************************/
double rate(double x);
float getPt(unsigned int ptCode);
double binRate(double pt1, double pt2);
int  getBinNo(double pt);
int getPtCode(double pt);
void getBinBounds(int binNo, double & pt1, double & pt2);
void initRates(TEffMap & effMap);
void ratesVsPtcut(TEffMap & effMap);
void rateVsTower(TEffMap & effMap,int ptCut);
void draw(TEffMap & effMap, int tower1, int tower2, TCanvas  & c1, int padNo=0);
void actCurves(TEffMap & effMap);
void drawActCur(TEffMap & effMap, int towerMin, int towerMax, TCanvas  & c1, int ptCut, int padNo=0);
int eta2tower(double eta);
void drawKar(TEffMap & effMap, int ptCut, int option=0);
void effAndPurity(TEffMap & effMap, int option, int towerBeg, int towerEnd);
/*************************************************
*
* Read the data, call functions for drawing
*
**************************************************/
int main(int argc, char *argv[]){
   
   std::string fname; 
   if (argc!=2){
      fname = "/tmp/fruboes/muons.txt";
      std::cout<< " Trying to use defualt: " << fname << std::endl;
   } else {
      fname = argv[1];
      std::cout<< " File: " << fname << std::endl;
   }
        

   std::ifstream fIn(fname.c_str());
   if (!fIn.is_open()) {
       std::cout << "could not open the file" << std::endl;  
       return 1;
   }

   double etaGen;
   double phiGen;
   double ptGen;
   int towerRec, phiRec, ptCodeRec, qual, muonsFound;
   TEffMap effMap;
   TEffMap effMapMC;
   int read = 0;
   //int toRead = 1000000;
   int toRead = -1;
   int ghost = 0;
//   while (fIn >> etaGen >> phiGen >> ptGen
//              >> towerRec >> phiRec >> ptCodeRec >> qual 
//              >> muonsFound)

  // 0.453722 -2.06621 53.1866 3 95 31 3 0

 /*                        m_outfileR << etaGen << " " << phiGen << " " << ptGen
                                    << " "    << towerRec << " " << phiRec << " " << ptCodeRec << " " << qual
                                    << " "    << ghost
                                    << std::endl;
*/

   while( fIn  >> etaGen >> phiGen >> ptGen
       >> towerRec >> phiRec >> ptCodeRec >> qual
       >> ghost)


   {
      if (read>toRead && toRead!=-1)
         break;

      if (ghost!=0) continue; // TODO: add ghost histo
      if (std::abs(etaGen)>2.1) continue; 
      if (ptGen < 1.7)  continue ; 
  
      ++read;
      if (read%100000==0){
         std::cout << " Read: " << read << std::endl;
      }
      int t = towerRec;
      if (ptCodeRec == 0) { // avoid putting events without muons in tower = 0
         t = eta2tower(etaGen);
      }
      /* 
      if (read < 1000) {
       std::cout << "p "<<  t << " " << std::abs(t) << " " << " " << ptGen << " " << getBinNo(ptGen) << std::endl;
       std::cout << "pMC "<< etaGen << " " << eta2tower(etaGen) << " " << ptGen << " " << getBinNo(ptGen) << std::endl;
       std::cout << " qual "  << qual << " ptCodeRec " << ptCodeRec << std::endl;
      }*/

      SMuonBinPos p(std::abs(t), getBinNo(ptGen) );
      SMuonBinPos pMC( eta2tower(etaGen) , getBinNo(ptGen) );
   //   ++(effMap[p].m_stats.at(ptCodeRec)); // ptCodeRec==0 - no muon found
      ++(effMap[p].m_qual2stats[qual].at(ptCodeRec));
      ++(effMapMC[pMC].m_qual2stats[qual].at(ptCodeRec));
   }
   std::cout << " Finished. Read: " << read << std::endl;
 
   TFile * hfile = new TFile("plots.root","RECREATE","Demo ROOT file with histograms");  

   // initilize some data after readout 
   initRates(effMap);
   initRates(effMapMC);

   // set style for plots
   setTDRStyle();   

   // Draw rate Vs pt cut plots. Plenty of plots 
   ratesVsPtcut(effMap);
   
   // draw rate vs tower for variuos cuts (9,13,...) 
   rateVsTower(effMap,9);
   rateVsTower(effMap,13);
   rateVsTower(effMap,18);
   rateVsTower(effMap,24);
   rateVsTower(effMap,31);
   // draw efficiency for various cuts of towers. Plenty of plots
   actCurves(effMap);
 
   // draw efficiency for different cuts 
   //  (note 1 means no cut) 
   drawKar(effMapMC, 1);
   drawKar(effMapMC, 5);
   drawKar(effMapMC, 10);
   drawKar(effMapMC, 15);
   drawKar(effMapMC, 20);
   drawKar(effMapMC, 25);
   drawKar(effMapMC, 30);
   drawKar(effMapMC, 31);
   
   // Same as above, draw with "text" option
   drawKar(effMapMC, 1,1);
   drawKar(effMapMC, 5,1);
   drawKar(effMapMC, 10,1);
   drawKar(effMapMC, 15,1);
   drawKar(effMapMC, 20,1);
   drawKar(effMapMC, 25,1);
   drawKar(effMapMC, 30,1);
   drawKar(effMapMC, 31,1);
   
   // effciency and purity
   effAndPurity(effMapMC,1,0,8); // barrel
   effAndPurity(effMapMC,1,9,16);// endacp

   // save to root file 
   hfile->Write();
   
   return 0;
}
/*************************************************
*
*  
*
**************************************************/
void actCurves(TEffMap & effMap){
   
   std::vector<int> cuts;
   cuts.push_back(5);
   cuts.push_back(10);
   cuts.push_back(15);
   cuts.push_back(20);
   cuts.push_back(25);
   cuts.push_back(29);
   cuts.push_back(30);
   cuts.push_back(31);
   
   TCanvas c1("actCanvas","",600,500);
   std::stringstream ss;
   std::vector<int>::iterator itC = cuts.begin();
   
   for(;itC!=cuts.end();++itC){
   
      // Barrel
      c1.Clear();
      drawActCur(effMap, 0, 7, c1, *itC);
      ss.clear();
      ss.str("");
      ss << "akt" << *itC << "t0_7.pdf";
      c1.Print(ss.str().c_str());
      
      
      // Endcap
      c1.Clear();
      drawActCur(effMap, 8, 16, c1, *itC);
      ss.clear();
      ss.str("");
      ss << "akt" << *itC << "t8_16.pdf";
      c1.Print(ss.str().c_str());
      
      // All towers    
      for (int tower = 0; tower < 17; ++tower){
         c1.Clear();
         drawActCur(effMap, tower, tower, c1, *itC);
         ss.clear();
         ss.str("");
         ss << "akt" << *itC << "t" << tower <<".pdf";
         c1.Print(ss.str().c_str());
      }
      
   }
   
   
}

/*************************************************
*
* Rates vs ptCut  
*
**************************************************/
void ratesVsPtcut(TEffMap & effMap){
  
   
   TCanvas c1;
   draw(effMap, 0, 7, c1);
   c1.Print("t0_7.pdf");
   
   c1.Clear();
   draw(effMap, 8, 16, c1);
   c1.Print("t8_16.pdf");
   
   for (int i = 0; i<17; ++i){
      c1.Clear();
      draw(effMap, i, i, c1);
      std::stringstream ss;
      ss << "t" << i << ".pdf";
      c1.Print(ss.str().c_str());
   }
   
   c1.Clear();
   c1.Divide(2,3);
   for (int i = 0; i<6; ++i){
      draw(effMap, i, i, c1, i+1);
   }
   c1.Print("t0t5.pdf");
   
   c1.Clear();
   c1.Divide(2,3);
   for (int i = 6; i<12; ++i){
      draw(effMap, i, i, c1, i+1-6);
   }
   c1.Print("t6t11.pdf");
   
   c1.Clear();
   c1.Divide(2,3);
   for (int i = 12; i<17; ++i){
      draw(effMap, i, i, c1, i+1-12);
   }
   c1.Print("t12t16.pdf");

   return;
}
/*************************************************
*
* Rates vs tower
*
**************************************************/
void rateVsTower(TEffMap & effMap,int ptCodeCut){

   
   double pt = pts[ptCodeCut];
   

   
   std::vector<double> rateVsTower(17,0);
   
   
   TEffMap::iterator it=effMap.begin();
   TEffMap::iterator itend=effMap.end();
   
   /*
   typedef std::vector<int> TQlCnt; 
   TQlCnt qlCnt(8,0); // 8 qualities max
   typedef std::vector<TQlCnt> TTow2QlCnt; 
   TTow2QlCnt tow2qlCnt(17,qlCnt);
   */
   
   typedef std::vector<int> TQinTowerCnt; // count occurences of given quality in towers
   TQinTowerCnt qInTCnt(17,0); 
   typedef std::vector<TQinTowerCnt> TQl2Tow; 
   TQl2Tow tow2qlCnt(4,qInTCnt);// 4 qualities max
   
   for(;it!=itend;++it){
      int tower = it->first.m_tower;
      double oldR = rateVsTower.at(tower);
      rateVsTower.at(tower)+=it->second.m_rate*it->second.m_stats.eff(ptCodeCut);
      /*
      if (rateVsTower.at(tower)/oldR > 10 ) {
           std::cout << oldR << " " << rateVsTower.at(tower) 
                     << " m_rate " << it->second.m_rate 
                     << " eff " << it->second.m_stats.eff(ptCodeCut)
                     << " m_binNo " << it->first.m_binNo
                     << " m_tower " << it->first.m_tower
                     << std::endl;
      }*/


      std::map<int, PTStatistics>::iterator it1 = it->second.m_qual2stats.begin();
      std::map<int, PTStatistics>::iterator it1end = it->second.m_qual2stats.end();
      for (;it1!=it1end;++it1){
         //tow2qlCnt.at(tower).at(it1->first)+=it1->second.sum(ptCodeCut);
         tow2qlCnt.at(it1->first).at(tower)+=it1->second.sum(ptCodeCut);
      }
   }

   std::stringstream name;
   name << "aRate_ptCodeCut_" << ptCodeCut;
   
   TH1D *hist = new TH1D(name.str().c_str(), "a", 17, -0.5, 16.5);
   
   TCanvas c1;
   
   //c1.cd(padNo);
   c1.SetTicks(1,1);
   c1.SetLogy();
   c1.SetGrid();


   hist->SetStats(false);
   hist->GetYaxis()->SetTitle("Rate [Hz]");
   hist->GetXaxis()->SetTitle("Tower");

   for(unsigned int i=0; i<rateVsTower.size();++i){
            hist->SetBinContent(i+1,rateVsTower.at(i));
   }
   TLegend * leg = new TLegend(0.2,0.87,0.6,0.94);
   std::stringstream stitle;
   stitle << "Rate in towers with ptCode cut " << ptCodeCut << "(" << pt << "GeV/c)";
   leg->SetHeader(stitle.str().c_str());

   hist->Draw();
   leg->Draw();
 
   
   std::stringstream ss;
   ss << "rateVsTow_cut" << ptCodeCut << ".pdf";
   c1.Print(ss.str().c_str());
   //delete hist;
   delete leg;
   
   // draw stacked rates with qualities shown
   TQinTowerCnt sum(17,0);
   for (unsigned int qual = 0; qual< tow2qlCnt.size();++qual){
      for (unsigned int tower = 0; tower< tow2qlCnt.at(qual).size();++tower){
         sum.at(tower)+=tow2qlCnt.at(qual).at(tower);
      }
   }

   THStack *hs = new THStack(ss.str().c_str(),"blabbla");
   
   //leg = new TLegend(0.4,0.6,0.89,0.89);
   leg = new TLegend(0.2,0.6,0.4,0.94);
   
   leg->SetHeader("Jakosc:");
   
   
   for (unsigned int qual = 0; qual< tow2qlCnt.size();++qual){
//   for (unsigned int qual = tow2qlCnt.size()-1; qual >= 0;--qual){
      std::stringstream name;
      name << "Rate_ptCodeCut_" << ptCodeCut << "_qual_" << qual;
      TH1D *hist = new TH1D(name.str().c_str(), "a", 17, -0.5, 16.5);
      bool empty = true;
      for (unsigned int tower = 0; tower< tow2qlCnt.at(qual).size();++tower){
         if (sum.at(tower)!=0){
            double bc = double(tow2qlCnt.at(qual).at(tower))/sum.at(tower)*rateVsTower.at(tower);
            hist->SetBinContent(tower+1,bc);
            empty=false;
         }
      }
      if (!empty){
         hist->SetFillColor(qual+1);
         hs->Add(hist);
         std::stringstream lname;
         lname << qual;
         leg->AddEntry(hist,lname.str().c_str(),"f");
      }

   }
   

   c1.Clear();
   hs->Draw();
   leg->Draw();
   hs->GetYaxis()->SetTitle("Czestosc [Hz]");
   hs->GetXaxis()->SetTitle("Wieza");
   std::stringstream ssa;
   ssa << "crateVsTow_cut" << ptCodeCut << ".pdf";
   c1.Print(ssa.str().c_str());
   
   return;
}
/*************************************************
*
* Muon rate vs ptCut in whole detector region
*
**************************************************/
#include <cmath>
double rate(double x){


   /*
   double ret = 0;
   double a = -0.235801;
   double b = -2.82346;
   double c = 17.162;

   //f1(x) =  x**(a*log(x)) *(x**b)*exp(c)
  //ret = std::pow( x,a*std::log(x) ) * std::pow(x,b)*std::exp(c)l
   ret = std::pow( x,a*std::log(x) ) * std::pow(x,b)*std::exp(c);
   return ret;
*/


 const double lum = 2.0e33; //defoult is 1.0e34;
 const double dabseta = 1.0;
 const double dpt = 1.0;
 const double afactor = 1.0e-34*lum*dabseta*dpt;
 const double a  = 2*1.3084E6;
 const double mu=-0.725;
 const double sigma=0.4333;
 const double s2=2*sigma*sigma; 
 double ptlog10;
 ptlog10 = std::log10(x);
 double ex = (ptlog10-mu)*(ptlog10-mu)/s2;
 double rate = (a * exp(-ex) * afactor); 

 return rate;

}
/*************************************************
*
* Int. rate  between pt1 and pt2 for given tower
*
**************************************************/
double binRate(double pt1, double pt2, int tower){


         
   tower = std::abs(tower);
   if (tower > 16){
      std::cout << "Bad tower " << tower << std::endl;
      throw "Bad tower";
   }
   
   
   double etaTowerSize = etas[tower+1]-etas[tower];
   double detEta = 2.1;

   //double ret = rate(pt2)-rate(pt1);
   double ret =( rate(pt2)-rate(pt1) ) /2/(pt2-pt1) ;
   
   if (ret<0){
      ret = -ret;
   }

   return ret*etaTowerSize/detEta;

}
/*************************************************
*
* Converts pt into bin number. 
*
*     -> Should be method of SMuonBinPos
*     -> getBinBounds() has the same internal data 
*
**************************************************/
int  getBinNo(double pt){
   
   //return getPtCode(pt);
   
   int ret = 0;
   int nbins = 1000;
   //int nbins = 200;
   double ptLow = 1.49;
   //double ptHigh = 201.;
   double ptHigh = 200.;
   double delta = ptHigh-ptLow;
   double binSize = delta/nbins;
   
   while (ret < nbins){
   
      double lowEdge = ptLow+ret*binSize;
      if (lowEdge>pt)
         break;
      
      ++ret;
   }

   return ret;
}
/*************************************************
*
* Converts bin number into pt range.
*
*     -> Should be method of SMuonBinPos
*     -> getBinNo() has the same internal data
*
**************************************************/
void getBinBounds(int binNo, double & pt1, double & pt2){

   /*
   pt1 = getPt(binNo);
   if(binNo == 31) 
      pt2 = 200.;
   else  
      pt2 = getPt(binNo+1);
         
   return;      
   */

   int nbins = 1000;
   //int nbins = 200;
   double ptLow = 1.49;
   double ptHigh = 200.;
   double delta = ptHigh-ptLow;
   double binSize = delta/nbins;
   pt1 = ptLow+binNo*binSize;
   pt2 = ptLow+(binNo+1)*binSize;

}
/*************************************************
*
* Initilize data after readout
*
**************************************************/
void initRates(TEffMap & effMap){

   TEffMap::iterator it=effMap.begin();
   TEffMap::iterator itend=effMap.end();
   
   for(;it!=itend;++it){
      double p1 = -1;
      double p2 = -1;
      getBinBounds(it->first.m_binNo,p1,p2);
      it->second.m_rate = binRate(p1,p2,it->first.m_tower);
      /*
      if (it->second.m_rate > 1000000) {
        std::cout << " initRates "<< p1 << " " << p2 << " " << it->first.m_tower << " " << binRate(p1,p2,it->first.m_tower) << std::endl; 
      }*/


     //*
      std::map<int, PTStatistics>::iterator it1 = it->second.m_qual2stats.begin();
      std::map<int, PTStatistics>::iterator it1end = it->second.m_qual2stats.end();
      for (;it1!=it1end;++it1){
         it->second.m_stats.update(it1->second);
      }//*/


      
   }


}
/*******************************************************************************************
*
*  Draws rate vs ptCut hist
*
*******************************************************************************************/
void draw(TEffMap & effMap, int towerMin, int towerMax, TCanvas  & c1, int padNo){

   //std::cout << "D " << towerMin << " " << towerMax << std::endl;
   
   std::vector<double> rateVsPtCut(32,0);

   TEffMap::iterator it=effMap.begin();
   TEffMap::iterator itend=effMap.end();
   
   for(;it!=itend;++it){
      int tower = it->first.m_tower;
      if ( tower < towerMin || tower > towerMax){
         continue;
      }
      for(int i=0;i<32;++i){ // `i` is ptCut
         rateVsPtCut.at(i)+=it->second.m_rate*it->second.m_stats.eff(i);
      }
      
   }


   std::stringstream ss,fname;
   
   if (towerMin==towerMax){
      ss << "Tower: " << towerMin;
   } else {
      ss << "Towers: " << towerMin << "..." << towerMax;
   }
   

   
   c1.cd(padNo);
   c1.SetTicks(1,1);
   c1.SetLogy();
   c1.SetGrid();

   fname << "RateInT" << towerMin << "_" << towerMax << ".pdf";
   TH1D *hist = new TH1D(fname.str().c_str(), ss.str().c_str(), 31, 0.5, 31.5);
   hist->SetStats(false);
   hist->GetYaxis()->SetTitle("czestosc [Hz]");
   hist->GetXaxis()->SetTitle("ptCodeCut");

   for(unsigned int i=1; i<rateVsPtCut.size();++i){
      //std::cout << i << " " << rateVsPtCut.at(i) << std::endl;
      hist->SetBinContent(i,rateVsPtCut.at(i));
   }
   //TLegend * leg = new TLegend(0.84,0.88,0.97,0.93);
   TLegend * leg = new TLegend(0.8,0.88,0.97,0.93);
   leg->SetHeader(ss.str().c_str());
   

   hist->Draw();
   leg->Draw();
   
   //delete hist;
   //delete leg;
}
/*******************************************************************************************
*
*  Draws effIciency vs ptGen for various ptCuts
*
*******************************************************************************************/
#include "TAxis.h"
void drawActCur(TEffMap & effMap, int towerMin, int towerMax, TCanvas  & c1, int ptCut, int padNo){

   
   //std::cout << "Akt " << towerMin << " " << towerMax << std::endl;

   int bins = getBinNo(10000000)+1; // returns max_bin bin no 0 is the first one
   //std::vector<double> effVsPt(bins,0);
   //std::vector<int> cnt(bins,0);
   
   std::vector<int> rec(bins,0);
   std::vector<int> all(bins,0);
   
   

   TEffMap::iterator it=effMap.begin();
   TEffMap::iterator itend=effMap.end();
   
   for(;it!=itend;++it){
      int tower = it->first.m_tower;
      if ( tower < towerMin || tower > towerMax){
         continue;
      }
      int bin = it->first.m_binNo;
      //effVsPt.at(bin)+=it->second.m_stats.eff(ptCut);
      //++cnt.at(bin);
      rec.at(bin)+=it->second.m_stats.sum(ptCut);
      all.at(bin)+=it->second.m_stats.sum();
   }


   std::stringstream ss,fname;
   
   if (towerMin==towerMax){
      ss << "Tower: " << towerMin;
   } else {
      ss << "Towers: " << towerMin << "..." << towerMax;
   }
   
   c1.cd(padNo);
   c1.SetTicks(1,1);
//   c1.SetLogy();
   c1.SetLogx();
   c1.SetGrid();

   fname << "akt" <<  ptCut << "inT" << towerMin << "_" << towerMax << ".pdf";
   
   //double ptLow = -1, ptHigh = -1, trash = -1;
   //getBinBounds(0,ptLow,trash);
   //getBinBounds(bins-1,trash,ptHigh);
   //TH1D *hist = new TH1D(fname.str().c_str(), ss.str().c_str(), bins, ptLow, ptHigh);
   
   double xbins[1000];
   double ptLow=0, ptHigh=0;
   for(int i = 0; i < bins; ++i){
      getBinBounds(i,ptLow,ptHigh);
      xbins[i]=ptLow;
   }
   xbins[bins]=ptHigh; // Fill the high edge of last bin
   
   TH1D *hist = new TH1D(fname.str().c_str(), ss.str().c_str(), bins, xbins);
   hist->SetStats(false);
   hist->GetYaxis()->SetTitle("efektywnosc");
   hist->GetXaxis()->SetTitle("pT [GeV/c]");
   hist->SetAxisRange(1.5,200.);

   //for(unsigned int i=0; i<effVsPt.size();++i){
   for(unsigned int i=0; i<all.size();++i){
      /*
      if(cnt.at(i)!=0){
         hist->SetBinContent(i+1,effVsPt.at(i)/cnt.at(i));
      }*/
      int div = all.at(i);
      if (div > 0){
         hist->SetBinContent(i+1,double(rec.at(i))/div);
      }
   }

   /*
   double pts[30] = {
//      0.0,  0.01,    
//      1.5,  2.0, 2.5,  3.0,  3.5,  4.0,  4.5, 
      2.0, 2.5,  3.0,  3.5,  4.0,  4.5,  // 6
      5.,   6.,   7.,   8.,   // 4
      10.,  12., 14.,  16.,  18., // 5 
      20.,  25.,  30., 35.,  40.,  45., // 5
      50.,  60.,  70., 80.,  90.,  100., 120., 140., 200.};//9

   */

   std::stringstream fnew;
   fnew << "aktR" <<  ptCut << "inT" << towerMin << "_" << towerMax << ".pdf";
   //hist = (TH1D *)hist->Rebin(5, fnew.str().c_str());
   //hist->Scale(0.2);
   hist->Draw();
   hist ->GetYaxis()->SetRangeUser(0.,1.);

   /*
//   hist ->GetXaxis()->SetRangeUser(2.0,200.0);
   //*/

   
   
   //TLegend * leg = new TLegend(0.84,0.88,0.97,0.93);
   //TLegend * leg = new TLegend(0.8,0.88,0.97,0.93);
   //leg->SetHeader(ss.str().c_str());
   //leg->Draw();
   
   
   //delete hist;
   //delete leg;
}
/*************************************************
*
*
*
**************************************************/
void drawKar(TEffMap & effMap, int ptCut, int option){
   
   static TEffMap newmap;
   static bool firstRun = true;
         
   if (firstRun){
      //std::cout << " FIRST run" << std::endl;
      firstRun = false;
      TEffMap::iterator it=effMap.begin();
      TEffMap::iterator itend=effMap.end();
      int tower = 0, bin = 0;
      double r1 = 0, r2 = 0;
      for(;it!=itend;++it){
         tower = it->first.m_tower;
         bin = it->first.m_binNo;
         getBinBounds(bin, r1, r2);
         int ptCode = getPtCode( (r1+r2)/2 );
         /*std::cout << r1 
                   << " " << r2
                   << " " << (r1+r2)/2
                   << " " << ptCode 
                   << std::endl;*/
         SMuonBinPos p(tower, ptCode );
         // copy the statistics into proper bin of newmap 
         newmap[p].m_stats.update(it->second.m_stats);
      }
   }
   
   // Fill the histogram
   std::stringstream sname;
   if (option == 0){
      sname << "effkar" << ptCut << ".pdf";
   } else { // (option == 1){
      sname << "effkarTT" << ptCut << ".pdf";
   }
   
   int m_TOWER_COUNT = 17, m_PT_CODE_MAX = 31; // From RPCConst
   TH2D *eff = new TH2D( sname.str().c_str(), "",
                         m_TOWER_COUNT, -0.5, m_TOWER_COUNT-0.5, //Towers
                         m_PT_CODE_MAX+1, -0.5, m_PT_CODE_MAX+0.5);// Pt's

   TEffMap::iterator it=newmap.begin();
   TEffMap::iterator itend=newmap.end();
   for(;it!=itend;++it){
      // XXX Tower may be negative...
      /*std::cout << it->first.m_tower
                << " " << it->first.m_binNo
                << " " <<it->second.m_stats.eff(ptCut)
                << std::endl;*/
      eff->SetBinContent(it->first.m_tower+1, it->first.m_binNo+1,it->second.m_stats.eff(ptCut));
   }

   // draw it and save to file
   //TCanvas c1("canbaaas","",800,600);
   TCanvas c1;
   gStyle->SetPalette(1,0);
   eff->SetStats(false);
   if (option == 0){
     eff->Draw("COLZ");
   } else { // (option == 1){
      eff->Draw("TEXT");
   }
   eff-> SetMaximum(1);
   eff-> SetMinimum(0);
   eff->GetXaxis()->SetTitle("tower generated");
   eff->GetYaxis()->SetTitle("ptCode generated");
   c1.Print(sname.str().c_str());

   
}
/*************************************************
*
*
*
**************************************************/
int eta2tower(double eta){

   bool neg = (eta < 0 ? true : false);
   if (neg) {
      eta = - eta;
   }
   if (eta < 0){
      std::cout << "Eta neg " << eta << std::endl;
      throw "Eta negative";
    }

   if (eta > 2.101) {
      std::cout << "Eta big " << eta << std::endl;
      throw "Eta to big";
   }
   
   
   int tower = 0;
   while (tower < 16 && eta > etas[tower+1] ) ++tower;
      
   return (neg ? -tower : tower);
 
}
/*************************************************
*
*
*
**************************************************/
int getPtCode(double pt){

   int ret = 0;
   while ( pt>pts[ret+1]  && ret < 31) ++ret;

   return ret;
}

float getPt(unsigned int ptCode){

   if (ptCode > 31){
      std::cout << "Ptcode big " << ptCode << std::endl;
      throw "Bad ptCode";
   }
   
   return pts[ptCode];
}
/*************************************************
*
* Draw efficiency and purity. Both use rate() for 
*  weighting. Since rate goes exp. down with pt
*  it may be wrong for efficiency plot 
*
**************************************************/
void effAndPurity(TEffMap & effMap, int option, int towerBeg, int towerEnd){

   TEffMap::iterator it=effMap.begin();
   TEffMap::iterator itend=effMap.end();

   
   // Rate wieghted calculations
   typedef std::vector<double> TPtCode2Rate;
   typedef std::vector<TPtCode2Rate> TTower2PtCode;
   
   TPtCode2Rate ini(32,0); // 31 ptCodes
   TTower2PtCode accepted(17,ini); // 17 towers  // #1
   TTower2PtCode shouldBeAccepted(17,ini); // 17 towers // #2
   TTower2PtCode wasAndShouldBeAccepted(17,ini); // 17 towers  // #3
   
   float ptCut = 0;
   double pt1 = 0, pt2 = 0;
   int tower = -1;
   for(;it!=itend;++it){
   
      for (int ptCodeCut = 1; ptCodeCut < 32; ++ptCodeCut){ // 
        ptCut = getPt(ptCodeCut); 
        tower = std::abs(it->first.m_tower);
        getBinBounds(it->first.m_binNo,pt1,pt2);
        
        double aRate = 0;
        int sum = it->second.m_stats.sum();
        if (sum != 0){
          aRate = (double)it->second.m_stats.sum(ptCodeCut); 
          aRate/=sum;
          aRate*=it->second.m_rate;
          accepted[tower][ptCodeCut]+=aRate;
          
        } 

        
        if ( (pt1+pt2)/2 > ptCut){
           shouldBeAccepted[tower][ptCodeCut]+=it->second.m_rate;
           wasAndShouldBeAccepted[tower][ptCodeCut]+=aRate;
           

        }
      }
   

   }
   
   
   
   std::stringstream pname;
   if (option == 0){
      pname << "pur"<< towerBeg << "_" << towerEnd << ".pdf";
   } else { // (option == 1){
      pname << "purTT"<< towerBeg << "_" << towerEnd << ".pdf";;
   }
   
   std::stringstream ename;
   if (option == 0){
      ename << "eff"<< towerBeg << "_" << towerEnd << ".pdf";;
   } else { // (option == 1){
      ename << "effTT"<< towerBeg << "_" << towerEnd << ".pdf";;
   }

   
   
   int m_TOWER_COUNT = towerEnd-towerBeg+1; 
   int m_PT_CODE_MAX = 31; // From RPCConst
   int ptCodeMin = 5;
   TH2D *eff = new TH2D( ename.str().c_str(), "",
                         m_TOWER_COUNT, towerBeg-0.5, towerEnd+0.5, //Towers
                         m_PT_CODE_MAX+1-ptCodeMin, ptCodeMin-0.5, m_PT_CODE_MAX+0.5);// Pt's
   
   TH2D *pur = new TH2D( pname.str().c_str(), "",
                         m_TOWER_COUNT, towerBeg-0.5, towerEnd+0.5, //Towers
                         m_PT_CODE_MAX+1-ptCodeMin, ptCodeMin-0.5, m_PT_CODE_MAX+0.5);// Pt's


   
   double div = 0;
   for (int ptCodeCut = ptCodeMin; ptCodeCut < 32; ++ptCodeCut){ 
      for (int tower = towerBeg; tower <= towerEnd; ++tower){ 
         
         double dEff = -1;
         // use unweighted values for efficiency calculations
         div = shouldBeAccepted[tower][ptCodeCut];
         if (div > 0) {
           dEff = wasAndShouldBeAccepted[tower][ptCodeCut];
           dEff /= div;
         } 
         
         double dPur = -1;
         // use weighted values for efficiency calculations
         div = accepted[tower][ptCodeCut];
         if (div > 0){
            dPur = wasAndShouldBeAccepted[tower][ptCodeCut];
            dPur /= div;
         }      
         
         eff->SetBinContent(tower-towerBeg+1, ptCodeCut+1-ptCodeMin, dEff);
         pur->SetBinContent(tower-towerBeg+1, ptCodeCut+1-ptCodeMin, dPur);

      }
   
   }

   //TCanvas c1("effCanv", "", 800, 500);
   TCanvas c1;
   gStyle->SetPalette(1,0);
   eff->SetStats(false);
   if (option == 0){
      eff->Draw("COLZ");
   } else { // (option == 1){
      eff->Draw("TEXT");
   }
   eff-> SetMaximum(1);
   eff-> SetMinimum(0);
   eff-> SetMarkerSize(2);
   eff->GetXaxis()->SetTitle("Wieza");
   eff->GetXaxis()->SetTickLength(0);
   eff->GetYaxis()->SetTitle("ptCodeCut");
   eff->GetYaxis()->SetTickLength(0.01);
   c1.Print(ename.str().c_str());
   
   TCanvas c2("purCanv", "", 800, 500);
   //TCanvas c2;
   c2.SetGridy();
   gStyle->SetPalette(1,0);
   pur->SetStats(false);
   if (option == 0){
      pur->Draw("COLZ");
   } else { // (option == 1){
      pur->Draw("TEXT");
   }
   pur-> SetMaximum(1);
   pur-> SetMinimum(0);
   pur-> SetMarkerSize(1.7);
   //pur-> Scale(1000);
   pur->GetXaxis()->SetTitle("Wieza");
   pur->GetXaxis()->SetTickLength(0);
   pur->GetYaxis()->SetTitle("ptCodeCut");
   pur->GetYaxis()->SetTickLength(0.01);
   c2.Print(pname.str().c_str());
   
   
}
