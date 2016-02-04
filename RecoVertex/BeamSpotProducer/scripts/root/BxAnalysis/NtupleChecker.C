//-----------------------------------------------------
//  Authors: Lorenzo Uplegger  : uplegger@cern.ch
//           Sushil s. Chauhan : sushil@fnal.gov
//
//   Last modified: 30 October 2010
//------------------------------------------------------
// This script read the pv  ntuples from Beam Spot 
// area and make plots for each bx using pv Fitter.
// It also create  a canvas with all the bx plotted for
// each beam spot variable in the output file. 
//-----------------------------------------------------

#include "RecoVertex/BeamSpotProducer/interface/BeamSpotTreeData.h"
#include "RecoVertex/BeamSpotProducer/interface/FcnBeamSpotFitPV.h"
#include "BSFitData.h"

#include "TFitterMinuit.h"
#include "Minuit2/FCNBase.h"
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>
#include <TString.h>
#include <TList.h>
#include <TSystemFile.h>
#include <TSystemDirectory.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <iostream>
#include <fstream>

using namespace std;

//--------Global definiton of some funtions
bool runPVFitter(std::map< int, std::vector<BeamSpotFitPVData> > bxMap_, int NFits, int tmpLumiCounter);
void PlotHistoBX(TString OFN, TString ODN);
void DefineHistStyle(TH1F *h1, int bx);
void FillHist(TH1F* h, int fitN, double pos, double posError, TString lab);
void FillnPVInfo(TH1F* h2,int fitN, float npv, TString lab );
void PlotAllBunches(TCanvas* myCanvas,TH1F* hist[],int bunchN, std::map<int,int> tmpMap_);


std::map<int, std::map< int, std::vector<BeamSpotFitPVData> > > StoreRunLSPVdata_;
std::map<int, map<int, std::vector<BSFitData> > > FitDone_Results_;
std::map<int, TString > FitDone_RunLumiRange_;
std::map<int, map<int, float> > bxMapPVSize_;

ofstream outdata;



//-----Here is the main funtion---------------------------
void NtupleChecker(){

  //----------------------------------------------//
  //                Input parameters              //
  //----------------------------------------------// 
  Int_t beginRunNumber = 162924;           //give -1 if do not want to set lower run limit
  Int_t endRunNumber   = 162930;               //give -1 if do not want to set upper run limit

  Int_t beginLSNumber  = -1;
  Int_t endLSNumber    = -1;

  Int_t FitNLumi       = 100;

  TString OutPutFileName ="BxAnalysis_Fill_1718.root";
  TString OutPutDir      ="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/burkett/BxNtuples/";
  //---------------------------------------------//



  //save few things to ouput log file 
  outdata.open("LogFile.dat");
  cout<<"-----------Step - 1 : Storing Info from Root file to a vector-----------------------"<<endl;
  outdata<<" -----------Step - 1 : Storing Info from Root file to a vector-----------------------"<<endl;

  //clear the map before storing
  StoreRunLSPVdata_.clear();
  FitDone_RunLumiRange_.clear();
  bxMapPVSize_.clear();
  FitDone_Results_.clear();
 
  //set direcotry structure
  //TString path = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/BxNtuples/";
  TString path = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/burkett/BxNtuples/";
  TSystemDirectory sourceDir("hi",path); 
  TList* fileList = sourceDir.GetListOfFiles(); 
  TIter next(fileList);
  TSystemFile* fileName;

  //define file numbers
  int fileNumber = 1;
  int maxFiles = -1;
  BeamSpotTreeData aData;


//-------------------Store all input in a map------------------------- 
  //check filename
  while ((fileName = (TSystemFile*)next()) && fileNumber >  maxFiles){

    if(TString(fileName->GetName()) == "." || TString(fileName->GetName()) == ".."  ){
      continue;
    }
    

    //set tree 
    TTree* aTree = 0;
    TFile file(path+fileName->GetName(),"READ");//STARTUP
    cout << "Opening file: " << path+fileName->GetName() << endl;
    file.cd();
    aTree = (TTree*)file.Get("PrimaryVertices");

    cout << (100*fileNumber)/(fileList->GetSize()-2) << "% of files done." << endl;
    ++fileNumber;
    if(aTree == 0){
      cout << "Can't find the tree" << endl;
      continue;
    }//check if file is not found

  
    aData.setBranchAddress(aTree);
    //ok start reading the tree one b; one
    for(unsigned int entry=0; entry<aTree->GetEntries(); entry++){
     aTree->GetEntry(entry);

     //for all runs and all LS
    if( beginRunNumber == -1 && endRunNumber == -1 ){
        if(beginLSNumber== -1 && endLSNumber == -1){
        BeamSpotFitPVData Input=aData.getPvData();
        StoreRunLSPVdata_[aData.getRun()][aData.getLumi()].push_back(Input);
        }}


     //for Selected Runs and Slected or All LS
     if((aData.getRun()>= beginRunNumber && aData.getRun()<= endRunNumber) ||  (aData.getRun()>= beginRunNumber && endRunNumber == -1) || (beginRunNumber==-1 && aData.getRun()<= endRunNumber) ){
        if((aData.getLumi()<= beginLSNumber && aData.getLumi()>= endLSNumber) || (beginLSNumber ==-1 && endLSNumber ==-1) ){
        BeamSpotFitPVData Input=aData.getPvData(); 
        StoreRunLSPVdata_[aData.getRun()][aData.getLumi()].push_back(Input); 
        }}   

     }//loop over entries of tree

    file.Close();

  }//loop over all the files

  //All the runs
   for( std::map<int, std::map< int, std::vector<BeamSpotFitPVData> > >::iterator p=StoreRunLSPVdata_.begin(); p!=StoreRunLSPVdata_.end(); ++p)
     {  for( std::map< int, std::vector<BeamSpotFitPVData> >::iterator q=p->second.begin(); q!=p->second.end(); ++q)
        {  cout<<" Run Number= "<<p->first<<"   LS Number= "<<q->first<<endl;
           outdata<<" Run Number= "<<p->first<<"   LS Number= "<<q->first<<endl;
        }
     }


  
if(StoreRunLSPVdata_.size()==0)outdata<<" EIther the file is  missing or  it does not contain any data!!!!! Please check  "<<endl;

cout<<"-----------Step - 2 : Now Running the PVFitter for each Bunch Xrossing-----------------------"<<endl;
outdata<<"-----------Step - 2 : Now Running the PVFitter for each Bunch Xrossing-----------------------"<<endl;

//-----------------Run the fitter and store the reulsts for plotting ----------------------------

   Int_t Fit_Done = 0;
   std::vector<BeamSpotFitPVData> pvStore_;
   std::map< int, std::vector<BeamSpotFitPVData> > bxMap_;    //this map for each bx for N lumi
   bxMap_.clear();   

   int Lumi_lo, Lumi_up,RunNumber;
   bool pv_fit_Ok;
   Int_t LastRun=0;
   Int_t LastLumi,LumiCounter,tmpLastLumi;
         LumiCounter=0;


 for( std::map<int, std::map< int, std::vector<BeamSpotFitPVData> > >::iterator RunIt=StoreRunLSPVdata_.begin(); RunIt!=StoreRunLSPVdata_.end(); ++RunIt)
     {  
        if(StoreRunLSPVdata_.size()==0) continue;
 
        if(LastRun==0)LastRun=RunIt->first;
        LumiCounter=0;    
        RunNumber=RunIt->first;
          
        std::map< int, std::vector<BeamSpotFitPVData> >::iterator LastLumiIt;

    for( std::map< int, std::vector<BeamSpotFitPVData> >::iterator LumiIt=RunIt->second.begin(); LumiIt!=RunIt->second.end(); ++LumiIt)
      { 
          LastLumiIt = RunIt->second.end();
          LastLumiIt--;  

          if(LumiCounter==0)Lumi_lo=LumiIt->first;
          LumiCounter++;  

          bool RemainingLSFit=false;

          if((LumiCounter == FitNLumi)  || (LumiIt->first == LastLumiIt->first  && LumiCounter != FitNLumi && LumiCounter >0) ){
                             RemainingLSFit=true;
                             Lumi_up=LumiIt->first;}


    //now loop over pv vectors hence: ->Run->LS->PV/Bx   
    for(size_t pvIt=0; pvIt < (LumiIt->second).size(); pvIt++)
      {
        int bx = (LumiIt->second)[pvIt].bunchCrossing;
        bxMap_[bx].push_back((LumiIt->second)[pvIt]);
        //if(RunNumber ==140123 && LumiIt->first==93 ) cout<<"Has-------->  bx = "<<bx<<endl;
     
     }
 
	if((RemainingLSFit) && (LumiCounter != 0))
         {    RemainingLSFit=false;
              //cout<<"Run Number ="<<RunNumber<<",    Lumi Range = "<<Lumi_lo<<" - "<<Lumi_up<<endl;
              int tmpLumiCounter=0;
              tmpLumiCounter = LumiCounter;

              LumiCounter=0;
              Fit_Done++; 
              if(runPVFitter(bxMap_,Fit_Done,tmpLumiCounter)){

                //store the run : LS range as Tstring
                Char_t RunLSRange[300];
                sprintf(RunLSRange,"%d%s%d%s%d%s",RunNumber,":",Lumi_lo," - ",Lumi_up,"\0");
                TString RunLSRange_(RunLSRange);
                FitDone_RunLumiRange_[Fit_Done]=RunLSRange_;

               }

            bxMap_.clear();

         }//check if Desired LS are collected to perform the fit
         
    
       }//loop over Lumi/pvdata map
    

  } //loop over Run map


cout<<"-----------Step - 3 : Now Filling the histograms for each Bunch Xrossing-----------------------"<<endl;
outdata<<"-----------Step - 3 : Now Filling the histograms for each Bunch Xrossing-----------------------"<<endl;

//-------------------------------Plot the histograms and store them in a root file----------------
       
     PlotHistoBX(OutPutDir, OutPutFileName );

outdata.close(); 
}//NtupleChecker ends here




//--------------------Plot histo after the fit results-----------------------------
void  PlotHistoBX(TString ODN, TString OFN){

   TFile f1(ODN+"/"+OFN,"recreate");


  std::map<int, TString>::iterator RLS;
  RLS=FitDone_RunLumiRange_.begin();
  

  std::map<int, map<int, std::vector<BSFitData> > >::iterator tmpIt=FitDone_Results_.begin();
  std::map<int, std::vector<BSFitData> >::iterator tmpBxIt;
  //tmpBxIt = (tmpIt->second).begin();


  Int_t PointsToPlot, bunchN,bunchN_previous;
  bunchN=0;
  PointsToPlot = FitDone_Results_.size();
  //cout<<"Total Fit Points ="<<PointsToPlot<<endl;

 //get the number of bunches
 bunchN_previous=0; 
 for(std::map<int, map<int, std::vector<BSFitData> > >::iterator tmpIt=FitDone_Results_.begin();tmpIt!=FitDone_Results_.end();++tmpIt){
 
  if((tmpIt->second).size()==0) continue; 
 
  bunchN = max((Int_t)(tmpIt->second).size(), bunchN_previous);
  if(bunchN > bunchN_previous)tmpBxIt=(tmpIt->second).begin();
  bunchN_previous=(tmpIt->second).size();
  }
  


 TH1F* h_X_bx_[bunchN];
 TH1F* h_Y_bx_[bunchN];
 TH1F* h_Z_bx_[bunchN];
 TH1F* h_widthX_bx_[bunchN];
 TH1F* h_widthY_bx_[bunchN];
 TH1F* h_widthZ_bx_[bunchN];
 TH1F* h_nPV_bx_[bunchN];

 //create a map to store index of histo and bx N
  std::map<int, int> bxNIndexMap_;
  bxNIndexMap_.clear();

  //define the histo
   for(int x=0; x < bunchN; x++){

    bxNIndexMap_[(tmpBxIt->first)]=x;    


   Char_t XName[254];
   sprintf(XName,"%s%d%s","h_X_bx_",tmpBxIt->first,"\0");
   TString XName_(XName);
   h_X_bx_[x] =new TH1F(XName_,XName_,PointsToPlot,0.,PointsToPlot);
   h_X_bx_[x]->GetYaxis()->SetTitle("BS Fit X  (cm)"); 
   DefineHistStyle(h_X_bx_[x],x); 

   Char_t YName[254];
   sprintf(YName,"%s%d%s","h_Y_bx_",tmpBxIt->first,"\0");
   TString YName_(YName);
   h_Y_bx_[x] =new TH1F(YName_,YName_,PointsToPlot,0.,PointsToPlot);
   h_Y_bx_[x]->GetYaxis()->SetTitle("BS Fit Y  (cm)");
   DefineHistStyle(h_Y_bx_[x],x);

   Char_t ZName[254];
   sprintf(ZName,"%s%d%s","h_Z_bx_",tmpBxIt->first,"\0");
   TString ZName_(ZName);
   h_Z_bx_[x] =new TH1F(ZName_,ZName_,PointsToPlot,0.,PointsToPlot);
   h_Z_bx_[x]->GetYaxis()->SetTitle("BS Fit Z  (cm)");
   DefineHistStyle(h_Z_bx_[x],x);

   Char_t WXName[254];
   sprintf(WXName,"%s%d%s","h_widthX_bx_",tmpBxIt->first,"\0");
   TString WXName_(WXName);
   h_widthX_bx_[x] =new TH1F(WXName_,WXName_,PointsToPlot,0.,PointsToPlot);
   h_widthX_bx_[x]->GetYaxis()->SetTitle("BS Fit #sigma X  (cm)");
   DefineHistStyle(h_widthX_bx_[x],x);
  
   Char_t WYName[254];
   sprintf(WYName,"%s%d%s","h_widthY_bx_",tmpBxIt->first,"\0");
   TString WYName_(WYName);
   h_widthY_bx_[x] =new TH1F(WYName_,WYName_,PointsToPlot,0.,PointsToPlot);
   h_widthY_bx_[x]->GetYaxis()->SetTitle("BS Fit #sigma Y  (cm)"); 
   DefineHistStyle(h_widthY_bx_[x],x);



   Char_t WZName[254];
   sprintf(WZName,"%s%d%s","h_widthZ_bx_",tmpBxIt->first,"\0");
   TString WZName_(WZName);
   h_widthZ_bx_[x] =new TH1F(WZName_,WZName_,PointsToPlot,0.,PointsToPlot);
   h_widthZ_bx_[x]->GetYaxis()->SetTitle("BS Fit #sigma Z  (cm)");
   DefineHistStyle(h_widthZ_bx_[x],x);

   //histo for PV # for each bx

   Char_t PVSize[254];
   sprintf(PVSize,"%s%d%s","h_nPV_bx_",tmpBxIt->first,"\0");
   TString PVSize_(PVSize);
   h_nPV_bx_[x] =new TH1F(PVSize_,PVSize_,PointsToPlot,0.,PointsToPlot);
   h_nPV_bx_[x]->GetYaxis()->SetTitle("# of Primary Vertices / LS");
   DefineHistStyle(h_nPV_bx_[x],x);

   tmpBxIt++;
   }



  //Lets create an iteratior for nPV information
  map<int, map<int, float > >::iterator bxnpvIt = bxMapPVSize_.begin(); 
    

  int bxfit=0;

  for( std::map<int, map<int, std::vector<BSFitData> > >::iterator FitIt=FitDone_Results_.begin(); FitIt!=FitDone_Results_.end(); ++FitIt){
               
            if(FitDone_Results_.size()==0) continue; 
            bxfit=0;

            //this for bx-nPV map
            std::map<int, float >::iterator npvIt = bxnpvIt->second.begin();

    for( std::map<int, std::vector<BSFitData> >::iterator BxIt=FitIt->second.begin(); BxIt!=FitIt->second.end(); ++BxIt){
          for(size_t bsfit=0; bsfit < (BxIt->second).size(); bsfit++){
                cout<<"Fit #  = "<<FitIt->first<<"  Run:LS1-LS2  ="<<(RLS->second)<<"       Bx Number ="<< BxIt->first<<"     X0 = "<<((BxIt->second)[bsfit].xyz[0])<<endl;
                outdata<<"Fit #  = "<<FitIt->first<<"  Run:LS1-LS2  ="<<(RLS->second)<<"       Bx Number ="<< BxIt->first<<"     X0 = "<<((BxIt->second)[bsfit].xyz[0])<<endl;
  
               //if bx is not equal to bunchN then!! e.g when fitting for one LS only and it is missing some of the bunches
    
               if(((FitIt->second).size()) < bunchN){
                  for( std::map<int, int> ::iterator bx_in = bxNIndexMap_.begin(); bx_in!=bxNIndexMap_.end(); ++bx_in){

                  if(bx_in->first == BxIt->first){  
                                                     FillHist(h_X_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyz[0],(BxIt->second)[bsfit].xyzErr[0],RLS->second);
                                                     FillHist(h_Y_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyz[1],(BxIt->second)[bsfit].xyzErr[1],RLS->second);
                                                     FillHist(h_Z_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyz[2],(BxIt->second)[bsfit].xyzErr[2],RLS->second);

                                                     FillHist(h_widthX_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyzwidth[0],(BxIt->second)[bsfit].xyzwidthErr[0],RLS->second);
                                                     FillHist(h_widthY_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyzwidth[1],(BxIt->second)[bsfit].xyzwidthErr[1],RLS->second);
                                                     FillHist(h_widthZ_bx_[bx_in->second],FitIt->first,(BxIt->second)[bsfit].xyzwidth[2],(BxIt->second)[bsfit].xyzwidthErr[2],RLS->second);
                                                     //Fill PV Info
                                                     FillnPVInfo(h_nPV_bx_[bx_in->second],FitIt->first, npvIt->second,RLS->second);   

                                                 }
                                                  else{
                                                        FillHist(h_X_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);
                                                        FillHist(h_Y_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);
                                                        FillHist(h_Z_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);

                                                        FillHist(h_widthX_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);
                                                        FillHist(h_widthY_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);
                                                        FillHist(h_widthZ_bx_[bx_in->second],FitIt->first,0.0,0.0,RLS->second);
                                                        FillnPVInfo(h_nPV_bx_[bx_in->second],FitIt->first, 0.0,RLS->second );
         

                                                      }
                 }//loop over bx index map
               }//if some bx are missing in this LS range

              if(((FitIt->second).size()) == bunchN){            
               FillHist(h_X_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyz[0],(BxIt->second)[bsfit].xyzErr[0],RLS->second);
               FillHist(h_Y_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyz[1],(BxIt->second)[bsfit].xyzErr[1],RLS->second);
               FillHist(h_Z_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyz[2],(BxIt->second)[bsfit].xyzErr[2],RLS->second);

               FillHist(h_widthX_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyzwidth[0],(BxIt->second)[bsfit].xyzwidthErr[0],RLS->second);
               FillHist(h_widthY_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyzwidth[1],(BxIt->second)[bsfit].xyzwidthErr[1],RLS->second);
               FillHist(h_widthZ_bx_[bxfit],FitIt->first,(BxIt->second)[bsfit].xyzwidth[2],(BxIt->second)[bsfit].xyzwidthErr[2],RLS->second);

               FillnPVInfo(h_nPV_bx_[bxfit],FitIt->first,npvIt->second,RLS->second);
               }//


           } //loop over position errors
            bxfit++;
            npvIt++;
        }//Loop over bx

       RLS++;
       bxnpvIt++;

    }//Loop over each fit


 TDirectory *X0 = f1.mkdir("X0");
 TDirectory *Y0 = f1.mkdir("Y0");
 TDirectory *Z0 = f1.mkdir("Z0");
 TDirectory *SigmaZ0 = f1.mkdir("Sigma_Z0");
 TDirectory *SigmaX0 = f1.mkdir("width_X0");
 TDirectory *SigmaY0 = f1.mkdir("Width_Y0");
 TDirectory *nPV     = f1.mkdir("bx_nPV");



for(int t=0; t< bunchN; t++){
X0->cd();
h_X_bx_[t]->Write();
Y0->cd();
h_Y_bx_[t]->Write();
Z0->cd();
h_Z_bx_[t]->Write();
SigmaX0->cd();
h_widthX_bx_[t]->Write();
SigmaY0->cd();
h_widthY_bx_[t]->Write();
SigmaZ0->cd();
h_widthZ_bx_[t]->Write();

nPV->cd();
h_nPV_bx_[t]->Write();

}

//Plot all bx on the same canvas and save in root file
TCanvas *All_X= new TCanvas("All_X","",7,8,699,499);
PlotAllBunches(All_X, h_X_bx_, bunchN, bxNIndexMap_);
X0->cd();
All_X->Write();
TCanvas *All_Y= new TCanvas("All_Y","",7,8,699,499);
PlotAllBunches(All_Y, h_Y_bx_, bunchN, bxNIndexMap_);
Y0->cd();
All_Y->Write();
TCanvas *All_Z= new TCanvas("All_Z","",7,8,699,499);
PlotAllBunches(All_Z, h_Z_bx_, bunchN, bxNIndexMap_);
Z0->cd();
All_Z->Write();
TCanvas *All_widthX= new TCanvas("All_widthX","",7,8,699,499);
PlotAllBunches(All_widthX, h_widthX_bx_, bunchN, bxNIndexMap_);
SigmaX0->cd();
All_widthX->Write();
TCanvas *All_widthY= new TCanvas("All_widthY","",7,8,699,499);
PlotAllBunches(All_widthY, h_widthY_bx_, bunchN, bxNIndexMap_);
SigmaY0->cd();
All_widthY->Write();
TCanvas *All_widthZ= new TCanvas("All_widthZ","",7,8,699,499);
PlotAllBunches(All_widthZ, h_widthZ_bx_, bunchN, bxNIndexMap_);
SigmaZ0->cd();
All_widthZ->Write();

TCanvas *All_nPV= new TCanvas("All_nPV","",7,8,699,499);
PlotAllBunches(All_nPV,h_nPV_bx_, bunchN, bxNIndexMap_);
nPV->cd();
All_nPV->Write();


 cout<<"The PV fit is performed for all the  "<<bunchN<<"  bunches"<<endl;
 outdata<<"The PV fit is performed for all the  "<<bunchN<<"  bunches"<<endl;

 if(bunchN> 1000 || bunchN < 0)cout<<"Something Went Wrong OR there is no input to the fit!!!! "<<endl;
 if(bunchN> 1000 || bunchN < 0)outdata<<"Something Went Wrong OR there is no input to the fit!!!! "<<endl;

f1.cd();
//write the root file
f1.Close();

//clear these vectors
bxNIndexMap_.clear();
FitDone_Results_.clear();
FitDone_RunLumiRange_.clear();
bxMapPVSize_.clear();
}//PlotHistoBX ends here


//---------Define Hstogram Styles-----
void DefineHistStyle(TH1F *h1, int bx){
 bx=bx+1;
// if(bx>8) bx=bx-8;

 h1->SetFillColor(0);
 h1->SetStats(0);
 h1->GetYaxis()->SetTitleOffset(1.20);
 h1->SetOption("e1");
 h1->GetYaxis()->CenterTitle();
 h1->SetTitleOffset(1.2);
 h1->SetLineWidth(2);
 
 h1->SetMarkerStyle((20+bx));
 if(bx > 10)h1->SetMarkerStyle((20+bx-10));
 h1->SetMarkerSize(0.9);
 h1->SetMarkerColor(bx);
 if(bx > 9)h1->SetMarkerColor(bx-9);
 h1->SetLineColor(bx);
 if(bx > 9)h1->SetLineColor(bx-9);
 //remove yellow
 if(bx==5){h1->SetMarkerColor(bx*9);
           h1->SetLineColor(bx*9);}

 h1->SetLineStyle(1);
 if(bx>10)h1->SetLineStyle(bx-9);


 bx=0;

}
//------------------------


void FillHist(TH1F* h, int fitN, double pos, double posError, TString lab){

h->SetBinContent(fitN,pos);
h->SetBinError(fitN,posError);
h->GetXaxis()->SetBinLabel(fitN,lab);

}


void FillnPVInfo(TH1F* h2,int fitN, float npv, TString lab ){

h2->SetBinContent(fitN,npv);
//h->SetBinError(fitN,posError);
h2->GetXaxis()->SetBinLabel(fitN,lab);

 npv=0.0;

}




//--------------put all bx on same canvas

void PlotAllBunches(TCanvas* myCanvas,TH1F* hist[], int bunchN, std::map<int,int> tmpMap_){


 TString legT;
         legT = "";
 TLegend *leg = new TLegend(0.72,0.72,0.90,0.90, legT);
         leg->SetFillColor(0);
         myCanvas->SetFillColor(0);

 std::map<int,int>::const_iterator iter=tmpMap_.begin();

 for(int i=0;i<bunchN;i++){

   Char_t nbx[50];
   sprintf(nbx,"%s%d%s"," bx # ",(iter->first),"\0");
   TString nbx_(nbx);

   leg->AddEntry(hist[i],nbx_,"PL");  

   if(i==0)hist[i]->Draw();   
   else{hist[i]->Draw("SAMES");}
   iter++;

  }

  leg->Draw();
  
} 




//---------------------------------------------------------------------
//------------------Here we define the Fitting module------------------
//---------------------------------------------------------------------
bool runPVFitter(std::map< int, std::vector<BeamSpotFitPVData> > bxMap_, int NFits, int tmpLumiCounter){

float errorScale_    = 0.9;
float sigmaCut_      = 5.0;
bool fit_ok          = true;
Int_t minNrVertices_ = 10;


for ( std::map<int,std::vector<BeamSpotFitPVData> >::const_iterator pvStore = bxMap_.begin();
pvStore!=bxMap_.end(); ++pvStore) {


   //fill number of pv for each bx crossing:
  bxMapPVSize_[NFits][pvStore->first]=(((Float_t)(pvStore->second).size()/(Float_t)tmpLumiCounter));

  //cout<<"  For bx ="<<pvStore->first<<"     PV # =  "<<pvStore->second.size()<<endl;
  if ( (pvStore->second).size() <= minNrVertices_) {
       cout << " Not enough PVs, Setting to zero ->"<<(pvStore->second).size() << std::endl;
        fit_ok = false;
        //continue; 
    }


//LL function and fitter
FcnBeamSpotFitPV* fcn = new FcnBeamSpotFitPV(pvStore->second);
TFitterMinuit minuitx;
minuitx.SetMinuitFCN(fcn);
// fit parameters: positions, widths, x-y correlations, tilts in xz and yz
minuitx.SetParameter(0,"x",0.,0.02,-10.,10.);
minuitx.SetParameter(1,"y",0.,0.02,-10.,10.);
minuitx.SetParameter(2,"z",0.,0.20,-30.,30.);
minuitx.SetParameter(3,"ex",0.015,0.01,0.,10.);
minuitx.SetParameter(4,"corrxy",0.,0.02,-1.,1.);
minuitx.SetParameter(5,"ey",0.015,0.01,0.,10.);
minuitx.SetParameter(6,"dxdz",0.,0.0002,-0.1,0.1);
minuitx.SetParameter(7,"dydz",0.,0.0002,-0.1,0.1);
minuitx.SetParameter(8,"ez",1.,0.1,0.,30.);
minuitx.SetParameter(9,"scale",errorScale_,errorScale_/10.,errorScale_/2.,errorScale_*2.);
     
// first iteration without correlations
     
int ierr=0;
minuitx.FixParameter(4);
minuitx.FixParameter(6);
minuitx.FixParameter(7);
minuitx.FixParameter(9);
minuitx.SetMaxIterations(100);

minuitx.SetPrintLevel(0);
minuitx.CreateMinimizer();
ierr = minuitx.Minimize();

if ( ierr ) {
cout << "3D beam spot fit failed in 1st iteration" <<endl;
fit_ok =false;
//continue; 
}
      
     
// refit with harder selection on vertices

fcn->setLimits(minuitx.GetParameter(0)-sigmaCut_*minuitx.GetParameter(3),
minuitx.GetParameter(0)+sigmaCut_*minuitx.GetParameter(3),
minuitx.GetParameter(1)-sigmaCut_*minuitx.GetParameter(5),
minuitx.GetParameter(1)+sigmaCut_*minuitx.GetParameter(5),
minuitx.GetParameter(2)-sigmaCut_*minuitx.GetParameter(8),
minuitx.GetParameter(2)+sigmaCut_*minuitx.GetParameter(8));
ierr = minuitx.Minimize();
if (ierr) {
cout << "3D beam spot fit failed in 2nd iteration" << endl;
fit_ok = false;
//continue;
}

// refit with correlations
minuitx.ReleaseParameter(4);
minuitx.ReleaseParameter(6);
minuitx.ReleaseParameter(7);
ierr = minuitx.Minimize();

if ( ierr ) {
cout << "3D beam spot fit failed in 3rd iteration: Setting to zero" <<endl;
fit_ok = false;
//continue;
}

BSFitData PVFitDataForNLumi;

PVFitDataForNLumi.xyz[0] = minuitx.GetParameter(0);
PVFitDataForNLumi.xyz[1] = minuitx.GetParameter(1);
PVFitDataForNLumi.xyz[2] = minuitx.GetParameter(2);
PVFitDataForNLumi.xyzErr[0] = minuitx.GetParError(0);
PVFitDataForNLumi.xyzErr[1] = minuitx.GetParError(1);
PVFitDataForNLumi.xyzErr[2] = minuitx.GetParError(2);

PVFitDataForNLumi.xyzwidth[0] = minuitx.GetParameter(3);
PVFitDataForNLumi.xyzwidth[1] = minuitx.GetParameter(5);
PVFitDataForNLumi.xyzwidth[2] = minuitx.GetParameter(8);

PVFitDataForNLumi.xyzwidthErr[0] = minuitx.GetParError(3);
PVFitDataForNLumi.xyzwidthErr[1]= minuitx.GetParError(5);
PVFitDataForNLumi.xyzwidthErr[2] = minuitx.GetParError(8);





if(!fit_ok){//set zero if fit fails//or not enough PV
PVFitDataForNLumi.xyz[0] = 0.0;
PVFitDataForNLumi.xyz[1] = 0.0;
PVFitDataForNLumi.xyz[2] = 0.0;
PVFitDataForNLumi.xyzErr[0] = 0.0;
PVFitDataForNLumi.xyzErr[1] = 0.0;
PVFitDataForNLumi.xyzErr[2] = 0.0;

PVFitDataForNLumi.xyzwidth[0] = 0.0;
PVFitDataForNLumi.xyzwidth[1] = 0.0;
PVFitDataForNLumi.xyzwidth[2] = 0.0;

PVFitDataForNLumi.xyzwidthErr[0] = 0.0;
PVFitDataForNLumi.xyzwidthErr[1] = 0.0;
PVFitDataForNLumi.xyzwidthErr[2] = 0.0;
}

fit_ok=true; //now safe to put it ture
/*
cout<<" Fitted X="<< minuitx.GetParameter(0)<<"    Fitted WidthX ="<<minuitx.GetParameter(3)<<endl;
cout<<" Fitted Y="<< minuitx.GetParameter(1)<<"    Fitted WidthY ="<<minuitx.GetParameter(5)<<endl;
cout<<" Fitted Z="<< minuitx.GetParameter(2)<<"    Fitted WidthZ ="<<minuitx.GetParameter(8)<<endl;
*/


//store beam spot fit result as a function of bx
FitDone_Results_[NFits][pvStore->first].push_back(PVFitDataForNLumi);
  
 }//loop over map

}//endof runBXFiiter
