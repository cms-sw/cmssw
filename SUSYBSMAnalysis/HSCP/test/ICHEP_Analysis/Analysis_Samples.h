
#ifndef HSCP_ANALYSIS_SAMPLE
#define HSCP_ANALYSIS_SAMPLE

//double             IntegratedLuminosity = 0.0084;
//double             IntegratedLuminosity = 0.035915;
//double             IntegratedLuminosity = 0.0678774;
//double             IntegratedLuminosity = 0.198149471;
//double             IntegratedLuminosity = 0.706883819;
double               IntegratedLuminosity = 1.092666355;

struct stSignal{
   string Type;
   string Name;
   string Legend;
   double Mass;
   double XSec;
   bool   MakePlot;
   int    NChargedHSCP;

   stSignal(); 
   stSignal(string Type_, string Name_, string Legend_, double Mass_, int NChargedHSCP_, bool MakePlot_, double XSec_){Type=Type_; Name=Name_; Legend=Legend_; Mass=Mass_; NChargedHSCP=NChargedHSCP_; MakePlot=MakePlot_; XSec=XSec_;}
};

void GetSignalDefinition(std::vector<stSignal>& signals){
// signals.push_back(stSignal("Gluino", "Gluino200"   , "#tilde{g} 200"       , 200, -1, 1, 326.500000) ); //LO
// signals.push_back(stSignal("Gluino", "Gluino300"   , "#tilde{g} 300"       , 300, -1, 0,  27.690000) ); //LO
// signals.push_back(stSignal("Gluino", "Gluino400"   , "#tilde{g} 400"       , 400, -1, 0,   3.910000) ); //LO
// signals.push_back(stSignal("Gluino", "Gluino500"   , "#tilde{g} 500"       , 500, -1, 1,   0.754600) ); //LO
// signals.push_back(stSignal("Gluino", "Gluino600"   , "#tilde{g} 600"       , 600, -1, 0,   0.170700) ); //LO
// signals.push_back(stSignal("Gluino", "Gluino900"   , "#tilde{g} 900"       , 900, -1, 1,   0.003942) ); //LO
   signals.push_back(stSignal("Gluino", "Gluino200"   , "#tilde{g} 200"       , 200, -1, 1, 606.000000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino300"   , "#tilde{g} 300"       , 300, -1, 0,  57.200000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino400"   , "#tilde{g} 400"       , 400, -1, 0,   8.980000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino500"   , "#tilde{g} 500"       , 500, -1, 1,   1.870000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino600"   , "#tilde{g} 600"       , 600, -1, 0,   0.465000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino900"   , "#tilde{g} 900"       , 900, -1, 1,   0.012800) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino200_0C", "#tilde{g} 200 0C"    , 200,  0, 0, 606.000000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino300_0C", "#tilde{g} 300 0C"    , 300,  0, 0,  57.200000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino400_0C", "#tilde{g} 400 0C"    , 400,  0, 0,   8.980000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino500_0C", "#tilde{g} 500 0C"    , 500,  0, 0,   1.870000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino600_0C", "#tilde{g} 600 0C"    , 600,  0, 0,   0.465000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino900_0C", "#tilde{g} 900 0C"    , 900,  0, 0,   0.012800) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino200_1C", "#tilde{g} 200 1C"    , 200,  1, 0, 606.000000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino300_1C", "#tilde{g} 300 1C"    , 300,  1, 0,  57.200000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino400_1C", "#tilde{g} 400 1C"    , 400,  1, 0,   8.980000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino500_1C", "#tilde{g} 500 1C"    , 500,  1, 0,   1.870000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino600_1C", "#tilde{g} 600 1C"    , 600,  1, 0,   0.465000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino900_1C", "#tilde{g} 900 1C"    , 900,  1, 0,   0.012800) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino200_2C", "#tilde{g} 200 2C"    , 200,  2, 0, 606.000000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino300_2C", "#tilde{g} 300 2C"    , 300,  2, 0,  57.200000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino400_2C", "#tilde{g} 400 2C"    , 400,  2, 0,   8.980000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino500_2C", "#tilde{g} 500 2C"    , 500,  2, 0,   1.870000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino600_2C", "#tilde{g} 600 2C"    , 600,  2, 0,   0.465000) ); //NLO
   signals.push_back(stSignal("Gluino", "Gluino900_2C", "#tilde{g} 900 2C"    , 900,  2, 0,   0.012800) ); //NLO
// signals.push_back(stSignal("Stop"  , "Stop130"     , "#tilde{t}_{1} 130"   , 130, -1, 0, 65.540000) ); //LO
// signals.push_back(stSignal("Stop"  , "Stop200"     , "#tilde{t}_{1} 200"   , 200, -1, 0,  6.832000) ); //LO
// signals.push_back(stSignal("Stop"  , "Stop300"     , "#tilde{t}_{1} 300"   , 300, -1, 0,  0.647800) ); //LO
// signals.push_back(stSignal("Stop"  , "Stop500"     , "#tilde{t}_{1} 500"   , 500, -1, 0,  0.022920) ); //LO
// signals.push_back(stSignal("Stop"  , "Stop800"     , "#tilde{t}_{1} 800"   , 800, -1, 0,  0.000542) ); //LO
// signals.push_back(stSignal("MGStop", "MGStop130"   , "#tilde{t}_{1} 130"   , 130, -1, 1, 73.270000) ); //LO
// signals.push_back(stSignal("MGStop", "MGStop200"   , "#tilde{t}_{1} 200"   , 200, -1, 1,  7.65100)  ); //LO
// signals.push_back(stSignal("MGStop", "MGStop300"   , "#tilde{t}_{1} 300"   , 300, -1, 0,  0.754000) ); //LO
// signals.push_back(stSignal("MGStop", "MGStop500"   , "#tilde{t}_{1} 500"   , 500, -1, 0,  0.026490) ); //LO
// signals.push_back(stSignal("MGStop", "MGStop800"   , "#tilde{t}_{1} 800"   , 800, -1, 1,  0.000621) ); //LO
// signals.push_back(stSignal("Stop"  , "Stop130"     , "#tilde{t}_{1} 130"   , 130, -1, 0, 65.540000*1.4927) ); //NLO
// signals.push_back(stSignal("Stop"  , "Stop200"     , "#tilde{t}_{1} 200"   , 200, -1, 0,  6.832000*1.5623) ); //NLO
// signals.push_back(stSignal("Stop"  , "Stop300"     , "#tilde{t}_{1} 300"   , 300, -1, 0,  0.647800*1.6378) ); //NLO
// signals.push_back(stSignal("Stop"  , "Stop500"     , "#tilde{t}_{1} 500"   , 500, -1, 0,  0.022920*1.7654) ); //NLO
// signals.push_back(stSignal("Stop"  , "Stop800"     , "#tilde{t}_{1} 800"   , 800, -1, 0,  0.000542*1.9887) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop130"   , "#tilde{t}_{1} 130"   , 130, -1, 1, 73.270000*1.4927) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop200"   , "#tilde{t}_{1} 200"   , 200, -1, 1,  7.651000*1.5623) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop300"   , "#tilde{t}_{1} 300"   , 300, -1, 0,  0.754000*1.6378) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop500"   , "#tilde{t}_{1} 500"   , 500, -1, 0,  0.026490*1.7654) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop800"   , "#tilde{t}_{1} 800"   , 800, -1, 1,  0.000621*1.9887) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop130_0C", "#tilde{t}_{1} 130 0C", 130,  0, 0, 73.270000*1.4927) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop200_0C", "#tilde{t}_{1} 200 0C", 200,  0, 0,  7.651000*1.5623) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop300_0C", "#tilde{t}_{1} 300 0C", 300,  0, 0,  0.754000*1.6378) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop500_0C", "#tilde{t}_{1} 500 0C", 500,  0, 0,  0.026490*1.7654) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop800_0C", "#tilde{t}_{1} 800 0C", 800,  0, 0,  0.000621*1.9887) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop130_1C", "#tilde{t}_{1} 130 1C", 130,  1, 0, 73.270000*1.4927) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop200_1C", "#tilde{t}_{1} 200 1C", 200,  1, 0,  7.651000*1.5623) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop300_1C", "#tilde{t}_{1} 300 1C", 300,  1, 0,  0.754000*1.6378) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop500_1C", "#tilde{t}_{1} 500 1C", 500,  1, 0,  0.026490*1.7654) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop800_1C", "#tilde{t}_{1} 800 1C", 800,  1, 0,  0.000621*1.9887) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop130_2C", "#tilde{t}_{1} 130 2C", 130,  2, 0, 73.270000*1.4927) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop200_2C", "#tilde{t}_{1} 200 2C", 200,  2, 0,  7.651000*1.5623) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop300_2C", "#tilde{t}_{1} 300 2C", 300,  2, 0,  0.754000*1.6378) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop500_2C", "#tilde{t}_{1} 500 2C", 500,  2, 0,  0.026490*1.7654) ); //NLO
   signals.push_back(stSignal("MGStop", "MGStop800_2C", "#tilde{t}_{1} 800 2C", 800,  2, 0,  0.000621*1.9887) ); //NLO
   signals.push_back(stSignal("Stau"  , "Stau100"     , "#tilde{#tau}_{1} 100", 100, -1, 1,  1.326000) ); //LO
   signals.push_back(stSignal("Stau"  , "Stau126"     , "#tilde{#tau}_{1} 126", 126, -1, 0,  0.330000) ); //LO
   signals.push_back(stSignal("Stau"  , "Stau156"     , "#tilde{#tau}_{1} 156", 156, -1, 0,  0.105000) ); //LO
   signals.push_back(stSignal("Stau"  , "Stau200"     , "#tilde{#tau}_{1} 200", 200, -1, 1,  0.025000) ); //LO
   signals.push_back(stSignal("Stau"  , "Stau247"     , "#tilde{#tau}_{1} 247", 247, -1, 0,  0.008000) ); //LO
   signals.push_back(stSignal("Stau"  , "Stau308"     , "#tilde{#tau}_{1} 308", 308, -1, 1,  0.002000) ); //LO
}

struct stMC{
   string Name;
   double ILumi;
   double MaxPtHat;
   double MaxEvent;

   stMC();
   stMC(string Name_, double ILumi_, double MaxPtHat_, int MaxEvent_){Name = Name_; ILumi = ILumi_; MaxPtHat = MaxPtHat_; MaxEvent = MaxEvent_;}
};

void GetMCDefinition(std::vector<stMC>& MC){
//   MC.push_back(stMC("MC_MB"   , 0.000759) );
//   MC.push_back(stMC("MC_QCD30", 0.0678)   );

   MC.push_back(stMC("MC_MB"   , 0.000759    , 30, -1 ));
//   MC.push_back(stMC("MC_MB"   , 0.000759*0.5, 30, -1 ));
//   MC.push_back(stMC("MC_PPMUX", 0.07327 *0.5, 30, -1 ));
   MC.push_back(stMC("MC_QCD30", 0.08432     , 80, -1 ));
   MC.push_back(stMC("MC_QCD80", 2.8436      , -1, -1 ));
}


void GetInputFiles(std::vector<string>& inputFiles, string SampleName){
//   string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_3_6_1_patch4/";
   string BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_3_6_1_patch4/10_08_22/";
//   std::cout<<"BASE DATASET DIRECTORY FIXED TO: " << BaseDirectory << " For Sample " << SampleName << std::endl;

   if(SampleName=="Data"){
      if(rand()%2==0){
         inputFiles.push_back(BaseDirectory + "Data_131511_to_135802.root");
         inputFiles.push_back(BaseDirectory + "Data_135821_to_137436.root");
         inputFiles.push_back(BaseDirectory + "Data_137437_to_139239.root");
         inputFiles.push_back(BaseDirectory + "Data_139347_to_139411.root");
         inputFiles.push_back(BaseDirectory + "Data_139455_to_139980.root");
         inputFiles.push_back(BaseDirectory + "Data_139981_to_140182.root");
         inputFiles.push_back(BaseDirectory + "Data_140331_to_141887.root");
         inputFiles.push_back(BaseDirectory + "Data_141888_to_142776.root");
         inputFiles.push_back(BaseDirectory + "Data_142928_to_143179.root");
      }else{
         inputFiles.push_back(BaseDirectory + "Data_131511_to_135802_B.root");
         inputFiles.push_back(BaseDirectory + "Data_135821_to_137436_B.root");
         inputFiles.push_back(BaseDirectory + "Data_137437_to_139239_B.root");
         inputFiles.push_back(BaseDirectory + "Data_139347_to_139411_B.root");
         inputFiles.push_back(BaseDirectory + "Data_139455_to_139980_B.root");
         inputFiles.push_back(BaseDirectory + "Data_139981_to_140182_B.root");
         inputFiles.push_back(BaseDirectory + "Data_140331_to_141887_B.root");
         inputFiles.push_back(BaseDirectory + "Data_141888_to_142776_B.root");
         inputFiles.push_back(BaseDirectory + "Data_142928_to_143179_B.root");
      }
   }else if(SampleName=="MC_MB"){
      inputFiles.push_back(BaseDirectory + "MC_MB.root");
   }else if(SampleName=="MC_PPMUX"){
      inputFiles.push_back(BaseDirectory + "MC_PPMUX.root");
   }else if(SampleName=="MC_QCD30"){
      inputFiles.push_back(BaseDirectory + "MC_QCD30.root");
   }else if(SampleName=="MC_QCD80"){
      if(rand()%2==0){
         inputFiles.push_back(BaseDirectory + "MC_QCD80.root");
      }else{
         inputFiles.push_back(BaseDirectory + "MC_QCD80_B.root");
      }

/*      inputFiles.push_back(BaseDirectory + "MC_QCD80_1.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_2.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_3.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_4.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_5.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_6.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_7.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_8.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_9.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_10.root");
      inputFiles.push_back(BaseDirectory + "MC_QCD80_11.root");
*/
   }else if(SampleName=="Gluino200" || SampleName=="Gluino200_0C" || SampleName=="Gluino200_1C" || SampleName=="Gluino200_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino200.root");
   }else if(SampleName=="Gluino300" || SampleName=="Gluino300_0C" || SampleName=="Gluino300_1C" || SampleName=="Gluino300_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino300.root");
   }else if(SampleName=="Gluino400" || SampleName=="Gluino400_0C" || SampleName=="Gluino400_1C" || SampleName=="Gluino400_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino400.root");
   }else if(SampleName=="Gluino500" || SampleName=="Gluino500_0C" || SampleName=="Gluino500_1C" || SampleName=="Gluino500_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino500.root");
   }else if(SampleName=="Gluino600" || SampleName=="Gluino600_0C" || SampleName=="Gluino600_1C" || SampleName=="Gluino600_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino600.root");
   }else if(SampleName=="Gluino900" || SampleName=="Gluino900_0C" || SampleName=="Gluino900_1C" || SampleName=="Gluino900_2C"){
      inputFiles.push_back(BaseDirectory + "Gluino900.root");
   }else if(SampleName=="MGStop130" || SampleName=="MGStop130_0C" || SampleName=="MGStop130_1C" || SampleName=="MGStop130_2C"){
      inputFiles.push_back(BaseDirectory + "MGStop130.root");
   }else if(SampleName=="MGStop200" || SampleName=="MGStop200_0C" || SampleName=="MGStop200_1C" || SampleName=="MGStop200_2C"){
      inputFiles.push_back(BaseDirectory + "MGStop200.root");
   }else if(SampleName=="MGStop300" || SampleName=="MGStop300_0C" || SampleName=="MGStop300_1C" || SampleName=="MGStop300_2C"){
      inputFiles.push_back(BaseDirectory + "MGStop300.root");
   }else if(SampleName=="MGStop500" || SampleName=="MGStop500_0C" || SampleName=="MGStop500_1C" || SampleName=="MGStop500_2C"){
      inputFiles.push_back(BaseDirectory + "MGStop500.root");
   }else if(SampleName=="MGStop800" || SampleName=="MGStop800_0C" || SampleName=="MGStop800_1C" || SampleName=="MGStop800_2C"){
      inputFiles.push_back(BaseDirectory + "MGStop800.root");
   }else if(SampleName=="Stau100"){
      inputFiles.push_back(BaseDirectory + "Stau100.root");
   }else if(SampleName=="Stau126"){
      inputFiles.push_back(BaseDirectory + "Stau126.root");
   }else if(SampleName=="Stau156"){
      inputFiles.push_back(BaseDirectory + "Stau156.root");
   }else if(SampleName=="Stau200"){
      inputFiles.push_back(BaseDirectory + "Stau200.root");
   }else if(SampleName=="Stau247"){
      inputFiles.push_back(BaseDirectory + "Stau247.root");
   }else if(SampleName=="Stau308"){
      inputFiles.push_back(BaseDirectory + "Stau308.root");
   }else if(SampleName=="Stop130"){
      inputFiles.push_back(BaseDirectory + "Stop130.root");
   }else if(SampleName=="Stop200"){
      inputFiles.push_back(BaseDirectory + "Stop200.root");
   }else if(SampleName=="Stop300"){
      inputFiles.push_back(BaseDirectory + "Stop300.root");
   }else if(SampleName=="Stop500"){
      inputFiles.push_back(BaseDirectory + "Stop500.root");
   }else if(SampleName=="Stop800"){
      inputFiles.push_back(BaseDirectory + "Stop800.root");
   }else{
      printf("\n\n\n!!!UNKOWN SAMPLE:%s!!!\n\n\n",SampleName.c_str());
   }

}

#endif
