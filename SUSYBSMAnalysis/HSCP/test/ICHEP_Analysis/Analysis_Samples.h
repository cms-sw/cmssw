

struct stSignal{
   string Type;
   string Name;
   double Mass;
   double XSec;

   stSignal(); 
   stSignal(string Type_, string Name_, double Mass_, double XSec_){Type=Type_; Name=Name_; Mass=Mass_; XSec=XSec_;}
};

void GetSignalDefinition(std::vector<stSignal>& signals){
   signals.push_back(stSignal("Gluino", "Gluino200", 200, 326.500000) );
   signals.push_back(stSignal("Gluino", "Gluino300", 300,  27.690000) );
   signals.push_back(stSignal("Gluino", "Gluino400", 400,   3.910000) );
   signals.push_back(stSignal("Gluino", "Gluino500", 500,   0.754600) );
   signals.push_back(stSignal("Gluino", "Gluino600", 600,   0.170700) );
   signals.push_back(stSignal("Gluino", "Gluino900", 900,   0.003942) );
   signals.push_back(stSignal("Stop"  , "Stop130"  , 130,  65.540000) );
   signals.push_back(stSignal("Stop"  , "Stop200"  , 200,   6.832000) );
   signals.push_back(stSignal("Stop"  , "Stop300"  , 300,   0.647800) );
   signals.push_back(stSignal("Stop"  , "Stop500"  , 500,   0.022920) );
   signals.push_back(stSignal("Stop"  , "Stop800"  , 800,   0.000542) );
   signals.push_back(stSignal("MGStop", "MGStop130", 130, 181.590000) );
   signals.push_back(stSignal("MGStop", "MGStop200", 200,  22.230000) );
   signals.push_back(stSignal("MGStop", "MGStop300", 300,   2.470000) );
   signals.push_back(stSignal("MGStop", "MGStop500", 500,   0.064000) );
   signals.push_back(stSignal("MGStop", "MGStop800", 800,   0.001560) );
   signals.push_back(stSignal("Stau"  , "Stau100"  , 100,   1.326000) );
   signals.push_back(stSignal("Stau"  , "Stau126"  , 126,   0.330000) );
   signals.push_back(stSignal("Stau"  , "Stau156"  , 156,   0.105000) );
   signals.push_back(stSignal("Stau"  , "Stau200"  , 200,   0.025000) );
   signals.push_back(stSignal("Stau"  , "Stau247"  , 247,   0.008000) );
   signals.push_back(stSignal("Stau"  , "Stau308"  , 308,   0.002000) );
}

void GetInputFiles(std::vector<string>& inputFiles, string SampleName){

   if(SampleName=="Data"){
      inputFiles.push_back("InputFiles/Data.root");
   }else if(SampleName=="MC"){
      inputFiles.push_back("InputFiles/MC.root");
   }else if(SampleName=="Gluino200"){
      inputFiles.push_back("InputFiles/Gluino200.root");
   }else if(SampleName=="Gluino300"){
      inputFiles.push_back("InputFiles/Gluino300.root");
   }else if(SampleName=="Gluino400"){
      inputFiles.push_back("InputFiles/Gluino400.root");
   }else if(SampleName=="Gluino500"){
      inputFiles.push_back("InputFiles/Gluino500.root");
   }else if(SampleName=="Gluino600"){
      inputFiles.push_back("InputFiles/Gluino600.root");
   }else if(SampleName=="Gluino900"){
      inputFiles.push_back("InputFiles/Gluino900.root");
   }else if(SampleName=="MGStop130"){
      inputFiles.push_back("InputFiles/MGStop130.root");
   }else if(SampleName=="MGStop200"){
      inputFiles.push_back("InputFiles/MGStop200.root");
   }else if(SampleName=="MGStop300"){
      inputFiles.push_back("InputFiles/MGStop300.root");
   }else if(SampleName=="MGStop500"){
      inputFiles.push_back("InputFiles/MGStop500.root");
   }else if(SampleName=="MGStop800"){
      inputFiles.push_back("InputFiles/MGStop800.root");
   }else if(SampleName=="Stau100"){
      inputFiles.push_back("InputFiles/Stau100.root");
   }else if(SampleName=="Stau126"){
      inputFiles.push_back("InputFiles/Stau126.root");
   }else if(SampleName=="Stau156"){
      inputFiles.push_back("InputFiles/Stau156.root");
   }else if(SampleName=="Stau200"){
      inputFiles.push_back("InputFiles/Stau200.root");
   }else if(SampleName=="Stau247"){
      inputFiles.push_back("InputFiles/Stau247.root");
   }else if(SampleName=="Stau308"){
      inputFiles.push_back("InputFiles/Stau308.root");
   }else if(SampleName=="Stop130"){
      inputFiles.push_back("InputFiles/Stop130.root");
   }else if(SampleName=="Stop200"){
      inputFiles.push_back("InputFiles/Stop200.root");
   }else if(SampleName=="Stop300"){
      inputFiles.push_back("InputFiles/Stop300.root");
   }else if(SampleName=="Stop500"){
      inputFiles.push_back("InputFiles/Stop500.root");
   }else if(SampleName=="Stop800"){
      inputFiles.push_back("InputFiles/Stop800.root");
   }else{
      printf("\n\n\n!!!UNKOWN SAMPLE:%s!!!\n\n\n",SampleName.c_str());
   }

}
