{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> caloJetBarrel;
vector<float> caloJetEndcaps;
vector<float> jptJetBarrel;
vector<float> jptJetEndcaps;
vector<float> pfJetBarrel;
vector<float> pfJetEndcaps;
vector<float> pfNewJetBarrel;
vector<float> pfNewJetEndcaps;
vector<float> jetEnergy;
vector<float> jetPt;
vector<float> jetAtlas;
vector<float> jetAtlasPt;
//vector<float> jetPtMin;
//vector<float> jetPtMax;

TFile filePF("JetBenchmark_Full_310pre2.root");
//TH2F* histPF2Barrel = (TH2F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPtvsPt"));
//TH2F* histPF2Endcap = (TH2F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPtvsPt"));


jetPt.push_back(17.5);
jetPt.push_back(29.8);
jetPt.push_back(49.6);
jetPt.push_back(69.6);
jetPt.push_back(89.5);
jetPt.push_back(123.);
jetPt.push_back(173.);
jetPt.push_back(223.);
jetPt.push_back(274.);
jetPt.push_back(346.);
jetPt.push_back(445.);
jetPt.push_back(620.);


/*
jetPtMin.push_back(7);
jetPtMin.push_back(10);
jetPtMin.push_back(15);
jetPtMin.push_back(20);
jetPtMin.push_back(25);
jetPtMin.push_back(30);
jetPtMin.push_back(35);
jetPtMin.push_back(40);
jetPtMin.push_back(45);
jetPtMin.push_back(50);
jetPtMin.push_back(60);
jetPtMin.push_back(75);
jetPtMin.push_back(90);
jetPtMin.push_back(110);
jetPtMin.push_back(140);
jetPtMin.push_back(170);
jetPtMin.push_back(200);

jetPtMax.push_back(10);
jetPtMax.push_back(15);
jetPtMax.push_back(20);
jetPtMax.push_back(25);
jetPtMax.push_back(30);
jetPtMax.push_back(35);
jetPtMax.push_back(40);
jetPtMax.push_back(45);
jetPtMax.push_back(50);
jetPtMax.push_back(60);
jetPtMax.push_back(75);
jetPtMax.push_back(90);
jetPtMax.push_back(110);
jetPtMax.push_back(140);
jetPtMax.push_back(170);
jetPtMax.push_back(200);
jetPtMax.push_back(250);

for ( unsigned ipt=0; ipt<jetPtMin.size(); ++ipt ) { 

  jetPt.push_back(jetPtMin[ipt]+jetPtMax[ipt]);
  TH1F* hb = (TH1F*) (histPF2Barrel->ProjectionY("",jetPtMin[ipt],jetPtMax[ipt])->Clone());
  TH1F* he = (TH1F*) (histPF2Endcap->ProjectionY("",jetPtMin[ipt],jetPtMax[ipt])->Clone());
  hb->Fit("gaus");
  he->Fit("gaus");
  TF1* gausb = hb->GetFunction( "gaus" );
  TF1* gause = he->GetFunction( "gaus" );
  pfNewJetBarrel.push_back( gausb->GetParameter(2)/(1.+gausb->GetParameter(1)));
  pfNewJetEndcaps.push_back( gause->GetParameter(2)/(1.+gause->GetParameter(1)));
}
jetPt.push_back(620.);
*/

jetAtlasPt.push_back(17.);
jetAtlasPt.push_back(25.);
jetAtlasPt.push_back(40.);
jetAtlasPt.push_back(60.);
jetAtlasPt.push_back(90.);
jetAtlasPt.push_back(110.);
jetAtlasPt.push_back(130.);
jetAtlasPt.push_back(160.);
jetAtlasPt.push_back(200.);
jetAtlasPt.push_back(300.);
jetAtlasPt.push_back(400.);
jetAtlasPt.push_back(600.);

jetAtlas.push_back((18.2+18.0)/200.);
jetAtlas.push_back((18.0+16.0)/200.);
jetAtlas.push_back((15.0+13.5)/200.);
jetAtlas.push_back((13.2+11.2)/200.);
jetAtlas.push_back((11.0+9.5)/200.);
jetAtlas.push_back((9.8+8.2)/200.);
jetAtlas.push_back((8.5+7.2)/200.);
jetAtlas.push_back((7.6+6.0)/200.);
jetAtlas.push_back((7.0+5.2)/200.);
jetAtlas.push_back((5.5+4.5)/200.);
jetAtlas.push_back((4.5+4.2)/200.);
jetAtlas.push_back((4.5+3.8)/200.);

  // 1 - 1.3
  /*
  jptJetEndcaps.push_back((0.414+0.416)/2.);
  jptJetEndcaps.push_back((0.361+0.356)/2.);
  jptJetEndcaps.push_back((0.334+0.338)/2.);
  jptJetEndcaps.push_back((0.281+0.281)/2.);
  jptJetEndcaps.push_back((0.207+0.203)/2.);
  jptJetEndcaps.push_back((0.162+0.160)/2.);
  jptJetEndcaps.push_back((0.144+0.145)/2.);
  jptJetEndcaps.push_back((0.129+0.131)/2.);
  jptJetEndcaps.push_back((0.117+0.118)/2.);
  jptJetEndcaps.push_back((0.103+0.105)/2.);
  jptJetEndcaps.push_back((0.093+0.089)/2.);
  jptJetEndcaps.push_back((0.080+0.083)/2.);
  jptJetEndcaps.push_back((0.071+0.068)/2.);
  jptJetEndcaps.push_back((0.063+0.063)/2.);
  jptJetEndcaps.push_back((0.057+0.057)/2.);
  jptJetEndcaps.push_back((0.048+0.050)/2.);
  jptJetEndcaps.push_back((0.049+0.052)/2.);
  jptJetEndcaps.push_back((0.046+0.044)/2.);
  jptJetEndcaps.push_back((0.047+0.042)/2.);
  jptJetEndcaps.push_back((0.010-8.268)/2.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);
  */
 
  // 0 - 1.4
  jptJetEndcaps.push_back((0.414+0.416)/2.);
  jptJetEndcaps.push_back((0.361+0.356)/2.);
  jptJetEndcaps.push_back((0.344+0.344)/2.);
  jptJetEndcaps.push_back((0.281+0.281)/2.);
  jptJetEndcaps.push_back((0.214+0.214)/2.);
  jptJetEndcaps.push_back((0.170+0.170)/2.);
  jptJetEndcaps.push_back((0.151+0.151)/2.);
  jptJetEndcaps.push_back((0.135+0.135)/2.);
  jptJetEndcaps.push_back((0.120+0.120)/2.);
  jptJetEndcaps.push_back((0.107+0.107)/2.);
  jptJetEndcaps.push_back((0.090+0.090)/2.);
  jptJetEndcaps.push_back((0.079+0.079)/2.);
  jptJetEndcaps.push_back((0.068+0.068)/2.);
  jptJetEndcaps.push_back((0.059+0.059)/2.);
  jptJetEndcaps.push_back((0.052+0.052)/2.);
  jptJetEndcaps.push_back((0.044+0.044)/2.);
  jptJetEndcaps.push_back((0.041+0.041)/2.);
  jptJetEndcaps.push_back((0.040+0.040)/2.);
  jptJetEndcaps.push_back((0.035+0.035)/2.);
  jptJetEndcaps.push_back((0.010-8.268)/2.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);
  jptJetEndcaps.push_back(-1.);

  // 0 - 1.3
  /*
  jptJetBarrel.push_back((0.305+0.307)/2.);
  jptJetBarrel.push_back((0.289+0.282)/2.);
  jptJetBarrel.push_back((0.268+0.271)/2.);
  jptJetBarrel.push_back((0.253+0.250)/2.);
  jptJetBarrel.push_back((0.207+0.209)/2.);
  jptJetBarrel.push_back((0.168+0.166)/2.);
  jptJetBarrel.push_back((0.152+0.150)/2.);
  jptJetBarrel.push_back((0.137+0.138)/2.);
  jptJetBarrel.push_back((0.121+0.121)/2.);
  jptJetBarrel.push_back((0.109+0.107)/2.);
  jptJetBarrel.push_back((0.098+0.096)/2.);
  jptJetBarrel.push_back((0.087+0.087)/2.);
  jptJetBarrel.push_back((0.078+0.078)/2.);
  jptJetBarrel.push_back((0.067+0.067)/2.);
  jptJetBarrel.push_back((0.060+0.061)/2.);
  jptJetBarrel.push_back((0.055+0.055)/2.);
  jptJetBarrel.push_back((0.049+0.051)/2.);
  jptJetBarrel.push_back((0.044+0.046)/2.);
  jptJetBarrel.push_back((0.042+0.040)/2.);
  jptJetBarrel.push_back((0.039+0.038)/2.);
  jptJetBarrel.push_back((0.035+0.036)/2.);
  jptJetBarrel.push_back((0.035+0.035)/2.);
  jptJetBarrel.push_back((0.034+0.035)/2.);
  */
  
  // 0 - 1.4
  jptJetBarrel.push_back((0.305+0.307)/2.);
  jptJetBarrel.push_back((0.289+0.282)/2.);
  jptJetBarrel.push_back((0.268+0.271)/2.);
  jptJetBarrel.push_back((0.257+0.257)/2.);
  jptJetBarrel.push_back((0.210+0.210)/2.);
  jptJetBarrel.push_back((0.169+0.169)/2.);
  jptJetBarrel.push_back((0.152+0.152)/2.);
  jptJetBarrel.push_back((0.139+0.139)/2.);
  jptJetBarrel.push_back((0.123+0.123)/2.);
  jptJetBarrel.push_back((0.112+0.112)/2.);
  jptJetBarrel.push_back((0.100+0.100)/2.);
  jptJetBarrel.push_back((0.088+0.088)/2.);
  jptJetBarrel.push_back((0.079+0.079)/2.);
  jptJetBarrel.push_back((0.069+0.069)/2.);
  jptJetBarrel.push_back((0.061+0.061)/2.);
  jptJetBarrel.push_back((0.055+0.055)/2.);
  jptJetBarrel.push_back((0.051+0.051)/2.);
  jptJetBarrel.push_back((0.046+0.046)/2.);
  jptJetBarrel.push_back((0.043+0.043)/2.);
  jptJetBarrel.push_back((0.039+0.039)/2.);
  jptJetBarrel.push_back((0.035+0.036)/2.);
  jptJetBarrel.push_back((0.035+0.035)/2.);
  jptJetBarrel.push_back((0.034+0.035)/2.);

/*
jptJetEndcaps.push_back((0.414+0.416)/2.);
jptJetEndcaps.push_back((0.361+0.356)/2.);
jptJetEndcaps.push_back((0.334+0.338)/2.);
jptJetEndcaps.push_back((0.281+0.281)/2.);
jptJetEndcaps.push_back((0.207+0.203)/2.);
jptJetEndcaps.push_back((0.162+0.160)/2.);
jptJetEndcaps.push_back((0.144+0.145)/2.);
jptJetEndcaps.push_back((0.129+0.131)/2.);
jptJetEndcaps.push_back((0.117+0.118)/2.);
jptJetEndcaps.push_back((0.103+0.105)/2.);
jptJetEndcaps.push_back((0.093+0.089)/2.);
jptJetEndcaps.push_back((0.080+0.083)/2.);
jptJetEndcaps.push_back((0.071+0.068)/2.);
jptJetEndcaps.push_back((0.063+0.063)/2.);
jptJetEndcaps.push_back((0.057+0.057)/2.);
jptJetEndcaps.push_back((0.048+0.050)/2.);
jptJetEndcaps.push_back((0.049+0.052)/2.);
jptJetEndcaps.push_back((0.046+0.044)/2.);
jptJetEndcaps.push_back((0.047+0.042)/2.);
jptJetEndcaps.push_back((0.010-8.268)/2.);
jptJetEndcaps.push_back(-1.);
jptJetEndcaps.push_back(-1.);
jptJetEndcaps.push_back(-1.);

jptJetBarrel.push_back((0.305+0.307)/2.);
jptJetBarrel.push_back((0.289+0.282)/2.);
jptJetBarrel.push_back((0.268+0.271)/2.);
jptJetBarrel.push_back((0.253+0.250)/2.);
jptJetBarrel.push_back((0.207+0.209)/2.);
jptJetBarrel.push_back((0.168+0.166)/2.);
jptJetBarrel.push_back((0.152+0.150)/2.);
jptJetBarrel.push_back((0.137+0.138)/2.);
jptJetBarrel.push_back((0.121+0.121)/2.);
jptJetBarrel.push_back((0.109+0.107)/2.);
jptJetBarrel.push_back((0.098+0.096)/2.);
jptJetBarrel.push_back((0.087+0.087)/2.);
jptJetBarrel.push_back((0.078+0.078)/2.);
jptJetBarrel.push_back((0.067+0.067)/2.);
jptJetBarrel.push_back((0.060+0.061)/2.);
jptJetBarrel.push_back((0.055+0.055)/2.);
jptJetBarrel.push_back((0.049+0.051)/2.);
jptJetBarrel.push_back((0.044+0.046)/2.);
jptJetBarrel.push_back((0.042+0.040)/2.);
jptJetBarrel.push_back((0.039+0.038)/2.);
jptJetBarrel.push_back((0.035+0.036)/2.);
jptJetBarrel.push_back((0.035+0.035)/2.);
jptJetBarrel.push_back((0.034+0.035)/2.);
*/

// 0 - 1.3
/*
caloJetEndcaps.push_back((0.527+0.536)/2.);
caloJetEndcaps.push_back((0.446+0.455)/2.);
caloJetEndcaps.push_back((0.398+0.403)/2.);
caloJetEndcaps.push_back((0.318+0.320)/2.);
caloJetEndcaps.push_back((0.245+0.240)/2.);
caloJetEndcaps.push_back((0.188+0.188)/2.);
caloJetEndcaps.push_back((0.155+0.156)/2.);
caloJetEndcaps.push_back((0.131+0.135)/2.);
caloJetEndcaps.push_back((0.121+0.122)/2.);
caloJetEndcaps.push_back((0.112+0.106)/2.);
caloJetEndcaps.push_back((0.097+0.098)/2.);
caloJetEndcaps.push_back((0.085+0.087)/2.);
caloJetEndcaps.push_back((0.077+0.074)/2.);
caloJetEndcaps.push_back((0.071+0.069)/2.);
caloJetEndcaps.push_back((0.061+0.063)/2.);
caloJetEndcaps.push_back((0.054+0.054)/2.);
caloJetEndcaps.push_back((0.050+0.050)/2.);
caloJetEndcaps.push_back((0.048+0.047)/2.);
caloJetEndcaps.push_back((0.064+0.049)/2.);
caloJetEndcaps.push_back((0.014+0.000)/2.);
caloJetEndcaps.push_back(-1.);
caloJetEndcaps.push_back(-1.);
caloJetEndcaps.push_back(-1.);
*/

// 0 -1.4		
caloJetEndcaps.push_back((0.527+0.536)/2.);
caloJetEndcaps.push_back((0.446+0.455)/2.);
caloJetEndcaps.push_back((0.398+0.403)/2.);
caloJetEndcaps.push_back((0.318+0.320)/2.);
caloJetEndcaps.push_back((0.263+0.263)/2.);
caloJetEndcaps.push_back((0.201+0.201)/2.);
caloJetEndcaps.push_back((0.168+0.168)/2.);
caloJetEndcaps.push_back((0.142+0.142)/2.);
caloJetEndcaps.push_back((0.126+0.126)/2.);
caloJetEndcaps.push_back((0.113+0.113)/2.);
caloJetEndcaps.push_back((0.099+0.099)/2.);
caloJetEndcaps.push_back((0.087+0.087)/2.);
caloJetEndcaps.push_back((0.076+0.076)/2.);
caloJetEndcaps.push_back((0.066+0.066)/2.);
caloJetEndcaps.push_back((0.059+0.059)/2.);
caloJetEndcaps.push_back((0.049+0.049)/2.);
caloJetEndcaps.push_back((0.045+0.045)/2.);
caloJetEndcaps.push_back((0.040+0.040)/2.);
caloJetEndcaps.push_back((0.064+0.049)/2.);
caloJetEndcaps.push_back((0.014+0.000)/2.);
caloJetEndcaps.push_back(-1.);
caloJetEndcaps.push_back(-1.);
caloJetEndcaps.push_back(-1.);
			
caloJetBarrel.push_back((0.612+0.614)/2.);
caloJetBarrel.push_back((0.559+0.549)/2.);
caloJetBarrel.push_back((0.484+0.481)/2.);
caloJetBarrel.push_back((0.403+0.405)/2.);
caloJetBarrel.push_back((0.314+0.312)/2.);
caloJetBarrel.push_back((0.250+0.241)/2.);
caloJetBarrel.push_back((0.199+0.204)/2.);
caloJetBarrel.push_back((0.177+0.178)/2.);
caloJetBarrel.push_back((0.157+0.159)/2.);
caloJetBarrel.push_back((0.145+0.136)/2.);
caloJetBarrel.push_back((0.127+0.127)/2.);
caloJetBarrel.push_back((0.116+0.119)/2.);
caloJetBarrel.push_back((0.102+0.101)/2.);
caloJetBarrel.push_back((0.086+0.086)/2.);
caloJetBarrel.push_back((0.074+0.075)/2.);
caloJetBarrel.push_back((0.064+0.065)/2.);
caloJetBarrel.push_back((0.057+0.060)/2.);
caloJetBarrel.push_back((0.052+0.054)/2.);
caloJetBarrel.push_back((0.047+0.048)/2.);
caloJetBarrel.push_back((0.044+0.043)/2.);
caloJetBarrel.push_back((0.039+0.040)/2.);
caloJetBarrel.push_back((0.038+0.038)/2.);
caloJetBarrel.push_back((0.036+0.036)/2.);


pfJetEndcaps.push_back((0.310+0.311)/2.);
pfJetEndcaps.push_back((0.276+0.279)/2.);
pfJetEndcaps.push_back((0.254+0.255)/2.);
pfJetEndcaps.push_back((0.216+0.218)/2.);
pfJetEndcaps.push_back((0.178+0.179)/2.);
pfJetEndcaps.push_back((0.153+0.152)/2.);
pfJetEndcaps.push_back((0.135+0.136)/2.);
pfJetEndcaps.push_back((0.119+0.121)/2.);
pfJetEndcaps.push_back((0.108+0.113)/2.);
pfJetEndcaps.push_back((0.100+0.100)/2.);
pfJetEndcaps.push_back((0.092+0.088)/2.);
pfJetEndcaps.push_back((0.079+0.083)/2.);
pfJetEndcaps.push_back((0.072+0.072)/2.);
pfJetEndcaps.push_back((0.061+0.063)/2.);
pfJetEndcaps.push_back((0.055+0.056)/2.);
pfJetEndcaps.push_back((0.050+0.049)/2.);
pfJetEndcaps.push_back((0.046+0.049)/2.);
pfJetEndcaps.push_back((0.047+0.046)/2.);
pfJetEndcaps.push_back((0.052+0.046)/2.);
pfJetEndcaps.push_back((0.017+0.046)/2.);
pfJetEndcaps.push_back(-1.);
pfJetEndcaps.push_back(-1.);
pfJetEndcaps.push_back(-1.);


pfJetBarrel.push_back((0.236+0.236)/2.);
pfJetBarrel.push_back((0.217+0.213)/2.);			       
pfJetBarrel.push_back((0.199+0.199)/2.);			       
pfJetBarrel.push_back((0.180+0.179)/2.);			       
pfJetBarrel.push_back((0.155+0.154)/2.);			       
pfJetBarrel.push_back((0.135+0.138)/2.);			       
pfJetBarrel.push_back((0.124+0.131)/2.);
pfJetBarrel.push_back((0.117+0.121)/2.);
pfJetBarrel.push_back((0.115+0.113)/2.);
pfJetBarrel.push_back((0.109+0.110)/2.);
pfJetBarrel.push_back((0.102+0.104)/2.);
pfJetBarrel.push_back((0.096+0.097)/2.);
pfJetBarrel.push_back((0.088+0.088)/2.);
pfJetBarrel.push_back((0.077+0.077)/2.);
pfJetBarrel.push_back((0.067+0.064)/2.);
pfJetBarrel.push_back((0.056+0.057)/2.);
pfJetBarrel.push_back((0.053+0.053)/2.);
pfJetBarrel.push_back((0.047+0.048)/2.);
pfJetBarrel.push_back((0.043+0.042)/2.);
pfJetBarrel.push_back((0.039+0.040)/2.);
pfJetBarrel.push_back((0.037+0.038)/2.);
pfJetBarrel.push_back((0.036+0.036)/2.);
pfJetBarrel.push_back((0.035+0.035)/2.);

/*
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(0.160);
pfNewJetEndcaps.push_back(0.137);
pfNewJetEndcaps.push_back(0.124);
pfNewJetEndcaps.push_back(0.114);
pfNewJetEndcaps.push_back(0.103);
pfNewJetEndcaps.push_back(0.090);
pfNewJetEndcaps.push_back(0.078);
pfNewJetEndcaps.push_back(0.071);
pfNewJetEndcaps.push_back(0.067);
pfNewJetEndcaps.push_back(0.060);
pfNewJetEndcaps.push_back(0.045);
pfNewJetEndcaps.push_back(0.038);
pfNewJetEndcaps.push_back(0.041);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
pfNewJetEndcaps.push_back(-1.);
*/

pfNewJetEndcaps.push_back(0.180903);
pfNewJetEndcaps.push_back(0.144401);
pfNewJetEndcaps.push_back(0.117234);
pfNewJetEndcaps.push_back(0.0959775);
pfNewJetEndcaps.push_back(0.0880382);
pfNewJetEndcaps.push_back(0.0791627);
pfNewJetEndcaps.push_back(0.0692823);
pfNewJetEndcaps.push_back(0.0609338);
pfNewJetEndcaps.push_back(0.0547117);
pfNewJetEndcaps.push_back(0.0501923);
pfNewJetEndcaps.push_back(0.0449842);
pfNewJetEndcaps.push_back(0.0399663);

/*
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);			       
pfNewJetBarrel.push_back(-1.);			       
pfNewJetBarrel.push_back(-1.);			       
pfNewJetBarrel.push_back(0.146);			       
pfNewJetBarrel.push_back(0.129);			       
pfNewJetBarrel.push_back(0.120);
pfNewJetBarrel.push_back(0.111);
pfNewJetBarrel.push_back(0.105);
pfNewJetBarrel.push_back(0.102);
pfNewJetBarrel.push_back(0.095);
pfNewJetBarrel.push_back(0.088);
pfNewJetBarrel.push_back(0.076);
pfNewJetBarrel.push_back(0.071);
pfNewJetBarrel.push_back(0.058);
pfNewJetBarrel.push_back(0.051);
pfNewJetBarrel.push_back(0.050);
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);
pfNewJetBarrel.push_back(-1.);
*/

pfNewJetBarrel.push_back(0.1458);
pfNewJetBarrel.push_back(0.142328);
pfNewJetBarrel.push_back(0.120926);
pfNewJetBarrel.push_back(0.108925);
pfNewJetBarrel.push_back(0.0990595);
pfNewJetBarrel.push_back(0.0902588);
pfNewJetBarrel.push_back(0.0794255);
pfNewJetBarrel.push_back(0.072775);
pfNewJetBarrel.push_back(0.0692695);
pfNewJetBarrel.push_back(0.0622623);
pfNewJetBarrel.push_back(0.058287);
pfNewJetBarrel.push_back(0.0533926);

jetEnergy.push_back(7.5);
jetEnergy.push_back(11);
jetEnergy.push_back(13.5);
jetEnergy.push_back(17.5);
jetEnergy.push_back(23.5);
jetEnergy.push_back(31.);
jetEnergy.push_back(40.);
jetEnergy.push_back(51.);
jetEnergy.push_back(64.5);
jetEnergy.push_back(81.);
jetEnergy.push_back(105.);
jetEnergy.push_back(135.);
jetEnergy.push_back(175.);
jetEnergy.push_back(250.);
jetEnergy.push_back(350.);
jetEnergy.push_back(475.);
jetEnergy.push_back(650.);
jetEnergy.push_back(875.);
jetEnergy.push_back(1250.);
jetEnergy.push_back(1750.);
jetEnergy.push_back(2250.);
jetEnergy.push_back(2750.);
jetEnergy.push_back(3250.);

TGraph* grCaloBarrel = new TGraph ( 23, &jetEnergy[0], &caloJetBarrel[0] );
TGraph* grCaloEndcap = new TGraph ( 23, &jetEnergy[0], &caloJetEndcaps[0] );
TGraph* grJptBarrel = new TGraph ( 23, &jetEnergy[0], &jptJetBarrel[0] );
TGraph* grJptEndcap = new TGraph ( 23, &jetEnergy[0], &jptJetEndcaps[0] );
TGraph* grPfBarrel = new TGraph ( 23, &jetEnergy[0], &pfJetBarrel[0] );
TGraph* grPfEndcap = new TGraph ( 23, &jetEnergy[0], &pfJetEndcaps[0] );
//TGraph* grPfNewBarrel = new TGraph ( 23, &jetEnergy[0], &pfNewJetBarrel[0] );
//TGraph* grPfNewEndcap = new TGraph ( 23, &jetEnergy[0], &pfNewJetEndcaps[0] );
TGraph* grPfNewBarrel = new TGraph ( jetPt.size(), &jetPt[0], &pfNewJetBarrel[0] );
TGraph* grPfNewEndcap = new TGraph ( jetPt.size(), &jetPt[0], &pfNewJetEndcaps[0] );
TGraph* grAtlasBarrel = new TGraph ( 12, &jetAtlasPt[0], &jetAtlas[0] );

TCanvas *cBarrel = new TCanvas();
FormatPad(cBarrel,false);
cBarrel->cd();

TH2F *h = new TH2F("Barrel","", 
		   100, 15., 700., 100, 0.0, 0.45 );

FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet-Energy Resolution");
gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();
h->SetStats(0);
h->Draw();

gPad->SetGridx();
gPad->SetGridy();

TF1* pfBarrel = new TF1("pfBarrel","[0]+[1]/sqrt(x)+[2]/x+[3]/x/sqrt(x)",15,700);
TF1* caloBarrel = new TF1("caloBarrel","[0]+[1]/sqrt(x)+[2]/x+[3]/x/sqrt(x)",15,700);
pfBarrel->SetParameters(0.05.,1.0,1,1);
pfBarrel->FixParameter(3,0.);
caloBarrel->SetParameters(0.05,1.,1,1);
//caloBarrel->FixParameter(3,0.);
pfBarrel->SetLineColor(2);
caloBarrel->SetLineColor(4);
grPfNewBarrel->Fit("pfBarrel","","",15,700);
grCaloBarrel->Fit("caloBarrel","","",15,700);

grCaloBarrel->SetMarkerColor(4);						
grCaloBarrel->SetMarkerStyle(25);
grCaloBarrel->SetMarkerSize(1.2);
grCaloBarrel->SetLineWidth(2);
grCaloBarrel->SetLineColor(4);
grCaloBarrel->Draw("P");

grJptBarrel->SetMarkerColor(1);						
grJptBarrel->SetMarkerStyle(23);
grJptBarrel->SetMarkerSize(1.2);
grJptBarrel->SetLineWidth(2);
grJptBarrel->SetLineColor(1);
//grJptBarrel->Draw("CP");

/*
gPad->SaveAs("BarrelResolution.pdf");

TCanvas *cBarrelPF = new TCanvas();
cBarrelPF->cd();

gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();
h->Draw();
*/

grPfNewBarrel->SetMarkerColor(2);						
grPfNewBarrel->SetMarkerStyle(22);
grPfNewBarrel->SetMarkerSize(1.2);						
grPfNewBarrel->SetLineWidth(2);
grPfNewBarrel->SetLineColor(2);
grPfNewBarrel->Draw("P");

grAtlasBarrel->SetMarkerColor(3);						
grAtlasBarrel->SetMarkerStyle(21);
grAtlasBarrel->SetMarkerSize(1.2);						
grAtlasBarrel->SetLineWidth(2);
grAtlasBarrel->SetLineColor(3);
//grAtlasBarrel->Draw("CP");

TLegend *leg=new TLegend(0.55,0.65,0.85,0.85);
leg->AddEntry(grCaloBarrel, "Corrected Calo-Jets", "lp");
//leg->AddEntry(grJptBarrel, "JPT-Corrected Jets", "lp");
//leg->SetTextSize(0.03);
//leg->Draw();
//TLegend *leg=new TLegend(0.55,0.55,0.85,0.65);
leg->AddEntry(grPfNewBarrel, "Particle-Flow Jets", "lp");
//leg->AddEntry(grAtlasBarrel, "ATLAS", "lp");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(150,0.26,"0 < |#eta| < 1.5");

gPad->SaveAs("BarrelResolutionPFAndCalo.png");
gPad->SaveAs("BarrelResolutionPFAndCalo.pdf");

TCanvas *cEndcap = new TCanvas();
FormatPad(cEndcap,false);
cEndcap->cd();

TH2F *h2 = new TH2F("Endcap","", 
		   100, 15., 700., 100, 0.0, 0.45 );
FormatHisto(h2,sback);
h2->SetTitle( "CMS Preliminary" );
h2->SetXTitle("p_{T} [GeV/c]" );
h2->SetYTitle("Jet-Energy resolution");
gPad->SetLogx();
h2->SetStats(0);
h2->Draw();

gPad->SetGridx();
gPad->SetGridy();

TF1* pfEndcap = new TF1("pfEndcap","[0]+[1]/sqrt(x)+[2]/x+[3]/x/sqrt(x)",15,700);
TF1* caloEndcap = new TF1("caloEndcap","[0]+[1]/sqrt(x)+[2]/x+[3]/x/sqrt(x)",15,700);
pfEndcap->SetParameters(0.05.,1.0,1,1);
pfEndcap->FixParameter(3,0.);
caloEndcap->SetParameters(0.05,1.,1,1);
//caloEndcap->FixParameter(3,0.);
pfEndcap->SetLineColor(2);
caloEndcap->SetLineColor(4);
grPfNewEndcap->Fit("pfEndcap","","",15,700);
grCaloEndcap->Fit("caloEndcap","","",15,700);

grCaloEndcap->SetMarkerColor(4);						
grCaloEndcap->SetMarkerStyle(25);
grCaloEndcap->SetMarkerSize(1.2);
grCaloEndcap->Draw("P");
grCaloEndcap->SetLineWidth(2);
grCaloEndcap->SetLineColor(4);

grJptEndcap->SetMarkerColor(1);						
grJptEndcap->SetMarkerStyle(23);
grJptEndcap->SetMarkerSize(1.2);
grJptEndcap->SetLineWidth(2);
grJptEndcap->SetLineColor(1);
//grJptEndcap->Draw("CP");


/*
gPad->SaveAs("EndcapResolution.pdf");

TCanvas *cEndcapPF = new TCanvas();
cEndcapPF->cd();

gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();
h2->Draw();
*/

grPfNewEndcap->SetMarkerColor(2);						
grPfNewEndcap->SetMarkerStyle(22);
grPfNewEndcap->SetMarkerSize(1.2);						
grPfNewEndcap->SetLineWidth(2);
grPfNewEndcap->SetLineColor(2);
grPfNewEndcap->Draw("P");

TLegend *leg=new TLegend(0.55,0.65,0.85,0.85);
leg->AddEntry(grCaloEndcap, "Corrected Calo-Jets", "lp");
//leg->AddEntry(grJptEndcap, "JPT-Corrected Jets", "lp");
//leg->SetTextSize(0.03);
//leg->Draw();
//TLegend *leg=new TLegend(0.55,0.55,0.85,0.65);
leg->AddEntry(grPfNewEndcap, "Particle-Flow Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(150,0.26,"1.5 < |#eta| < 2.5");

gPad->SaveAs("EndcapResolutionPFAndCalo.png");
gPad->SaveAs("EndcapResolutionPFAndCalo.pdf");


}
