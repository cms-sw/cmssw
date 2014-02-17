/** \file GlobalDigisHistogrammer.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2011/09/12 09:11:30 $
 *  $Revision: 1.9 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalDigis/interface/GlobalDigisHistogrammer.h"
#include "DQMServices/Core/interface/DQMStore.h"

GlobalDigisHistogrammer::GlobalDigisHistogrammer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), theCSCStripPedestalSum(0),
  theCSCStripPedestalCount(0), count(0)
{
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_GlobalDigisHistogrammer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  outputfile = iPSet.getParameter<std::string>("outputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  GlobalDigisSrc_ = iPSet.getParameter<edm::InputTag>("GlobalDigisSrc");
  //ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  //ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  //ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  //HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");
  //SiStripSrc_ = iPSet.getParameter<edm::InputTag>("SiStripSrc"); 
  //SiPxlSrc_ = iPSet.getParameter<edm::InputTag>("SiPxlSrc");
  //MuDTSrc_ = iPSet.getParameter<edm::InputTag>("MuDTSrc");
  //MuCSCStripSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCStripSrc");
  //MuCSCWireSrc_ = iPSet.getParameter<edm::InputTag>("MuCSCWireSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  //produces<PGlobalDigi>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDHistogrammer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    OutputFile    = " << outputfile << "\n"
      << "    DoOutput      = " << doOutput << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    Global Src    = " << GlobalDigisSrc_ << "\n"

      << "===============================\n";
  }

  //Put in analyzer stuff here.... Pasted from Rec Hits... 

  dbe = 0;
dbe = edm::Service<DQMStore>().operator->();
if (dbe) {
    if (verbosity > 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
}
if (dbe) {
    if (verbosity > 0 ) dbe->showDirStructure();
  }

//monitor elements 

//Si Strip  ***Done***
 if(dbe)
   {
std::string SiStripString[19] = {"TECW1", "TECW2", "TECW3", "TECW4", "TECW5", "TECW6", "TECW7", "TECW8", "TIBL1", "TIBL2", "TIBL3", "TIBL4", "TIDW1", "TIDW2", "TIDW3", "TOBL1", "TOBL2", "TOBL3", "TOBL4"};
for(int i = 0; i<19; ++i)
{
  mehSiStripn[i]=0;
  mehSiStripADC[i]=0;
  mehSiStripStrip[i]=0;
}
dbe->setCurrentFolder("GlobalDigisV/SiStrips");
for(int amend = 0; amend < 19; ++amend)
{ 
  mehSiStripn[amend] = dbe->book1D("hSiStripn_"+SiStripString[amend], SiStripString[amend]+"  Digis",500,0.,1000.);
  mehSiStripn[amend]->setAxisTitle("Number of Digis",1);
  mehSiStripn[amend]->setAxisTitle("Count",2);
  mehSiStripADC[amend] = dbe->book1D("hSiStripADC_"+SiStripString[amend],SiStripString[amend]+" ADC",150,0.0,300.);
  mehSiStripADC[amend]->setAxisTitle("ADC",1);
  mehSiStripADC[amend]->setAxisTitle("Count",2);
  mehSiStripStrip[amend] = dbe->book1D("hSiStripStripADC_"+SiStripString[amend],SiStripString[amend]+" Strip",200,0.0,800.);
  mehSiStripStrip[amend]->setAxisTitle("Strip Number",1);
  mehSiStripStrip[amend]->setAxisTitle("Count",2);
}


//HCal  **DONE**
std::string HCalString[4] = {"HB", "HE", "HO","HF"}; 
float calnUpper[4] = {3000.,3000.,3000.,2000.}; float calnLower[4]={2000.,2000.,2000.,1000.}; 
float SHEUpper[4]={0.05,.05,0.05,20};
float SHEvAEEUpper[4] = {5000, 5000, 5000, 20}; float SHEvAEELower[4] = {-5000, -5000, -5000, -20}; 
int SHEvAEEnBins[4] = {200,200,200,40};
double ProfileUpper[4] = {1.,1.,1.,20.};  

for(int i =0; i<4; ++i)
{
  mehHcaln[i]=0;
  mehHcalAEE[i]=0;
  mehHcalSHE[i]=0;
  mehHcalAEESHE[i]=0;
  mehHcalSHEvAEE[i]=0;
}
dbe->setCurrentFolder("GlobalDigisV/HCals");
 
for(int amend = 0; amend < 4; ++amend)
{
  mehHcaln[amend] = dbe->book1D("hHcaln_"+HCalString[amend],HCalString[amend]+"  digis", 1000, calnLower[amend], calnUpper[amend]);
  mehHcaln[amend]->setAxisTitle("Number of Digis",1);
  mehHcaln[amend]->setAxisTitle("Count",2);
  mehHcalAEE[amend] = dbe->book1D("hHcalAEE_"+HCalString[amend],HCalString[amend]+"Cal AEE", 60, -10., 50.);
  mehHcalAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
  mehHcalAEE[amend]->setAxisTitle("Count",2);
  mehHcalSHE[amend] = dbe->book1D("hHcalSHE_"+HCalString[amend],HCalString[amend]+"Cal SHE", 100, 0.0, SHEUpper[amend]);
  mehHcalSHE[amend]->setAxisTitle("Simulated Hit Energy",1);
  mehHcalSHE[amend]->setAxisTitle("Count",2);
  mehHcalAEESHE[amend] = dbe->book1D("hHcalAEESHE_"+HCalString[amend], HCalString[amend]+"Cal AEE/SHE", SHEvAEEnBins[amend], SHEvAEELower[amend], SHEvAEEUpper[amend]);
  mehHcalAEESHE[amend]->setAxisTitle("ADC / SHE",1);
  mehHcalAEESHE[amend]->setAxisTitle("Count",2);
  
  //************  Not sure how to do Profile ME **************
  mehHcalSHEvAEE[amend] = dbe->bookProfile("hHcalSHEvAEE_"+HCalString[amend],HCalString[amend]+"Cal SHE vs. AEE", 60, (float)-10., (float)50., 100, (float)0., (float)ProfileUpper[amend],"");
  mehHcalSHEvAEE[amend]->setAxisTitle("AEE / SHE",1);
  mehHcalSHEvAEE[amend]->setAxisTitle("SHE",2);

}




//Ecal **Done **
std::string ECalString[2] = {"EB","EE"}; 

for(int i =0; i<2; ++i)
{
  mehEcaln[i]=0;
  mehEcalAEE[i]=0;
  mehEcalSHE[i]=0;
  mehEcalMaxPos[i]=0;
  mehEcalMultvAEE[i]=0;
  mehEcalSHEvAEESHE[i]=0;
}
dbe->setCurrentFolder("GlobalDigisV/ECals");
 
for(int amend = 0; amend < 2; ++amend)
{
  mehEcaln[amend] = dbe->book1D("hEcaln_"+ECalString[amend],ECalString[amend]+"  digis", 300, 1000., 4000.);
  mehEcaln[amend]->setAxisTitle("Number of Digis",1);
  mehEcaln[amend]->setAxisTitle("Count",2);
  mehEcalAEE[amend] = dbe->book1D("hEcalAEE_"+ECalString[amend],ECalString[amend]+"Cal AEE", 100, 0., 1.);
  mehEcalAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
  mehEcalAEE[amend]->setAxisTitle("Count",2);
  mehEcalSHE[amend] = dbe->book1D("hEcalSHE_"+ECalString[amend],ECalString[amend]+"Cal SHE", 50, 0., 5.);
  mehEcalSHE[amend]->setAxisTitle("Simulated Hit Energy",1);
  mehEcalSHE[amend]->setAxisTitle("Count",2);
  mehEcalMaxPos[amend] = dbe->book1D("hEcalMaxPos_"+ECalString[amend],ECalString[amend]+"Cal MaxPos",10, 0., 10.);
  mehEcalMaxPos[amend]->setAxisTitle("Maximum Position",1);
  mehEcalMaxPos[amend]->setAxisTitle("Count",2);
  
  //************  Not sure how to do Profile ME **************
  mehEcalSHEvAEESHE[amend] = dbe->bookProfile("hEcalSHEvAEESHE_"+ECalString[amend],ECalString[amend]+"Cal SHE vs. AEE/SHE",100, (float)0., (float)10., 50, (float)0., (float)5.,"");
  mehEcalSHEvAEESHE[amend]->setAxisTitle("AEE / SHE",1);
  mehEcalSHEvAEESHE[amend]->setAxisTitle("SHE",2);
  mehEcalMultvAEE[amend] = dbe->bookProfile("hEcalMultvAEE_"+ECalString[amend],ECalString[amend]+"Cal Multi vs. AEE", 100, (float)0., (float)10., 400, (float)0., (float)4000.,"");
  mehEcalMultvAEE[amend]->setAxisTitle("Analog Equivalent Energy",1);
  mehEcalMultvAEE[amend]->setAxisTitle("Number of Digis",2);



}
  mehEcaln[2] = 0;
  mehEcaln[2] = dbe->book1D("hEcaln_ES","ESCAL  digis", 100, 0., 500.);
  mehEcaln[2]->setAxisTitle("Number of Digis",1);
  mehEcaln[2]->setAxisTitle("Count",2);
  std::string ADCNumber[3] = {"0", "1", "2"};
  for(int i =0; i<3; ++i)
    {
      mehEScalADC[i] = 0;
      mehEScalADC[i] = dbe->book1D("hEcalADC"+ADCNumber[i]+"_ES","ESCAL  ADC"+ADCNumber[i], 150, 950., 1500.);
      mehEScalADC[i]->setAxisTitle("ADC"+ADCNumber[i],1);
      mehEScalADC[i]->setAxisTitle("Count",2);

    }

//Si Pixels ***DONE***  
std::string SiPixelString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", "FWD2n", "FWD2p"};
for(int j =0; j<7; ++j)
{
  mehSiPixeln[j]=0;
  mehSiPixelADC[j]=0;
  mehSiPixelRow[j]=0;
  mehSiPixelCol[j]=0;
}

dbe->setCurrentFolder("GlobalDigisV/SiPixels");
for(int amend = 0; amend < 7; ++amend)
{
  if(amend<3) mehSiPixeln[amend] = dbe->book1D("hSiPixeln_"+SiPixelString[amend],SiPixelString[amend]+" Digis",50,0.,100.);
  else mehSiPixeln[amend] = dbe->book1D("hSiPixeln_"+SiPixelString[amend],SiPixelString[amend]+" Digis",25,0.,50.);
  mehSiPixeln[amend]->setAxisTitle("Number of Digis",1);
  mehSiPixeln[amend]->setAxisTitle("Count",2);
  mehSiPixelADC[amend] = dbe->book1D("hSiPixelADC_"+SiPixelString[amend],SiPixelString[amend]+" ADC",150,0.0,300.);
  mehSiPixelADC[amend]->setAxisTitle("ADC",1);
  mehSiPixelADC[amend]->setAxisTitle("Count",2);
  mehSiPixelRow[amend] = dbe->book1D("hSiPixelRow_"+SiPixelString[amend],SiPixelString[amend]+" Row",100,0.0,100.);
  mehSiPixelRow[amend]->setAxisTitle("Row Number",1);
  mehSiPixelRow[amend]->setAxisTitle("Count",2);
  mehSiPixelCol[amend] = dbe->book1D("hSiPixelColumn_"+SiPixelString[amend],SiPixelString[amend]+" Column",200,0.0,500.);
  mehSiPixelCol[amend]->setAxisTitle("Column Number",1);
  mehSiPixelCol[amend]->setAxisTitle("Count",2);
}
//Muons ***DONE****
dbe->setCurrentFolder("GlobalDigisV/Muons");
std::string MuonString[4] = {"MB1", "MB2", "MB3", "MB4"};

for(int i =0; i < 4; ++i)
{
  mehDtMuonn[i] = 0;
  mehDtMuonLayer[i] = 0;
  mehDtMuonTime[i] = 0;
  mehDtMuonTimevLayer[i] = 0;
}

for(int j = 0; j < 4; ++j)
{
  mehDtMuonn[j] = dbe->book1D("hDtMuonn_"+MuonString[j],MuonString[j]+"  digis",25, 0., 50.);
  mehDtMuonn[j]->setAxisTitle("Number of Digis",1);
  mehDtMuonn[j]->setAxisTitle("Count",2);
  mehDtMuonLayer[j] = dbe->book1D("hDtLayer_"+MuonString[j],MuonString[j]+"  Layer",12, 1., 13.);
  mehDtMuonLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer",1);
  mehDtMuonLayer[j]->setAxisTitle("Count",2);
  mehDtMuonTime[j] = dbe->book1D("hDtMuonTime_"+MuonString[j],MuonString[j]+"  Time",300, 400., 1000.);
  mehDtMuonTime[j]->setAxisTitle("Time",1);
  mehDtMuonTime[j]->setAxisTitle("Count",2);
  mehDtMuonTimevLayer[j] = dbe->bookProfile("hDtMuonTimevLayer_"+MuonString[j],MuonString[j]+"  Time vs. Layer",12, 1., 13., 300, 400., 1000.,"");
  mehDtMuonTimevLayer[j]->setAxisTitle("4 * (SuperLayer - 1) + Layer",1);
  mehDtMuonTimevLayer[j]->setAxisTitle("Time",2);
}

//  ****  Have to do CSC and RPC now *****
//CSC 
mehCSCStripn = 0;
mehCSCStripn = dbe->book1D("hCSCStripn","CSC Strip digis",25, 0., 50.);
mehCSCStripn->setAxisTitle("Number of Digis",1);
mehCSCStripn->setAxisTitle("Count",2);

mehCSCStripADC = 0;
mehCSCStripADC = dbe->book1D("hCSCStripADC","CSC Strip ADC", 110, 0., 1100.);
mehCSCStripADC->setAxisTitle("ADC",1);
mehCSCStripADC->setAxisTitle("Count",2);

mehCSCWiren = 0;
mehCSCWiren = dbe->book1D("hCSCWiren","CSC Wire digis",25, 0., 50.);
mehCSCWiren->setAxisTitle("Number of Digis",1);
mehCSCWiren->setAxisTitle("Count",2);



mehCSCWireTime = 0;
mehCSCWiren = dbe->book1D("hCSCWireTime","CSC Wire Time",10, 0., 10.);
mehCSCWiren->setAxisTitle("Time",1);
mehCSCWiren->setAxisTitle("Count",2);
 

}

}

  // set default constants
  // ECal

  //ECalgainConv_[0] = 0.;
  //ECalgainConv_[1] = 1.;
  //ECalgainConv_[2] = 2.;
  //ECalgainConv_[3] = 12.;  
  //ECalbarrelADCtoGeV_ = 0.035;
  //ECalendcapADCtoGeV_ = 0.06;



GlobalDigisHistogrammer::~GlobalDigisHistogrammer() 
{
  if (doOutput)
    if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
}

void GlobalDigisHistogrammer::beginJob( void )
{
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_beginJob";

  // setup calorimeter constants from service
  //edm::ESHandle<EcalADCToGeVConstant> pAgc;
  //iSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  //const EcalADCToGeVConstant* agc = pAgc.product();
  
  //EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  // ECalgainConv_[0] = 0.;
  // ECalgainConv_[1] = 1.;
  // // ECalgainConv_[2] = defaultRatios->gain12Over6() ;
  //ECalgainConv_[3] = ECalgainConv_[2]*(defaultRatios->gain6Over1()) ;

  //delete defaultRatios;

  //ECalbarrelADCtoGeV_ = agc->getEBValue();
  //ECalendcapADCtoGeV_ = agc->getEEValue();

  //if (verbosity >= 0) {
  // edm::LogInfo(MsgLoggerCat) 
  // << "Modified Calorimeter gain constants: g0 = " << ECalgainConv_[0]
  //<< ", g1 = " << ECalgainConv_[1] << ", g2 = " << ECalgainConv_[2]
  // << ", g3 = " << ECalgainConv_[3];
  // edm::LogInfo(MsgLoggerCat)
  //  << "Modified Calorimeter ADCtoGeV constants: barrel = " 
  //  << ECalbarrelADCtoGeV_ << ", endcap = " << ECalendcapADCtoGeV_;
  //}

  // clear storage vectors
  //clear();
  return;
}

void GlobalDigisHistogrammer::endJob()
{
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalDigisHistogrammer::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisHistogrammer_analyze";

  // keep track of number of events processed
  ++count;

  // get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << ", event " << nevt
      << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt%frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << ", event " << nevt
	<< " (" << count << " events total)";
    }
  }

  // clear event holders
  //clear();

  // look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat)
	<< "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");      

      for (unsigned int i = 0; i < AllProv.size(); ++i) {
	eventout += "\n       ******************************";
	eventout += "\n       Module       : ";
	//eventout += (AllProv[i]->product).moduleLabel();
	eventout += AllProv[i]->moduleLabel();
	eventout += "\n       ProductID    : ";
	//eventout += (AllProv[i]->product).productID_.id_;
	eventout += AllProv[i]->productID().id();
	eventout += "\n       ClassName    : ";
	//eventout += (AllProv[i]->product).fullClassName_;
	eventout += AllProv[i]->className();
	eventout += "\n       InstanceName : ";
	//eventout += (AllProv[i]->product).productInstanceName_;
	eventout += AllProv[i]->productInstanceName();
	eventout += "\n       BranchName   : ";
	//eventout += (AllProv[i]->product).branchName_;
	eventout += AllProv[i]->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
      getAllProvenances = false;
    }
edm::Handle<PGlobalDigi> srcGlobalDigis;
  iEvent.getByLabel(GlobalDigisSrc_,srcGlobalDigis);
  if (!srcGlobalDigis.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find PGlobalDigis in event!";
    return;

  }

    

    int nEBCalDigis = srcGlobalDigis->getnEBCalDigis();
    int nEECalDigis = srcGlobalDigis->getnEECalDigis();
    int nESCalDigis = srcGlobalDigis->getnESCalDigis();

    int nHBCalDigis = srcGlobalDigis->getnHBCalDigis();
    int nHECalDigis = srcGlobalDigis->getnHECalDigis();
    int nHOCalDigis = srcGlobalDigis->getnHOCalDigis();
    int nHFCalDigis = srcGlobalDigis->getnHFCalDigis();        

    int nTIBL1Digis = srcGlobalDigis->getnTIBL1Digis();    
    int nTIBL2Digis = srcGlobalDigis->getnTIBL2Digis();    
    int nTIBL3Digis = srcGlobalDigis->getnTIBL3Digis();    
    int nTIBL4Digis = srcGlobalDigis->getnTIBL4Digis();    
    int nTOBL1Digis = srcGlobalDigis->getnTOBL1Digis();    
    int nTOBL2Digis = srcGlobalDigis->getnTOBL2Digis();    
    int nTOBL3Digis = srcGlobalDigis->getnTOBL3Digis();    
    int nTOBL4Digis = srcGlobalDigis->getnTOBL4Digis();    
    int nTIDW1Digis = srcGlobalDigis->getnTIDW1Digis();    
    int nTIDW2Digis = srcGlobalDigis->getnTIDW2Digis();    
    int nTIDW3Digis = srcGlobalDigis->getnTIDW3Digis();    
    int nTECW1Digis = srcGlobalDigis->getnTECW1Digis();    
    int nTECW2Digis = srcGlobalDigis->getnTECW2Digis();    
    int nTECW3Digis = srcGlobalDigis->getnTECW3Digis();  
    int nTECW4Digis = srcGlobalDigis->getnTECW4Digis();    
    int nTECW5Digis = srcGlobalDigis->getnTECW5Digis();    
    int nTECW6Digis = srcGlobalDigis->getnTECW6Digis();  
    int nTECW7Digis = srcGlobalDigis->getnTECW7Digis();    
    int nTECW8Digis = srcGlobalDigis->getnTECW8Digis();

    int nBRL1Digis = srcGlobalDigis->getnBRL1Digis();    
    int nBRL2Digis = srcGlobalDigis->getnBRL2Digis();    
    int nBRL3Digis = srcGlobalDigis->getnBRL3Digis();       
    int nFWD1nDigis = srcGlobalDigis->getnFWD1nDigis();
    int nFWD1pDigis = srcGlobalDigis->getnFWD1pDigis();    
    int nFWD2nDigis = srcGlobalDigis->getnFWD2nDigis();    
    int nFWD2pDigis = srcGlobalDigis->getnFWD2pDigis(); 
  
    int nMB1Digis = srcGlobalDigis->getnMB1Digis();  
    int nMB2Digis = srcGlobalDigis->getnMB2Digis();  
    int nMB3Digis = srcGlobalDigis->getnMB3Digis();  
    int nMB4Digis = srcGlobalDigis->getnMB4Digis();  

    int nCSCstripDigis = srcGlobalDigis->getnCSCstripDigis();

    int nCSCwireDigis = srcGlobalDigis->getnCSCwireDigis();

  // get Ecal info
    std::vector<PGlobalDigi::ECalDigi> EECalDigis = 
      srcGlobalDigis->getEECalDigis();
    mehEcaln[0]->Fill((float)nEECalDigis);
    for (unsigned int i = 0; i < EECalDigis.size(); ++i) {
      mehEcalAEE[0]->Fill(EECalDigis[i].AEE);
      mehEcalMaxPos[0]->Fill(EECalDigis[i].maxPos);
      mehEcalMultvAEE[0]->Fill(EECalDigis[i].AEE,(float)nEECalDigis,1);
      if (EECalDigis[i].SHE != 0.) {
	mehEcalSHE[0]->Fill(EECalDigis[i].SHE);
	mehEcalSHEvAEESHE[0]->
	  Fill(EECalDigis[i].AEE/EECalDigis[i].SHE,EECalDigis[i].SHE,1);
      }
    }
    
    std::vector<PGlobalDigi::ECalDigi> EBCalDigis = 
      srcGlobalDigis->getEBCalDigis();
    mehEcaln[1]->Fill((float)nEBCalDigis);
    for (unsigned int i = 0; i < EBCalDigis.size(); ++i) {
      mehEcalAEE[1]->Fill(EBCalDigis[i].AEE);
      mehEcalMaxPos[1]->Fill(EBCalDigis[i].maxPos);
      mehEcalMultvAEE[1]->Fill(EBCalDigis[i].AEE,(float)nEBCalDigis,1);
      if (EBCalDigis[i].SHE != 0.) {
	mehEcalSHE[1]->Fill(EBCalDigis[i].SHE);
	mehEcalSHEvAEESHE[1]->
	  Fill(EBCalDigis[i].AEE/EBCalDigis[i].SHE,EBCalDigis[i].SHE,1);
      }
    }
    
    std::vector<PGlobalDigi::ESCalDigi> ESCalDigis = 
      srcGlobalDigis->getESCalDigis();   
    mehEcaln[2]->Fill((float)nESCalDigis);
    for (unsigned int i = 0; i < ESCalDigis.size(); ++i) {
      mehEScalADC[0]->Fill(ESCalDigis[i].ADC0);
      mehEScalADC[1]->Fill(ESCalDigis[i].ADC1);
      mehEScalADC[2]->Fill(ESCalDigis[i].ADC2);
    }
    
    // Get HCal info
    std::vector<PGlobalDigi::HCalDigi> HBCalDigis = 
      srcGlobalDigis->getHBCalDigis();
    mehHcaln[0]->Fill((float)nHBCalDigis);
    for (unsigned int i = 0; i < HBCalDigis.size(); ++i) {
      mehHcalAEE[0]->Fill(HBCalDigis[i].AEE);
      if (HBCalDigis[i].SHE != 0.) {
	mehHcalSHE[0]->Fill(HBCalDigis[i].SHE);
	mehHcalAEESHE[0]->Fill(HBCalDigis[i].AEE/HBCalDigis[i].SHE);
	mehHcalSHEvAEE[0]->
	  Fill(HBCalDigis[i].AEE,HBCalDigis[i].SHE,1);
      }
    }
    std::vector<PGlobalDigi::HCalDigi> HECalDigis = 
      srcGlobalDigis->getHECalDigis();
    mehHcaln[1]->Fill((float)nHECalDigis);
    for (unsigned int i = 0; i < HECalDigis.size(); ++i) {
      mehHcalAEE[1]->Fill(HECalDigis[i].AEE);
      if (HECalDigis[i].SHE != 0.) {
	mehHcalSHE[1]->Fill(HECalDigis[i].SHE);
	mehHcalAEESHE[1]->Fill(HECalDigis[i].AEE/HECalDigis[i].SHE);
	mehHcalSHEvAEE[1]->
	  Fill(HECalDigis[i].AEE,HECalDigis[i].SHE,1);
      }
    }

    std::vector<PGlobalDigi::HCalDigi> HOCalDigis = 
      srcGlobalDigis->getHOCalDigis();
    mehHcaln[2]->Fill((float)nHOCalDigis);
    for (unsigned int i = 0; i < HOCalDigis.size(); ++i) {
      mehHcalAEE[2]->Fill(HOCalDigis[i].AEE);
      if (HOCalDigis[i].SHE != 0.) {
	mehHcalSHE[2]->Fill(HOCalDigis[i].SHE);
	mehHcalAEESHE[2]->Fill(HOCalDigis[i].AEE/HOCalDigis[i].SHE);
	mehHcalSHEvAEE[2]->
	  Fill(HOCalDigis[i].AEE,HOCalDigis[i].SHE,1);
      }
    }

    std::vector<PGlobalDigi::HCalDigi> HFCalDigis = 
      srcGlobalDigis->getHFCalDigis();
    mehHcaln[3]->Fill((float)nHFCalDigis);
    for (unsigned int i = 0; i < HFCalDigis.size(); ++i) {
      mehHcalAEE[3]->Fill(HFCalDigis[i].AEE);
      if (HFCalDigis[i].SHE != 0.) {
	mehHcalSHE[3]->Fill(HFCalDigis[i].SHE);
	mehHcalAEESHE[3]->Fill(HFCalDigis[i].AEE/HFCalDigis[i].SHE);
	mehHcalSHEvAEE[3]->
	  Fill(HFCalDigis[i].AEE,HFCalDigis[i].SHE,1);
      }
    }

    // get SiStrip info
    std::vector<PGlobalDigi::SiStripDigi> TIBL1Digis =
      srcGlobalDigis->getTIBL1Digis();      
    mehSiStripn[0]->Fill((float)nTIBL1Digis);
    for (unsigned int i = 0; i < TIBL1Digis.size(); ++i) {
      mehSiStripADC[0]->Fill(TIBL1Digis[i].ADC);
      mehSiStripStrip[0]->Fill(TIBL1Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL2Digis =
      srcGlobalDigis->getTIBL2Digis();      
    mehSiStripn[1]->Fill((float)nTIBL2Digis);
    for (unsigned int i = 0; i < TIBL2Digis.size(); ++i) {
      mehSiStripADC[1]->Fill(TIBL2Digis[i].ADC);
      mehSiStripStrip[1]->Fill(TIBL2Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL3Digis =
      srcGlobalDigis->getTIBL3Digis();      
    mehSiStripn[2]->Fill((float)nTIBL3Digis);
    for (unsigned int i = 0; i < TIBL3Digis.size(); ++i) {
      mehSiStripADC[2]->Fill(TIBL3Digis[i].ADC);
      mehSiStripStrip[2]->Fill(TIBL3Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIBL4Digis =
      srcGlobalDigis->getTIBL4Digis();      
    mehSiStripn[3]->Fill((float)nTIBL4Digis);
    for (unsigned int i = 0; i < TIBL4Digis.size(); ++i) {
      mehSiStripADC[3]->Fill(TIBL4Digis[i].ADC);
      mehSiStripStrip[3]->Fill(TIBL4Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL1Digis =
      srcGlobalDigis->getTOBL1Digis();      
    mehSiStripn[4]->Fill((float)nTOBL1Digis);
    for (unsigned int i = 0; i < TOBL1Digis.size(); ++i) {
      mehSiStripADC[4]->Fill(TOBL1Digis[i].ADC);
      mehSiStripStrip[4]->Fill(TOBL1Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL2Digis =
      srcGlobalDigis->getTOBL2Digis();      
    mehSiStripn[5]->Fill((float)nTOBL2Digis);
    for (unsigned int i = 0; i < TOBL2Digis.size(); ++i) {
      mehSiStripADC[5]->Fill(TOBL2Digis[i].ADC);
      mehSiStripStrip[5]->Fill(TOBL2Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL3Digis =
      srcGlobalDigis->getTOBL3Digis();      
    mehSiStripn[6]->Fill((float)nTOBL3Digis);
    for (unsigned int i = 0; i < TOBL3Digis.size(); ++i) {
      mehSiStripADC[6]->Fill(TOBL3Digis[i].ADC);
      mehSiStripStrip[6]->Fill(TOBL3Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TOBL4Digis =
      srcGlobalDigis->getTOBL4Digis();      
    mehSiStripn[7]->Fill((float)nTOBL4Digis);
    for (unsigned int i = 0; i < TOBL4Digis.size(); ++i) {
      mehSiStripADC[7]->Fill(TOBL4Digis[i].ADC);
      mehSiStripStrip[7]->Fill(TOBL4Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW1Digis =
      srcGlobalDigis->getTIDW1Digis();      
    mehSiStripn[8]->Fill((float)nTIDW1Digis);
    for (unsigned int i = 0; i < TIDW1Digis.size(); ++i) {
      mehSiStripADC[8]->Fill(TIDW1Digis[i].ADC);
      mehSiStripStrip[8]->Fill(TIDW1Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW2Digis =
      srcGlobalDigis->getTIDW2Digis();      
    mehSiStripn[9]->Fill((float)nTIDW2Digis);
    for (unsigned int i = 0; i < TIDW2Digis.size(); ++i) {
      mehSiStripADC[9]->Fill(TIDW2Digis[i].ADC);
      mehSiStripStrip[9]->Fill(TIDW2Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TIDW3Digis =
      srcGlobalDigis->getTIDW3Digis();      
    mehSiStripn[10]->Fill((float)nTIDW3Digis);
    for (unsigned int i = 0; i < TIDW3Digis.size(); ++i) {
      mehSiStripADC[10]->Fill(TIDW3Digis[i].ADC);
      mehSiStripStrip[10]->Fill(TIDW3Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW1Digis =
      srcGlobalDigis->getTECW1Digis();      
    mehSiStripn[11]->Fill((float)nTECW1Digis);
    for (unsigned int i = 0; i < TECW1Digis.size(); ++i) {
      mehSiStripADC[11]->Fill(TECW1Digis[i].ADC);
      mehSiStripStrip[11]->Fill(TECW1Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW2Digis =
      srcGlobalDigis->getTECW2Digis();      
    mehSiStripn[12]->Fill((float)nTECW2Digis);
    for (unsigned int i = 0; i < TECW2Digis.size(); ++i) {
      mehSiStripADC[12]->Fill(TECW2Digis[i].ADC);
      mehSiStripStrip[12]->Fill(TECW2Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW3Digis =
      srcGlobalDigis->getTECW3Digis();      
    mehSiStripn[13]->Fill((float)nTECW3Digis);
    for (unsigned int i = 0; i < TECW3Digis.size(); ++i) {
      mehSiStripADC[13]->Fill(TECW3Digis[i].ADC);
      mehSiStripStrip[13]->Fill(TECW3Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW4Digis =
      srcGlobalDigis->getTECW4Digis();      
    mehSiStripn[14]->Fill((float)nTECW4Digis);
    for (unsigned int i = 0; i < TECW4Digis.size(); ++i) {
      mehSiStripADC[14]->Fill(TECW4Digis[i].ADC);
      mehSiStripStrip[14]->Fill(TECW4Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW5Digis =
      srcGlobalDigis->getTECW5Digis();      
    mehSiStripn[15]->Fill((float)nTECW5Digis);
    for (unsigned int i = 0; i < TECW5Digis.size(); ++i) {
      mehSiStripADC[15]->Fill(TECW5Digis[i].ADC);
      mehSiStripStrip[15]->Fill(TECW5Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW6Digis =
      srcGlobalDigis->getTECW6Digis();      
    mehSiStripn[16]->Fill((float)nTECW6Digis);
    for (unsigned int i = 0; i < TECW6Digis.size(); ++i) {
      mehSiStripADC[16]->Fill(TECW6Digis[i].ADC);
      mehSiStripStrip[16]->Fill(TECW6Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW7Digis =
      srcGlobalDigis->getTECW7Digis();      
    mehSiStripn[17]->Fill((float)nTECW7Digis);
    for (unsigned int i = 0; i < TECW7Digis.size(); ++i) {
      mehSiStripADC[17]->Fill(TECW7Digis[i].ADC);
      mehSiStripStrip[17]->Fill(TECW7Digis[i].STRIP);
    }

    std::vector<PGlobalDigi::SiStripDigi> TECW8Digis =
      srcGlobalDigis->getTECW8Digis();      
    mehSiStripn[18]->Fill((float)nTECW8Digis);
    for (unsigned int i = 0; i < TECW8Digis.size(); ++i) {
      mehSiStripADC[18]->Fill(TECW8Digis[i].ADC);
      mehSiStripStrip[18]->Fill(TECW8Digis[i].STRIP);
    }

    // get SiPixel info
    std::vector<PGlobalDigi::SiPixelDigi> BRL1Digis =
      srcGlobalDigis->getBRL1Digis();      
    mehSiPixeln[0]->Fill((float)nBRL1Digis);
    for (unsigned int i = 0; i < BRL1Digis.size(); ++i) {
      mehSiPixelADC[0]->Fill(BRL1Digis[i].ADC);
      mehSiPixelRow[0]->Fill(BRL1Digis[i].ROW);
      mehSiPixelCol[0]->Fill(BRL1Digis[i].COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> BRL2Digis =
      srcGlobalDigis->getBRL2Digis();      
    mehSiPixeln[1]->Fill((float)nBRL2Digis);
    for (unsigned int i = 0; i < BRL2Digis.size(); ++i) {
      mehSiPixelADC[1]->Fill(BRL2Digis[i].ADC);
      mehSiPixelRow[1]->Fill(BRL2Digis[i].ROW);
      mehSiPixelCol[1]->Fill(BRL2Digis[i].COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> BRL3Digis =
      srcGlobalDigis->getBRL3Digis();      
    mehSiPixeln[2]->Fill((float)nBRL3Digis);
    for (unsigned int i = 0; i < BRL3Digis.size(); ++i) {
      mehSiPixelADC[2]->Fill(BRL3Digis[i].ADC);
      mehSiPixelRow[2]->Fill(BRL3Digis[i].ROW);
      mehSiPixelCol[2]->Fill(BRL3Digis[i].COLUMN); 
   }

    std::vector<PGlobalDigi::SiPixelDigi> FWD1pDigis =
      srcGlobalDigis->getFWD1pDigis();      
    mehSiPixeln[3]->Fill((float)nFWD1pDigis);
    for (unsigned int i = 0; i < FWD1pDigis.size(); ++i) {
      mehSiPixelADC[3]->Fill(FWD1pDigis[i].ADC);
      mehSiPixelRow[3]->Fill(FWD1pDigis[i].ROW);
      mehSiPixelCol[3]->Fill(FWD1pDigis[i].COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD1nDigis =
      srcGlobalDigis->getFWD1nDigis();      
    mehSiPixeln[4]->Fill((float)nFWD1nDigis);
    for (unsigned int i = 0; i < FWD1nDigis.size(); ++i) {
      mehSiPixelADC[4]->Fill(FWD1nDigis[i].ADC);
      mehSiPixelRow[4]->Fill(FWD1nDigis[i].ROW);
      mehSiPixelCol[4]->Fill(FWD1nDigis[i].COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD2pDigis =
      srcGlobalDigis->getFWD2pDigis();      
    mehSiPixeln[5]->Fill((float)nFWD2pDigis);
    for (unsigned int i = 0; i < FWD2pDigis.size(); ++i) {
      mehSiPixelADC[5]->Fill(FWD2pDigis[i].ADC);
      mehSiPixelRow[5]->Fill(FWD2pDigis[i].ROW);
      mehSiPixelCol[5]->Fill(FWD2pDigis[i].COLUMN);
    }

    std::vector<PGlobalDigi::SiPixelDigi> FWD2nDigis =
      srcGlobalDigis->getFWD2nDigis();      
    mehSiPixeln[6]->Fill((float)nFWD2nDigis);
    for (unsigned int i = 0; i < FWD2nDigis.size(); ++i) {
      mehSiPixelADC[6]->Fill(FWD2nDigis[i].ADC);
      mehSiPixelRow[6]->Fill(FWD2nDigis[i].ROW);
      mehSiPixelCol[6]->Fill(FWD2nDigis[i].COLUMN);
    }

    // get DtMuon info
    std::vector<PGlobalDigi::DTDigi> MB1Digis =
      srcGlobalDigis->getMB1Digis();      
    mehDtMuonn[0]->Fill((float)nMB1Digis);
    for (unsigned int i = 0; i < MB1Digis.size(); ++i) {
      float layer = 4.0 * (MB1Digis[i].SLAYER - 1.0) + MB1Digis[i].LAYER;
      mehDtMuonLayer[0]->Fill(layer);
      mehDtMuonTime[0]->Fill(MB1Digis[i].TIME);
      mehDtMuonTimevLayer[0]->Fill(layer,MB1Digis[i].TIME,1);
    }

    std::vector<PGlobalDigi::DTDigi> MB2Digis =
      srcGlobalDigis->getMB2Digis();      
    mehDtMuonn[1]->Fill((float)nMB2Digis);
    for (unsigned int i = 0; i < MB2Digis.size(); ++i) {
      float layer = 4.0 * (MB2Digis[i].SLAYER - 1.0) + MB2Digis[i].LAYER;
      mehDtMuonLayer[1]->Fill(layer);
      mehDtMuonTime[1]->Fill(MB2Digis[i].TIME);
      mehDtMuonTimevLayer[1]->Fill(layer,MB2Digis[i].TIME,1);
    }

    std::vector<PGlobalDigi::DTDigi> MB3Digis =
      srcGlobalDigis->getMB3Digis();      
    mehDtMuonn[2]->Fill((float)nMB3Digis);
    for (unsigned int i = 0; i < MB3Digis.size(); ++i) {
      float layer = 4.0 * (MB3Digis[i].SLAYER - 1.0) + MB3Digis[i].LAYER;
      mehDtMuonLayer[2]->Fill(layer);
      mehDtMuonTime[2]->Fill(MB3Digis[i].TIME);
      mehDtMuonTimevLayer[2]->Fill(layer,MB3Digis[i].TIME,1);
    }

    std::vector<PGlobalDigi::DTDigi> MB4Digis =
      srcGlobalDigis->getMB4Digis();      
    mehDtMuonn[3]->Fill((float)nMB4Digis);
    for (unsigned int i = 0; i < MB4Digis.size(); ++i) {
      float layer = 4.0 * (MB4Digis[i].SLAYER - 1.0) + MB4Digis[i].LAYER;
      mehDtMuonLayer[3]->Fill(layer);
      mehDtMuonTime[3]->Fill(MB4Digis[i].TIME);
      mehDtMuonTimevLayer[3]->Fill(layer,MB4Digis[i].TIME,1);
    }

    // get CSC Strip info
    std::vector<PGlobalDigi::CSCstripDigi> CSCstripDigis =
      srcGlobalDigis->getCSCstripDigis();      
    mehCSCStripn->Fill((float)nCSCstripDigis);
    for (unsigned int i = 0; i < CSCstripDigis.size(); ++i) {
      mehCSCStripADC->Fill(CSCstripDigis[i].ADC);
    }

    // get CSC Wire info
    std::vector<PGlobalDigi::CSCwireDigi> CSCwireDigis =
      srcGlobalDigis->getCSCwireDigis();      
    mehCSCWiren->Fill((float)nCSCwireDigis);
    for (unsigned int i = 0; i < CSCwireDigis.size(); ++i) {
      mehCSCWireTime->Fill(CSCwireDigis[i].TIME);
    }
 if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";

  } // end loop through events
}

//define this as a plug-in
//DEFINE_FWK_MODULE(GlobalDigisHistogrammer);
