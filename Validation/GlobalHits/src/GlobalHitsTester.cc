#include "Validation/GlobalHits/interface/GlobalHitsTester.h"
#include "DQMServices/Core/interface/DQMStore.h"

GlobalHitsTester::GlobalHitsTester(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), vtxunit(0), label(""), 
  getAllProvenances(false), printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalHitsTester_GlobalHitsTester";

  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");
 
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    VtxUnit       = " << vtxunit << "\n"
      << "    OutputFile    = " << outputfile << "\n"
      << "    DoOutput      = " << doOutput << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "===============================\n";
  }
}

GlobalHitsTester::~GlobalHitsTester() 
{
}

void GlobalHitsTester::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  meTestString = 0;
  meTestInt = 0;
  meTestFloat = 0;
  meTestTH1F = 0;
  meTestTH2F = 0;
  meTestTH3F = 0;
  meTestProfile1 = 0;
  meTestProfile2 = 0;
  Random = new TRandom3();

  ibooker.setCurrentFolder("GlobalTestV/String");
  meTestString = ibooker.bookString("TestString", "Hello World" );

  ibooker.setCurrentFolder("GlobalTestV/Int");
  meTestInt = ibooker.bookInt("TestInt");

  ibooker.setCurrentFolder("GlobalTestV/Float");
  meTestFloat = ibooker.bookFloat("TestFloat");

  ibooker.setCurrentFolder("GlobalTestV/TH1F");
  meTestTH1F = ibooker.book1D("Random1D", "Random1D", 100, -10., 10.);

  ibooker.setCurrentFolder("GlobalTestV/TH2F");
  meTestTH2F = ibooker.book2D("Random2D", "Random2D", 100, -10, 10., 100, -10.,
      10.);

  ibooker.setCurrentFolder("GlobalTestV/TH3F");
  meTestTH3F = ibooker.book3D("Random3D", "Random3D", 100, -10., 10., 100,
      -10., 10., 100, -10., 10.);

  ibooker.setCurrentFolder("GlobalTestV/TProfile");
  meTestProfile1 = ibooker.bookProfile("Profile1", "Profile1", 100, -10., 10.,
      100, -10., 10.);

  ibooker.setCurrentFolder("GlobalTestV/TProfile2D");
  meTestProfile2 = ibooker.bookProfile2D("Profile2", "Profile2", 100, -10.,
      10., 100, -10, 10., 100, -10., 10.);

  ibooker.tag(meTestTH1F, 1);
  ibooker.tag(meTestTH2F, 2);
  ibooker.tag(meTestTH3F, 3);
  ibooker.tag(meTestProfile1, 4);
  ibooker.tag(meTestProfile2, 5);
  ibooker.tag(meTestString, 6);
  ibooker.tag(meTestInt, 7);
  ibooker.tag(meTestFloat, 8);

}

void GlobalHitsTester::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  
  for(int i = 0; i < 1000; ++i) {
    RandomVal1 = Random->Gaus(0.,1.);
    RandomVal2 = Random->Gaus(0.,1.);
    RandomVal3 = Random->Gaus(0.,1.);
    
    meTestTH1F->Fill(RandomVal1);
    meTestTH2F->Fill(RandomVal1, RandomVal2);
    meTestTH3F->Fill(RandomVal1, RandomVal2, RandomVal3);
    meTestProfile1->Fill(RandomVal1, RandomVal2);
    meTestProfile2->Fill(RandomVal1, RandomVal2, RandomVal3);
  }

  meTestInt->Fill(100);
  meTestFloat->Fill(3.141592);
}
