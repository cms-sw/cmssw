#include "Validation/GlobalHits/interface/GlobalHitsTester.h"

GlobalHitsTester::GlobalHitsTester(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), vtxunit(0), label(""), 
  getAllProvenances(false), printProvenanceInfo(false), count(0)
{
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");
  
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  if(dbe){
    meTestString = 0;
    meTestInt = 0;
    meTestFloat = 0;
    meTestTH1F = 0;
    meTestTH2F = 0;
    meTestTH3F = 0;
    meTestProfile1 = 0;
    meTestProfile2 = 0;
    Random = new TRandom3();
    
    dbe->setCurrentFolder("GlobalTestV");

    meTestString = dbe->bookString("TestString","Hello World" );
    meTestInt = dbe->bookInt("TestInt");
    meTestFloat = dbe->bookFloat("TestFloat");
    meTestTH1F = dbe->book1D("Random1D", "Random1D", 100, -10., 10.);
    meTestTH2F = dbe->book2D("Random2D", "Random2D", 100, -10, 10., 100, -10., 
			     10.);
    meTestTH3F = dbe->book3D("Random3D", "Random3D", 100, -10., 10., 100, 
			     -10., 10., 100, -10., 10.);
    meTestProfile1 = dbe->bookProfile("Profile1", "Profile1", 100, -10., 10., 
				      100, -10., 10.);
    meTestProfile2 = dbe->bookProfile2D("Profile2", "Profile2", 100, -10., 
					10., 100, -10, 10., 100, -10., 10.);

    dbe->tag(meTestTH1F->getFullname(),1);
    dbe->tag(meTestTH2F->getFullname(),2);
    dbe->tag(meTestTH3F->getFullname(),3);
    dbe->tag(meTestProfile1->getFullname(),4);
    dbe->tag(meTestProfile2->getFullname(),5);
    dbe->tag(meTestString->getFullname(),6);
    dbe->tag(meTestInt->getFullname(),7);
    dbe->tag(meTestFloat->getFullname(),8);
  }
}

GlobalHitsTester::~GlobalHitsTester() 
{
  if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
}

void GlobalHitsTester::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void GlobalHitsTester::endJob()
{
  std::string MsgLoggerCat = "GlobalHitsTester_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalHitsTester::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  return;
}

void GlobalHitsTester::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  meTestInt->Fill(100);
  meTestFloat->Fill(3.141592);

  return;
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
}
