#include "Validation/GlobalRecHits/interface/GlobalRecHitsProducer.h"

GlobalRecHitsProducer::GlobalRecHitsProducer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_GlobalRecHitsProducer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  produces<PGlobalRecHit>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    Label         = " << label << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    ECalEBSrc     = " << ECalEBSrc_.label() 
      << ":" << ECalEBSrc_.instance() << "\n"
      << "    ECalEESrc     = " << ECalEESrc_.label() 
      << ":" << ECalEESrc_.instance() << "\n"
      << "===============================\n";
  }
}

GlobalRecHitsProducer::~GlobalRecHitsProducer() 
{
}

void GlobalRecHitsProducer::beginJob(const edm::EventSetup& iSetup)
{
  clear();
  return;
}

void GlobalRecHitsProducer::endJob()
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalRecHitsProducer::produce(edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_produce";

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
  clear();

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
	eventout += (AllProv[i]->product).moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += (AllProv[i]->product).productID_.id_;
	eventout += "\n       ClassName    : ";
	eventout += (AllProv[i]->product).fullClassName_;
	eventout += "\n       InstanceName : ";
	eventout += (AllProv[i]->product).productInstanceName_;
	eventout += "\n       BranchName   : ";
	eventout += (AllProv[i]->product).branchName_;
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

  // call fill functions
  // gather Ecal information from event
  fillECal(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";

  // produce object to put into event
  std::auto_ptr<PGlobalRecHit> pOut(new PGlobalRecHit);

  if (verbosity > 2)
    edm::LogInfo (MsgLoggerCat)
      << "Saving event contents:";

  // call store functions
  // store ECal information in produce
  storeECal(*pOut);

  // store information in event
  iEvent.put(pOut,label);

  return;
}

void GlobalRecHitsProducer::fillECal(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalRecHitsProducer::storeECal(PGlobalRecHit& product)
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_storeECal";

  return;
}

void GlobalRecHitsProducer::clear()
{
  std::string MsgLoggerCat = "GlobalRecHitsProducer_clear";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat)
      << "Clearing event holders"; 

  // reset electromagnetic info
  // EE info
  EECalADC.clear();
  EECalmaxPos.clear(); 
  EECalAEE.clear(); 
  EECalSHE.clear();
  // EB info
  EBCalADC.clear();
  EBCalmaxPos.clear(); 
  EBCalAEE.clear(); 
  EBCalSHE.clear();

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalRecHitsProducer);
