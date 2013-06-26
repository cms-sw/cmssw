
#include "AnalyzeTuples.h"

AnalyzeTuples::AnalyzeTuples(const edm::ParameterSet & iConfig) {
  std::cout<<"analyzetuples a buraya girdi"<<std::endl;
  edm::ParameterSet m_HS = iConfig.getParameter<edm::ParameterSet>("HFShowerLibrary");
  edm::FileInPath fp       = m_HS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  std::string emName       = m_HS.getParameter<std::string>("TreeEMID");
  std::string hadName      = m_HS.getParameter<std::string>("TreeHadID");
  std::string branchEvInfo = m_HS.getUntrackedParameter<std::string>("BranchEvt","HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo");
  std::string branchPre    = m_HS.getUntrackedParameter<std::string>("BranchPre","HFShowerPhotons_hfshowerlib_");
  std::string branchPost   = m_HS.getUntrackedParameter<std::string>("BranchPost","_R.obj");

  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  if (!hf->IsOpen()) { 
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree 
			      << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogInfo("HFShower") << "HFShowerLibrary: opening " << nTree 
			     << " successfully"; 
  }

  TTree* event = (TTree *) hf ->Get("Events");
  if (event) {
    std::string info = branchEvInfo + branchPost;
    TBranch *evtInfo = event->GetBranch(info.c_str());
    if (evtInfo) {
      loadEventInfo(evtInfo);
    } else {
      edm::LogError("HFShower") << "HFShowerLibrary: HFShowerLibrayEventInfo"
				<< " Branch does not exist in Event";
      throw cms::Exception("Unknown", "HFShowerLibrary")
	<< "Event information absent\n";
    }
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: Events Tree does not "
			      << "exist";
    throw cms::Exception("Unknown", "HFShowerLibrary")
      << "Events tree absent\n";
  }
  
  edm::LogInfo("HFShower") << "HFShowerLibrary: Library " << libVers 
			   << " ListVersion "	<< listVersion 
			   << " Events Total " << totEvents << " and "
			   << evtPerBin << " per bin";
  edm::LogInfo("HFShower") << "HFShowerLibrary: Energies (GeV) with " 
			   << nMomBin	<< " bins";
  for (int i=0; i<nMomBin; i++)
    edm::LogInfo("HFShower") << "HFShowerLibrary: pmom[" << i << "] = "
			     << pmom[i] << " GeV";

  std::string nameBr = branchPre + emName + branchPost;
  emBranch         = event->GetBranch(nameBr.c_str());
  nameBr           = branchPre + hadName + branchPost;
  hadBranch        = event->GetBranch(nameBr.c_str());
  edm::LogInfo("HFShower") << "HFShowerLibrary:Branch " << emName 
			   << " has " << emBranch->GetEntries() 
			   << " entries and Branch " << hadName 
			   << " has " << hadBranch->GetEntries() 
			   << " entries";
  edm::LogInfo("HFShower") << "HFShowerLibrary::No packing information -"
			   << " Assume x, y, z are not in packed form";
}


AnalyzeTuples::~AnalyzeTuples() { }

void AnalyzeTuples::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {
  for(int ibin = 0; ibin < 12;++ibin){
    int min = evtPerBin*(ibin);
    int max = evtPerBin*(ibin+1);
    for(int i = min; i < max;++i){
      getRecord(0,i);
      int npe_long = 0;
      int npe_short = 0;
      std::cout<<"phptons size"<<photon.size()<<std::endl;
      for(int j = 0; j < int(photon.size());++j){
	//int depth = 0;
	if(photon[j].z() < 0){
	  //depth = 2; 
	  ++npe_short;
	}else{
	  //depth = 1; 
	  ++npe_long;
	  std::cout<<photon[j].z()<<std::endl;
	}
      }
      hNPELongElec[ibin]->Fill(npe_long);
      std::cout<<ibin<<npe_long<<std::endl;
      hNPEShortElec[ibin]->Fill(npe_short);
    }
  }
  for(int ibin = 0; ibin < 12;++ibin){
    int min = evtPerBin*(ibin);
    int max = evtPerBin*(ibin+1);
    for(int i = min; i < max;++i){
      getRecord(1,i);
      int npe_long = 0;
      int npe_short = 0;
      for(int j = 0; j < int(photon.size());++j){
	//int depth = 0;
	if(photon[j].z() < 0){
	  //depth = 2; 
	  ++npe_short;
	}else{
	  //depth = 1; 
	  ++npe_long;
	}
      }
      hNPELongPion[ibin]->Fill(npe_long);
      hNPEShortPion[ibin]->Fill(npe_short);
    }
  }
}

void AnalyzeTuples::beginJob() {
    
  TFileDirectory HFDir = fs->mkdir("HF");
  char title[128];
  for(int i = 0; i < int(pmom.size());++i){
    sprintf(title,"NPELongElec_Mom_%i",int(pmom[i]));
    int maxBin = int(pmom[i]+50);
    hNPELongElec[i] = HFDir.make<TH1I>(title, "NPE Long", 140, 0, 140);
    sprintf(title,"NPEShortElec_Mom_%i",int(pmom[i]));
    hNPEShortElec[i] = HFDir.make<TH1I>(title, "NPE Short", maxBin, 0, maxBin);
    sprintf(title,"NPELongPion_Mom_%i",int(pmom[i]));
    hNPELongPion[i] = HFDir.make<TH1I>(title, "NPE Long", maxBin, 0, maxBin);
    sprintf(title,"NPEShortPion_Mom_%i",int(pmom[i]));
    hNPEShortPion[i] = HFDir.make<TH1I>(title, "NPE Short", maxBin, 0, maxBin);
  }
}

void AnalyzeTuples::endJob() {
}

void AnalyzeTuples::loadEventInfo(TBranch* branch) {

  std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
  branch->SetAddress(&eventInfoCollection);
  branch->GetEntry(0);
  edm::LogInfo("HFShower") << "HFShowerLibrary::loadEventInfo loads "
			   << " EventInfo Collection of size "
			   << eventInfoCollection.size() << " records";
  totEvents   = eventInfoCollection[0].totalEvents();
  nMomBin     = eventInfoCollection[0].numberOfBins();
  evtPerBin   = eventInfoCollection[0].eventsPerBin();
  libVers     = eventInfoCollection[0].showerLibraryVersion();
  listVersion = eventInfoCollection[0].physListVersion();
  pmom        = eventInfoCollection[0].energyBins();
}

void AnalyzeTuples::getRecord(int type, int record) {

  int nrc     = record-1;
  photon.clear();
  if (type > 0) {
    hadBranch->SetAddress(&photon);
    hadBranch->GetEntry(nrc);
  } else {
    emBranch->SetAddress(&photon);
    emBranch->GetEntry(nrc);
  }
#ifdef DebugLog
  int nPhoton = photon.size();
  LogDebug("HFShower") << "HFShowerLibrary::getRecord: Record " << record
		       << " of type " << type << " with " << nPhoton 
		       << " photons";
  for (int j = 0; j < nPhoton; j++) 
    LogDebug("HFShower") << "Photon " << j << " " << photon[j];
#endif
}

// define this as a plug-in
DEFINE_FWK_MODULE(AnalyzeTuples);
