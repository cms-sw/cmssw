/** \file GlobalHitsProdHistStripper.cc
 *
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/GlobalHits/interface/GlobalHitsProdHistStripper.h"

GlobalHitsProdHistStripper::GlobalHitsProdHistStripper(const edm::ParameterSet &iPSet)
    : fName(""),
      verbosity(0),
      frequency(0),
      vtxunit(0),
      getAllProvenances(false),
      printProvenanceInfo(false),
      outputfile(""),
      count(0) {
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_GlobalHitsProdHistStripper";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov = iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = nullptr;
  dbe = edm::Service<DQMStore>().operator->();

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDAnalyzer with parameter values:\n"
                               << "    Name           = " << fName << "\n"
                               << "    Verbosity      = " << verbosity << "\n"
                               << "    Frequency      = " << frequency << "\n"
                               << "    VtxUnit        = " << vtxunit << "\n"
                               << "    OutputFile     = " << outputfile << "\n"
                               << "    DoOutput      = " << doOutput << "\n"
                               << "    GetProv        = " << getAllProvenances << "\n"
                               << "    PrintProv      = " << printProvenanceInfo << "\n"
                               << "===============================\n";
  }
}

GlobalHitsProdHistStripper::~GlobalHitsProdHistStripper() {
  if (doOutput)
    if (!outputfile.empty() && dbe)
      dbe->save(outputfile);
}

void GlobalHitsProdHistStripper::beginJob(void) { return; }

void GlobalHitsProdHistStripper::endJob() {
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) << "Terminating having processed " << count << " runs.";
  return;
}

void GlobalHitsProdHistStripper::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_beginRun";
  // keep track of number of runs processed
  ++count;

  int nrun = iRun.run();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << " (" << count << " runs total)";
  } else if (verbosity == 0) {
    if (nrun % frequency == 0 || count == 1) {
      edm::LogInfo(MsgLoggerCat) << "Processing run " << nrun << " (" << count << " runs total)";
    }
  }

  if (getAllProvenances) {
    std::vector<const edm::StableProvenance *> AllProv;
    iRun.getAllStableProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat) << "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");

      for (unsigned int i = 0; i < AllProv.size(); ++i) {
        eventout += "\n       ******************************";
        eventout += "\n       Module       : ";
        eventout += AllProv[i]->moduleLabel();
        eventout += "\n       ProductID    : ";
        eventout += AllProv[i]->productID().id();
        eventout += "\n       ClassName    : ";
        eventout += AllProv[i]->className();
        eventout += "\n       InstanceName : ";
        eventout += AllProv[i]->productInstanceName();
        eventout += "\n       BranchName   : ";
        eventout += AllProv[i]->branchName();
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

  return;
}

void GlobalHitsProdHistStripper::endRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_endRun";

  edm::Handle<TH1F> histogram1D;
  std::vector<edm::Handle<TH1F>> allhistogram1D;
  iRun.getManyByType(allhistogram1D);

  me.resize(allhistogram1D.size());

  for (uint i = 0; i < allhistogram1D.size(); ++i) {
    histogram1D = allhistogram1D[i];
    if (!histogram1D.isValid()) {
      edm::LogWarning(MsgLoggerCat) << "Invalid histogram extracted from event.";
      continue;
    }

    me[i] = nullptr;

    /*
    std::cout << "Extracting histogram: " << std::endl
              << "       Module       : "
              << (histogram1D.provenance()->branchDescription()).moduleLabel()
              << std::endl
              << "       ProductID    : "
              <<
    (histogram1D.provenance()->branchDescription()).productID().id()
              << std::endl
              << "       ClassName    : "
              << (histogram1D.provenance()->branchDescription()).className()
              << std::endl
              << "       InstanceName : "
              <<
    (histogram1D.provenance()->branchDescription()).productInstanceName()
              << std::endl
              << "       BranchName   : "
              << (histogram1D.provenance()->branchDescription()).branchName()
              << std::endl;
    */

    if ((histogram1D.provenance()->branchDescription()).moduleLabel() != "globalhitsprodhist")
      continue;

    std::string histname = histogram1D->GetName();

    std::string subhist1 = histname.substr(1, 5);
    std::string subhist2 = histname.substr(1, 4);

    if (dbe) {
      if (subhist1 == "CaloE" || subhist1 == "CaloP") {
        dbe->setCurrentFolder("GlobalHitsV/ECal");
      } else if (subhist1 == "CaloH") {
        dbe->setCurrentFolder("GlobalHitsV/HCal");
      } else if (subhist1 == "Geant" || subhist2 == "MCG4" || subhist1 == "MCRGP") {
        dbe->setCurrentFolder("GlobalHitsV/MCGeant");
      } else if (subhist2 == "Muon") {
        dbe->setCurrentFolder("GlobalHitsV/Muon");
      } else if (subhist1 == "Track") {
        dbe->setCurrentFolder("GlobalHitsV/Tracker");
      }

      me[i] = dbe->book1D(histname,
                          histogram1D->GetTitle(),
                          histogram1D->GetXaxis()->GetNbins(),
                          histogram1D->GetXaxis()->GetXmin(),
                          histogram1D->GetXaxis()->GetXmax());
      me[i]->setAxisTitle(histogram1D->GetXaxis()->GetTitle(), 1);
      me[i]->setAxisTitle(histogram1D->GetYaxis()->GetTitle(), 2);
    }

    std::string mename = me[i]->getName();

    // std::cout << "Extracting histogram " << histname
    //	      << " into MonitorElement " << mename
    //	      << std::endl;

    for (Int_t x = 1; x <= histogram1D->GetXaxis()->GetNbins(); ++x) {
      Double_t binx = histogram1D->GetBinCenter(x);
      Double_t value = histogram1D->GetBinContent(x);
      me[i]->Fill(binx, value);
    }
  }
  return;
}
/*
if (iter != monitorElements.end()) {

  std::string mename = iter->second->getName();

  std::cout << "Extracting histogram " << histname
            << " into MonitorElement " << mename
            << std::endl;

  if (histname == "hGeantTrkE" || histname == "hGeantTrkPt") {
    std::cout << "Information stored in histogram pointer:"
              << std::endl;
    std::cout << histname << ":" << std::endl;
    std::cout << "  Entries: " << histogram1D->GetEntries()
              << std::endl;
    std::cout << "  Mean: " << histogram1D->GetMean() << std::endl;
    std::cout << "  RMS: " << histogram1D->GetRMS() << std::endl;
  }

  for (Int_t x = 1; x <= histogram1D->GetXaxis()->GetNbins(); ++x) {
    Double_t binx = histogram1D->GetBinCenter(x);
    Double_t value = histogram1D->GetBinContent(x);
    iter->second->Fill(binx,value);
  }

  if (histname == "hGeantTrkE" || histname == "hGeantTrkPt") {
    std::cout << "Information stored in monitor element:" << std::endl;
    std::cout << mename << ":" << std::endl;
    std::cout << "  Entries: "
              << iter->second->getEntries() << std::endl;
    std::cout << "  Mean: " << iter->second->getMean()
              << std::endl;
    std::cout << "  RMS: " << iter->second->getRMS()
              << std::endl;
              }
} // find in map
} // loop through getManyByType

return;
}
*/

void GlobalHitsProdHistStripper::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) { return; }
