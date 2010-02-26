#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/CodedException.h"
#include "Validation/RecoB/plugins/BTagPerformanceAnalyzerMC.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DataFormats/Common/interface/View.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace reco;
using namespace edm;
//using namespace BTagMCTools;

typedef std::pair<Jet, reco::JetFlavour> JetWithFlavour;

BTagPerformanceAnalyzerMC::BTagPerformanceAnalyzerMC(const edm::ParameterSet& pSet)
{
  init(pSet);
  bookHistos(pSet);
}

void BTagPerformanceAnalyzerMC::bookHistos(const edm::ParameterSet& pSet)
{
  //
  // Book all histograms.
  //

  if (update) {
    //
    // append the DQM file ... we should consider this experimental
    //    edm::Service<DQMStore>().operator->()->open(std::string((const char *)(inputFile)),"/");
    // removed; DQM framework will take care
  }

  // Rec jet
  double pRecJetMin  = 0.0 ;
  double pRecJetMax  = 99999.9 ;

  // parton p
//   double pPartonMin = 0.0    ;
//   double pPartonMax = 99999.9 ;

  // specify jet and parton kinematic cuts.
  jetSelector.setEtaMin            ( etaMin      ) ;
  jetSelector.setEtaMax            ( etaMax      ) ;
//   jetSelector.setPPartonMin        ( pPartonMin  ) ;
//   jetSelector.setPPartonMax        ( pPartonMax  ) ;
//   jetSelector.setPtPartonMin       ( ptPartonMin ) ;
//   jetSelector.setPtPartonMax       ( ptPartonMax ) ;
  jetSelector.setPtRecJetMin       ( ptRecJetMin ) ;
  jetSelector.setPtRecJetMax       ( ptRecJetMax ) ;
  jetSelector.setPRecJetMin        ( pRecJetMin  ) ;
  jetSelector.setPRecJetMax        ( pRecJetMax  ) ;
  jetSelector.setRatioMin          ( ratioMin) ;
  jetSelector.setRatioMax          ( ratioMax) ;

  // iterate over ranges:
  const int iEtaStart = -1                   ;  // this will be the inactive one
  const int iEtaEnd   = etaRanges.size() - 1 ;
  const int iPtStart  = -1                   ;  // this will be the inactive one
  const int iPtEnd    = ptRanges.size() - 1  ;
  setTDRStyle();

  TagInfoPlotterFactory theFactory;

  for (unsigned int iModule = 0; iModule != moduleConfig.size(); ++iModule) {

    string dataFormatType = "JetTag";
    if (moduleConfig[iModule].exists("type"))
    	 dataFormatType = moduleConfig[iModule].getParameter<string>("type");
    InputTag moduleLabel = moduleConfig[iModule].getParameter<InputTag>("label");
    if (dataFormatType == "JetTag") {
      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<JetTagPlotter*>()) ;
      // Contains plots for each bin of rapidity and pt.
	vector<BTagDifferentialPlot*> * differentialPlotsConstantEta = new vector<BTagDifferentialPlot*> () ;
	vector<BTagDifferentialPlot*> * differentialPlotsConstantPt  = new vector<BTagDifferentialPlot*> () ;
      if (finalize && mcPlots_ ){
	differentialPlots.push_back(vector<BTagDifferentialPlot*>());

	// the constant b-efficiency for the differential plots versus pt and eta
	double effBConst =
	  moduleConfig[iModule].getParameter<edm::ParameterSet>("parameters").getParameter<double>("effBConst");

	// the objects for the differential plots vs. eta,pt for

	for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	  BTagDifferentialPlot * etaConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constETA, moduleLabel.label());
	  differentialPlotsConstantEta->push_back ( etaConstDifferentialPlot );
	}

	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  // differentialPlots for this pt bin
	  BTagDifferentialPlot * ptConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constPT, moduleLabel.label());
	  differentialPlotsConstantPt->push_back ( ptConstDifferentialPlot );
	}
      }

      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {

	  EtaPtBin etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the genertic b tag plotter
	  JetTagPlotter *jetTagPlotter = new JetTagPlotter(
							   moduleLabel.label(), etaPtBin,
							   moduleConfig[iModule].getParameter<edm::ParameterSet>("parameters"),mcPlots_,update,finalize);
	  binJetTagPlotters.back().push_back ( jetTagPlotter ) ;

	  // Add to the corresponding differential plotters
	  if (finalize && mcPlots_){
	    (*differentialPlotsConstantEta)[iEta+1]->addBinPlotter ( jetTagPlotter ) ;
	    (*differentialPlotsConstantPt )[iPt+1] ->addBinPlotter ( jetTagPlotter ) ;
	  }
	}
      }

      // the objects for the differential plots vs. eta, pt: collect all from constant eta and constant pt
      if (finalize && mcPlots_){
	differentialPlots.back().reserve(differentialPlotsConstantEta->size()+differentialPlotsConstantPt->size()) ;
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantEta->begin(), differentialPlotsConstantEta->end());
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantPt->begin(), differentialPlotsConstantPt->end());

	edm::LogInfo("Info")
	  << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();

	// the intermediate ones are no longer needed
	delete differentialPlotsConstantEta ;
	delete differentialPlotsConstantPt  ;
      }
    } else {
      // tag info retrievel is deferred (needs availability of EventSetup)
      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<BaseTagInfoPlotter*>()) ;

      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  EtaPtBin etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the tagInfo plotter

	  BaseTagInfoPlotter *jetTagPlotter = theFactory.buildPlotter(
								      dataFormatType, moduleLabel.label(), etaPtBin,
								      moduleConfig[iModule].getParameter<edm::ParameterSet>("parameters"), update, mcPlots_,finalize);
	  binTagInfoPlotters.back().push_back ( jetTagPlotter ) ;
          binTagInfoPlottersToModuleConfig[jetTagPlotter] = iModule;
	}
      }

      edm::LogInfo("Info")
	<< "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
    }
  }
}

EtaPtBin BTagPerformanceAnalyzerMC::getEtaPtBin(int iEta, int iPt)
{
  // DEFINE BTagBin:
  bool    etaActive_ , ptActive_;
  double  etaMin_, etaMax_, ptMin_, ptMax_ ;

  if ( iEta != -1 ) {
    etaActive_ = true ;
    etaMin_    = etaRanges[iEta]   ;
    etaMax_    = etaRanges[iEta+1] ;
  }
  else {
    etaActive_ = false ;
    etaMin_    = etaRanges[0]   ;
    etaMax_    = etaRanges[etaRanges.size() - 1]   ;
  }

  if ( iPt != -1 ) {
    ptActive_ = true ;
    ptMin_    = ptRanges[iPt]   ;
    ptMax_    = ptRanges[iPt+1] ;
  }
  else {
    ptActive_ = false ;
    ptMin_    = ptRanges[0]	;
    ptMax_    = ptRanges[ptRanges.size() - 1]	;
  }
  return EtaPtBin(etaActive_ , etaMin_ , etaMax_ ,
			ptActive_  , ptMin_  , ptMax_ );
}

void BTagPerformanceAnalyzerMC::init(const edm::ParameterSet& iConfig)
{
  // Get histogram plotting options from configuration.

  moduleConfig = iConfig.getParameter< vector<edm::ParameterSet> >("tagConfig");

  update = iConfig.getParameter<bool>( "update" );
  inputFile = TString(iConfig.getParameter<std::string>( "inputfile" ));

  finalize = iConfig.getParameter< bool >("finalizePlots");
  finalizeOnly = iConfig.getParameter< bool >("finalizeOnly");
  mcPlots_ = iConfig.getParameter< bool >("mcPlots");

  produceEps = iConfig.getParameter< bool >("produceEps");
  producePs = iConfig.getParameter< bool >("producePs");
  psBaseName = TString(iConfig.getParameter<std::string>( "psBaseName" ));
  epsBaseName = TString(iConfig.getParameter<std::string>( "epsBaseName" ));

  allHisto = iConfig.getParameter<bool>( "allHistograms" );

  partonKinematics = iConfig.getParameter< bool >("partonKinematics");

  // eta jet
  etaMin = iConfig.getParameter<double>("etaMin");
  etaMax = iConfig.getParameter<double>("etaMax");

  // rec. jet
  ptRecJetMin = iConfig.getParameter<double>("ptRecJetMin");
  ptRecJetMax = iConfig.getParameter<double>("ptRecJetMax");

  // parton pt
  ptPartonMin = iConfig.getParameter<double>("ptPartonMin");
  ptPartonMax = iConfig.getParameter<double>("ptPartonMax");

  // ratio of largest lepton momentum in jet to jet energy
  ratioMin = iConfig.getParameter<double>("ratioMin");
  ratioMax = iConfig.getParameter<double>("ratioMax");

  // DEFINITION OF BINS

  // eta and pt ranges (bins for differential plots)
  etaRanges = iConfig.getParameter< vector<double> >("etaRanges");
  ptRanges = iConfig.getParameter< vector<double> >("ptRanges");

  jetMCSrc = iConfig.getParameter<edm::InputTag>("jetMCSrc");
  slInfoTag = iConfig.getParameter<edm::InputTag>("softLeptonInfo");

  jetCorrector = CorrectJet(iConfig.getParameter<std::string>("jetCorrection"));
  jetMatcher = MatchJet(iConfig.getParameter<edm::ParameterSet>("recJetMatching"));
  jetMatcher.setThreshold(0.25 * ptRecJetMin);

  /// needed for lepton specific plots
  electronPlots = false;
  muonPlots = false;
  tauPlots = false;

  switch(iConfig.getParameter<unsigned int>("leptonPlots")) {
    case 11: electronPlots = true; break;
    case 13: muonPlots = true;     break;
    case 15: tauPlots = true;      break;
  }
}

BTagPerformanceAnalyzerMC::~BTagPerformanceAnalyzerMC()
{
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      delete binJetTagPlotters[iJetLabel][iPlotter];
    }
    if (finalize  && mcPlots_){
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	   iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	delete *iPlotter;
      }
    }
  }
  for (unsigned int iJetLabel = 0; iJetLabel != binTagInfoPlotters.size(); ++iJetLabel) {
    int plotterSize =  binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      delete binTagInfoPlotters[iJetLabel][iPlotter];
    }
  }
}

void BTagPerformanceAnalyzerMC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventInitialized = false;

  if (finalizeOnly == true) return;

  edm::Handle<JetFlavourMatchingCollection> jetMC;
  FlavourMap flavours;
  LeptonMap leptons;

  iEvent.getByLabel(jetMCSrc, jetMC);
  for (JetFlavourMatchingCollection::const_iterator iter = jetMC->begin();
       iter != jetMC->end(); iter++) {
    unsigned int fl = std::abs(iter->second.getFlavour());
    flavours.insert(std::make_pair(iter->first, fl));
    const reco::JetFlavour::Leptons &lep = iter->second.getLeptons();
    leptons.insert(std::make_pair(iter->first, lep));
  }

  edm::Handle<reco::SoftLeptonTagInfoCollection> infoHandle;
  iEvent.getByLabel(slInfoTag, infoHandle);

// Look first at the jetTags

  for (unsigned int iJetLabel = 0; iJetLabel != jetTagInputTags.size(); ++iJetLabel) {
    edm::Handle<reco::JetTagCollection> tagHandle;
    iEvent.getByLabel(jetTagInputTags[iJetLabel], tagHandle);
    const reco::JetTagCollection & tagColl = *(tagHandle.product());
    LogDebug("Info") << "Found " << tagColl.size() << " B candidates in collection " << jetTagInputTags[iJetLabel];

    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (JetTagCollection::const_iterator tagI = tagColl.begin();
	 tagI != tagColl.end(); ++tagI) {
      // Identify parton associated to jet.

      /// needed for lepton specific plots
      if (flavours[tagI->first] == 5 &&
          ((electronPlots && !leptons[tagI->first].electron) ||
           (muonPlots && !leptons[tagI->first].muon) ||
           (tauPlots && !leptons[tagI->first].tau)))
        continue;

      JetWithFlavour jetWithFlavour;
      if (!getJetWithFlavour(tagI->first, flavours, jetWithFlavour, iSetup))
        continue;
      if (!jetSelector(jetWithFlavour.first, std::abs(jetWithFlavour.second.getFlavour()), infoHandle))
        continue;

      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	      bool inBin = false;
	      if (partonKinematics)
                inBin = binJetTagPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(jetWithFlavour.second.getLorentzVector().Eta(),
                                                                                 jetWithFlavour.second.getLorentzVector().Pt());
	      else
                inBin = binJetTagPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(jetWithFlavour.first);
	      // Fill histograms if in desired pt/rapidity bin.
	      if (inBin)
	        binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag(jetWithFlavour.first, tagI->second, std::abs(jetWithFlavour.second.getFlavour()));
      }
    }
  }

// Now look at the TagInfos

  for (unsigned int iJetLabel = 0; iJetLabel != tiDataFormatType.size(); ++iJetLabel) {
    int plotterSize = binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter)
      binTagInfoPlotters[iJetLabel][iPlotter]->setEventSetup(iSetup);

    vector<edm::InputTag> & inputTags = tagInfoInputTags[iJetLabel];
    if (inputTags.empty()) {
      // deferred retrieval of input tags
      BaseTagInfoPlotter *firstPlotter = binTagInfoPlotters[iJetLabel][0];
      int iModule = binTagInfoPlottersToModuleConfig[firstPlotter];
      vector<string> labels = firstPlotter->tagInfoRequirements();
      if (labels.empty())
        labels.push_back("label");
      for (vector<string>::const_iterator iLabels = labels.begin();
           iLabels != labels.end(); ++iLabels) {
        edm::InputTag inputTag =
        	moduleConfig[iModule].getParameter<InputTag>(*iLabels);
        inputTags.push_back(inputTag);
      }
    }

    unsigned int nInputTags = inputTags.size();
    vector< edm::Handle< View<BaseTagInfo> > > tagInfoHandles(nInputTags);
    edm::ProductID jetProductID;
    unsigned int nTagInfos = 0;
    for (unsigned int iInputTags = 0; iInputTags < inputTags.size(); ++iInputTags) {
      edm::Handle< View<BaseTagInfo> > & tagInfoHandle = tagInfoHandles[iInputTags];
      iEvent.getByLabel(inputTags[iInputTags], tagInfoHandle);
      unsigned int size = tagInfoHandle->size();
      LogDebug("Info") << "Found " << size << " B candidates in collection " << inputTags[iInputTags];

      edm::ProductID thisProductID = (size > 0) ? (*tagInfoHandle)[0].jet().id() : edm::ProductID();
      if (iInputTags == 0) {
        jetProductID = thisProductID;
        nTagInfos = size;
      } else if (jetProductID != thisProductID)
        throw cms::Exception("Configuration") << "TagInfos are referencing a different jet collection." << endl;
      else if (nTagInfos != size)
        throw cms::Exception("Configuration") << "TagInfo collections are having a different size." << endl;
    }

    for (unsigned int iTagInfos = 0; iTagInfos < nTagInfos; ++iTagInfos) {
      vector<const BaseTagInfo*> baseTagInfos(nInputTags);
      edm::RefToBase<Jet> jetRef;
      for (unsigned int iTagInfo = 0; iTagInfo < nInputTags; iTagInfo++) {
        const BaseTagInfo &baseTagInfo = (*tagInfoHandles[iTagInfo])[iTagInfos];
        if (iTagInfo == 0)
          jetRef = baseTagInfo.jet();
        else if (baseTagInfo.jet() != jetRef)
          throw cms::Exception("Configuration") << "TagInfos pointing to different jets." << endl;
        baseTagInfos[iTagInfo] = &baseTagInfo;
      }

      // Identify parton associated to jet.

      /// needed for lepton specific plots
      if (flavours[jetRef] == 5 &&
          ((electronPlots && !leptons[jetRef].electron) ||
           (muonPlots && !leptons[jetRef].muon) ||
           (tauPlots && !leptons[jetRef].tau)))
         continue;

      JetWithFlavour jetWithFlavour;
      if (!getJetWithFlavour(jetRef, flavours, jetWithFlavour, iSetup))
        continue;
      if (!jetSelector(jetWithFlavour.first, std::abs(jetWithFlavour.second.getFlavour()), infoHandle))
        continue;

      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	      bool inBin = false;
	      if (partonKinematics)
               inBin = binTagInfoPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(jetWithFlavour.second.getLorentzVector().Eta(),
                                                                                 jetWithFlavour.second.getLorentzVector().Pt());
	      else
               inBin = binTagInfoPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*jetRef);
	      // Fill histograms if in desired pt/rapidity bin.
	      if (inBin)
	        binTagInfoPlotters[iJetLabel][iPlotter]->analyzeTag(baseTagInfos, std::abs(jetWithFlavour.second.getFlavour()));
      }
    }
  }
}

bool  BTagPerformanceAnalyzerMC::getJetWithFlavour(	edm::RefToBase<Jet> jetRef, FlavourMap flavours,
	JetWithFlavour & jetWithFlavour, const edm::EventSetup & es)
{

  edm::ProductID recProdId = jetRef.id();
  edm::ProductID refProdId = (flavours.begin() == flavours.end())
    ? recProdId
    : flavours.begin()->first.id();

  if (!eventInitialized) {
    jetCorrector.setEventSetup(es);
    if (recProdId != refProdId) {
      edm::RefToBaseVector<Jet> refJets;
      for(FlavourMap::const_iterator iter = flavours.begin();
          iter != flavours.end(); ++iter)
        refJets.push_back(iter->first);
      const edm::RefToBaseProd<Jet> recJetsProd(jetRef);
      edm::RefToBaseVector<Jet> recJets;
      for(unsigned int i = 0; i < recJetsProd->size(); i++)
        recJets.push_back(edm::RefToBase<Jet>(recJetsProd, i));
      jetMatcher.matchCollections(refJets, recJets, es);
    }
    eventInitialized = true;
  }

  if (recProdId != refProdId) {
    jetRef = jetMatcher(jetRef);
    if (jetRef.isNull())
      return false;
  }

  jetWithFlavour.first = jetCorrector(*jetRef);

  jetWithFlavour.second = reco::JetFlavour(jetWithFlavour.first.p4(), math::XYZPoint (0,0,0), flavours[jetRef]);

  LogTrace("Info") << "Found jet with flavour "<<jetWithFlavour.second.getFlavour()<<endl;
  LogTrace("Info") << jetWithFlavour.first.p()<<" , "<< jetWithFlavour.first.pt()<<" - "
   << jetWithFlavour.second.getLorentzVector().P()<<" , "<< jetWithFlavour.second.getLorentzVector().Pt()<<endl;

  return true;
}

void BTagPerformanceAnalyzerMC::endJob()
{
  if (!finalize) return;
  setTDRStyle();
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
       binJetTagPlotters[iJetLabel][iPlotter]->finalize();
      //      binJetTagPlotters[iJetLabel][iPlotter]->write(allHisto);
      if (producePs)  (*binJetTagPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps) (*binJetTagPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
    
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	   iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	(**iPlotter).process();
	if (producePs)  (**iPlotter).psPlot(psBaseName);
	if (produceEps) (**iPlotter).epsPlot(epsBaseName);
	//      (**iPlotter).write(allHisto);
      }
  }
  for (unsigned int iJetLabel = 0; iJetLabel != binTagInfoPlotters.size(); ++iJetLabel) {
    int plotterSize =  binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binTagInfoPlotters[iJetLabel][iPlotter]->finalize();
      //      binTagInfoPlotters[iJetLabel][iPlotter]->write(allHisto);
      if (producePs)  (*binTagInfoPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps) (*binTagInfoPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceAnalyzerMC);
