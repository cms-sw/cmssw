#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/RecoB/plugins/BTagPerformanceAnalyzerMC.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DataFormats/Common/interface/View.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace reco;
using namespace edm;
using namespace std;
using namespace RecoBTag;
//using namespace BTagMCTools;

typedef std::pair<Jet, reco::JetFlavour> JetWithFlavour;

BTagPerformanceAnalyzerMC::BTagPerformanceAnalyzerMC(const edm::ParameterSet& pSet) :
  partonKinematics(pSet.getParameter< bool >("partonKinematics")),
  ptPartonMin(pSet.getParameter<double>("ptPartonMin")),
  ptPartonMax(pSet.getParameter<double>("ptPartonMax")),
  jetSelector(
    pSet.getParameter<double>("etaMin"),
    pSet.getParameter<double>("etaMax"),
    pSet.getParameter<double>("ptRecJetMin"),
    pSet.getParameter<double>("ptRecJetMax"),
    0.0, 99999.0,
    pSet.getParameter<double>("ratioMin"),
    pSet.getParameter<double>("ratioMax")
  ),
  etaRanges(pSet.getParameter< vector<double> >("etaRanges")),
  ptRanges(pSet.getParameter< vector<double> >("ptRanges")),
  produceEps(pSet.getParameter< bool >("produceEps")),
  producePs(pSet.getParameter< bool >("producePs")),
  inputFile(pSet.getParameter<std::string>( "inputfile" )),
  update(pSet.getParameter<bool>( "update" )),
  allHisto(pSet.getParameter<bool>( "allHistograms" )),
  finalize(pSet.getParameter< bool >("finalizePlots")),
  finalizeOnly(pSet.getParameter< bool >("finalizeOnly")),
  ptHatWeight(pSet.getParameter< bool >("applyPtHatWeight")),
  jetMCSrc(pSet.getParameter<edm::InputTag>("jetMCSrc")),
  slInfoTag(pSet.getParameter<edm::InputTag>("softLeptonInfo")),
  moduleConfig(pSet.getParameter< vector<edm::ParameterSet> >("tagConfig")),
  flavPlots_(pSet.getParameter< std::string >("flavPlots")),
  makeDiffPlots_(pSet.getParameter< bool >("differentialPlots")),
  jetCorrector(pSet.getParameter<std::string>("jetCorrection")),
  jetMatcher(pSet.getParameter<edm::ParameterSet>("recJetMatching"))
{
  //mcPlots_ : 1=b+c+l+ni; 2=all+1; 3=1+d+u+s+g; 4=3+all . Default is 2. Don't use 0.
  if(flavPlots_.find("dusg")<15){
    if(flavPlots_.find("noall")<15) mcPlots_ = 3;
    else mcPlots_ = 4;
  }
  else{
    if(flavPlots_.find("noall")<15) mcPlots_ = 1;
    else mcPlots_ = 2;
  }
  if(makeDiffPlots_) mcPlots_ = 4; //create differentialPlots only for all histograms 
  double ptRecJetMin = pSet.getParameter<double>("ptRecJetMin");
  jetMatcher.setThreshold(0.25 * ptRecJetMin);
  switch(pSet.getParameter<unsigned int>("leptonPlots")) {
    case 11: electronPlots = true; muonPlots = false; tauPlots = false; break;
    case 13: muonPlots = true; electronPlots = false; tauPlots = false; break;
    case 15: tauPlots = true; electronPlots = false; tauPlots = false; break;
    default: electronPlots = false; muonPlots = false; tauPlots = false;
  }
}

void BTagPerformanceAnalyzerMC::beginRun(const edm::Run & run, const edm::EventSetup & es)
{
  bookHistos();
}

void BTagPerformanceAnalyzerMC::bookHistos()
{
  //
  // Book all histograms.
  //

  //if (update) {
    //
    // append the DQM file ... we should consider this experimental
    //    edm::Service<DQMStore>().operator->()->open(std::string((const char *)(inputFile)),"/");
    // removed; DQM framework will take care
  //}

  // parton p
//   double pPartonMin = 0.0    ;
//   double pPartonMax = 99999.9 ;


  // iterate over ranges:
  const int iEtaStart = -1                   ;  // this will be the inactive one
  const int iEtaEnd   = etaRanges.size() - 1 ;
  const int iPtStart  = -1                   ;  // this will be the inactive one
  const int iPtEnd    = ptRanges.size() - 1  ;
  setTDRStyle();

  TagInfoPlotterFactory theFactory;
  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin();
       iModule != moduleConfig.end(); ++iModule) {

    const string& dataFormatType = iModule->exists("type") ?
                                   iModule->getParameter<string>("type") :
                                   "JetTag";
    if (dataFormatType == "JetTag") {
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");

      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<JetTagPlotter*>()) ;
      // Contains plots for each bin of rapidity and pt.
	vector<BTagDifferentialPlot*> * differentialPlotsConstantEta = new vector<BTagDifferentialPlot*> () ;
	vector<BTagDifferentialPlot*> * differentialPlotsConstantPt  = new vector<BTagDifferentialPlot*> () ;
      if (finalize && mcPlots_==4 && makeDiffPlots_){
	differentialPlots.push_back(vector<BTagDifferentialPlot*>());

	// the constant b-efficiency for the differential plots versus pt and eta
	const double& effBConst =
	  			iModule->getParameter<edm::ParameterSet>("parameters").getParameter<double>("effBConst");

	// the objects for the differential plots vs. eta,pt for
	for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	  BTagDifferentialPlot * etaConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constETA, folderName);
	  differentialPlotsConstantEta->push_back ( etaConstDifferentialPlot );
	}

	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  // differentialPlots for this pt bin
	  BTagDifferentialPlot * ptConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constPT, folderName);
	  differentialPlotsConstantPt->push_back ( ptConstDifferentialPlot );
	}
      }
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {

	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the genertic b tag plotter
	  JetTagPlotter *jetTagPlotter = new JetTagPlotter(folderName, etaPtBin,
							   iModule->getParameter<edm::ParameterSet>("parameters"),mcPlots_,update,finalize);
	  binJetTagPlotters.back().push_back ( jetTagPlotter ) ;

	  // Add to the corresponding differential plotters
	  if (finalize && mcPlots_==4 && makeDiffPlots_){
	    (*differentialPlotsConstantEta)[iEta+1]->addBinPlotter ( jetTagPlotter ) ;
	    (*differentialPlotsConstantPt )[iPt+1] ->addBinPlotter ( jetTagPlotter ) ;
	  }
	}
      }
      // the objects for the differential plots vs. eta, pt: collect all from constant eta and constant pt
      if (finalize && mcPlots_==4 && makeDiffPlots_){
	differentialPlots.back().reserve(differentialPlotsConstantEta->size()+differentialPlotsConstantPt->size()) ;
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantEta->begin(), differentialPlotsConstantEta->end());
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantPt->begin(), differentialPlotsConstantPt->end());

	edm::LogInfo("Info")
	  << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();

	// the intermediate ones are no longer needed
	delete differentialPlotsConstantEta ;
	delete differentialPlotsConstantPt  ;
      }
    } else if(dataFormatType == "TagCorrelation") {
        const InputTag& label1 = iModule->getParameter<InputTag>("label1");
        const InputTag& label2 = iModule->getParameter<InputTag>("label2");
        tagCorrelationInputTags.push_back(std::pair<edm::InputTag, edm::InputTag>(label1, label2));
        binTagCorrelationPlotters.push_back(vector<TagCorrelationPlotter*>());

        // eta loop
        for ( int iEta = iEtaStart ; iEta != iEtaEnd ; ++iEta) {
          // pt loop
          for( int iPt = iPtStart ; iPt != iPtEnd ; ++iPt) {
            const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
            // Instantiate the generic b tag correlation plotter
            TagCorrelationPlotter* tagCorrelationPlotter = new TagCorrelationPlotter(label1.label(), label2.label(), etaPtBin,
                                                                                     iModule->getParameter<edm::ParameterSet>("parameters"),
                                                                                     mcPlots_, update);
            binTagCorrelationPlotters.back().push_back(tagCorrelationPlotter);
          }
        }
    } else {
      // tag info retrievel is deferred (needs availability of EventSetup)
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");
      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<BaseTagInfoPlotter*>()) ;
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the tagInfo plotter

	  BaseTagInfoPlotter *jetTagPlotter = theFactory.buildPlotter(dataFormatType, moduleLabel.label(), 
			             etaPtBin, iModule->getParameter<edm::ParameterSet>("parameters"), folderName, 
                                     update, mcPlots_,finalize);
	  binTagInfoPlotters.back().push_back ( jetTagPlotter ) ;
          binTagInfoPlottersToModuleConfig.insert(make_pair(jetTagPlotter, iModule - moduleConfig.begin()));
	}
      }
      edm::LogInfo("Info")
	<< "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
    }
  }
}

EtaPtBin BTagPerformanceAnalyzerMC::getEtaPtBin(const int& iEta, const int& iPt)
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

BTagPerformanceAnalyzerMC::~BTagPerformanceAnalyzerMC()
{
  for (vector<vector<JetTagPlotter*> >::iterator iJetLabel = binJetTagPlotters.begin();
       iJetLabel != binJetTagPlotters.end(); ++iJetLabel) 
    for (vector<JetTagPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;

  if (finalize  && mcPlots_==4 && makeDiffPlots_) {
    for(vector<vector<BTagDifferentialPlot*> >::iterator iJetLabel = differentialPlots.begin();
        iJetLabel != differentialPlots.end(); ++iJetLabel)
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = iJetLabel->begin();
           iPlotter != iJetLabel->end(); ++ iPlotter) 
	delete *iPlotter;
  }

  for (vector<vector<TagCorrelationPlotter*> >::iterator iJetLabel = binTagCorrelationPlotters.begin(); 
       iJetLabel != binTagCorrelationPlotters.end(); ++iJetLabel) 
    for (vector<TagCorrelationPlotter* >::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;
    
  
  for (vector<vector<BaseTagInfoPlotter*> >::iterator iJetLabel = binTagInfoPlotters.begin(); 
       iJetLabel != binTagInfoPlotters.end(); ++iJetLabel) 
    for (vector<BaseTagInfoPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;
    
}

void BTagPerformanceAnalyzerMC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventInitialized = false;

  float weight = 1; // event weight

  if (ptHatWeight) {

    /* APPLY PTHAT EVENT  WEIGHT */

    edm::Handle<GenEventInfoProduct> genInfoHandle;
    iEvent.getByLabel("generator", genInfoHandle);
    
    if( genInfoHandle.isValid() ) {
      weight = weight*static_cast<float>(genInfoHandle->weight());
    
    }
  }
    
  LogDebug("Info") << "Event weight is: " << weight;
    
  if (finalizeOnly) return;
    
  edm::Handle<JetFlavourMatchingCollection> jetMC;
  FlavourMap flavours;
  LeptonMap leptons;

  iEvent.getByLabel(jetMCSrc, jetMC);
  for (JetFlavourMatchingCollection::const_iterator iter = jetMC->begin();
       iter != jetMC->end(); ++iter) {
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
	        binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag(jetWithFlavour.first, tagI->second, std::abs(jetWithFlavour.second.getFlavour()),weight);
      }
    }
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag(weight);
    }
  }

// Now look at Tag Correlations
  for (unsigned int iJetLabel = 0; iJetLabel != tagCorrelationInputTags.size(); ++iJetLabel) {
    const std::pair<edm::InputTag, edm::InputTag>& inputTags = tagCorrelationInputTags[iJetLabel];
    edm::Handle<reco::JetTagCollection> tagHandle1;
    iEvent.getByLabel(inputTags.first, tagHandle1);
    const reco::JetTagCollection& tagColl1 = *(tagHandle1.product());

    edm::Handle<reco::JetTagCollection> tagHandle2;
    iEvent.getByLabel(inputTags.second, tagHandle2);
    const reco::JetTagCollection& tagColl2 = *(tagHandle2.product());

    int plotterSize = binTagCorrelationPlotters[iJetLabel].size();
    for (JetTagCollection::const_iterator tagI = tagColl1.begin(); tagI != tagColl1.end(); ++tagI) {
      
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

      for(int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
        bool inBin = false;
        if (partonKinematics)
          inBin = binTagCorrelationPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(jetWithFlavour.second.getLorentzVector().Eta(),
                                                                                   jetWithFlavour.second.getLorentzVector().Pt());

        else
          inBin = binTagCorrelationPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(jetWithFlavour.first);

        if(inBin)
        {
          double discr2 = tagColl2[tagI->first];
          binTagCorrelationPlotters[iJetLabel][iPlotter]->analyzeTags(tagI->second, discr2, std::abs(jetWithFlavour.second.getFlavour()),weight);
        }
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
	        binTagInfoPlotters[iJetLabel][iPlotter]->analyzeTag(baseTagInfos, std::abs(jetWithFlavour.second.getFlavour()),weight);
      }
    }
  }
}

bool  BTagPerformanceAnalyzerMC::getJetWithFlavour(	edm::RefToBase<Jet> jetRef, const FlavourMap& flavours,
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

//  jetWithFlavour.second = reco::JetFlavour(jetWithFlavour.first.p4(), math::XYZPoint (0,0,0), flavours.at(jetRef));
  auto itFound = flavours.find(jetRef);
  unsigned int flavour = itFound != flavours.end()? itFound->second : 0;
  jetWithFlavour.second = reco::JetFlavour(jetWithFlavour.first.p4(), math::XYZPoint (0,0,0), flavour);

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
   
      if(makeDiffPlots_) { 
        for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	     iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	  (*iPlotter)->process();
	  if (producePs)  (*iPlotter)->psPlot(psBaseName);
	  if (produceEps) (*iPlotter)->epsPlot(epsBaseName);
	  //      (**iPlotter).write(allHisto);
        }
      }
  }
  for (vector<vector<BaseTagInfoPlotter*> >::iterator iJetLabel = binTagInfoPlotters.begin();
       iJetLabel != binTagInfoPlotters.end(); ++iJetLabel) {
    for (vector<BaseTagInfoPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) {
      (*iPlotter)->finalize();
      //      binTagInfoPlotters[iJetLabel][iPlotter]->write(allHisto);
      if (producePs)  (*iPlotter)->psPlot(psBaseName);
      if (produceEps) (*iPlotter)->epsPlot(epsBaseName);
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceAnalyzerMC);
