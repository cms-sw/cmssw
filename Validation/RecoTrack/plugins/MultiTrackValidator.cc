#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleNumberOfLayers.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "CommonTools/Utils/interface/associationMapFilterValues.h"
#include<type_traits>


#include "TMath.h"
#include <TF1.h>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
//#include <iostream>

using namespace std;
using namespace edm;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;

MultiTrackValidator::MultiTrackValidator(const edm::ParameterSet& pset):
  MultiTrackValidatorBase(pset,consumesCollector()),
  parametersDefinerIsCosmic_(parametersDefiner == "CosmicParametersDefinerForTP"),
  calculateDrSingleCollection_(pset.getUntrackedParameter<bool>("calculateDrSingleCollection")),
  doPlotsOnlyForTruePV_(pset.getUntrackedParameter<bool>("doPlotsOnlyForTruePV")),
  doSummaryPlots_(pset.getUntrackedParameter<bool>("doSummaryPlots")),
  doSimPlots_(pset.getUntrackedParameter<bool>("doSimPlots")),
  doSimTrackPlots_(pset.getUntrackedParameter<bool>("doSimTrackPlots")),
  doRecoTrackPlots_(pset.getUntrackedParameter<bool>("doRecoTrackPlots")),
  dodEdxPlots_(pset.getUntrackedParameter<bool>("dodEdxPlots")),
  doPVAssociationPlots_(pset.getUntrackedParameter<bool>("doPVAssociationPlots"))
{
  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<MTVHistoProducerAlgoForTracker>(psetForHistoProducerAlgo, consumesCollector());

  dirName_ = pset.getParameter<std::string>("dirName");
  UseAssociators = pset.getParameter< bool >("UseAssociators");

  if(dodEdxPlots_) {
    m_dEdx1Tag = consumes<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx1Tag"));
    m_dEdx2Tag = consumes<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx2Tag"));
  }

  if(doPlotsOnlyForTruePV_ || doPVAssociationPlots_) {
    label_tv = consumes<TrackingVertexCollection>(pset.getParameter< edm::InputTag >("label_tv"));
    recoVertexToken_ = consumes<edm::View<reco::Vertex> >(pset.getUntrackedParameter<edm::InputTag>("label_vertex"));
    vertexAssociatorToken_ = consumes<reco::VertexToTrackingVertexAssociator>(pset.getUntrackedParameter<edm::InputTag>("vertexAssociator"));
  }

  tpSelector = TrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
					pset.getParameter<double>("minRapidityTP"),
					pset.getParameter<double>("maxRapidityTP"),
					pset.getParameter<double>("tipTP"),
					pset.getParameter<double>("lipTP"),
					pset.getParameter<int>("minHitTP"),
					pset.getParameter<bool>("signalOnlyTP"),
					pset.getParameter<bool>("intimeOnlyTP"),
					pset.getParameter<bool>("chargedOnlyTP"),
					pset.getParameter<bool>("stableOnlyTP"),
					pset.getParameter<std::vector<int> >("pdgIdTP"));

  cosmictpSelector = CosmicTrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
						    pset.getParameter<double>("minRapidityTP"),
						    pset.getParameter<double>("maxRapidityTP"),
						    pset.getParameter<double>("tipTP"),
						    pset.getParameter<double>("lipTP"),
						    pset.getParameter<int>("minHitTP"),
						    pset.getParameter<bool>("chargedOnlyTP"),
						    pset.getParameter<std::vector<int> >("pdgIdTP"));


  ParameterSet psetVsEta = psetForHistoProducerAlgo.getParameter<ParameterSet>("TpSelectorForEfficiencyVsEta");
  dRtpSelector = TrackingParticleSelector(psetVsEta.getParameter<double>("ptMin"),
					  psetVsEta.getParameter<double>("minRapidity"),
					  psetVsEta.getParameter<double>("maxRapidity"),
					  psetVsEta.getParameter<double>("tip"),
					  psetVsEta.getParameter<double>("lip"),
					  psetVsEta.getParameter<int>("minHit"),
					  psetVsEta.getParameter<bool>("signalOnly"),
					  psetVsEta.getParameter<bool>("intimeOnly"),
					  psetVsEta.getParameter<bool>("chargedOnly"),
					  psetVsEta.getParameter<bool>("stableOnly"),
					  psetVsEta.getParameter<std::vector<int> >("pdgId"));

  dRtpSelectorNoPtCut = TrackingParticleSelector(0.0,
                                                 psetVsEta.getParameter<double>("minRapidity"),
                                                 psetVsEta.getParameter<double>("maxRapidity"),
                                                 psetVsEta.getParameter<double>("tip"),
                                                 psetVsEta.getParameter<double>("lip"),
                                                 psetVsEta.getParameter<int>("minHit"),
                                                 psetVsEta.getParameter<bool>("signalOnly"),
                                                 psetVsEta.getParameter<bool>("intimeOnly"),
                                                 psetVsEta.getParameter<bool>("chargedOnly"),
                                                 psetVsEta.getParameter<bool>("stableOnly"),
                                                 psetVsEta.getParameter<std::vector<int> >("pdgId"));

  useGsf = pset.getParameter<bool>("useGsf");

  _simHitTpMapTag = mayConsume<SimHitTPAssociationProducer::SimHitTPAssociationList>(pset.getParameter<edm::InputTag>("simHitTpMapTag"));

  if(calculateDrSingleCollection_) {
    labelTokenForDrCalculation = consumes<edm::View<reco::Track> >(pset.getParameter<edm::InputTag>("trackCollectionForDrCalculation"));
  }

  if(UseAssociators) {
    for (auto const& src: associators) {
      associatorTokens.push_back(consumes<reco::TrackToTrackingParticleAssociator>(src));
    }
  } else {   
    for (auto const& src: associators) {
      associatormapStRs.push_back(consumes<reco::SimToRecoCollection>(src));
      associatormapRtSs.push_back(consumes<reco::RecoToSimCollection>(src));
    }
  }
}


MultiTrackValidator::~MultiTrackValidator() {}


void MultiTrackValidator::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const& setup) {

  const auto minColl = -0.5;
  const auto maxColl = label.size()-0.5;
  const auto nintColl = label.size();

  auto binLabels = [&](MonitorElement *me) {
    TH1 *h = me->getTH1();
    for(size_t i=0; i<label.size(); ++i) {
      h->GetXaxis()->SetBinLabel(i+1, label[i].label().c_str());
    }
    return me;
  };

  //Booking histograms concerning with simulated tracks
  if(doSimPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + "simulation");

    histoProducerAlgo_->bookSimHistos(ibook);

    ibook.cd();
    ibook.setCurrentFolder(dirName_);
  }

  for (unsigned int ww=0;ww<associators.size();ww++){
    ibook.cd();
    // FIXME: these need to be moved to a subdirectory whose name depends on the associator
    ibook.setCurrentFolder(dirName_);

    if(doSummaryPlots_) {
      if(doSimTrackPlots_) {
        h_assoc_coll.push_back(binLabels( ibook.book1D("num_assoc(simToReco)_coll", "N of associated (simToReco) tracks vs track collection", nintColl, minColl, maxColl) ));
        h_simul_coll.push_back(binLabels( ibook.book1D("num_simul_coll", "N of simulated tracks vs track collection", nintColl, minColl, maxColl) ));

        h_assoc_coll_allPt.push_back(binLabels( ibook.book1D("num_assoc(simToReco)_coll_allPt", "N of associated (simToReco) tracks vs track collection", nintColl, minColl, maxColl) ));
        h_simul_coll_allPt.push_back(binLabels( ibook.book1D("num_simul_coll_allPt", "N of simulated tracks vs track collection", nintColl, minColl, maxColl) ));

      }
      if(doRecoTrackPlots_) {
        h_reco_coll.push_back(binLabels( ibook.book1D("num_reco_coll", "N of reco track vs track collection", nintColl, minColl, maxColl) ));
        h_assoc2_coll.push_back(binLabels( ibook.book1D("num_assoc(recoToSim)_coll", "N of associated (recoToSim) tracks vs track collection", nintColl, minColl, maxColl) ));
        h_looper_coll.push_back(binLabels( ibook.book1D("num_duplicate_coll", "N of associated (recoToSim) looper tracks vs track collection", nintColl, minColl, maxColl) ));
        h_pileup_coll.push_back(binLabels( ibook.book1D("num_pileup_coll", "N of associated (recoToSim) pileup tracks vs track collection", nintColl, minColl, maxColl) ));
      }
    }

    for (unsigned int www=0;www<label.size();www++){
      ibook.cd();
      InputTag algo = label[www];
      string dirName=dirName_;
      if (algo.process()!="")
        dirName+=algo.process()+"_";
      if(algo.label()!="")
        dirName+=algo.label()+"_";
      if(algo.instance()!="")
        dirName+=algo.instance()+"_";
      if (dirName.find("Tracks")<dirName.length()){
        dirName.replace(dirName.find("Tracks"),6,"");
      }
      string assoc= associators[ww].label();
      if (assoc.find("Track")<assoc.length()){
        assoc.replace(assoc.find("Track"),5,"");
      }
      dirName+=assoc;
      std::replace(dirName.begin(), dirName.end(), ':', '_');

      ibook.setCurrentFolder(dirName.c_str());

      if(doSimTrackPlots_) {
        histoProducerAlgo_->bookSimTrackHistos(ibook);
        if(doPVAssociationPlots_) histoProducerAlgo_->bookSimTrackPVAssociationHistos(ibook);
      }

      //Booking histograms concerning with reconstructed tracks
      if(doRecoTrackPlots_) {
        histoProducerAlgo_->bookRecoHistos(ibook);
        if (dodEdxPlots_) histoProducerAlgo_->bookRecodEdxHistos(ibook);
        if (doPVAssociationPlots_) histoProducerAlgo_->bookRecoPVAssociationHistos(ibook);
      }
    }//end loop www
  }// end loop ww
}


void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){
  using namespace reco;

  LogDebug("TrackValidator") << "\n====================================================" << "\n"
                             << "Analyzing new event" << "\n"
                             << "====================================================\n" << "\n";


  edm::ESHandle<ParametersDefinerForTP> parametersDefinerTPHandle;
  setup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTPHandle);
  //Since we modify the object, we must clone it
  auto parametersDefinerTP = parametersDefinerTPHandle->clone();

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByToken(label_tp_effic,TPCollectionHeff);
  TrackingParticleCollection const & tPCeff = *(TPCollectionHeff.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByToken(label_tp_fake,TPCollectionHfake);


  if(parametersDefinerIsCosmic_) {
    edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
    //warning: make sure the TP collection used in the map is the same used in the MTV!
    event.getByToken(_simHitTpMapTag,simHitsTPAssoc);
    parametersDefinerTP->initEvent(simHitsTPAssoc);
    cosmictpSelector.initEvent(simHitsTPAssoc);
  }

  const reco::Vertex::Point *thePVposition = nullptr;
  const TrackingVertex::LorentzVector *theSimPVPosition = nullptr;
  if(doPlotsOnlyForTruePV_ || doPVAssociationPlots_) {
    edm::Handle<TrackingVertexCollection> htv;
    event.getByToken(label_tv, htv);

    edm::Handle<edm::View<reco::Vertex> > hvertex;
    event.getByToken(recoVertexToken_, hvertex);

    edm::Handle<reco::VertexToTrackingVertexAssociator> hvassociator;
    event.getByToken(vertexAssociatorToken_, hvassociator);

    auto v_r2s = hvassociator->associateRecoToSim(hvertex, htv);
    auto pvPtr = hvertex->refAt(0);
    if(!(pvPtr->isFake() || pvPtr->ndof() < 0)) { // skip junk vertices
      auto pvFound = v_r2s.find(pvPtr);
      if(pvFound != v_r2s.end()) {
        int simPVindex = -1;
        int i=0;
        for(const auto& vertexRefQuality: pvFound->val) {
          const TrackingVertex& tv = *(vertexRefQuality.first);
          if(tv.eventId().event() == 0 && tv.eventId().bunchCrossing() == 0) {
            simPVindex = i;
          }
          ++i;
        }
        if(simPVindex >= 0) {
          if(doPVAssociationPlots_) {
            thePVposition = &(pvPtr->position());
            theSimPVPosition = &(pvFound->val[simPVindex].first->position());
          }
        }
        else if(doPlotsOnlyForTruePV_)
          return;
      }
      else if(doPlotsOnlyForTruePV_)
        return;
    }
    else if(doPlotsOnlyForTruePV_)
      return;
  }

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByToken(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot const & bs = *recoBeamSpotHandle;

  edm::Handle< std::vector<PileupSummaryInfo> > puinfoH;
  event.getByToken(label_pileupinfo,puinfoH);
  PileupSummaryInfo puinfo;

  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }

  /*
  edm::Handle<TrackingVertexCollection> tvH;
  event.getByToken(label_tv,tvH);
  TrackingVertexCollection const & tv = *tvH;
  */

  // Calculate the number of 3D layers for TPs
  //
  // I would have preferred to produce the ValueMap to Event and read
  // it from there, but there would have been quite some number of
  // knock-on effects, and again the fact that we take two TP
  // collections do not support Ref<TP>'s would have complicated that.
  //
  // In principle we could use the SimHitTPAssociationList read above
  // for parametersDefinerIsCosmic_, but since we don't currently
  // support Ref<TP>s, we can't in general use it since eff/fake TP
  // collections can, in general, be different.
  TrackingParticleNumberOfLayers tpNumberOfLayersAlgo(event, simHitTokens_);
  auto nlayers_tPCeff_ptrs = tpNumberOfLayersAlgo.calculate(TPCollectionHeff, setup);
  const auto& nLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nTrackerLayers>(nlayers_tPCeff_ptrs));
  const auto& nPixelLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nPixelLayers>(nlayers_tPCeff_ptrs));
  const auto& nStripMonoAndStereoLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nStripMonoAndStereoLayers>(nlayers_tPCeff_ptrs));

  // Precalculate TP selection (for efficiency), and momentum and vertex wrt PCA
  //
  // TODO: ParametersDefinerForTP ESProduct needs to be changed to
  // EDProduct because of consumes.
  //
  // In principle, we could just precalculate the momentum and vertex
  // wrt PCA for all TPs for once and put that to the event. To avoid
  // repetitive calculations those should be calculated only once for
  // each TP. That would imply that we should access TPs via Refs
  // (i.e. View) in here, since, in general, the eff and fake TP
  // collections can be different (and at least HI seems to use that
  // feature). This would further imply that the
  // RecoToSimCollection/SimToRecoCollection should be changed to use
  // View<TP> instead of vector<TP>, and migrate everything.
  //
  // Or we could take only one input TP collection, and do another
  // TP-selection to obtain the "fake" collection like we already do
  // for "efficiency" TPs.
  std::vector<size_t> selected_tPCeff;
  std::vector<std::tuple<TrackingParticle::Vector, TrackingParticle::Point>> momVert_tPCeff;
  selected_tPCeff.reserve(tPCeff.size());
  momVert_tPCeff.reserve(tPCeff.size());
  int nIntimeTPs = 0;
  if(parametersDefinerIsCosmic_) {
    for(size_t j=0; j<tPCeff.size(); ++j) {
      TrackingParticleRef tpr(TPCollectionHeff, j);

      TrackingParticle::Vector momentum = parametersDefinerTP->momentum(event,setup,tpr);
      TrackingParticle::Point vertex = parametersDefinerTP->vertex(event,setup,tpr);
      if(doSimPlots_) {
        histoProducerAlgo_->fill_generic_simTrack_histos(momentum, vertex, tpr->eventId().bunchCrossing());
      }
      if(tpr->eventId().bunchCrossing() == 0)
        ++nIntimeTPs;

      if(cosmictpSelector(tpr,&bs,event,setup)) {
        selected_tPCeff.push_back(j);
        momVert_tPCeff.emplace_back(momentum, vertex);
      }
    }
  }
  else {
    size_t j=0;
    for(auto const& tp: tPCeff) {

      // TODO: do we want to fill these from all TPs that include IT
      // and OOT (as below), or limit to IT+OOT TPs passing tpSelector
      // (as it was before)? The latter would require another instance
      // of tpSelector with intimeOnly=False.
      if(doSimPlots_) {
        histoProducerAlgo_->fill_generic_simTrack_histos(tp.momentum(), tp.vertex(), tp.eventId().bunchCrossing());
      }
      if(tp.eventId().bunchCrossing() == 0)
        ++nIntimeTPs;

      if(tpSelector(tp)) {
        selected_tPCeff.push_back(j);
	TrackingParticleRef tpr(TPCollectionHeff, j);
        TrackingParticle::Vector momentum = parametersDefinerTP->momentum(event,setup,tpr);
        TrackingParticle::Point vertex = parametersDefinerTP->vertex(event,setup,tpr);
        momVert_tPCeff.emplace_back(momentum, vertex);
      }
      ++j;
    }
  }
  if(doSimPlots_) {
    histoProducerAlgo_->fill_simTrackBased_histos(nIntimeTPs);
  }

  //calculate dR for TPs
  float dR_tPCeff[tPCeff.size()];
  {
    float etaL[tPCeff.size()], phiL[tPCeff.size()];
    for(size_t iTP: selected_tPCeff) {
      //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
      auto const& tp2 = tPCeff[iTP];
      auto  && p = tp2.momentum();
      etaL[iTP] = etaFromXYZ(p.x(),p.y(),p.z());
      phiL[iTP] = atan2f(p.y(),p.x());
    }
    auto i=0U;
    for ( auto const & tp : tPCeff) {
      double dR = std::numeric_limits<double>::max();
      if(dRtpSelector(tp)) {//only for those needed for efficiency!
        auto  && p = tp.momentum();
        float eta = etaFromXYZ(p.x(),p.y(),p.z());
        float phi = atan2f(p.y(),p.x());
        for(size_t iTP: selected_tPCeff) {
          //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
	  if (i==iTP) {continue;}
          auto dR_tmp = reco::deltaR2(eta, phi, etaL[iTP], phiL[iTP]);
          if (dR_tmp<dR) dR=dR_tmp;
        }  // ttp2 (iTP)
      }
      dR_tPCeff[i++] = std::sqrt(dR);
    }  // tp
  }

  edm::Handle<View<Track> >  trackCollectionForDrCalculation;
  if(calculateDrSingleCollection_) {
    event.getByToken(labelTokenForDrCalculation, trackCollectionForDrCalculation);
  }

  // dE/dx
  // at some point this could be generalized, with a vector of tags and a corresponding vector of Handles
  // I'm writing the interface such to take vectors of ValueMaps
  std::vector<const edm::ValueMap<reco::DeDxData> *> v_dEdx;
  if(dodEdxPlots_) {
    edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx1Handle;
    edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx2Handle;
    event.getByToken(m_dEdx1Tag, dEdx1Handle);
    event.getByToken(m_dEdx2Tag, dEdx2Handle);
    v_dEdx.push_back(dEdx1Handle.product());
    v_dEdx.push_back(dEdx2Handle.product());
  }

  int w=0; //counter counting the number of sets of histograms
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++, w++){ // need to increment w here, since there are many continues in the loop body
      //
      //get collections from the event
      //
      edm::Handle<View<Track> >  trackCollection;
      if(!event.getByToken(labelToken[www], trackCollection)&&ignoremissingtkcollection_)continue;

      reco::RecoToSimCollection const * recSimCollP=nullptr;
      reco::SimToRecoCollection const * simRecCollP=nullptr;
      reco::RecoToSimCollection recSimCollL;
      reco::SimToRecoCollection simRecCollL;

      //associate tracks
      LogTrace("TrackValidator") << "Analyzing "
                                 << label[www] << " with "
                                 << associators[ww] <<"\n";
      if(UseAssociators){
        edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
        event.getByToken(associatorTokens[ww], theAssociator);

	LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
	recSimCollL = std::move(theAssociator->associateRecoToSim(trackCollection,
                                                                   TPCollectionHfake));
         recSimCollP = &recSimCollL;
	LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
	simRecCollL = std::move(theAssociator->associateSimToReco(trackCollection,
                                                                   TPCollectionHeff));
        simRecCollP = &simRecCollL;
      }
      else{
	Handle<reco::SimToRecoCollection > simtorecoCollectionH;
	event.getByToken(associatormapStRs[ww], simtorecoCollectionH);
	simRecCollP = simtorecoCollectionH.product();

        // We need to filter the associations of the current track
        // collection only from SimToReco collection, otherwise the
        // SimToReco histograms get false entries
        simRecCollL = associationMapFilterValues(*simRecCollP, *trackCollection);
        simRecCollP = &simRecCollL;

	Handle<reco::RecoToSimCollection > recotosimCollectionH;
	event.getByToken(associatormapRtSs[ww],recotosimCollectionH);
	recSimCollP = recotosimCollectionH.product();

        // In general, we should filter also the RecoToSim collection.
        // But, that would require changing the input type of TPs to
        // View<TrackingParticle>, and either replace current
        // associator interfaces with (View<Track>, View<TP>) or
        // adding the View,View interface (same goes for
        // RefToBaseVector,RefToBaseVector). Since there is currently
        // no compelling-enough use-case, we do not filter the
        // RecoToSim collection here. If an association using a subset
        // of the Sim collection is needed, user has to produce such
        // an association explicitly.
      }

      reco::RecoToSimCollection const & recSimColl = *recSimCollP;
      reco::SimToRecoCollection const & simRecColl = *simRecCollP;
 


      // ########################################################
      // fill simulation histograms (LOOP OVER TRACKINGPARTICLES)
      // ########################################################

      //compute number of tracks per eta interval
      //
      LogTrace("TrackValidator") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      int ats(0);  	  //This counter counts the number of simTracks that are "associated" to recoTracks
      int st(0);    	  //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )
      unsigned sts(0);   //This counter counts the number of simTracks surviving the bunchcrossing cut
      unsigned asts(0);  //This counter counts the number of simTracks that are "associated" to recoTracks surviving the bunchcrossing cut

      //loop over already-selected TPs for tracking efficiency
      for(size_t i=0; i<selected_tPCeff.size(); ++i) {
        size_t iTP = selected_tPCeff[i];
	TrackingParticleRef tpr(TPCollectionHeff, iTP);
	const TrackingParticle& tp = tPCeff[iTP];

        auto const& momVert = momVert_tPCeff[i];
	TrackingParticle::Vector momentumTP;
	TrackingParticle::Point vertexTP;

	double dxySim(0);
	double dzSim(0);
        double dxyPVSim = 0;
        double dzPVSim = 0;
	double dR=dR_tPCeff[iTP];

	//---------- THIS PART HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------
	//If the TrackingParticle is collison like, get the momentum and vertex at production state
	if(!parametersDefinerIsCosmic_)
	  {
	    momentumTP = tp.momentum();
	    vertexTP = tp.vertex();
	    //Calcualte the impact parameters w.r.t. PCA
	    const TrackingParticle::Vector& momentum = std::get<TrackingParticle::Vector>(momVert);
	    const TrackingParticle::Point& vertex = std::get<TrackingParticle::Point>(momVert);
	    dxySim = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
	    dzSim = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2())
	      * momentum.z()/sqrt(momentum.perp2());

            if(theSimPVPosition) {
              // As in TrackBase::dxy(Point) and dz(Point)
              dxyPVSim = -(vertex.x()-theSimPVPosition->x())*sin(momentum.phi()) + (vertex.y()-theSimPVPosition->y())*cos(momentum.phi());
              dzPVSim = vertex.z()-theSimPVPosition->z() - ( (vertex.x()-theSimPVPosition->x()) + (vertex.y()-theSimPVPosition->y()) )/sqrt(momentum.perp2()) * momentum.z()/sqrt(momentum.perp2());
            }
	  }
	//If the TrackingParticle is comics, get the momentum and vertex at PCA
	else
	  {
	    momentumTP = std::get<TrackingParticle::Vector>(momVert);
	    vertexTP = std::get<TrackingParticle::Point>(momVert);
	    dxySim = (-vertexTP.x()*sin(momentumTP.phi())+vertexTP.y()*cos(momentumTP.phi()));
	    dzSim = vertexTP.z() - (vertexTP.x()*momentumTP.x()+vertexTP.y()*momentumTP.y())/sqrt(momentumTP.perp2())
	      * momentumTP.z()/sqrt(momentumTP.perp2());

            // Do dxy and dz vs. PV make any sense for cosmics? I guess not
	  }
	//---------- THE PART ABOVE HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------

        //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) ), but only for in-time TPs
        if(tp.eventId().bunchCrossing() == 0) {
          st++;
        }

	// in the coming lines, histos are filled using as input
	// - momentumTP
	// - vertexTP
	// - dxySim
	// - dzSim
        if(!doSimTrackPlots_)
          continue;

	// ##############################################
	// fill RecoAssociated SimTracks' histograms
	// ##############################################
	const reco::Track* matchedTrackPointer=0;
	if(simRecColl.find(tpr) != simRecColl.end()){
	  auto const & rt = simRecColl[tpr];
	  if (rt.size()!=0) {
	    ats++; //This counter counts the number of simTracks that have a recoTrack associated
	    // isRecoMatched = true; // UNUSED
	    matchedTrackPointer = rt.begin()->first.get();
	    LogTrace("TrackValidator") << "TrackingParticle #" << st
                                       << " with pt=" << sqrt(momentumTP.perp2())
                                       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  LogTrace("TrackValidator")
	    << "TrackingParticle #" << st
	    << " with pt,eta,phi: "
	    << sqrt(momentumTP.perp2()) << " , "
	    << momentumTP.eta() << " , "
	    << momentumTP.phi() << " , "
	    << " NOT associated to any reco::Track" << "\n";
	}




        int nSimHits = tp.numberOfTrackerHits();
        int nSimLayers = nLayers_tPCeff[tpr];
        int nSimPixelLayers = nPixelLayers_tPCeff[tpr];
        int nSimStripMonoAndStereoLayers = nStripMonoAndStereoLayers_tPCeff[tpr];
        histoProducerAlgo_->fill_recoAssociated_simTrack_histos(w,tp,momentumTP,vertexTP,dxySim,dzSim,dxyPVSim,dzPVSim,nSimHits,nSimLayers,nSimPixelLayers,nSimStripMonoAndStereoLayers,matchedTrackPointer,puinfo.getPU_NumInteractions(), dR, thePVposition);
          sts++;
          if(matchedTrackPointer)
            asts++;
          if(doSummaryPlots_) {
            if(dRtpSelectorNoPtCut(tp)) {
              h_simul_coll_allPt[ww]->Fill(www);
              if (matchedTrackPointer) {
                h_assoc_coll_allPt[ww]->Fill(www);
              }

              if(dRtpSelector(tp)) {
                h_simul_coll[ww]->Fill(www);
                if (matchedTrackPointer) {
                  h_assoc_coll[ww]->Fill(www);
                }
              }
            }
          }




      } // End  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){

      // ##############################################
      // fill recoTracks histograms (LOOP OVER TRACKS)
      // ##############################################
      if(!doRecoTrackPlots_)
        continue;
      LogTrace("TrackValidator") << "\n# of reco::Tracks with "
                                 << label[www].process()<<":"
                                 << label[www].label()<<":"
                                 << label[www].instance()
                                 << ": " << trackCollection->size() << "\n";

      int sat(0); //This counter counts the number of recoTracks that are associated to SimTracks from Signal only
      int at(0); //This counter counts the number of recoTracks that are associated to SimTracks
      int rT(0); //This counter counts the number of recoTracks in general

      //calculate dR for tracks
      const edm::View<Track> *trackCollectionDr = trackCollection.product();
      if(calculateDrSingleCollection_) {
        trackCollectionDr = trackCollectionForDrCalculation.product();
      }
      float dR_trk[trackCollection->size()];
      int i=0;
      float etaL[trackCollectionDr->size()];
      float phiL[trackCollectionDr->size()];
      for (auto const & track2 : *trackCollectionDr) {
         auto  && p = track2.momentum();
         etaL[i] = etaFromXYZ(p.x(),p.y(),p.z());
         phiL[i] = atan2f(p.y(),p.x());
         ++i;
      }
      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){
	auto const &  track = (*trackCollection)[i];
	auto dR = std::numeric_limits<float>::max();
        auto  && p = track.momentum();
        float eta = etaFromXYZ(p.x(),p.y(),p.z());
        float phi = atan2f(p.y(),p.x());
	for(View<Track>::size_type j=0; j<trackCollectionDr->size(); ++j){
	  auto dR_tmp = reco::deltaR2(eta, phi, etaL[j], phiL[j]);
	  if ( (dR_tmp<dR) & (dR_tmp>std::numeric_limits<float>::min())) dR=dR_tmp;
	}
	dR_trk[i] = std::sqrt(dR);
      }

      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){

	RefToBase<Track> track(trackCollection, i);
	rT++;
 
	bool isSigSimMatched(false);
	bool isSimMatched(false);
        bool isChargeMatched(true);
        int numAssocRecoTracks = 0;
	int nSimHits = 0;
	double sharedFraction = 0.;

        auto tpFound = recSimColl.find(track);
        isSimMatched = tpFound != recSimColl.end();
        if (isSimMatched) {
            const auto& tp = tpFound->val;
	    nSimHits = tp[0].first->numberOfTrackerHits();
            sharedFraction = tp[0].second;
            if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
            if(simRecColl.find(tp[0].first) != simRecColl.end()) numAssocRecoTracks = simRecColl[tp[0].first].size();
	    at++;
	    for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite){
              TrackingParticle trackpart = *(tp[tp_ite].first);
	      if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0)){
	      	isSigSimMatched = true;
		sat++;
		break;
	      }
            }
	    LogTrace("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
                                       << " associated with quality:" << tp.begin()->second <<"\n";
	} else {
	  LogTrace("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
                                     << " NOT associated to any TrackingParticle" << "\n";
	}

	double dR=dR_trk[i];
	histoProducerAlgo_->fill_generic_recoTrack_histos(w,*track,bs.position(), thePVposition, isSimMatched,isSigSimMatched, isChargeMatched, numAssocRecoTracks, puinfo.getPU_NumInteractions(), nSimHits, sharedFraction, dR);
        if(doSummaryPlots_) {
          h_reco_coll[ww]->Fill(www);
          if(isSimMatched) {
            h_assoc2_coll[ww]->Fill(www);
            if(numAssocRecoTracks>1) {
              h_looper_coll[ww]->Fill(www);
            }
            else if(!isSigSimMatched) {
              h_pileup_coll[ww]->Fill(www);
            }
          }
        }

	// dE/dx
	if (dodEdxPlots_) histoProducerAlgo_->fill_dedx_recoTrack_histos(w,track, v_dEdx);


	//Fill other histos
	if (!isSimMatched) continue;

	histoProducerAlgo_->fill_simAssociated_recoTrack_histos(w,*track);

	TrackingParticleRef tpr = tpFound->val.begin()->first;

	/* TO BE FIXED LATER
	if (associators[ww]=="trackAssociatorByChi2"){
	  //association chi2
	  double assocChi2 = -tp.begin()->second;//in association map is stored -chi2
	  h_assochi2[www]->Fill(assocChi2);
	  h_assochi2_prob[www]->Fill(TMath::Prob((assocChi2)*5,5));
	}
	else if (associators[ww]=="quickTrackAssociatorByHits"){
	  double fraction = tp.begin()->second;
	  h_assocFraction[www]->Fill(fraction);
	  h_assocSharedHit[www]->Fill(fraction*track->numberOfValidHits());
	}
	*/


	//Get tracking particle parameters at point of closest approach to the beamline
	TrackingParticle::Vector momentumTP = parametersDefinerTP->momentum(event,setup,tpr);
	TrackingParticle::Point vertexTP = parametersDefinerTP->vertex(event,setup,tpr);
	int chargeTP = tpr->charge();

	histoProducerAlgo_->fill_ResoAndPull_recoTrack_histos(w,momentumTP,vertexTP,chargeTP,
							     *track,bs.position());


	//TO BE FIXED
	//std::vector<PSimHit> simhits=tpr.get()->trackPSimHit(DetId::Tracker);
	//nrecHit_vs_nsimHit_rec2sim[w]->Fill(track->numberOfValidHits(), (int)(simhits.end()-simhits.begin() ));

      } // End of for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){

      histoProducerAlgo_->fill_trackBased_histos(w,at,rT,st);

      LogTrace("TrackValidator") << "Total Simulated: " << st << "\n"
                                 << "Total Associated (simToReco): " << ats << "\n"
                                 << "Total Reconstructed: " << rT << "\n"
                                 << "Total Associated (recoToSim): " << at << "\n"
                                 << "Total Fakes: " << rT-at << "\n";
    } // End of  for (unsigned int www=0;www<label.size();www++){
  } //END of for (unsigned int ww=0;ww<associators.size();ww++){

}
