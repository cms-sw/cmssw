#include "Validation/HGCalValidation/interface/HGCalValidator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

HGCalValidator::HGCalValidator(const edm::ParameterSet& pset):
  label(pset.getParameter< std::vector<edm::InputTag> >("label")),
  doCaloParticlePlots_(pset.getUntrackedParameter<bool>("doCaloParticlePlots")),
  dolayerclustersPlots_(pset.getUntrackedParameter<bool>("dolayerclustersPlots")),
  cummatbudinxo_(pset.getParameter<edm::FileInPath>("cummatbudinxo"))
{
  
  //In this way we can easily generalize to associations between other objects also. 
  const edm::InputTag& label_cp_effic_tag = pset.getParameter< edm::InputTag >("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter< edm::InputTag >("label_cp_fake");

  label_cp_effic = consumes<std::vector<CaloParticle> >(label_cp_effic_tag);
  label_cp_fake = consumes<std::vector<CaloParticle> >(label_cp_fake_tag);

  simVertices_ = consumes<std::vector<SimVertex>>(pset.getParameter<edm::InputTag>("simVertices"));
  
  recHitsEE_ = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  recHitsFH_ = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  recHitsBH_ = consumes<HGCRecHitCollection>(edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));

  for (auto& itag : label) {
    labelToken.push_back(consumes<reco::CaloClusterCollection>(itag));
  }

  cpSelector = CaloParticleSelector(pset.getParameter<double>("ptMinCP"),
				    pset.getParameter<double>("ptMaxCP"),
				    pset.getParameter<double>("minRapidityCP"),
				    pset.getParameter<double>("maxRapidityCP"),
				    pset.getParameter<int>("minHitCP"),
				    pset.getParameter<double>("tipCP"),
				    pset.getParameter<double>("lipCP"),
				    pset.getParameter<bool>("signalOnlyCP"),
				    pset.getParameter<bool>("intimeOnlyCP"),
				    pset.getParameter<bool>("chargedOnlyCP"),
				    pset.getParameter<bool>("stableOnlyCP"),
				    pset.getParameter<std::vector<int> >("pdgIdCP"));
  
  tools_.reset(new hgcal::RecHitTools());
 
  particles_to_monitor_ = pset.getParameter<std::vector<int> >("pdgIdCP");
  totallayers_to_monitor_ = pset.getParameter<int>("totallayers_to_monitor");
  thicknesses_to_monitor_ = pset.getParameter<std::vector<int> >("thicknesses_to_monitor");
  
  //For the material budget file here
  std::ifstream fmb(cummatbudinxo_.fullPath().c_str());
  double thelay = 0.; double mbg = 0.;
  for (unsigned ilayer = 1; ilayer <= totallayers_to_monitor_; ++ilayer) {
    fmb >> thelay >> mbg;
    cummatbudg.insert( std::pair<double, double>( thelay , mbg ) ); 
  }
  
  fmb.close();



  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<HGVHistoProducerAlgo>(psetForHistoProducerAlgo);

  dirName_ = pset.getParameter<std::string>("dirName");

}


HGCalValidator::~HGCalValidator() {}


void HGCalValidator::bookHistograms(DQMStore::ConcurrentBooker& ibook, edm::Run const&, edm::EventSetup const& setup, Histograms& histograms) const {

  // rechittools_->getEventSetup(setup);
  // histoProducerAlgo_->setRecHitTools(rechittools_);
  
  if(doCaloParticlePlots_) {
    ibook.cd();

    for (auto const particle : particles_to_monitor_) {
      ibook.setCurrentFolder(dirName_ + "SelectedCaloParticles/" + std::to_string(particle));     
      histoProducerAlgo_->bookCaloParticleHistos(ibook, histograms.histoProducerAlgo, particle);
    }
    ibook.cd();
    ibook.setCurrentFolder(dirName_);
  }

  for (unsigned int www=0;www<label.size();www++){
    ibook.cd();
    InputTag algo = label[www];
    string dirName=dirName_;
    if (!algo.process().empty())
      dirName+=algo.process()+"_";
    std::cout << dirName << std::endl; 
    if(!algo.label().empty())
      dirName+=algo.label()+"_";
    std::cout << dirName << std::endl; 
    if(!algo.instance().empty())
      dirName+=algo.instance()+"_";
    std::cout << dirName << std::endl; 
    // if (dirName.find("Tracks")<dirName.length()){
    //   dirName.replace(dirName.find("Tracks"),6,"");
    // }

    if (dirName.size () > 0){dirName.resize(dirName.size() - 1);}
    
    std::cout << dirName << std::endl; 

    ibook.setCurrentFolder(dirName);

    //Booking histograms concerning with hgcal layer clusters
    if(dolayerclustersPlots_) {
      histoProducerAlgo_->bookClusterHistos(ibook, histograms.histoProducerAlgo, totallayers_to_monitor_, thicknesses_to_monitor_);
    }

  }//end loop www
}

void HGCalValidator::cpParametersAndSelection(const Histograms& histograms,
					      std::vector<CaloParticle> const & cPeff,
					      std::vector<SimVertex> const & simVertices,
					      std::vector<size_t>& selected_cPeff) const {
  selected_cPeff.reserve(cPeff.size());

  size_t j=0;
  for (auto const caloParticle : cPeff) {
    int id = caloParticle.pdgId();

    if(cpSelector(caloParticle,simVertices)) {
      selected_cPeff.push_back(j);
      if(doCaloParticlePlots_) {
	histoProducerAlgo_->fill_caloparticle_histos(histograms.histoProducerAlgo,id,caloParticle,simVertices);
      }
    }
    ++j;
  }//end of loop over caloparticles
  
}

void HGCalValidator::dqmAnalyze(const edm::Event& event, const edm::EventSetup& setup, const Histograms& histograms) const {
  using namespace reco;

  LogDebug("HGCalValidator") << "\n====================================================" << "\n"
                             << "Analyzing new event" << "\n"
                             << "====================================================\n" << "\n";

  edm::Handle<std::vector<SimVertex>> simVerticesHandle;
  event.getByToken(simVertices_, simVerticesHandle);
  std::vector<SimVertex> const & simVertices = *simVerticesHandle;

  edm::Handle<std::vector<CaloParticle> > caloParticleHandle;
  event.getByToken(label_cp_effic, caloParticleHandle);
  std::vector<CaloParticle> const & caloParticles = *caloParticleHandle;
  
  tools_->getEventSetup(setup);
  histoProducerAlgo_->setRecHitTools(tools_);

  edm::Handle<HGCRecHitCollection> recHitHandleEE;
  event.getByToken(recHitsEE_, recHitHandleEE);
  edm::Handle<HGCRecHitCollection> recHitHandleFH;
  event.getByToken(recHitsFH_, recHitHandleFH);
  edm::Handle<HGCRecHitCollection> recHitHandleBH;
  event.getByToken(recHitsBH_, recHitHandleBH);
  
  histoProducerAlgo_->fillHitMap(*recHitHandleEE,*recHitHandleFH,*recHitHandleBH);
  
  
  // ##############################################
  // fill caloparticles histograms 
  // ##############################################
  LogTrace("HGCalValidator") << "\n# of CaloParticles: " << caloParticles.size() << "\n";
  std::vector<size_t> selected_cPeff;
  cpParametersAndSelection(histograms, caloParticles, simVertices, selected_cPeff);
  
  int w=0; //counter counting the number of sets of histograms
  for (unsigned int www=0;www<label.size();www++, w++){ // need to increment w here, since there will be many continues in the loop body
    
    //get collections from the event
    edm::Handle<reco::CaloClusterCollection> clusterHandle;
    event.getByToken(labelToken[www],clusterHandle);
    const reco::CaloClusterCollection &clusters = *clusterHandle;

    // ##############################################
    // fill cluster histograms (LOOP OVER CLUSTERS)
    // ##############################################
    if(!dolayerclustersPlots_){continue;}

    histoProducerAlgo_->fill_generic_cluster_histos(histograms.histoProducerAlgo,w,clusters,caloParticles,cummatbudg,totallayers_to_monitor_, thicknesses_to_monitor_);

    for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {

      //std::cout << "TESTING HERE " << clusters[layerclusterIndex].eta() << std::endl;     
      histoProducerAlgo_->fill_cluster_histos(histograms.histoProducerAlgo,w,clusters[layerclusterIndex]);
      
    }

    LogTrace("HGCalValidator") << "\n# of layer clusters with "
			       << label[www].process()<<":"
			       << label[www].label()<<":"
			       << label[www].instance()
			       << ": " << clusters.size() << "\n";

  } // End of  for (unsigned int www=0;www<label.size();www++){
}

