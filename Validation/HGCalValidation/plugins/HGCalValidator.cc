#include "Validation/HGCalValidation/interface/HGCalValidator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

HGCalValidator::HGCalValidator(const edm::ParameterSet& pset):
  label(pset.getParameter< std::vector<edm::InputTag> >("label")),
  dolayerclustersPlots_(pset.getUntrackedParameter<bool>("dolayerclustersPlots"))
{
  
  //In this way we can easily generalize to associations between other objects also. 
  const edm::InputTag& label_cp_effic_tag = pset.getParameter< edm::InputTag >("label_cp_effic");
  const edm::InputTag& label_cp_fake_tag = pset.getParameter< edm::InputTag >("label_cp_fake");

  label_cp_effic = consumes<std::vector<CaloParticle> >(label_cp_effic_tag);
  label_cp_fake = consumes<std::vector<CaloParticle> >(label_cp_fake_tag);

  for (auto& itag : label) {
    labelToken.push_back(consumes<reco::CaloClusterCollection>(itag));
  }
  
  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<HGVHistoProducerAlgo>(psetForHistoProducerAlgo);

  dirName_ = pset.getParameter<std::string>("dirName");

}


HGCalValidator::~HGCalValidator() {}


void HGCalValidator::bookHistograms(DQMStore::ConcurrentBooker& ibook, edm::Run const&, edm::EventSetup const& setup, Histograms& histograms) const {

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
      histoProducerAlgo_->bookClusterHistos(ibook, histograms.histoProducerAlgo);
    }

  }//end loop www
}

void HGCalValidator::dqmAnalyze(const edm::Event& event, const edm::EventSetup& setup, const Histograms& histograms) const {
  using namespace reco;

  LogDebug("HGCalValidator") << "\n====================================================" << "\n"
                             << "Analyzing new event" << "\n"
                             << "====================================================\n" << "\n";


  int w=0; //counter counting the number of sets of histograms
  for (unsigned int www=0;www<label.size();www++, w++){ // need to increment w here, since there will be many continues in the loop body
    //
    //get collections from the event
    //
    
    edm::Handle<reco::CaloClusterCollection> clusterHandle;
    event.getByToken(labelToken[www],clusterHandle);
    const reco::CaloClusterCollection &clusters = *clusterHandle;
    
    // ##############################################
    // fill cluster histograms (LOOP OVER CLUSTERS)
    // ##############################################
    if(!dolayerclustersPlots_){continue;}

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
