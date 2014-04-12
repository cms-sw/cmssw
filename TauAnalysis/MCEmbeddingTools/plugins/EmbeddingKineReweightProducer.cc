#include "TauAnalysis/MCEmbeddingTools/plugins/EmbeddingKineReweightProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>
#include <math.h>

EmbeddingKineReweightProducer::EmbeddingKineReweightProducer(const edm::ParameterSet& cfg)
{
  srcGenDiTaus_ = cfg.getParameter<edm::InputTag>("srcGenDiTaus"); 

  edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
  if ( inputFileName.location() == edm::FileInPath::Unknown) 
    throw cms::Exception("EmbeddingReweightProducer") 
      << " Failed to find File = " << inputFileName << " !!\n";
  std::auto_ptr<TFile> inputFile(new TFile(inputFileName.fullPath().data()));

  edm::ParameterSet cfgLUTs = cfg.getParameter<edm::ParameterSet>("lutNames"); 
  std::vector<std::string> variables = cfgLUTs.getParameterNamesForType<std::string>();
  for ( std::vector<std::string>::const_iterator variable = variables.begin(); 
        variable != variables.end(); ++variable ) {
    std::string lutName = cfgLUTs.getParameter<std::string>(*variable);
    lutEntryType* lutEntry = 
      new lutEntryType(*inputFile, *variable, lutName);
    lutEntries_.push_back(lutEntry);
  }
  
  minWeight_ = cfg.getParameter<double>("minWeight"); 
  maxWeight_ = cfg.getParameter<double>("maxWeight");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  for ( std::vector<lutEntryType*>::iterator lutEntry = lutEntries_.begin();
	lutEntry != lutEntries_.end(); ++lutEntry ) {
    produces<double>((*lutEntry)->variableName_);
  }
}

EmbeddingKineReweightProducer::~EmbeddingKineReweightProducer()
{
  for ( std::vector<lutEntryType*>::iterator it = lutEntries_.begin();
	it != lutEntries_.end(); ++it ) {
    delete (*it);
  }
}

void EmbeddingKineReweightProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<EmbeddingKineReweightProducer::produce>:" << std::endl;
  }

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> genDiTaus;
  evt.getByLabel(srcGenDiTaus_, genDiTaus);
  if ( genDiTaus->size() != 1 )
    throw cms::Exception("EmbeddingKineReweightProducer") 
      << "Failed to find unique genDiTau object !!\n";  
  const reco::Candidate& genDiTau = genDiTaus->front();
  if ( verbosity_ ) {
    std::cout << "diTau: Pt = " << genDiTau.pt() << ", eta = " << genDiTau.eta() << ", phi = " << genDiTau.phi() << ", mass = " << genDiTau.mass() << std::endl;
  }

  for ( std::vector<lutEntryType*>::const_iterator lutEntry = lutEntries_.begin();
	lutEntry != lutEntries_.end(); ++lutEntry ) {
    double weight = (**lutEntry)(genDiTau);
    if ( weight < minWeight_ ) weight = minWeight_;
    if ( weight > maxWeight_ ) weight = maxWeight_;
    if ( verbosity_ ) {
      std::cout << " " << (*lutEntry)->variableName_ << " = " << weight << std::endl;
    }
    std::auto_ptr<double> weightPtr(new double(weight));
    evt.put(weightPtr, (*lutEntry)->variableName_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EmbeddingKineReweightProducer);


