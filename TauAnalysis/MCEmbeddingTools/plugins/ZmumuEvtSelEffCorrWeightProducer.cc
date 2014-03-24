#include "TauAnalysis/MCEmbeddingTools/plugins/ZmumuEvtSelEffCorrWeightProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TFile.h>

#include <vector>
#include <algorithm>
#include <math.h>

ZmumuEvtSelEffCorrWeightProducer::ZmumuEvtSelEffCorrWeightProducer(const edm::ParameterSet& cfg)
  : lutEfficiencyPt_(0),
    lutEffCorrEta_(0)
{
  srcSelectedMuons_ = cfg.getParameter<edm::InputTag>("selectedMuons");

  edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
  if ( inputFileName.location() == edm::FileInPath::Unknown) 
    throw cms::Exception("MuonRadiationCorrWeightProducer") 
      << " Failed to find File = " << inputFileName << " !!\n";
  std::auto_ptr<TFile> inputFile(new TFile(inputFileName.fullPath().data()));

  std::string lutEfficiencyPtName = cfg.getParameter<std::string>("lutEfficiencyPt");
  TH2* lutEfficiencyPt = dynamic_cast<TH2*>(inputFile->Get(lutEfficiencyPtName.data()));
  if ( !lutEfficiencyPt ) 
    throw cms::Exception("MuonRadiationCorrWeightProducer") 
      << " Failed to load LUT = " << lutEfficiencyPtName << " from file = " << inputFile->GetName() << " !!\n";
  lutEfficiencyPt_ = (TH2*)lutEfficiencyPt->Clone(std::string(lutEfficiencyPt->GetName()).append("_cloned").data());
  xAxisEfficiencyPt_ = lutEfficiencyPt_->GetXaxis();
  yAxisEfficiencyPt_ = lutEfficiencyPt_->GetYaxis();

  std::string lutEffCorrEtaName = cfg.getParameter<std::string>("lutEffCorrEta");
  TH2* lutEffCorrEta = dynamic_cast<TH2*>(inputFile->Get(lutEffCorrEtaName.data()));
  if ( !lutEffCorrEta ) 
    throw cms::Exception("MuonRadiationCorrWeightProducer") 
      << " Failed to load LUT = " << lutEffCorrEtaName << " from file = " << inputFile->GetName() << " !!\n";
  lutEffCorrEta_ = (TH2*)lutEffCorrEta->Clone(std::string(lutEffCorrEta->GetName()).append("_cloned").data());
  xAxisEffCorrEta_ = lutEffCorrEta_->GetXaxis();
  yAxisEffCorrEta_ = lutEffCorrEta_->GetYaxis();
  
  minWeight_ = cfg.getParameter<double>("minWeight"); 
  maxWeight_ = cfg.getParameter<double>("maxWeight");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<double>("weight");
  produces<double>("weightUp");
  produces<double>("weightDown");
}

ZmumuEvtSelEffCorrWeightProducer::~ZmumuEvtSelEffCorrWeightProducer() 
{
  delete lutEfficiencyPt_;
  delete lutEffCorrEta_;
}

namespace
{
  int findBin(TAxis* xAxis, double x)
  {
    int bin = xAxis->FindBin(x);
    if ( bin < 1                 ) bin = 1;
    if ( bin > xAxis->GetNbins() ) bin = xAxis->GetNbins();
    return bin;
  }

  double square(double x)
  {
    return x*x;
  }
}

void ZmumuEvtSelEffCorrWeightProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<ZmumuEvtSelEffCorrWeightProducer::produce>:" << std::endl;
    std::cout << " srcSelectedMuons = " << srcSelectedMuons_ << std::endl;
  }

  double weight = 1.;
  double weightUp = 2.*maxWeight_;
  double weightDown = 0.;
  
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);
  if ( muPlus.isNonnull() && muMinus.isNonnull() ) {
    if ( verbosity_ ) {
      std::cout << "Mu+: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
      std::cout << "Mu-: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
    }

    int binX_efficiencyPt = findBin(xAxisEfficiencyPt_, muPlus->pt());
    int binY_efficiencyPt = findBin(yAxisEfficiencyPt_, muMinus->pt());
    double efficiencyPt = lutEfficiencyPt_->GetBinContent(binX_efficiencyPt, binY_efficiencyPt);
    double efficiencyPtErr = lutEfficiencyPt_->GetBinError(binX_efficiencyPt, binY_efficiencyPt);

    int binX_effCorrEta = findBin(xAxisEffCorrEta_, muPlus->eta());
    int binY_effCorrEta = findBin(yAxisEffCorrEta_, muMinus->eta());
    double effCorrEta = lutEffCorrEta_->GetBinContent(binX_effCorrEta, binY_effCorrEta);
    double effCorrEtaErr = lutEffCorrEta_->GetBinError(binX_effCorrEta, binY_effCorrEta);

    double efficiency = efficiencyPt*effCorrEta;
    if ( efficiency > 0. ) {
      weight = 1./efficiency;
      if ( weight > maxWeight_ ) weight = maxWeight_;
      if ( weight < minWeight_ ) weight = minWeight_;
      weightUp = weight + TMath::Sqrt(square(efficiencyPtErr*effCorrEta) + square(efficiencyPt*effCorrEtaErr));
      if ( weightUp > (2.*maxWeight_) ) weightUp = 2.*maxWeight_;
      if ( weightUp < weight ) weightUp = weight;
      weightDown = weight - TMath::Sqrt(square(efficiencyPtErr*effCorrEta) + square(efficiencyPt*effCorrEtaErr));
      if ( weightDown > weight ) weightDown = weight;
      if ( weightDown < 0. ) weightDown = 0.;
    } else {
      weight = maxWeight_;
      weightUp = 2.*maxWeight_;
      weightDown = 0.;
    }
  }
  
  if ( verbosity_ ) {
    std::cout << "--> weight = " << weight << " + " << (weightUp - weight) << " - " << (weight - weightDown) << std::endl;
  }

  std::auto_ptr<double> weightPtr(new double(weight));
  evt.put(weightPtr, "weight");
  std::auto_ptr<double> weightUpPtr(new double(weightUp));
  evt.put(weightUpPtr, "weightUp");
  std::auto_ptr<double> weightDownPtr(new double(weightDown));
  evt.put(weightDownPtr, "weightDown");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZmumuEvtSelEffCorrWeightProducer);


