#include "TauAnalysis/MCEmbeddingTools/plugins/PileUpWeightProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <TFile.h>

PileUpWeightProducer::PileUpWeightProducer(const edm::ParameterSet& cfg):
  srcPileUpSummaryInfo_(cfg.getParameter<edm::InputTag>("srcPileUpSummaryInfo"))
{
  edm::FileInPath mcFileName = cfg.getParameter<edm::FileInPath>("sourceInputFile");
  if(!mcFileName.isLocal())
    throw cms::Exception("PileUpWeightProducer") << " Failed to find File = " << mcFileName << " !!\n";

  std::auto_ptr<TFile> mcFile(new TFile(mcFileName.fullPath().data()));
  TH1* mcPileUpHisto = dynamic_cast<TH1*>(mcFile->Get("pileup"));
  if(!mcPileUpHisto)
    throw cms::Exception("PileUpWeightProducer") << " No PileUp Histogram in source input file = " << mcFileName << " !!\n";

  edm::FileInPath dataFileName = cfg.getParameter<edm::FileInPath>("targetInputFile");
  if(!dataFileName.isLocal())
    throw cms::Exception("PileUpWeightProducer") << " Failed to find File = " << dataFileName << " !!\n";

  std::auto_ptr<TFile> dataFile(new TFile(dataFileName.fullPath().data()));
  TH1* dataPileUpHisto = dynamic_cast<TH1*>(dataFile->Get("pileup"));
  if(!dataPileUpHisto)
    throw cms::Exception("PileUpWeightProducer") << " No PileUp Histogram in target input file = " << mcFileName << " !!\n";

  dataPileUpHisto->Scale(1.0/dataPileUpHisto->Integral());
  mcPileUpHisto->Scale(1.0/mcPileUpHisto->Integral());

  weightHisto_ = static_cast<TH1*>(dataPileUpHisto->Clone(std::string(dataPileUpHisto->GetName()).append("_cloned").data()));
  weightHisto_->Divide(mcPileUpHisto);

  produces<double>("weight");
}

PileUpWeightProducer::~PileUpWeightProducer() 
{
  delete weightHisto_;
}

void PileUpWeightProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<std::vector<PileupSummaryInfo> > puInfoPtr;
  evt.getByLabel(srcPileUpSummaryInfo_, puInfoPtr);

  double numTruePileUpInteractions = -1.;
  for(std::vector<PileupSummaryInfo>::const_iterator iter = puInfoPtr->begin(); iter != puInfoPtr->end(); ++iter)
    if(iter->getBunchCrossing() == 0)
      numTruePileUpInteractions = iter->getTrueNumInteractions();

  if(numTruePileUpInteractions < 0)
    throw cms::Exception("PileUpWeightProducer") << " No PileUp information for BX = 0 present !!\n";

  const double weight = weightHisto_->GetBinContent(weightHisto_->FindBin(numTruePileUpInteractions));
  std::auto_ptr<double> weightPtr(new double(weight));
  evt.put(weightPtr, "weight");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PileUpWeightProducer);
