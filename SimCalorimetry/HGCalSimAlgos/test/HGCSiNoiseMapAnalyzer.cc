// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"


#include <TH1F.h>
#include <TH2F.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//STL headers
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#include "vdt/vdtMath.h"

using namespace std;

//
// class declaration
//
class HGCSiNoiseMapAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {

public:

  explicit HGCSiNoiseMapAnalyzer(const edm::ParameterSet&);
  ~HGCSiNoiseMapAnalyzer() override;
  
private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}
  
  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  std::map<DetId::Detector, HGCalSiNoiseMap *> noiseMaps_;
  std::map< std::pair<DetId::Detector,int>, TH1F *> layerN_, layerCCE_, layerNoise_, layerIleak_, layerSN_, layerF_;
  std::map< DetId::Detector, TH2F *> detN_, detCCE_, detNoise_, detIleak_, detSN_, detF_;
  std::map<HGCSiliconDetId::waferType,double> signalfC_;
};

//
HGCSiNoiseMapAnalyzer::HGCSiNoiseMapAnalyzer(const edm::ParameterSet& iConfig)
{
  usesResource("TFileService");
  fs_->file().cd();

  //configure the dose map
  std::string doseMapURL( iConfig.getParameter<std::string>("doseMap") );
  noiseMaps_[DetId::HGCalEE]=new HGCalSiNoiseMap;
  noiseMaps_[DetId::HGCalEE]->setDoseMap( doseMapURL );
  noiseMaps_[DetId::HGCalHSi]=new HGCalSiNoiseMap;
  noiseMaps_[DetId::HGCalHSi]->setDoseMap( doseMapURL );

  double ePerMipPerMicron(iConfig.getParameter<double>("ePerMipPerMicron"));
  double e2fc(1.60217646E-4);
  signalfC_[HGCSiliconDetId::waferType::HGCalFine]=ePerMipPerMicron*e2fc*120;
  signalfC_[HGCSiliconDetId::waferType::HGCalCoarseThin]=ePerMipPerMicron*e2fc*200;
  signalfC_[HGCSiliconDetId::waferType::HGCalCoarseThick]=ePerMipPerMicron*e2fc*300;
}

//
HGCSiNoiseMapAnalyzer::~HGCSiNoiseMapAnalyzer()
{
}

//
void HGCSiNoiseMapAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& es)
{
  //get geometry
  edm::ESHandle<CaloGeometry> geom;
  es.get<CaloGeometryRecord>().get(geom);

  std::vector<DetId::Detector> dets={DetId::HGCalEE,DetId::HGCalHSi};
  for(auto &d:dets) {
    noiseMaps_[d]->setGeometry( geom->getSubdetectorGeometry(d, ForwardSubdetector::ForwardEmpty) );
    //sub-detector boundaries
    unsigned int nlay=noiseMaps_[d]->ddd()->layers(true);
    std::pair<double,double> ranZ=noiseMaps_[d]->ddd()->rangeZ(true);
    std::pair<double,double> ranRAtZmin=noiseMaps_[d]->ddd()->rangeR(ranZ.first,true);
    std::pair<double,double> ranRAtZmax=noiseMaps_[d]->ddd()->rangeR(ranZ.first,true);
    std::pair<double,double> ranR(ranRAtZmin.first-20,ranRAtZmax.second+20);

    const std::vector<DetId> &detIdVec = noiseMaps_[d]->geom()->getValidDetIds();
    cout << "Subdetector:" << d << " has " << detIdVec.size() << " valid cells" << endl
         << "\t" <<  ranR.first << "<r<" << ranR.second << "\t" << ranZ.first << "<z<" << ranZ.second << endl;

    //start histos
    TString baseName(Form("d%d_",d));
    TString title(d==DetId::HGCalEE ? "CEE" : "CEH_{Si}");    
    Int_t nbinsR(100);
    for(unsigned int ilay=0; ilay<nlay; ilay++) {

      //this layer histos
      int layer(ilay+1);
      std::pair<DetId::Detector,int> key(d,layer);
      TString layerBaseName(Form("%slayer%d_",baseName.Data(),layer));
      TString layerTitle(Form("%s %d",title.Data(),layer));
      layerTitle+= ";Radius [cm];";
      layerN_[key]     = fs_->make<TH1F>(layerBaseName+"ncells",  layerTitle+"Cells",               nbinsR, ranR.first, ranR.second);
      layerCCE_[key]   = fs_->make<TH1F>(layerBaseName+"cce",     layerTitle+"<CCE>",               nbinsR, ranR.first, ranR.second);                     
      layerNoise_[key] = fs_->make<TH1F>(layerBaseName+"noise",   layerTitle+"<Noise> [fC]",        nbinsR, ranR.first, ranR.second);
      layerIleak_[key] = fs_->make<TH1F>(layerBaseName+"ileak",   layerTitle+"<I_{leak}> [#muA]",   nbinsR, ranR.first, ranR.second);
      layerSN_[key]    = fs_->make<TH1F>(layerBaseName+"sn",      layerTitle+"<S/N>",               nbinsR, ranR.first, ranR.second);
      layerF_[key]     = fs_->make<TH1F>(layerBaseName+"fluence", layerTitle+"<F> [n_{eq}/cm^{2}]", nbinsR, ranR.first, ranR.second);
    }

    //sub-detector histos
    title+=";Layer;Radius [cm];";
    detN_[d]     = fs_->make<TH2F>(baseName+"ncells",  title+"Cells",               nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
    detCCE_[d]   = fs_->make<TH2F>(baseName+"cce",     title+"<CCE>",               nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
    detNoise_[d] = fs_->make<TH2F>(baseName+"noise",   title+"<Noise> [fC]",        nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
    detIleak_[d] = fs_->make<TH2F>(baseName+"ileak",   title+"<I_{leak}> [#muA]",   nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
    detSN_[d]    = fs_->make<TH2F>(baseName+"sn",      title+"<S/N>",               nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
    detF_[d]     = fs_->make<TH2F>(baseName+"fluence", title+"<F> [n_{eq}/cm^{2}]", nlay,1,nlay+1, nbinsR,ranR.first,ranR.second);
                    
    //fill histos
    for(const auto &cellId : detIdVec)
      {
        HGCSiliconDetId id(cellId.rawId());
        int layer = std::abs(id.layer());
        GlobalPoint pt = noiseMaps_[d]->geom()->getPosition(id);
        double r(pt.perp());
        
        HGCalSiNoiseMap::SiCellOpCharacteristics siop=noiseMaps_[d]->getSiCellOpCharacteristics(HGCalSiNoiseMap::q80fC,id,r);

        double S(signalfC_[HGCSiliconDetId::waferType(id.type())]);

        detN_[d]->Fill(layer,r,1);
        detCCE_[d]->Fill(layer,r,siop.cce);
        detNoise_[d]->Fill(layer,r,siop.noise);
        detSN_[d]->Fill(layer,r,S/siop.noise);
        detIleak_[d]->Fill(layer,r,siop.ileak);
        float fluence=exp(siop.lnfluence);
        detF_[d]->Fill(layer,r,fluence);

        std::pair<DetId::Detector,int> key(d,layer);
        layerN_[key]->Fill(r,1);        
        layerCCE_[key]->Fill(r,siop.cce);
        layerNoise_[key]->Fill(r,siop.noise);
        layerSN_[key]->Fill(r,S/siop.noise);
        layerIleak_[key]->Fill(r,siop.ileak);
        layerF_[key]->Fill(r,fluence);
      }    
  
    //normalize histos per cell counts
    detF_[d]->Divide(detN_[d]);
    detCCE_[d]->Divide(detN_[d]);
    detNoise_[d]->Divide(detN_[d]);
    detSN_[d]->Divide(detN_[d]);
    detIleak_[d]->Divide(detN_[d]);
    for(unsigned int ilay=0; ilay<nlay; ilay++) {
      int layer(ilay+1);
      std::pair<DetId::Detector,int> key(d,layer);
      layerF_[key]->Divide(layerN_[key]);
      layerCCE_[key]->Divide(layerN_[key]);
      layerNoise_[key]->Divide(layerN_[key]);
      layerSN_[key]->Divide(layerN_[key]);
      layerIleak_[key]->Divide(layerN_[key]);
    }
  }
  


}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCSiNoiseMapAnalyzer);
