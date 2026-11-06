#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "CondFormats/DataRecord/interface/HGCalAgeingCondsRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalAgeingCondsDevice.h"

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalAlgoBasedAgeingCondProducer : public ESProducer {
    public:
      //
      HGCalAlgoBasedAgeingCondProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
        auto cc = setWhatProduced(this);
        geoTkn_ = cc.consumes();
        denseIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("denseIndexer"));
        cellMappingToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellMappingSource"));

        //radiation map common to all the detector
        auto doseMapURL = iConfig.getParameter<std::string>("doseMapURL");
        auto doseMapAlgo = iConfig.getParameter<uint32_t>("doseMapAlgo");
        auto scaleByDoseFactor = iConfig.getParameter<double>("scaleByDoseFactor");

        //Si
        edm::ParameterSet siPSet = iConfig.getParameter<edm::ParameterSet>("si");
        std::vector<double> ileakParam(siPSet.getParameter<std::vector<double>>("ileakParam"));
        std::vector<double> cceParam120(siPSet.getParameter<edm::ParameterSet>("cceParams")
                                            .template getParameter<std::vector<double>>("cceParam120"));
        std::vector<double> cceParam200(siPSet.getParameter<edm::ParameterSet>("cceParams")
                                            .template getParameter<std::vector<double>>("cceParam200"));
        std::vector<double> cceParam300(siPSet.getParameter<edm::ParameterSet>("cceParams")
                                            .template getParameter<std::vector<double>>("cceParam300"));
        siNoiseMap_ = std::make_unique<HGCalSiNoiseMap<HGCSiliconDetId>>();
        siNoiseMap_->setDoseMap(doseMapURL, doseMapAlgo);
        siNoiseMap_->setFluenceScaleFactor(scaleByDoseFactor);
        siNoiseMap_->setIleakParam(ileakParam);
        siNoiseMap_->setCceParam(cceParam120, cceParam200, cceParam300);

        //init the noise map for the SiPM-on-tile
        edm::ParameterSet sipmPSet = iConfig.getParameter<edm::ParameterSet>("sipm");
        auto refIdark = sipmPSet.getParameter<double>("refIdark");
        auto refXtalk = sipmPSet.getParameter<double>("refXtalk");
        sipmNoiseMap_ = std::make_unique<HGCalSciNoiseMap>();
        sipmNoiseMap_->setDoseMap(doseMapURL, doseMapAlgo);
        sipmNoiseMap_->setFluenceScaleFactor(scaleByDoseFactor);
        sipmNoiseMap_->setReferenceDarkCurrent(refIdark);
        sipmNoiseMap_->setReferenceCrossTalk(refXtalk);
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::ESInputTag>("denseIndexer", edm::ESInputTag(""))->setComment("Dense indexer SoA source");
        desc.add<edm::ESInputTag>("cellMappingSource", edm::ESInputTag(""))->setComment("Cell mapping source");

        //radiation map parameters
        desc.add<std::string>("doseMapURL")->setComment("dose map file");
        desc.add<uint32_t>("doseMapAlgo")->setComment("fluence algo to use");
        desc.add<double>("scaleByDoseFactor")->setComment("dose scaling factor");

        //si ageing parameters
        edm::ParameterSetDescription siDesc;
        siDesc.add<std::vector<double>>("ileakParam");
        edm::ParameterSetDescription cceDesc;
        cceDesc.add<std::vector<double>>("cceParam120");
        cceDesc.add<std::vector<double>>("cceParam200");
        cceDesc.add<std::vector<double>>("cceParam300");
        siDesc.add<edm::ParameterSetDescription>("cceParams", cceDesc);
        desc.add<edm::ParameterSetDescription>("si", siDesc)->setComment("Si sensor main ageing parameters");

        //sipm-on-tile ageing parameters
        edm::ParameterSetDescription sipmDesc;
        sipmDesc.add<double>("refIdark");
        sipmDesc.add<double>("refXtalk");
        desc.add<edm::ParameterSetDescription>("sipm", sipmDesc)
            ->setComment("SiPM-on-tile sensor main ageing parameters");

        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalAgeingCondsHost> produce(const HGCalAgeingCondsRcd& iRecord) {
        //get the geometry record
        const auto& geom = iRecord.get(geoTkn_);

        //get the dense indexer in use
        auto const& denseIndexInfo = iRecord.get(denseIndexTkn_);
        auto const& denseIndexInfo_view = denseIndexInfo.const_view();

        //get the cell mapping info
        auto const& cell_info = iRecord.get(cellMappingToken_);
        auto const& cell_info_view = cell_info.const_view();

        //declare the dense index info collection to be produced
        //the size is determined by the module indexer
        const int nIndices = denseIndexInfo_view.metadata().size();
        HGCalAgeingCondsHost ageingConds(cms::alpakatools::host(), nIndices);
        auto ac_view = ageingConds.view();
        for (int i = 0; i < nIndices; i++) {
          //get the information on this cell
          auto di_row = denseIndexInfo_view[i];
          DetId did(di_row.detid());
          auto cell_info_idx = di_row.cellInfoIdx();
          auto chType = cell_info_view[cell_info_idx].t();
          bool isSiPM = cell_info_view[cell_info_idx].isSiPM();

          // unconnected channels
          if (chType < 0 || did.det() == 0) {
            ac_view[i].signal() = 0;
            ac_view[i].signalScale() = 0;
            ac_view[i].enc() = 0;
            ac_view[i].enc_s() = 0;
            ac_view[i].enc_p() = 0;
            ac_view[i].ileak() = 0;
            ac_view[i].xtalk() = 0;
            ac_view[i].ntotalPE() = 0;
            ac_view[i].gain() = 0;
            ac_view[i].toa_thr() = 0;
            ac_view[i].tot_thr() = 0;
            ac_view[i].adc_lsb() = 0;
            ac_view[i].tot_lsb() = 0;
            ac_view[i].ln_f() = 0;
            ac_view[i].ln_dose() = 0;
          }
          //SiPM-on-tile
          else if (isSiPM) {
            HGCScintillatorDetId sciid(did);
            sipmNoiseMap_->setGeometry(geom.getSubdetectorGeometry(did.det(), ForwardSubdetector::ForwardEmpty));

            auto sipmop = sipmNoiseMap_->scaleByDose(sciid);
            ac_view[i].signal() = sipmop.s;
            ac_view[i].signalScale() = sipmop.lySF;
            ac_view[i].enc() = sipmop.n;
            ac_view[i].enc_s() = sipmop.n;
            ac_view[i].enc_p() = 0.;  //FIXME
            ac_view[i].ileak() = 0.;  //FIXME
            ac_view[i].xtalk() = sipmop.xtalk;
            ac_view[i].ntotalPE() = sipmop.ntotalPE;
            ac_view[i].gain() = sipmop.gain;
            ac_view[i].toa_thr() = 0.;  //FIXME
            ac_view[i].tot_thr() = sipmNoiseMap_->getMaxADCPerGain()[sipmop.gain] - 1e-6;
            ac_view[i].adc_lsb() = sipmNoiseMap_->getLSBPerGain()[sipmop.gain];
            ac_view[i].tot_lsb() = 0.;  //FIXME
            ac_view[i].ln_f() = sipmop.lnfluence;
            ac_view[i].ln_dose() = sipmop.lndose;
          }
          //Si
          else {
            HGCSiliconDetId siid(did);
            siNoiseMap_->setGeometry(geom.getSubdetectorGeometry(did.det(), ForwardSubdetector::ForwardEmpty));
            auto siop = siNoiseMap_->getSiCellOpCharacteristics(
                siid, HGCalSiNoiseMap<HGCSiliconDetId>::GainRange_t::AUTO, 10, chType);
            ac_view[i].signal() = siop.mipfC;
            ac_view[i].signalScale() = siop.core.cce;
            ac_view[i].enc() = siop.core.noise;
            ac_view[i].enc_s() = siop.enc_s;
            ac_view[i].enc_p() = siop.enc_p;
            ac_view[i].ileak() = siop.ileak;
            ac_view[i].xtalk() = 0;
            ac_view[i].ntotalPE() = 0;
            ac_view[i].gain() = siop.core.gain;
            ac_view[i].toa_thr() = siop.toa_thr;
            ac_view[i].tot_thr() = siop.tot_thr;
            ac_view[i].adc_lsb() = siop.adc_lsb;
            ac_view[i].tot_lsb() = siop.tot_lsb;
            ac_view[i].ln_f() = siop.lnfluence;
            ac_view[i].ln_dose() = siop.lndose;
          }
        }

        //return denseIdxInfo;
        return ageingConds;

      }  // end of produce()

    private:
      edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoTkn_;
      edm::ESGetToken<hgcal::HGCalDenseIndexInfoHost, HGCalDenseIndexInfoRcd> denseIndexTkn_;
      edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellMappingToken_;
      std::unique_ptr<HGCalSiNoiseMap<HGCSiliconDetId>> siNoiseMap_;
      std::unique_ptr<HGCalSciNoiseMap> sipmNoiseMap_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalAlgoBasedAgeingCondProducer);
