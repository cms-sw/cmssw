#include <utility>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"

namespace {
  struct dictionary {
    std::vector<PGlobalSimHit::Vtx>                         dummy1;
    std::vector<PGlobalSimHit::Trk>                         dummy2;
    std::vector<PGlobalSimHit::CalHit>                      dummy3;
    std::vector<PGlobalSimHit::FwdHit>                      dummy4;
    std::vector<PGlobalSimHit::BrlHit>                      dummy5;
    edm::Wrapper<PGlobalSimHit>                             theValidData1;

    std::vector<PGlobalDigi::ECalDigi>                      dummy6;
    std::vector<PGlobalDigi::ESCalDigi>                     dummy7;
    std::vector<PGlobalDigi::HCalDigi>                      dummy8;
    std::vector<PGlobalDigi::SiStripDigi>                   dummy9;
    std::vector<PGlobalDigi::SiPixelDigi>                   dummy10;
    std::vector<PGlobalDigi::DTDigi>                        dummy11;
    std::vector<PGlobalDigi::CSCstripDigi>                  dummy12;
    std::vector<PGlobalDigi::CSCwireDigi>                   dummy13;
    edm::Wrapper<PGlobalDigi>                               theValidData2;

    std::vector<PGlobalRecHit::ECalRecHit>                  dummy14;
    std::vector<PGlobalRecHit::HCalRecHit>                  dummy15;
    std::vector<PGlobalRecHit::SiStripRecHit>               dummy16;
    std::vector<PGlobalRecHit::SiPixelRecHit>               dummy17;
    std::vector<PGlobalRecHit::DTRecHit>                    dummy18;
    std::vector<PGlobalRecHit::CSCRecHit>                   dummy19;
    std::vector<PGlobalRecHit::RPCRecHit>                   dummy20;
    edm::Wrapper<PGlobalRecHit>                             theValidData3;

    edm::Wrapper<PEcalValidInfo>                            theValidData4;

    PHcalValidInfoLayer                                     theLayer;
    PHcalValidInfoNxN                                       theNxN;
    PHcalValidInfoJets                                      theJets;
    edm::Wrapper<PHcalValidInfoLayer>                       theValidData5;
    edm::Wrapper<PHcalValidInfoNxN>                         theValidData6;
    edm::Wrapper<PHcalValidInfoJets>                        theValidData7;

    std::vector<PMuonSimHit::Vtx>                           dummy21;
    std::vector<PMuonSimHit::Trk>                           dummy22;
    std::vector<PMuonSimHit::CSC>                           dummy23;
    std::vector<PMuonSimHit::DT>                            dummy24;
    std::vector<PMuonSimHit::RPC>                           dummy25;
    edm::Wrapper<PMuonSimHit>                               theValidData8;

    std::vector<PTrackerSimHit::Vtx>                        dummy26;
    std::vector<PTrackerSimHit::Trk>                        dummy27;
    std::vector<PTrackerSimHit::Hit>                        dummy28;
    edm::Wrapper<PTrackerSimHit>                            theValidData9;
    
    //std::pair<float,float>                                p;
    
    MaterialAccountingStep                                  mastep;
    std::vector<MaterialAccountingStep>                     mastep_v;
    edm::Wrapper<std::vector<MaterialAccountingStep> >      mastep_w;

    MaterialAccountingDetector                              madet;
    std::vector<MaterialAccountingDetector>                 madet_v;
    edm::Wrapper<std::vector<MaterialAccountingDetector> >  madet_w;

    MaterialAccountingTrack                                 matrack;
    std::vector<MaterialAccountingTrack>                    matrack_v;
    edm::Wrapper<std::vector<MaterialAccountingTrack> >     matrack_w;
  };
}
