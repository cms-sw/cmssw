
/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: PhysicsTools/TagAndProbe
 *
 *
 * Authors:
 *   Giovanni Petrucciani, UCSD - Giovanni.Petrucciani@cern.ch
 *
 * Description:
 *   - Matches a given object with other objects using deltaR-matching.
 *   - For example: can match a photon with track within a given deltaR.
 *   - Saves collection of the reference vectors of matched objects.
 * History:
 *   
 * Kalanand Mishra, Fermilab - kalanand@fnal.gov
 * Extended the class to compute deltaR with respect to any object 
 * (i.e., Candidate, Jet, Muon, Electron, or Photon). The previous 
 * version of this class could deltaR only with respect to reco::Candidates.
 * This didn't work if one wanted to apply selection cuts on the Candidate's 
 * RefToBase object.
 *****************************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class PV : public edm::EDProducer {
    public:
        explicit PV(const edm::ParameterSet & iConfig);
        virtual ~PV() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
        double GetPUWeight(const edm::Event& iEvent, const std::string& pileup, edm::LumiReWeighting& LumiWeightsMC);
    private:
        edm::InputTag probes_;            
        bool makePV;
        bool makeSAEta;
        bool makeSAPt;
        bool makeWeight;
        bool isData;

        std::vector< float > BgLumiMC;
        std::vector< float > TrueDist;
        edm::LumiReWeighting LumiWeightsMC;
};

PV::PV(const edm::ParameterSet & iConfig) :
  probes_(iConfig.getParameter<edm::InputTag>("probes")),
  makePV(iConfig.getParameter<bool>("makePV")),
  makeSAEta(iConfig.getParameter<bool>("makeSAEta")),
  makeSAPt(iConfig.getParameter<bool>("makeSAPt")),
  makeWeight(iConfig.getParameter<bool>("makeWeight")),
  isData(iConfig.getParameter<bool>("isData"))
{
    produces<edm::ValueMap<float> >();

    //for initializing PileupReweighting utility.
    const   float Pileup_MC_Summer2012[60] = { 2.560E-06, 5.239E-06, 1.420E-05, 5.005E-05, 1.001E-04, 2.705E-04, 1.999E-03, 6.097E-03, 1.046E-02, 1.383E-02, 1.685E-02, 2.055E-02, 2.572E-02, 3.262E-02, 4.121E-02, 4.977E-02, 5.539E-02, 5.725E-02, 5.607E-02, 5.312E-02, 5.008E-02, 4.763E-02, 4.558E-02, 4.363E-02, 4.159E-02, 3.933E-02, 3.681E-02, 3.406E-02, 3.116E-02, 2.818E-02, 2.519E-02, 2.226E-02, 1.946E-02, 1.682E-02, 1.437E-02, 1.215E-02, 1.016E-02, 8.400E-03, 6.873E-03, 5.564E-03, 4.457E-03, 3.533E-03, 2.772E-03, 2.154E-03, 1.656E-03, 1.261E-03, 9.513E-04, 7.107E-04, 5.259E-04, 3.856E-04, 2.801E-04, 2.017E-04, 1.439E-04, 1.017E-04, 7.126E-05, 4.948E-05, 3.405E-05, 2.322E-05, 1.570E-05, 5.005E-06};

    const   float TrueDist2012_f[60] = {1.05858e-06 ,2.79007e-06 ,5.66022e-06 ,2.21761e-05 ,4.333e-05 ,0.00021475 ,0.00127484 ,0.00380513 ,0.00859346 ,0.0164099 ,0.0277558 ,0.0411688 ,0.0518905 ,0.0579633 ,0.0615463 ,0.0640369 ,0.0648159 ,0.0639443 ,0.0622142 ,0.0598481 ,0.0571089 ,0.0543368 ,0.0516781 ,0.0487896 ,0.0449614 ,0.0397967 ,0.0335265 ,0.0267498 ,0.0201118 ,0.0141912 ,0.00941021 ,0.00590948 ,0.00354911 ,0.00204957 ,0.00113529 ,0.000598229 ,0.00029732 ,0.000138844 ,6.11323e-05 ,2.5644e-05 ,1.04009e-05 ,4.139e-06 ,1.63291e-06 ,6.41399e-07 ,2.50663e-07 ,9.71641e-08 ,3.72356e-08 ,1.40768e-08 ,5.24657e-09 ,1.92946e-09 ,7.01358e-10 ,2.52448e-10 ,9.00753e-11 ,3.18556e-11 ,1.11511e-11 ,3.85524e-12 ,1.31312e-12 ,4.3963e-13 ,1.44422e-13 ,4.64971e-14};

    for(int i=0; i<60; ++i) BgLumiMC.push_back(Pileup_MC_Summer2012[i]);
    for(int i=0; i<60; ++i) TrueDist.push_back(TrueDist2012_f[i]);

    LumiWeightsMC = edm::LumiReWeighting(BgLumiMC, TrueDist);
}


PV::~PV()
{
}

void 
PV::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes;
    iEvent.getByLabel(probes_,  probes);

    Handle< std::vector<reco::Vertex> > vertexCollHandle;
    iEvent.getByLabel("offlinePrimaryVertices",  vertexCollHandle);
    if(!vertexCollHandle.isValid()){printf("Vertex Collection NotFound\n");return;}
    const std::vector<reco::Vertex>& vertexColl = *vertexCollHandle;

    Handle<View<reco::Muon> > muons;
    iEvent.getByLabel(probes_, muons);

    std::vector<float> values;

    // fill
    int i=0;
    View<reco::Candidate>::const_iterator probe, endprobes = probes->end();
    for (probe = probes->begin(); probe != endprobes; ++probe) {
      RefToBase<reco::Muon> muonRef = muons->refAt(i);
      if(makePV) values.push_back(vertexColl.size());
      else if(makeSAEta) {
	if(probe->isStandAloneMuon()) values.push_back(muonRef->standAloneMuon()->eta());
	else values.push_back(999.);
      }
      else if(makeSAPt) {
        if(probe->isStandAloneMuon()) values.push_back(muonRef->standAloneMuon()->pt());
        else values.push_back(0.);
      }
      else if(makeWeight) {
	if(isData) values.push_back(1.);
	else values.push_back(GetPUWeight(iEvent, "S10", LumiWeightsMC));
      }
      i++;
    }

    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap);
}

double PV::GetPUWeight(const edm::Event& iEvent, const std::string& pileup, edm::LumiReWeighting& LumiWeightsMC){
  edm::Handle<std::vector< PileupSummaryInfo > >  PupInfo;
  iEvent.getByLabel(edm::InputTag("addPileupInfo"), PupInfo);
  if(!PupInfo.isValid()){printf("PileupSummaryInfo Collection NotFound\n");return 1.0;}
  double PUWeight_thisevent=1;
  std::vector<PileupSummaryInfo>::const_iterator PVI;
  int npv = -1; float Tnpv = -1;

  if(pileup=="S4"){
    float sum_nvtx = 0;
    for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
      npv = PVI->getPU_NumInteractions();
      sum_nvtx += float(npv);
    }
    float ave_nvtx = sum_nvtx/3.;
    PUWeight_thisevent = LumiWeightsMC.weight( ave_nvtx );
  }else if(pileup=="S3"){
    for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
      int BX = PVI->getBunchCrossing();
      if(BX == 0) {
	npv = PVI->getPU_NumInteractions();
	continue;
      }
    }
    PUWeight_thisevent = LumiWeightsMC.weight( npv );
  }else if(pileup=="S10"){
    for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
      int BX = PVI->getBunchCrossing();
      if(BX == 0) {
	Tnpv = PVI->getTrueNumInteractions();
	continue;
      }
    }
    PUWeight_thisevent = LumiWeightsMC.weight( Tnpv );
  }

  return PUWeight_thisevent;
}


////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PV);          

