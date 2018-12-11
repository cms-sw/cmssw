/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

#include <map>
#include <set>
#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSAcceptancePlotter : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSAcceptancePlotter( const edm::ParameterSet& );
    ~CTPPSAcceptancePlotter();

  private:
    virtual void beginJob() override;

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    virtual void endJob() override;

    edm::EDGetTokenT<edm::HepMCProduct> tokenHepMC;
    edm::EDGetTokenT< std::vector<CTPPSLocalTrackLite> > tokenTracks;

    unsigned int rpId_45_N, rpId_45_F, rpId_56_N, rpId_56_F;

    std::string outputFile;

    struct SingleArmPlots
    {
      TH1D *h_xi_all = NULL, *h_xi_acc = NULL;

      void Init()
      {
        h_xi_all = new TH1D("", ";#xi", 100, 0., 0.25);
        h_xi_acc = new TH1D("", ";#xi", 100, 0., 0.25);
      }

      void Fill(double xi, bool acc)
      {
        if (h_xi_all == NULL)
          Init();

        h_xi_all->Fill(xi); 
        if (acc)
          h_xi_acc->Fill(xi); 
      }

      void Write() const
      {
        h_xi_all->Write("h_xi_all");
        h_xi_acc->Write("h_xi_acc");

        TH1D *h_xi_rat = new TH1D(*h_xi_acc);
        h_xi_rat->Divide(h_xi_all);
        h_xi_rat->Write("h_xi_rat");
      }
    };

    std::vector<std::set<unsigned int>> singleArmConfigurations;
    std::map<std::set<unsigned int>, SingleArmPlots> singleArmPlots;

    struct DoubleArmPlots
    {
      TH1D *h_m_all = NULL, *h_m_acc = NULL;
      TH2D *h2_xi_45_vs_xi_56_all, *h2_xi_45_vs_xi_56_acc;
      TH2D *h2_y_vs_m_all, *h2_y_vs_m_acc;

      void Init()
      {
        h_m_all = new TH1D("", ";m   (GeV)", 100, 0., 2500.);
        h_m_acc = new TH1D("", ";m   (GeV)", 100, 0., 2500.);

        h2_xi_45_vs_xi_56_all = new TH2D("", ";xi_56;xi_45", 25, 0., 0.25, 25, 0., 0.25);
        h2_xi_45_vs_xi_56_acc = new TH2D("", ";xi_56;xi_45", 25, 0., 0.25, 25, 0., 0.25);

        h2_y_vs_m_all = new TH2D("", ";m   (GeV);y", 25, 0., 2500., 25, -1.5, +1.5);
        h2_y_vs_m_acc = new TH2D("", ";m   (GeV);y", 25, 0., 2500., 25, -1.5, +1.5);
      }

      void Fill(double xi_45, double xi_56, bool acc)
      {
        if (h_m_all == NULL)
          Init();

        const double p_nom = 6500.;
        const double m = 2. * p_nom * sqrt(xi_45 * xi_56);
        const double y = log(xi_45 / xi_56) / 2.;

        h_m_all->Fill(m);
        h2_xi_45_vs_xi_56_all->Fill(xi_56, xi_45);
        h2_y_vs_m_all->Fill(m, y);
        
        if (acc)
        {
          h_m_acc->Fill(m); 
          h2_xi_45_vs_xi_56_acc->Fill(xi_56, xi_45);
          h2_y_vs_m_acc->Fill(m, y);
        }
      }

      void Write() const
      {
        h_m_all->Write("h_m_all");
        h_m_acc->Write("h_m_acc");

        TH1D *h_m_rat = new TH1D(*h_m_acc);
        h_m_rat->Divide(h_m_all);
        h_m_rat->Write("h_m_rat");


        h2_xi_45_vs_xi_56_all->Write("h2_xi_45_vs_xi_56_all");
        h2_xi_45_vs_xi_56_acc->Write("h2_xi_45_vs_xi_56_acc");

        TH2D *h2_xi_45_vs_xi_56_rat = new TH2D(*h2_xi_45_vs_xi_56_acc);
        h2_xi_45_vs_xi_56_rat->Divide(h2_xi_45_vs_xi_56_all);
        h2_xi_45_vs_xi_56_rat->Write("h2_xi_45_vs_xi_56_rat");


        h2_y_vs_m_all->Write("h2_y_vs_m_all");
        h2_y_vs_m_acc->Write("h2_y_vs_m_acc");

        TH2D *h2_y_vs_m_rat = new TH2D(*h2_y_vs_m_acc);
        h2_y_vs_m_rat->Divide(h2_y_vs_m_all);
        h2_y_vs_m_rat->Write("h2_y_vs_m_rat");
      }
    };

    std::vector<std::set<unsigned int>> doubleArmConfigurations;
    std::map<std::set<unsigned int>, DoubleArmPlots> doubleArmPlots;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSAcceptancePlotter::CTPPSAcceptancePlotter(const edm::ParameterSet& iConfig) :
  tokenHepMC( consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagHepMC")) ),
  tokenTracks( consumes< std::vector<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "tagTracks" ) ) ),

  rpId_45_N(iConfig.getParameter<unsigned int>("rpId_45_N")),
  rpId_45_F(iConfig.getParameter<unsigned int>("rpId_45_F")),
  rpId_56_N(iConfig.getParameter<unsigned int>("rpId_56_N")),
  rpId_56_F(iConfig.getParameter<unsigned int>("rpId_56_F")),

  outputFile(iConfig.getParameter<string>("outputFile"))
{
  singleArmConfigurations = {
    { rpId_45_N },
    { rpId_45_F },
    { rpId_56_N },
    { rpId_56_F },
    { rpId_45_N, rpId_45_F },
    { rpId_56_N, rpId_56_F },
  };

  doubleArmConfigurations = {
    { rpId_45_N, rpId_56_N },
    { rpId_45_F, rpId_56_F },
    { rpId_45_N, rpId_45_F, rpId_56_N, rpId_56_F },
  };
}

//----------------------------------------------------------------------------------------------------

CTPPSAcceptancePlotter::~CTPPSAcceptancePlotter()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSAcceptancePlotter::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
  // get input
  edm::Handle<edm::HepMCProduct> hHepMC;
  iEvent.getByToken(tokenHepMC, hHepMC);
  HepMC::GenEvent *hepMCEvent = (HepMC::GenEvent *) hHepMC->GetEvent();

  edm::Handle< std::vector<CTPPSLocalTrackLite> > hTracks;
  iEvent.getByToken(tokenTracks, hTracks);

  // extract protons
  bool proton_45_set = false;
  bool proton_56_set = false;
  FourVector mom_45, mom_56;

  for (auto it = hepMCEvent->particles_begin(); it != hepMCEvent->particles_end(); ++it)
  {
    const auto &part = *it;

    // accept only stable non-beam protons
    if (part->pdg_id() != 2212)
      continue;

    if (part->status() != 1)
      continue;

    if (part->is_beam())
      continue;

    const auto &mom = part->momentum();

    if (mom.e() < 4500.)
      continue;

    if (mom.z() > 0)
    {
      // 45
      if (proton_45_set)
      {
        LogError("CTPPSAcceptancePlotter") << "Multiple protons found in sector 45.";
        return;
      }

      proton_45_set = true;
      mom_45 = mom;
    } else {
      // 56
      if (proton_56_set)
      {
        LogError("CTPPSAcceptancePlotter") << "Multiple protons found in sector 56.";
        return;
      }

      proton_56_set = true;
      mom_56 = mom;
    }
  }

  // stop if protons missing
  if (!proton_45_set || !proton_56_set)
    return;

  // calculate xi's
  const double p_nom = 6500.;
  const double xi_45 = (p_nom - mom_45.e()) / p_nom;
  const double xi_56 = (p_nom - mom_56.e()) / p_nom;

  // process tracks
  map<unsigned int, bool> trackPresent;
  for (const auto& trk : *hTracks)
  {
    CTPPSDetId rpId(trk.getRPId());
    unsigned int rpDecId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();
    trackPresent[rpDecId] = true;
  }

  // update plots
  for (const auto rpIds : singleArmConfigurations)
  {
    bool acc = true;
    signed int arm = -1;
    for (const auto rpId : rpIds)
    {
      acc &= trackPresent[rpId];
      arm = rpId / 100;
    }

    if (arm < 0)
      continue;

    const double xi = (arm == 0) ? xi_45 : xi_56;
    
    singleArmPlots[rpIds].Fill(xi, acc);
  }

  for (const auto rpIds : doubleArmConfigurations)
  {
    bool acc = true;
    for (const auto rpId : rpIds)
      acc &= trackPresent[rpId];

    doubleArmPlots[rpIds].Fill(xi_45, xi_56, acc);
  }
}


//----------------------------------------------------------------------------------------------------

void CTPPSAcceptancePlotter::beginJob()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSAcceptancePlotter::endJob()
{
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");

  for (const auto &p : singleArmPlots)
  {
    string dirName;
    for (const auto &rpId : p.first)
    {
      if (dirName.size() > 0)
        dirName += ",";
      char buf[100];
      sprintf(buf, "%u", rpId);
      dirName += buf;
    }

    gDirectory = f_out->mkdir(dirName.c_str());
    p.second.Write();
  }

  for (const auto &p : doubleArmPlots)
  {
    string dirName;
    for (const auto &rpId : p.first)
    {
      if (dirName.size() > 0)
        dirName += ",";
      char buf[100];
      sprintf(buf, "%u", rpId);
      dirName += buf;
    }

    gDirectory = f_out->mkdir(dirName.c_str());
    p.second.Write();
  }

  delete f_out;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSAcceptancePlotter);
