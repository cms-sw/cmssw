/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TFile.h"
#include "TH1D.h"

#include <map>

//----------------------------------------------------------------------------------------------------

class CTPPSBeamSmearingValidator : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSBeamSmearingValidator( const edm::ParameterSet& );

    ~CTPPSBeamSmearingValidator() override {}

  private:
    void analyze(const edm::Event&, const edm::EventSetup&) override;

    void endJob() override;

    edm::EDGetTokenT<edm::HepMCProduct> tokenBeforeSmearing_;
    edm::EDGetTokenT<edm::HepMCProduct> tokenAfterSmearing_;

    std::string outputFile_;

    std::unique_ptr<TH1D> h_de_vtx_x_, h_de_vtx_y_, h_de_vtx_z_;

    struct SectorPlots
    {
      std::unique_ptr<TH1D> h_de_th_x, h_de_th_y, h_de_p;

      SectorPlots() :
        h_de_th_x(new TH1D("", ";#Delta#theta_{x}   (rad)", 100, 0., 0.)),
        h_de_th_y(new TH1D("", ";#Delta#theta_{y}   (rad)", 100, 0., 0.)),
        h_de_p(new TH1D("", ";#Deltap   (GeV)", 100, 0., 0.)) {}

      void write() const
      {
        h_de_th_x->Write("h_de_th_x");
        h_de_th_y->Write("h_de_th_y");
        h_de_p->Write("h_de_p");
      }
    };

    std::map<unsigned int, SectorPlots> sectorPlots_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

CTPPSBeamSmearingValidator::CTPPSBeamSmearingValidator(const edm::ParameterSet& iConfig) :
  tokenBeforeSmearing_( consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagBeforeSmearing")) ),
  tokenAfterSmearing_( consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("tagAfterSmearing")) ),
  outputFile_(iConfig.getParameter<string>("outputFile")),
  h_de_vtx_x_(new TH1D("h_de_vtx_x", ";#Delta vtx_{x}   (mm)", 100, 0., 0.)),
  h_de_vtx_y_(new TH1D("h_de_vtx_y", ";#Delta vtx_{y}   (mm)", 100, 0., 0.)),
  h_de_vtx_z_(new TH1D("h_de_vtx_z", ";#Delta vtx_{z}   (mm)", 100, 0., 0.))
{}

//----------------------------------------------------------------------------------------------------

void CTPPSBeamSmearingValidator::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
  // get input
  edm::Handle<edm::HepMCProduct> hBeforeSmearing;
  iEvent.getByToken(tokenBeforeSmearing_, hBeforeSmearing);
  HepMC::GenEvent *orig = (HepMC::GenEvent *) hBeforeSmearing->GetEvent();

  edm::Handle<edm::HepMCProduct> hAfterSmearing;
  iEvent.getByToken(tokenAfterSmearing_, hAfterSmearing);
  HepMC::GenEvent *smear = (HepMC::GenEvent *) hAfterSmearing->GetEvent();

  // vertices
  GenEvent::vertex_const_iterator vold, vnew;
  for (vold = orig->vertices_begin(), vnew = smear->vertices_begin();
      vold != orig->vertices_end() && vnew != smear->vertices_end(); ++vold, ++vnew)
  {
    const FourVector &vo = (*vold)->position();
    const FourVector &vn = (*vnew)->position();

    // HepMC gives vertex in mm
    h_de_vtx_x_->Fill(vn.x() - vo.x());
    h_de_vtx_y_->Fill(vn.y() - vo.y());
    h_de_vtx_z_->Fill(vn.z() - vo.z());
  }

  // particles
  GenEvent::particle_const_iterator pold, pnew;
  for (pold = orig->particles_begin(), pnew = smear->particles_begin();
      pold != orig->particles_end() && pnew != smear->particles_end(); ++pold, ++pnew)
  {
    FourVector o = (*pold)->momentum(), n = (*pnew)->momentum();

    // determine direction region
    signed int idx = -1;
    const double thetaLim = 0.01; // rad
    double th = o.theta();

    if (th < thetaLim)
      idx = 0;
    if (th > (M_PI - thetaLim))
      idx = 1;

    if (idx < 0)
      continue;

    /*
        cout << "particle\n\told: [" << o.x() << ", " << o.y() << ", " << o.z() << ", " << o.t()
        << "]\n\tnew: [" << n.x() << ", " << n.y() << ", " << n.z() << ", " << n.t()
        << "]\n\tregion: " << idx << endl;
    */

    // fill histograms
    auto &sp = sectorPlots_[idx];

    double othx = o.x() / o.rho(), othy = o.y() / o.rho();
    double nthx = n.x() / n.rho(), nthy = n.y() / n.rho();

    sp.h_de_p->Fill(n.rho() - o.rho());

    sp.h_de_th_x->Fill(nthx - othx);
    sp.h_de_th_y->Fill(nthy - othy);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSBeamSmearingValidator::endJob()
{
  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  h_de_vtx_x_->Write();
  h_de_vtx_y_->Write();
  h_de_vtx_z_->Write();

  gDirectory = f_out->mkdir("sector 45");
  sectorPlots_[0].write();

  gDirectory = f_out->mkdir("sector 56");
  sectorPlots_[1].write();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSBeamSmearingValidator);

