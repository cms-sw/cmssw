/****************************************************************************
 *
 * This is a part of CTPPS validation software
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "TFile.h"
#include "TH2D.h"
#include "TProfile.h"

#include <map>

//----------------------------------------------------------------------------------------------------

class CTPPSTrackDistributionPlotter : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSTrackDistributionPlotter( const edm::ParameterSet& );
    ~CTPPSTrackDistributionPlotter();

  private:
    virtual void beginJob() override;

    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;

    virtual void endJob() override;

    edm::EDGetTokenT< std::vector<CTPPSLocalTrackLite> > tracksToken_;

    std::string outputFile;

    struct RPPlots
    {
      TH2D *h2_y_vs_x;
      TProfile *p_y_vs_x;

      void init()
      {
        h2_y_vs_x = new TH2D("", "", 300, -10., +50., 300, -30, +30.);
        p_y_vs_x = new TProfile("", "", 300, -10., +50.);
      }

      void fill(double x, double y)
      {
        if (h2_y_vs_x == NULL)
          init();

        h2_y_vs_x->Fill(x, y);
        p_y_vs_x->Fill(x, y);
      }

      void write() const
      {
        h2_y_vs_x->Write("h2_y_vs_x");
        p_y_vs_x->Write("p_y_vs_x");
      }
    };

    std::map<unsigned int, RPPlots> rpPlots;
};

//----------------------------------------------------------------------------------------------------

CTPPSTrackDistributionPlotter::CTPPSTrackDistributionPlotter( const edm::ParameterSet& iConfig ) :
  tracksToken_( consumes< std::vector<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "tracksTag" ) ) ),
  outputFile( iConfig.getParameter<std::string>("outputFile") )
{
}

//----------------------------------------------------------------------------------------------------

CTPPSTrackDistributionPlotter::~CTPPSTrackDistributionPlotter()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSTrackDistributionPlotter::analyze( const edm::Event& iEvent, const edm::EventSetup& )
{
  // get input
  edm::Handle< std::vector<CTPPSLocalTrackLite> > tracks;
  iEvent.getByToken( tracksToken_, tracks );

  // process tracks
  for (const auto& trk : *tracks)
  {
    CTPPSDetId rpId(trk.getRPId());
    unsigned int rpDecId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();
    rpPlots[rpDecId].fill(trk.getX(), trk.getY());
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSTrackDistributionPlotter::beginJob()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSTrackDistributionPlotter::endJob()
{
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");
  
  for (const auto it : rpPlots)
  {
    gDirectory = f_out->mkdir(Form("RP %u", it.first));
    it.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSTrackDistributionPlotter );
