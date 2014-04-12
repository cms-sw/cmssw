/*!
  \file PFTauRenderPlugin
  \brief Display Plugin for Pixel DQM Histograms
  \author P.Merkel
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraphPolar.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include <cassert>
#include <string>

using namespace std;

class PFTauRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject & o, const VisDQMImgInfo & )
    {
      return ((o.name.find( "RecoTauV/" ) != std::string::npos ) && (o.name.find( "Eff" ) != std::string::npos ) ); //Size and SumPt are already configured
    }

  virtual void preDraw( TCanvas * canvas, const VisDQMObject & o, const VisDQMImgInfo & , VisDQMRenderInfo & renderInfo)
    {
      canvas->cd();
      TH1* obj = dynamic_cast<TH1*>( o.object );
      if(!obj) return; //nothing to do for TH2
      //general setings
      //drawing options
      gStyle->SetOptStat(0);
      renderInfo.drawOptions = "E0";
      if(o.name.find( "Rejection" ) != std::string::npos ) canvas->SetLogy();
      if(o.name.find( "RealData"  ) != std::string::npos ) canvas->SetLogy();

      //titles and axis
      string discriminator = stripDicriminator(o.name);
      string variable = stripVar(o.name);
      obj->SetTitle((discriminator+" fake rate vs "+variable).c_str());
      obj->GetXaxis()->SetTitle(variable.c_str());
      obj->GetYaxis()->SetTitle("fake rate");
      double min = (canvas->GetLogy() ) ? 0.001 : 0.;
      double max = (canvas->GetLogy() ) ? 2.    : 1.2;
      obj->GetYaxis()->SetRangeUser(min,max);
      obj->SetMarkerStyle(20);
    }

  virtual void postDraw( TCanvas *, const VisDQMObject &, const VisDQMImgInfo & )
    {
    }

private:

  string stripDicriminator(string name)
  {
    return name.substr(name.rfind("/")+1,name.rfind("Eff")-name.rfind("/")-1);
  }
  string stripVar(string name)
  {
    return name.substr(name.rfind("Eff")+3);
  }
};

static PFTauRenderPlugin instance;
