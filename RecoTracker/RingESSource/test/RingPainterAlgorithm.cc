//
// Package:         RecoTracker/RingESSource/test
// Class:           RingPainter
// 
// Description:     paints rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:01:46 $
// $Revision: 1.1 $
//

#include <vector>
#include <sstream>

#include "RecoTracker/RingESSource/test/RingPainterAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TCanvas.h"
#include "TH2D.h"
#include "TGaxis.h"
#include "TPaveLabel.h"

RingPainterAlgorithm::RingPainterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 

  pictureName_ = conf_.getUntrackedParameter<std::string>("PictureName");

}

RingPainterAlgorithm::~RingPainterAlgorithm() {
}


void RingPainterAlgorithm::run(const Rings* rings)
{

  TCanvas *canvas = new TCanvas;

  canvas->SetTopMargin(0.02);

  canvas->SetFrameBorderMode(0);
  canvas->SetBorderMode(0);
  canvas->SetFillColor(0);

  canvas->Draw();

  TH2D *histo = new TH2D("histo","",100,-280,280,100,0,120);
  histo->SetStats(kFALSE);
  histo->GetXaxis()->SetTitle("z [cm]");
  histo->GetXaxis()->SetTitleOffset(0.9);
  histo->GetXaxis()->SetTitleSize(0.05);
  histo->GetYaxis()->SetTitle("r [cm]");
  histo->GetYaxis()->SetTitleOffset(1.0);
  histo->GetYaxis()->SetTitleSize(0.05);


  histo->Draw();

  //draw an axis on the right side
  TGaxis *axis = new TGaxis(280,0,280,120,
			    0,120,510,"+L");
  axis->SetTitle("r [cm]");
  axis->SetTitleSize(0.05);
  axis->SetLabelSize(0.05);
  axis->Draw();


  std::vector<TPaveLabel*> labels;


  for (Rings::const_iterator ring = rings->begin();
       ring != rings->end();
       ++ring ) {
    std::ostringstream text;
    text << ring->second.getindex();
    TPaveLabel *label = new TPaveLabel(ring->second.getzmin(),
				       ring->second.getrmin(),
				       ring->second.getzmax(),
				       ring->second.getrmax(),
				       text.str().c_str());
    label->SetFillColor(41);
    label->SetBorderSize(1);
    labels.push_back(label);
    label->Draw("SAME");
  }
  
  canvas->SaveAs(pictureName_.c_str());

}
