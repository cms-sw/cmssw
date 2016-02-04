//
// Package:         RecoTracker/RingESSource/test
// Class:           RoadPainter
// 
// Description:     paints rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:15:01 $
// $Revision: 1.1 $
//

#include <vector>
#include <sstream>

#include "RecoTracker/RoadMapESSource/test/RoadPainterAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TCanvas.h"
#include "TH2D.h"
#include "TGaxis.h"
#include "TPaveLabel.h"

RoadPainterAlgorithm::RoadPainterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 

  pictureName_ = conf_.getUntrackedParameter<std::string>("PictureName");

}

RoadPainterAlgorithm::~RoadPainterAlgorithm() {
}


void RoadPainterAlgorithm::run(const Rings* rings, const Roads *roads)
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


  //draw an axis on the right side
  TGaxis *axis = new TGaxis(280,0,280,120,
			    0,120,510,"+L");
  axis->SetTitle("r [cm]");
  axis->SetTitleSize(0.05);
  axis->SetLabelSize(0.05);
  axis->Draw();

  // variables
  std::vector<TPaveLabel*> allRings;
  std::ostringstream text;

  for (Rings::const_iterator ring = rings->begin();
       ring != rings->end();
       ++ring ) {
    text.str("");
    text << ring->second.getindex();
    TPaveLabel *label = new TPaveLabel(ring->second.getzmin(),
				       ring->second.getrmin(),
				       ring->second.getzmax(),
				       ring->second.getrmax(),
				       text.str().c_str());
    label->SetFillColor(41);
    label->SetBorderSize(1);
    allRings.push_back(label);
  }

  // generate filenames
  std::string begin_pictureName = pictureName_ + ".ps[";
  std::string pictureName       = pictureName_ + ".ps";
  std::string end_pictureName   = pictureName_ + ".ps]";

  // vector for storing seed rings for final plot
  std::vector<TPaveLabel*> innerSeedRings;
  std::vector<TPaveLabel*> outerSeedRings;

  // open file
  canvas->Print(begin_pictureName.c_str());

  for ( Roads::const_iterator road = roads->begin();
	road != roads->end();
	++road ) {

    // clear canvas
    canvas->Clear();

    // draw histogram
    histo->Draw();

    // draw all rings

    for ( std::vector<TPaveLabel*>::iterator ringLabel = allRings.begin();
	  ringLabel != allRings.end();
	  ++ringLabel ) {
      (*ringLabel)->Draw("SAME");
    }

    const Roads::RoadSeed *seed = &(road->first);
    const Roads::RoadSet  *set  = &(road->second);

    // loop over roadset and draw roads
    unsigned int fillColor = 38;
    for ( Roads::RoadSet::const_iterator layer = set->begin(); layer != set->end(); ++layer ) {
      if ( fillColor == 38 ) {
	fillColor = 4;
      } else {
	fillColor = 38;
      }
      for ( std::vector<const Ring*>::const_iterator ring = layer->begin();
	    ring != layer->end();
	    ++ring ) {
	text.str("");
	text << (*ring)->getindex();
	TPaveLabel *ringLabel = new TPaveLabel((*ring)->getzmin(),
					       (*ring)->getrmin(),
					       (*ring)->getzmax(),
					       (*ring)->getrmax(),
					       text.str().c_str());
	ringLabel->SetFillColor(fillColor);
	ringLabel->SetBorderSize(1);
	ringLabel->Draw("SAME");
      }
    }
    
    // draw inner seed rings
    for (std::vector<const Ring*>::const_iterator ring = seed->first.begin();
	 ring != seed->first.end();
	 ++ring ) {
      text.str("");
      text << (*ring)->getindex();
      TPaveLabel *seed1 = new TPaveLabel((*ring)->getzmin(),
					 (*ring)->getrmin(),
					 (*ring)->getzmax(),
					 (*ring)->getrmax(),
					 text.str().c_str());
      seed1->SetFillColor(2);
      seed1->SetBorderSize(1);
      seed1->Draw("SAME");
      innerSeedRings.push_back(seed1);
    }

    // draw outer seed rings
    for (std::vector<const Ring*>::const_iterator ring = seed->second.begin();
	 ring != seed->second.end();
	 ++ring ) {
      text.str("");
      text << (*ring)->getindex();
      TPaveLabel *seed2 = new TPaveLabel((*ring)->getzmin(),
					 (*ring)->getrmin(),
					 (*ring)->getzmax(),
					 (*ring)->getrmax(),
					 text.str().c_str());
      seed2->SetFillColor(8);
      seed2->SetBorderSize(1);
      seed2->Draw("SAME");
      outerSeedRings.push_back(seed2);
    }
  
    // save into picture file
    canvas->Print(pictureName.c_str());

  }

  // draw rings and all seed rings
  // clear canvas
  canvas->Clear();

  // draw histogram
  histo->Draw();

  // draw all rings
  for ( std::vector<TPaveLabel*>::iterator ringLabel = allRings.begin();
	ringLabel != allRings.end();
	++ringLabel ) {
    (*ringLabel)->Draw("SAME");
  }

  // draw all inner seed rings
  for ( std::vector<TPaveLabel*>::iterator ringLabel = innerSeedRings.begin();
	ringLabel != innerSeedRings.end();
	++ringLabel ) {
    (*ringLabel)->SetFillColor(2);
    (*ringLabel)->SetBorderSize(1);
    (*ringLabel)->Draw("SAME");
  }

  // draw all outer seed rings
  for ( std::vector<TPaveLabel*>::iterator ringLabel = outerSeedRings.begin();
	ringLabel != outerSeedRings.end();
	++ringLabel ) {
    (*ringLabel)->SetFillColor(4);
    (*ringLabel)->SetBorderSize(1);
    (*ringLabel)->Draw("SAME");
  }

  // save into picture file
  canvas->Print(pictureName.c_str());

  // close picture file
  canvas->Print(end_pictureName.c_str());

}
