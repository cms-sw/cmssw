#!/bin/csh

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  
#  
#  "Validation/RecoEgamma/test/PhotonValidator_cfg.py
#
# The default list of histograms (configurable) is based on version VXX-XX-XX
# of Validation/RecoEgamma
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================
setenv TYPE Photons
setenv CMSSWver1 3_8_0
setenv CMSSWver2 3_8_0
setenv OLDRELEASE 380
setenv NEWRELEASE 380
setenv OLDPRERELEASE pre7 
setenv NEWPRERELEASE pre8

setenv OLDRELEASE ${OLDRELEASE}${OLDPRERELEASE}
setenv NEWRELEASE ${NEWRELEASE}${NEWPRERELEASE}



#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test


setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test



#Name of sample (affects output directory name and htmldescription only) 

#setenv SAMPLE PhotonJetPt15
#setenv SAMPLE PhotonJetPt0-15
#setenv SAMPLE PhotonJetPt500toInf
#setenv SAMPLE PhotonJetPt80
#setenv SAMPLE PhotonJetPt470

setenv SAMPLE SingleGammaPt10IDEAL
#setenv SAMPLE SingleGammaPt35IDEAL
#setenv SAMPLE SingleGammaFlatPt10_100
#setenv SAMPLE H130GGgluonfusionSTARTUP
#setenv SAMPLE PhotonJets_Pt_10
#setenv SAMPLE GammaJets_Pt_80_120STARTUP
#setenv SAMPLE QCD_Pt_80_120STARTUP
#TYPE must be one ofPixelMatchGsfElectron, Photon 

#==============END BASIC CONFIGURATION==================


#Input root trees for the two cases to be compared 



#setenv OLDFILE /afs/cern.ch/user/n/nancy/scratch0/PreProductionValidation/CMSSW_3_1_1/src/Validation/RecoEgamma/test/results/${SAMPLE}_total.root
#setenv NEWFILE /afs/cern.ch/user/n/nancy/scratch0/PreProductionValidation/CMSSW_3_1_1/src/Validation/RecoEgamma/test/results/${SAMPLE}_total.root


if ($SAMPLE == SingleGammaPt10IDEAL) then

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt10.root

else if ($SAMPLE == SingleGammaPt35IDEAL) then 

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt35.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt35.root

else if ($SAMPLE == H130GGgluonfusionSTARTUP) then 

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_H130GGgluonfusion.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_H130GGgluonfusion.root


else if ($SAMPLE == PhotonJets_Pt_10) then

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_PhotonJets_Pt_10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_PhotonJets_Pt_10.root


else if ($SAMPLE ==  GammaJets_Pt_80_120STARTUP) then 

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_GammaJets_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_GammaJets_Pt_80_120.root


else if ($SAMPLE == QCD_Pt_80_120STARTUP) then 

setenv OLDFILE /data/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test/PhotonValidationRelVal${NEWRELEASE}_QCD_Pt_80_120.root
#setenv NEWFILE /data/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test/PhotonValidationRelVal${NEWRELEASE}_QCD_Pt_80_120.root


endif



#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation
cd $OUTPATH
if (! -d $NEWRELEASE) then
  mkdir $NEWRELEASE
endif
setenv OUTPATH $OUTPATH/$NEWRELEASE
cd $OUTPATH

if (! -d ${TYPE}) then
  mkdir ${TYPE}
endif
setenv OUTPATH $OUTPATH/${TYPE}
cd  $OUTPATH

if (! -d vs${OLDRELEASE}) then
  mkdir vs${OLDRELEASE}
endif
setenv OUTPATH $OUTPATH/vs${OLDRELEASE}

setenv OUTDIR $OUTPATH/${SAMPLE}
if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR


#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == Photons ) then


cat > efficiencyForPhotons <<EOF
  recoEffVsEta
  recoEffVsPhi
  recoEffVsEt
  deadChVsEta
  deadChVsPhi
  deadChVsEt
EOF

cat > scaledhistosForPhotons <<EOF

  gamgamMassAll
  gamgamMassBarrel
  gamgamMassEndcap
  gamgamMassNoConvAll
  gamgamMassNoConvBarrel
  gamgamMassNoConvEndcap
  gamgamMassConvAll
  gamgamMassConvBarrel
  gamgamMassConvEndcap
  scEta
  scPhi
  scEAll
  scEtAll
  phoEta
  phoPhi
  phoDEta
  phoDPhi
  phoEAll
  phoEtAll
  eResAll
  eResBarrel
  eResEndcap
  eResunconvAll
  eResunconvBarrel
  eResunconvEndcap
  eResconvAll
  eResconvBarrel
  eResconvEndcap
  r9All
  r9Barrel
  r9Endcap
  r1All
  r1Barrel
  r1Endcap
  r2All
  r2Barrel
  r2Endcap
  sigmaIetaIetaAll
  sigmaIetaIetaBarrel
  sigmaIetaIetaEndcap
  ecalRecHitSumEtConeDR04Barrel
  ecalRecHitSumEtConeDR04Endcap
  isoTrkSolidConeDR04All
  isoTrkSolidConeDR04Barrel
  isoTrkSolidConeDR04Endcap
  nTrkSolidConeDR04All
  nTrkSolidConeDR04Barrel
  nTrkSolidConeDR04Endcap



EOF

cat > scaledhistosForPhotonsLogScale <<EOF
  hOverEAll
  hOverEBarrel
  hOverEEndcap
  hcalTowerSumEtConeDR04Barrel
  hcalTowerSumEtConeDR04Endcap



EOF


cat > unscaledhistosForPhotons <<EOF
pEResVsR9All
pEResVsR9Barrel
pEResVsR9Endcap
scpEResVsR9All
scpEResVsR9Barrel
scpEResVsR9Endcap
pEResVsEtAll
pEResVsEtBarrel
pEResVsEtEndcap
pEcalRecHitSumEtConeDR04VsEtaAll
pEcalRecHitSumEtConeDR04VsEtBarrel
pEcalRecHitSumEtConeDR04VsEtEndcap
pHcalTowerSumEtConeDR04VsEtaAll
pHcalTowerSumEtConeDR04VsEtBarrel
pHcalTowerSumEtConeDR04VsEtEndcap

EOF



cat > 2dhistosForPhotons <<EOF
  R9VsEtaAll
  R1VsEtaAll
  R2VsEtaAll
  sigmaIetaIetaVsEtaAll
  hOverEVsEtaAll
  ecalRecHitSumEtConeDR04VsEtaAll
  hcalTowerSumEtConeDR04VsEtaAll
  isoTrkSolidConeDR04VsEtaAll
  nTrkSolidConeDR04VsEtaAll
  R9VsEtAll
  R1VsEtAll
  R2VsEtAll
  sigmaIetaIetaVsEtAll
  hOverEVsEtAll
  ecalRecHitSumEtConeDR04VsEtBarrel
  ecalRecHitSumEtConeDR04VsEtEndcap
  hcalTowerSumEtConeDR04VsEtBarrel
  hcalTowerSumEtConeDR04VsEtEndcap
  isoTrkSolidConeDR04VsEtAll
  nTrkSolidConeDR04VsEtAll
  eResVsR9All
  eResVsR9Barrel
  eResVsR9Endcap
  sceResVsR9All
  sceResVsR9Barrel
  sceResVsR9Endcap

EOF



cat > efficiencyForConvertedPhotons <<EOF

  convEffVsEtaTwoTracks
  convEffVsPhiTwoTracks
  convEffVsRTwoTracks
  convEffVsZTwoTracks
  convEffVsEtTwoTracks
  convEffVsEtaTwoTracksAndVtxProbGT0

EOF



cat > scaledhistosForConvertedPhotons <<EOF

  convEta2
  convPhi
  convEResAll
  convEResBarrel
  convEResEndcap
  PoverEtracksAll
  PoverEtracksBarrel 
  PoverEtracksEndcap
  mvaOutAll
  mvaOutBarrel
  mvaOutEndcap

EOF


cat > scaledhistosForConvertedPhotonsLogScale <<EOF
  EoverPtracksAll
  EoverPtracksBarrel 
  EoverPtracksEndcap


EOF



cat > unscaledhistosForConvertedPhotons <<EOF
pEoverEtrueVsEtaAll
pEoverPVsEtaAll
pEoverPVsRAll

EOF


cat > 2dhistosForConvertedPhotons <<EOF
  convVtxRvsZAll
EOF

cat > projectionsForConvertedPhotons <<EOF
   convVtxRvsZBarrel
   convVtxRvsZEndcap
EOF



cat > fakeRateForConvertedPhotons <<EOF

  convFakeRateVsEtaTwoTracks
  convFakeRateVsPhiTwoTracks
  convFakeRateVsRTwoTracks
  convFakeRateVsZTwoTracks
  convFakeRateVsEtTwoTracks

EOF



cat > scaledhistosForTracks <<EOF

tkChi2AllTracks
hTkPtPullAll
hTkPtPullBarrel
hTkPtPullEndcap
hDPhiTracksAtVtxAll
hDCotTracksAll
hDEtaTracksAtEcalAll
hDPhiTracksAtEcalAll
eBcOverTkPoutAll
eBcOverTkPoutBarrel
eBcOverTkPoutEndcap
zPVFromTracks
dzPVFromTracks
vtxChi2ProbAll
vtxChi2ProbBarrel
vtxChi2ProbEndcap

EOF

cat > unscaledhistosForTracks <<EOF
h_nHitsVsEtaAllTracks
h_nHitsVsRAllTracks
pChi2VsEtaAll
pChi2VsRAll
pDCotTracksVsEtaAll
pDCotTracksVsRAll
pDPhiTracksAtEcalVsEtaAll
pDPhiTracksAtEcalVsRAll
pConvVtxdRVsR
pConvVtxdRVsEta
pdzPVVsR


EOF



endif

#=================END CONFIGURATION=====================

if (-e validation.C) rm validation.C
touch validation.C
cat > begin.C <<EOF
{
TFile *file_old = TFile::Open("$OLDFILE");
TFile *file_new = TFile::Open("$NEWFILE");

EOF
cat begin.C >>& validation.C
rm begin.C

setenv N 1



foreach i (`cat efficiencyForPhotons`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
if ( $i==deadChVsEta ||  $i==deadChVsPhi ||  $i==deadChVsEt ) {
$i->SetMinimum(0.);
$i->SetMaximum(0.2);
} else if (  $i==recoEffVsEt ) {
$i->GetXaxis()->SetRangeUser(0.,200.);
} else {
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();

file_new->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
file_old->cd("DQMData/EgammaV/PhotonValidator/Photons");
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetMinimum(0.);
//if ( mnew > mold) 
// $i->SetMaximum(mnew+mnew*0.2);
//else 
//$i->SetMaximum(mold+mold*0.2);
//$i->SetMaximum(mold+mold*0.2);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
//$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
//$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosForPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->SetLogy(1);
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
Double_t nnew=$i->GetEntries();
file_old->cd("DQMData/EgammaV/PhotonValidator/Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel ||  $i==hcalTowerSumEtConeDR04Endcap  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat unscaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/Photons");
$i->SetStats(0);
if ( $i==pEcalRecHitSumEtConeDR04VsEtaAll   ) {  
$i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtBarrel ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.); 
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtaAll   ) 
{ $i->GetYaxis()->SetRangeUser(0.,0.3);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtBarrel   ) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtEndcap  ) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else  {
$i->SetMinimum(0.8);
$i->SetMaximum(1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat 2dhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat efficiencyForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosForConvertedPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->SetLogy(1);
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat unscaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.6);
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat fakeRateForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat 2dhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMarkerColor(kPink+8);
$i->Draw();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMarkerColor(kBlack);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat projectionsForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
if ($i==convVtxRvsZBarrel)
TH1D *tmp1$i= $i->ProjectionY();
else if ($i==convVtxRvsZEndcap)
TH1D *tmp1$i= $i->ProjectionX();
Double_t nold=tmp1$i->GetEntries();
Double_t mold=tmp1$i->GetMaximum();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
//TH1D *tmp2$i= $i->ProjectionY();
if ($i==convVtxRvsZBarrel)
TH1D *tmp2$i= $i->ProjectionY();
else if ($i==convVtxRvsZEndcap)
TH1D *tmp2$i= $i->ProjectionX();
Double_t nnew=tmp2$i->GetEntries();
Double_t mnew=tmp2$i->GetMaximum();
tmp1$i->SetStats(0);
tmp1$i->SetMinimum(0.);
if ( mnew > mold) 
tmp1$i->SetMaximum(mnew+mnew*0.2);
else 
tmp1$i->SetMaximum(mold+mold*0.2);
tmp1$i->SetLineColor(kPink+8);
tmp1$i->SetFillColor(kPink+8);
tmp1$i->SetLineWidth(3);
tmp1$i->Draw();
tmp2$i->SetStats(0);
tmp2$i->SetLineColor(kBlack);
tmp2$i->SetLineWidth(3);
tmp2$i->Scale(nold/nnew);
tmp2$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat unscaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
if ($i==pDCotTracksVsEtaAll ||  $i==pDCotTracksVsRAll ) {
$i->SetMinimum(-0.05);
$i->SetMaximum(0.05);
} else {
$i->SetMinimum(0.);
    if ( mnew > mold) 
    $i->SetMaximum(mnew+mnew*0.4);
    else 
    $i->SetMaximum(mold+mold*0.4);
}

$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/EgammaV/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
//$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





setenv NTOT `expr $N - 1`
setenv N 1
while ( $N <= $NTOT )
  cat temp$N.C >>& validation.C
  rm temp$N.C
  setenv N `expr $N + 1`
end

cat > end.C <<EOF
}
EOF
cat end.C >>& validation.C
rm end.C


if ( $TYPE == PixelMatchGsfElectron ) then
  setenv ANALYZER PixelMatchGsfElectronAnalyzer
  setenv CFG read_gsfElectrons
else if ( $TYPE == Photons ) then
  setenv ANALYZER PhotonValidator
  setenv CFG PhotonValidator_cfg
endif

if (-e validation.html) rm validation.html
if (-e validationPlotsTemplate.html) rm validationPlotsTemplate.html
cp ${CURRENTDIR}/validationPlotsTemplate.html validationPlotsTemplate.html
touch validation.html
cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$NEWRELEASE vs $OLDRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE vs $OLDRELEASE $TYPE validation
<br>
 $SAMPLE 
</h1>

<p>The following plots were made using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/src/$ANALYZER.cc">Validation/RecoEgamma/src/$ANALYZER</a>, 
using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/test/$CFG.py">Validation/RecoEgamma/test/$CFG.py</a>
<p>The script used to make the plots is <a href="validation.C">here</a>.
<br>
In all plots below, $OLDRELEASE is in purple , $NEWRELEASE in black. 
<br>
Click on the plots to see them enlarged.
<br>
Responsible: N. Marinelli
<br>
<br>



EOF
cat begin.html >>& validation.html
rm begin.html
cat  validationPlotsTemplate.html >>& validation.html
rm  validationPlotsTemplate.html 

rm scaledhistosForPhotons
rm unscaledhistosForPhotons
rm 2dhistosForPhotons
rm scaledhistosForConvertedPhotons
rm scaledhistosForConvertedPhotonsLogScale
rm unscaledhistosForConvertedPhotons
rm efficiencyForPhotons
rm efficiencyForConvertedPhotons
rm fakeRateForConvertedPhotons
rm 2dhistosForConvertedPhotons
rm projectionsForConvertedPhotons
rm scaledhistosForTracks
rm unscaledhistosForTracks
rm scaledhistosForPhotonsLogScale

#echo "Now paste the following into your terminal window:"
#echo ""
echo "cd $OUTDIR"
#echo " root -b"
#echo ".x validation.C"
#echo ".q"
#echo "cd $CURRENTDIR"
#echo ""


root -b -l -q validation.C
cd $CURRENTDIR
echo "Then you can view your valdation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/$OUTPATH/validation.html"
