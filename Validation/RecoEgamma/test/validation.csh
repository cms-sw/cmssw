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

#Input root trees for the two cases to be compared 
#setenv OLDFILE /data/test/CMSSW_2_2_1/src/Validation/RecoEgamma/test/PhotonValidationRelVal221_SingleGammaPt35.root
#setenv OLDFILE /data/test/CMSSW_3_0_0_pre6/src/Validation/RecoEgamma/test/PhotonValidationRelVal300pre6_SingleGammaPt35.root
#setenv NEWFILE /data/test/CMSSW_3_0_0_pre7/src/Validation/RecoEgamma/test/PhotonValidationRelVal300pre7_SingleGammaPt35.root

#setenv OLDFILE /data/test/CMSSW_3_1_0_pre2/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre2_SingleGammaPt10.root
#setenv NEWFILE /data/test/CMSSW_3_1_0_pre3/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre3_SingleGammaPt10.root

setenv OLDFILE /data/test/CMSSW_3_1_0_pre2/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre2_SingleGammaPt35.root
setenv NEWFILE /data/test/CMSSW_3_1_0_pre3/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre3_SingleGammaPt35.root

#setenv OLDFILE /data/test/CMSSW_3_1_0_pre2/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre2_H130GGgluonfusion.root
#setenv NEWFILE /data/test/CMSSW_3_1_0_pre3/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre3_H130GGgluonfusion.root

#setenv OLDFILE /data/test/CMSSW_3_1_0_pre2/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre2_GammaJets_Pt_80_120.root
#setenv NEWFILE /data/test/CMSSW_3_1_0_pre3/src/Validation/RecoEgamma/test/PhotonValidationRelVal310pre3_GammaJets_Pt_80_120.root



#setenv OLDRELEASE 221IDEAL
setenv OLDRELEASE 310pre2IDEAL
setenv NEWRELEASE 310pre3IDEAL
#Name of sample (affects output directory name and htmldescription only) 
setenv SAMPLE SingleGammaPt35
#setenv SAMPLE SingleGammaFlatPt10_100
#setenv SAMPLE H130GGgluonfusionSTARTUP
#setenv SAMPLE GammaJets_Pt_80_120STARTUP
#TYPE must be one ofPixelMatchGsfElectron, Photon 
setenv TYPE Photon


#==============END BASIC CONFIGURATION==================

#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation
cd $OUTPATH
if (! -d $NEWRELEASE) then
  mkdir $NEWRELEASE
endif
setenv OUTPATH $OUTPATH/$NEWRELEASE

setenv OUTDIR $OUTPATH/${SAMPLE}_${NEWRELEASE}_${OLDRELEASE}
if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR

#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == PixelMatchGsfElectron ) then

cat > scaledhistos <<EOF
  h_ele_PoPtrue   
  h_ele_EtaMnEtaTrue   
  h_ele_PhiMnPhiTrue 
  h_ele_vertexP 
  h_ele_vertexPt 
  h_ele_outerP_mode 
  h_ele_outerPt_mode 
  h_ele_vertexZ 
  h_ele_EoP 
  h_ele_EoPout 
  h_ele_dEtaCl_propOut 
  h_ele_dEtaSc_propVtx 
  h_ele_dPhiCl_propOut 
  h_ele_dPhiSc_propVtx 
  h_ele_HoE 
  h_ele_chi2 
  h_ele_foundHits 
  h_ele_lostHits 
  h_ele_PinMnPout_mode 
  h_ele_classes 
EOF

cat > unscaledhistos <<EOF
  h_ele_absetaEff
  h_ele_etaEff
  h_ele_ptEff
  h_ele_eta_bbremFrac 
  h_ele_eta_goldenFrac 
  h_ele_eta_narrowFrac 
  h_ele_eta_showerFrac 
EOF

else if ( $TYPE == Photon ) then

cat > scaledhistosForPhotons <<EOF
  scEta
  scPhi
  scEAll
  scEtAll
  phoDEta
  phoDPhi
  r9All
  r9Barrel
  r9Endcap
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

EOF


cat > scaledhistosForConvertedPhotons <<EOF

  convEta
  convPhi
  convEResAll
  convEResBarrel
  convEResEndcap
  EoverPtracksAll
  EoverPtracksBarrel 
  EoverPtracksEndcap
  PoverEtracksAll
  PoverEtracksBarrel 
  PoverEtracksEndcap


EOF




cat > unscaledhistosForConvertedPhotons <<EOF
pEoverEtrueVsEtaAll
pEoverPVsEtaAll

EOF

cat > efficiencyForPhotons <<EOF
  recoEffVsEta
  recoEffVsPhi

EOF

cat > efficiencyForConvertedPhotons <<EOF

  convEffVsEtaTwoTracks
  convEffVsPhiTwoTracks
  convEffVsRTwoTracks
  convEffVsZTwoTracks


EOF

cat > fakeRateForConvertedPhotons <<EOF

  convFakeRateVsEtaTwoTracks
  convFakeRateVsPhiTwoTracks
  convFakeRateVsRTwoTracks
  convFakeRateVsZTwoTracks


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

EOF

cat > unscaledhistosForTracks <<EOF
h_nHitsVsEtaAllTracks
h_nHitsVsRAllTracks


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



foreach i (`cat scaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/Egamma/PhotonValidator/Photons");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/Egamma/PhotonValidator/Photons");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kRed-8);
$i->SetFillColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->SetMinimum(0.);
$i->Scale(nold/nnew);
$i->Draw("esame");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat efficiencyForPhotons`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/Egamma/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat scaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kRed-8);
$i->SetFillColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("esame");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat unscaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.6);
$i->SetLineColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Draw("esame");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat efficiencyForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat fakeRateForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat scaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t mnew=$i->GetMaximum();
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kRed-8);
$i->SetFillColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("esame");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat unscaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(15.);
$i->SetLineColor(kRed-8);
$i->SetFillColor(kRed-8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("esame");
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
else if ( $TYPE == Photon ) then
  setenv ANALYZER PhotonValidator
  setenv CFG PhotonValidator_cfg
endif

if (-e validation.html) rm validation.html
touch validation.html
cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$NEWRELEASE vs $OLDRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE vs $OLDRELEASE $TYPE validation</h1>

<p>The following plots were made using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/src/$ANALYZER.cc">Validation/RecoEgamma/src/$ANALYZER</a>, 
using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/test/$CFG.py">Validation/RecoEgamma/test/$CFG.py</a>, using $SAMPLE as input.
<p>The script used to make the plots is <a href="validation.C">here</a>.

<p>In all plots below, $OLDRELEASE is in brown , $NEWRELEASE in black.


EOF
cat begin.html >>& validation.html
rm begin.html

setenv N 1
foreach i (`cat scaledhistosForPhotons  efficiencyForPhotons scaledhistosForConvertedPhotons unscaledhistosForConvertedPhotons  efficiencyForConvertedPhotons  fakeRateForConvertedPhotons scaledhistosForTracks unscaledhistosForTracks  `)
  cat > temp$N.html <<EOF
<br>
<p><img class="image" width="500" src="gifs/$i.gif">
EOF
  setenv N `expr $N + 1`
end

setenv N 1
while ( $N <= $NTOT )
  cat temp$N.html >>& validation.html
  rm temp$N.html
  setenv N `expr $N + 1`
end

cat > end.html <<EOF

</html>
EOF
cat end.html >>& validation.html
rm end.html

rm scaledhistosForPhotons
rm scaledhistosForConvertedPhotons
rm unscaledhistosForConvertedPhotons
rm efficiencyForPhotons
rm efficiencyForConvertedPhotons
rm fakeRateForConvertedPhotons
rm scaledhistosForTracks
rm unscaledhistosForTracks


echo "Now paste the following into your terminal window:"
echo ""
echo "cd $OUTDIR"
echo " root -b"
echo ".x validation.C"
echo ".q"
echo "cd $CURRENTDIR"
echo ""
echo "Then you can view your valdation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/validation/${NEWRELEASE}/${SAMPLE}_${NEWRELEASE}_${OLDRELEASE}/validation.html"
