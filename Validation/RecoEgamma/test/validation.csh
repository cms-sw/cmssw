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

setenv OLDFILE /data/test/CMSSW_2_1_X_2008-07-02-0000/src/Validation/RecoEgamma/test/PhotonValidationRelVal210_pre6_4.0T.root
setenv NEWFILE /data/test/CMSSW_2_1_X_2008-07-02-0000/src/Validation/RecoEgamma/test/PhotonValidationRelVal210_pre6_3.8T.root


setenv OLDRELEASE 210pre6IDEAL_TestValidation_40T
setenv NEWRELEASE 210pre6IDEAL_TestValidation_38T
#Name of sample (affects output directory name and htmldescription only) 
setenv SAMPLE SingleGammaPt35
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
file_old->cd("DQMData/Egamma/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
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
$i->SetMinimum(0.8);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/Egamma/PhotonValidator/Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(2);
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
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
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
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Draw("same");
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
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(2);
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
file_old->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
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
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonValidator/ConversionInfo");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
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
  setenv ANALYZER SimplePhotonAnalyzer
  setenv CFG SimplePhotonAnalyzer
else if ( $TYPE == Conversion ) then
  setenv ANALYZER SimpleConvertedPhotonAnalyzer
  setenv CFG SimpleConvertedPhotonAnalyzer
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


<p>In all plots below, $OLDRELEASE is in blue, $NEWRELEASE in red.

EOF
cat begin.html >>& validation.html
rm begin.html

setenv N 1
foreach i (`cat scaledhistosForPhotons  efficiencyForPhotons scaledhistosForConvertedPhotons unscaledhistosForConvertedPhotons  efficiencyForConvertedPhotons scaledhistosForTracks unscaledhistosForTracks  `)
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
