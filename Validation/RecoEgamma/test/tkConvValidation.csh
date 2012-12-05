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
setenv TYPE AllConversions
setenv RUNTYPE Central
setenv STARTUP True


setenv CMSSWver1 6_1_0
setenv CMSSWver2 6_1_0
setenv OLDRELEASE 6_1_0
setenv NEWRELEASE 6_1_0
setenv OLDPRERELEASE pre3
setenv NEWPRERELEASE pre7



if ( $STARTUP == True) then
setenv OLDGLOBALTAG START61_V1-v1
setenv NEWGLOBALTAG START61_V5A-v1
else 
setenv OLDGLOBALTAG START50_V8-v3
setenv NEWGLOBALTAG START50_V8-v3
endif




setenv OLDRELEASE ${OLDRELEASE}_${OLDPRERELEASE}
#setenv OLDRELEASE ${OLDRELEASE}
setenv NEWRELEASE ${NEWRELEASE}_${NEWPRERELEASE}
#setenv NEWRELEASE ${NEWRELEASE}

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test


#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test


#Name of sample (affects output directory name and htmlde


setenv PU False
#setenv SAMPLE SingleGammaPt10
#setenv SAMPLE SingleGammaPt35
#setenv SAMPLE QCD_Pt_80_120
#setenv SAMPLE QCD_Pt_20_30
setenv SAMPLE H130GGgluonfusion

if ( $RUNTYPE == Central ) then
setenv HISTOPATHNAME_Efficiencies DQMData/Run\ 1/EgammaV/Run\ summary/ConversionValidator/EfficienciesAndFakeRate
setenv HISTOPATHNAME_tkConversions DQMData/Run\ 1/EgammaV/Run\ summary/ConversionValidator/ConversionInfo
endif

if ( $RUNTYPE == Local ) then
setenv HISTOPATHNAME_Efficiencies DQMData/EgammaV/ConversionValidator/EfficienciesAndFakeRate
setenv HISTOPATHNAME_tkConversions DQMData/EgammaV/ConversionValidator/ConversionInfo
endif

#==============END BASIC CONFIGURATION==================
#Input root trees for the two cases to be compared 

if ($SAMPLE == SingleGammaPt10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt10.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt10__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif


else if ($SAMPLE == SingleGammaPt35) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt35.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt35.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt35__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif


else if ($SAMPLE == SingleGammaFlatPt10_100) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaFlatPt10To100.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaFlatPt10To100.root
endif 


else if ($SAMPLE == H130GGgluonfusion) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_H130GGgluonfusion.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_H130GGgluonfusion.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif



else if ($SAMPLE == PhotonJets_Pt_10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_PhotonJets_Pt_10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_PhotonJets_Pt_10.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif

else if ($SAMPLE ==  GammaJets_Pt_80_120) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_GammaJets_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_GammaJets_Pt_80_120.root
else if ( $RUNTYPE == Central ) then
endif

else if ($SAMPLE == QCD_Pt_80_120) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_QCD_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_QCD_Pt_80_120.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValQCD_Pt_80_120__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValQCD_Pt_80_120__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif

else if ($SAMPLE == QCD_Pt_20_30STARTUP) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/ConversionValidationRelVal${OLDRELEASE}_QCD_Pt_20_30.root
setenv NEWFILE ${WorkDir2}/ConversionValidationRelVal${NEWRELEASE}_QCD_Pt_20_30.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0003_R000000001__RelValQCD_Pt_20_30__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValQCD_Pt_20_30__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif
 
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

if ( $PU == True) then
setenv OUTDIR $OUTPATH/${SAMPLE}PU
else if ( $PU == False) then
setenv OUTDIR $OUTPATH/${SAMPLE}
endif

if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR
echo "cd $OUTDIR"

#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == AllConversions ) then

cat > efficiency <<EOF
  convEffVsEtaTwoTracks
  convEffVsPhiTwoTracks
  convEffVsRTwoTracks
  convEffVsZTwoTracks
  convEffVsEtTwoTracks
  convFakeRateVsEtaTwoTracks
  convFakeRateVsPhiTwoTracks
  convFakeRateVsRTwoTracks
  convFakeRateVsZTwoTracks
  convFakeRateVsEtTwoTracks
EOF



cat > scaledhistos <<EOF
convEta
convPhi
convZ
convVtxdEta
convVtxdPhi
convVtxdR
convVtxdZ
convVtxdX
convVtxdY
convVtxdR_barrel
convVtxdZ_barrel
convVtxdX_barrel
convVtxdY_barrel
convVtxdR_endcap
convVtxdZ_endcap
convVtxdX_endcap
convVtxdY_endcap
hDCotTracksAll
hDCotTracksBarrel
hDCotTracksEndcap
convPtResAll
convPtResBarrel
convPtResEndcap
hDPhiTracksAtVtxAll
hDPhiTracksAtVtxBarrel
hDPhiTracksAtVtxEndcap
hInvMassAll_AllTracks
hInvMassBarrel_AllTracks
hInvMassEndcap_AllTracks

hTkPtPullAll
hTkPtPullBarrel
hTkPtPullEndcap

nSharedHitsBarrel
nSharedHitsEndcap
nHitsBeforeVtxBarrel
nHitsBeforeVtxEndcap
maxNHitsBeforeVtxBarrel
maxNHitsBeforeVtxEndcap
maxDlClosestHitToVtxBarrel
maxDlClosestHitToVtxEndcap

EOF

cat > logYScaledHistos <<EOF
convRplot
convPt
hTkD0All
hDistMinAppTracksAll
hDistMinAppTracksBarrel
hDistMinAppTracksEndcap
vtxChi2ProbAll
vtxChi2ProbBarrel
vtxChi2ProbEndcap

EOF

cat > projections <<EOF
nHitsVsEtaAllTracks

EOF


cat > profiles <<EOF
pConvVtxdRVsR
pConvVtxdRVsEta
pConvVtxdXVsX
pConvVtxdYVsY
pConvVtxdZVsZ

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






foreach i (`cat scaledhistos`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i","c$i",430, 10, 700, 500);
c$i->SetFillColor(10);
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_old->cd("$HISTOPATHNAME_tkConversions");
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( $i==convVtxdR || $i==convVtxdX  ||$i==convVtxdY  ) {
$i->SetMaximum (2000);
} else if ( $i==convVtxdR_barrel ||  $i==convVtxdX_barrel ||  $i==convVtxdY_barrel ) {
$i->SetMaximum (1200);
} else if ( $i==convVtxdR_endcap  ||  $i==convVtxdX_endcap ||  $i==convVtxdY_endcap ) {
$i->SetMaximum (1000);
} else if ( $i==convVtxdZ_endcap ) {
$i->SetMaximum (500);
} else if ( $i==hDPhiTracksAtVtxAll ) {
$i->SetMaximum (2000);
} else if ( $i==hDPhiTracksAtVtxBarrel ||  $i==hDPhiTracksAtVtxEndcap ) {
$i->SetMaximum (1000);
} else if ( $i==hTkPtPullAll ) {
$i->SetMaximum (2*250);
} else if ( $i==hTkPtPullBarrel ) {
$i->SetMaximum (2*150);
}  else if ( $i==hTkPtPullEndcap ) {
$i->SetMaximum (2*200);
} else if ( $i==nSharedHitsBarrel ) {
$i->SetTitle(" Number of shared hits: barrel");
$i->SetMaximum(3000);
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==nSharedHitsEndcap ) {
$i->SetTitle(" Number of shared hits: endcap");
$i->SetMaximum(3200);
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==nHitsBeforeVtxBarrel ) {
$i->SetTitle(" Number of hits before the vtx: barrel");
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==nHitsBeforeVtxEndcap ) {
$i->SetTitle(" Number of hits before the vtx: endcap");
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==maxNHitsBeforeVtxBarrel ) {
$i->SetTitle(" Max number of hits before the vtx: barrel");
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==maxNHitsBeforeVtxEndcap ) {
$i->SetTitle(" Max number of hits before the vtx: endcap");
$i->GetXaxis()->SetRangeUser(-0.5,10.);
} else if ( $i==maxDlClosestHitToVtxBarrel ) {
$i->SetTitle(" Max Dl closest hit  to vtx: barrel");
} else if ( $i==maxDlClosestHitToVtxEndcap ) {
$i->SetTitle(" Max Dl closest hit  to vtx: endcap");


}

$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
//$i->SetLineWidth(3);
$i->Draw();
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
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




foreach i (`cat logYScaledHistos`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i","c$i",430, 10, 700, 500);
c$i->SetFillColor(10);
c$i->SetLogy(1);
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_old->cd("$HISTOPATHNAME_tkConversions");
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetMinimum(3.);
//$i->GetXaxis()->SetRangeUser(0.,60.);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
if ( $i==convRplot) {
$i->SetMarkerSize(0.08);
}
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat efficiency`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i","c$i",430, 10, 700, 500);
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/ConversionValidator/EfficienciesAndFakeRate");
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
if ( $i==convFakeRateVsRTwoTracks ) {
$i->GetYaxis()->SetRangeUser(0.,1.);
} else {
$i->GetYaxis()->SetRangeUser(0.,0.5);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1");
//file_new->cd("DQMData/EgammaV/ConversionValidator/EfficienciesAndFakeRate");
file_new->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
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


foreach i (`cat projections`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i","c$i",430, 10, 700, 500);
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_old->cd("$HISTOPATHNAME_tkConversions");
TH1D *tmp1$i= $i->ProjectionY();
Double_t nold=tmp1$i->GetEntries();
Double_t mold=tmp1$i->GetMaximum();
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
TH1D *tmp2$i= $i->ProjectionY();
Double_t nnew=tmp2$i->GetEntries();
Double_t mnew=tmp2$i->GetMaximum();
tmp1$i->SetStats(0);
tmp1$i->SetMinimum(0.);
tmp1$i->SetLineColor(kPink+8);
tmp1$i->SetFillColor(kPink+8);
tmp1$i->SetLineWidth(3);
tmp1$i->Draw();
tmp2$i->SetStats(0);
tmp2$i->SetLineColor(kBlack);
tmp2$i->SetMarkerColor(kBlack);
tmp2$i->SetMarkerStyle(20);
tmp2$i->SetMarkerSize(1);
tmp2$i->SetLineWidth(1);
tmp2$i->Scale(nold/nnew);
tmp2$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");


EOF
  setenv N `expr $N + 1`
end


foreach i (`cat profiles`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i","c$i",430, 10, 700, 500);
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_old->cd("$HISTOPATHNAME_tkConversions");
$i->SetStats(0);
if (  $i == pConvVtxdRVsR || $i == pConvVtxdXVsX ||  $i ==  pConvVtxdYVsY ) {
$i->GetYaxis()->SetRangeUser(-2.,2);
} else if ( $i == pConvVtxdZVsZ ) {
$i->GetYaxis()->SetRangeUser(-10.,10);
} else {
$i->GetYaxis()->SetRangeUser(-1.,1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/ConversionValidator/ConversionInfo");
file_new->cd("$HISTOPATHNAME_tkConversions");
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
else if ( $TYPE == AllConversions ) then
  setenv ANALYZER ConversionValidator
  setenv CFG ConversionValidator_cfg
endif

if (-e validation.html) rm validation.html
if (-e tkConvValidationPlotsTemplate.html) rm tkConvValidationPlotsTemplate.html
cp ${CURRENTDIR}/tkConvValidationPlotsTemplate.html tkConvValidationPlotsTemplate.html
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
cat  tkConvValidationPlotsTemplate.html >>& validation.html
rm   tkConvValidationPlotsTemplate.html 

rm efficiency
rm scaledhistos
rm profiles
rm logYScaledHistos
rm projections

root -b -l -q validation.C
cd $CURRENTDIR

python makeWebpage.py ${OUTDIR}/gifs  ${OLDRELEASE}  ${NEWRELEASE} ${SAMPLE}
echo "Then you can view your validation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/$OUTPATH/validation.html"
