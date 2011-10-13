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
setenv RUNTYPE Central
setenv STARTUP True
setenv CMSSWver1 4_2_0
setenv CMSSWver2 4_2_0
setenv OLDRELEASE 4_2_0
setenv NEWRELEASE 4_2_0
setenv OLDPRERELEASE pre4
setenv NEWPRERELEASE pre5



if ( $STARTUP == True) then
setenv OLDGLOBALTAG START42_V1-v1
setenv NEWGLOBALTAG START42_V3-v1
else
setenv OLDGLOBALTAG MC_42_V1-v1
setenv NEWGLOBALTAG MC_42_V3-v1
endif




setenv OLDRELEASE ${OLDRELEASE}_${OLDPRERELEASE}
#setenv OLDRELEASE ${OLDRELEASE}
setenv NEWRELEASE ${NEWRELEASE}_${NEWPRERELEASE}



#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test


setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test


#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test


#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

#Name of sample (affects output directory name and htmldescription only) 
setenv SAMPLE QCD_Pt_80_120STARTUP
#setenv SAMPLE QCD_Pt_20_30STARTUP


if ( $RUNTYPE == Central ) then
setenv HISTOPATHNAME_Efficiencies DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/Efficiencies
setenv HISTOPATHNAME_Photons DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/Photons
setenv HISTOPATHNAME_Conversions DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/ConversionInfo
setenv HISTOPATHNAME_Background DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/Background
endif

if ( $RUNTYPE == Local ) then
setenv HISTOPATHNAME_Efficiencies DQMData/EgammaV/PhotonValidator/Efficiencies
setenv HISTOPATHNAME_Photons DQMData/EgammaV/PhotonValidator/Photons
setenv HISTOPATHNAME_Conversions DQMData/EgammaV/PhotonValidator/ConversionInfo
setenv HISTOPATHNAME_Background DQMData/EgammaV/PhotonValidator/Background
endif
#==============END BASIC CONFIGURATION==================


#Input root trees for the two cases to be compared 


if ($SAMPLE == QCD_Pt_20_30STARTUP) then 

setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_QCD_Pt_20_30.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_QCD_Pt_20_30.root

else if ($SAMPLE == QCD_Pt_80_120STARTUP) then 
if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_QCD_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_QCD_Pt_80_120.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValQCD_Pt_80_120__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__GEN-SIM-RECO.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValQCD_Pt_80_120__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__GEN-SIM-RECO.root
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

cat > efficiencyForBkg <<EOF
  bkgEffVsEta
  bkgEffVsPhi
  bkgEffVsEt
  deadChVsEtaBkg
  deadChVsPhiBkg
  deadChVsEtBkg
EOF



cat > scaledhistosForBkg <<EOF

  nOfPhotons
  scBkgEta
  scBkgPhi
  scBkgEAll
  scBkgEtAll
  phoBkgEta
  phoBkgPhi
  phoBkgEAll
  phoBkgEtAll
  phoBkgDEta
  phoBkgDPhi
  isoTrkSolidConeDR04BkgBarrel
  isoTrkSolidConeDR04BkgEndcap
  nTrkSolidConeDR04BkgBarrel
  nTrkSolidConeDR04BkgEndcap
  
  convEtaBkg
  convPhiBkg
  PoverEtracksBkgAll
  PoverEtracksBkgBarrel 
  PoverEtracksBkgEndcap
  mvaOutBkgAll
  mvaOutBkgBarrel
  mvaOutBkgEndcap

  hDPhiTracksAtVtxBkgAll
  hDCotTracksBkgAll

EOF


cat > scaledhistosForBkgLogScale <<EOF
  hOverEBkgAll
  hOverEBkgBarrel
  hOverEBkgEndcap
  hcalTowerSumEtConeDR04BkgBarrel
  hcalTowerSumEtConeDR04BkgEndcap
  EoverPtracksBkgAll
  EoverPtracksBkgBarrel 
  EoverPtracksBkgEndcap
  r9BkgBarrel
  r9BkgEndcap
  r1BkgBarrel
  r1BkgEndcap
  r2BkgBarrel
  r2BkgEndcap
  sigmaIetaIetaBkgBarrel
  sigmaIetaIetaBkgEndcap
  ecalRecHitSumEtConeDR04BkgBarrel
  ecalRecHitSumEtConeDR04BkgEndcap


EOF



cat > unscaledhistosForBkg <<EOF

  pR1VsEtaBkgAll
  pR2VsEtaBkgAll
  pR1VsEtBkgAll
  pR2VsEtBkgAll
  pSigmaIetaIetaVsEtaBkgAll
  pSigmaIetaIetaVsEtBkgAll
  pHOverEVsEtaBkgAll
  pHOverEVsEtBkgAll
  pEcalRecHitSumEtConeDR04VsEtBkgBarrel
  pEcalRecHitSumEtConeDR04VsEtBkgEndcap
  pEcalRecHitSumEtConeDR04VsEtaBkgAll
  pHcalTowerSumEtConeDR04VsEtBkgBarrel
  pHcalTowerSumEtConeDR04VsEtBkgEndcap
  pHcalTowerSumEtConeDR04VsEtaBkgAll
  pIsoTrkSolidConeDR04VsEtBkgBarrel
  pIsoTrkSolidConeDR04VsEtBkgEndcap
  pIsoTrkSolidConeDR04VsEtaBkgAll
  pnTrkSolidConeDR04VsEtBkgBarrel
  pnTrkSolidConeDR04VsEtBkgEndcap
  p_nTrkSolidConeDR04VsEtaBkgAll
 


EOF

cat > 2DhistosForBkg <<EOF
  R9VsEtaBkgAll
  R9VsEtBkgAll
  hOverEVsEtaBkgAll
  hOverEVsEtBkgAll


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


foreach i (`cat efficiencyForBkg`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
if ( $i==deadChVsEtaBkg ||  $i==deadChVsPhiBkg ||  $i==deadChVsEtBkg ) {
$i->GetYaxis()->SetRangeUser(0.,0.2);

}else if ( $i==bkgEffVsEta ||  $i==bkgEffVsPhi ) {
$i->GetYaxis()->SetRangeUser(0.,0.4);
}else if (  $i==bkgEffVsEt ) {
$i->GetYaxis()->SetRangeUser(0.,1.);
} else {
$i->GetYaxis()->SetRangeUser(0.,1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1");

//file_new->cd("DQMData/EgammaV/PhotonValidator/Efficiencies");
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





foreach i (`cat scaledhistosForBkg`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/PhotonValidator/Background");
file_old->cd("$HISTOPATHNAME_Background");
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
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
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



foreach i (`cat unscaledhistosForBkg`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/PhotonValidator/Background");
file_old->cd("$HISTOPATHNAME_Background");
$i->SetStats(0);
if ( $i==pEcalRecHitSumEtConeDR04VsEtaBkgAll ||  $i==pHcalTowerSumEtConeDR04VsEtaBkgAll  ) {  
$i->GetYaxis()->SetRangeUser(0.,25.);
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtBkgBarrel || $i==pHcalTowerSumEtConeDR04VsEtBkgBarrel ) 
{ $i->GetYaxis()->SetRangeUser(0.,30.);
} else if (  $i==p_nTrkSolidConeDR04VsEtaBkgAll || $i==pnTrkSolidConeDR04VsEtBkgBarrel ||   $i==pnTrkSolidConeDR04VsEtBkgEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.);
} else if (   $i==pIsoTrkSolidConeDR04VsEtaBkgAll ||  $i==pIsoTrkSolidConeDR04VsEtBkgBarrel || $i==pIsoTrkSolidConeDR04VsEtBkgEndcap)
{ $i->GetYaxis()->SetRangeUser(0.,100.);
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtBkgEndcap || $i==pHcalTowerSumEtConeDR04VsEtBkgEndcap ) 
{$i->GetYaxis()->SetRangeUser(0.,30.);
} else if ( $i==pSigmaIetaIetaVsEtaBkgAll || $i==pSigmaIetaIetaVsEtBkgAll ||  $i==pHOverEVsEtaBkgAll ||  $i==pHOverEVsEtBkgAll ) 
{ $i->GetYaxis()->SetRangeUser(0.,0.1);
} else  {
$i->SetMinimum(0.);
$i->SetMaximum(1.);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
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



foreach i (`cat scaledhistosForBkgLogScale`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->SetLogy(1);
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/PhotonValidator/Background");
file_old->cd("$HISTOPATHNAME_Background");
if ( $i==hcalTowerSumEtConeDR04BkgBarrel ||  $i==hcalTowerSumEtConeDR04BkgEndcap  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
} else if ( $i==hOverEBkgBarrel || $i==hOverEBkgEndcap ) {
$i->GetXaxis()->SetRangeUser(0.,1.);
}
 
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
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





foreach i (`cat 2DhistosForBkg`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/PhotonValidator/Background");
file_old->cd("$HISTOPATHNAME_Background");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
$i->Draw();
//file_new->cd("DQMData/EgammaV/PhotonValidator/Background");
file_new->cd("$HISTOPATHNAME_Background");
$i->SetStats(0);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
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
else if ( $TYPE == Photons ) then
  setenv ANALYZER PhotonValidator
  setenv CFG PhotonValidator_cfg
endif

if (-e validation.html) rm validation.html
if (-e bkgValidationPlotsTemplate.html) rm bkgValidationPlotsTemplate.html
cp ${CURRENTDIR}/bkgValidationPlotsTemplate.html bkgValidationPlotsTemplate.html
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
cat  bkgValidationPlotsTemplate.html >>& validation.html
rm   bkgValidationPlotsTemplate.html 

rm efficiencyForBkg
rm scaledhistosForBkg
rm scaledhistosForBkgLogScale
rm unscaledhistosForBkg
rm 2DhistosForBkg

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
