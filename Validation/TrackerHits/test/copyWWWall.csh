#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

setenv WWWDIR /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/SimHits

if ( ! -d $WWWDIR) mkdir $WWWDIR

mkdir $WWWDIR/eps
mkdir $WWWDIR/eps/TK-strips
mkdir $WWWDIR/eps/TK-strips/eloss
mkdir $WWWDIR/eps/TK-strips/Localx
mkdir $WWWDIR/eps/TK-strips/Localy
mkdir $WWWDIR/eps/TK-strips/Entryx-Exitx
mkdir $WWWDIR/eps/TK-strips/Entryy-Exity
mkdir $WWWDIR/eps/TK-strips/Entryz-Exitz
mkdir $WWWDIR/eps/TK-pixels
mkdir $WWWDIR/eps/TK-pixels/eloss
mkdir $WWWDIR/eps/TK-pixels/Localx
mkdir $WWWDIR/eps/TK-pixels/Localy
mkdir $WWWDIR/eps/TK-pixels/Entryx-Exitx
mkdir $WWWDIR/eps/TK-pixels/Entryy-Exity
mkdir $WWWDIR/eps/TK-pixels/Entryz-Exitz
mkdir $WWWDIR/eps/TK-summary

echo "...Copying..."

mv plots/muon/eloss_T*_KS*.eps $WWWDIR/eps/TK-strips/eloss
mv plots/muon/pos_Entryx-Exitx_T*.eps $WWWDIR/eps/TK-strips/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_T*.eps $WWWDIR/eps/TK-strips/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_T*.eps $WWWDIR/eps/TK-strips/Entryz-Exitz
mv plots/muon/pos_Localy_T*.eps $WWWDIR/eps/TK-strips/Localy
mv plots/muon/pos_Localx_T*.eps $WWWDIR/eps/TK-strips/Localx

mv plots/muon/eloss_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/eloss
mv plots/muon/pos_Entryx-Exitx_*PIX*.eps $WWWDIR/eps/TK-pixels/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_*PIX*.eps $WWWDIR/eps/TK-pixels/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_*PIX*.eps $WWWDIR/eps/TK-pixels/Entryz-Exitz
mv plots/muon/pos_Localy_*PIX*.eps $WWWDIR/eps/TK-pixels/Localy
mv plots/muon/pos_Localx_*PIX*.eps $WWWDIR/eps/TK-pixels/Localx

mv plots/muon/summary*.eps $WWWDIR/eps/TK-summary

echo "...Done..."
