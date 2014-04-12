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
mkdir $WWWDIR/eps/ToF

mkdir $WWWDIR/gif
mkdir $WWWDIR/gif/TK-strips
mkdir $WWWDIR/gif/TK-strips/eloss
mkdir $WWWDIR/gif/TK-strips/Localx
mkdir $WWWDIR/gif/TK-strips/Localy
mkdir $WWWDIR/gif/TK-strips/Entryx-Exitx
mkdir $WWWDIR/gif/TK-strips/Entryy-Exity
mkdir $WWWDIR/gif/TK-strips/Entryz-Exitz
mkdir $WWWDIR/gif/TK-pixels
mkdir $WWWDIR/gif/TK-pixels/eloss
mkdir $WWWDIR/gif/TK-pixels/Localx
mkdir $WWWDIR/gif/TK-pixels/Localy
mkdir $WWWDIR/gif/TK-pixels/Entryx-Exitx
mkdir $WWWDIR/gif/TK-pixels/Entryy-Exity
mkdir $WWWDIR/gif/TK-pixels/Entryz-Exitz
mkdir $WWWDIR/gif/TK-summary
mkdir $WWWDIR/gif/ToF

echo "...Copying..."

mv plots/muon/eloss_T*_KS*.eps.gz $WWWDIR/eps/TK-strips/eloss
mv plots/muon/pos_Entryx-Exitx_T*.eps.gz $WWWDIR/eps/TK-strips/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_T*.eps.gz $WWWDIR/eps/TK-strips/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_T*.eps.gz $WWWDIR/eps/TK-strips/Entryz-Exitz
mv plots/muon/pos_Localy_T*.eps.gz $WWWDIR/eps/TK-strips/Localy
mv plots/muon/pos_Localx_T*.eps.gz $WWWDIR/eps/TK-strips/Localx

mv plots/muon/eloss_*PIX_KS*.eps.gz $WWWDIR/eps/TK-pixels/eloss
mv plots/muon/pos_Entryx-Exitx_*PIX*.eps.gz $WWWDIR/eps/TK-pixels/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_*PIX*.eps.gz $WWWDIR/eps/TK-pixels/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_*PIX*.eps.gz $WWWDIR/eps/TK-pixels/Entryz-Exitz
mv plots/muon/pos_Localy_*PIX*.eps.gz $WWWDIR/eps/TK-pixels/Localy
mv plots/muon/pos_Localx_*PIX*.eps.gz $WWWDIR/eps/TK-pixels/Localx

mv plots/muon/Tof.eps.gz       $WWWDIR/eps/ToF/
mv plots/muon/*summary*.eps.gz $WWWDIR/eps/TK-summary

mv plots/muon/eloss_T*_KS*.gif $WWWDIR/gif/TK-strips/eloss
mv plots/muon/pos_Entryx-Exitx_T*.gif $WWWDIR/gif/TK-strips/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_T*.gif $WWWDIR/gif/TK-strips/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_T*.gif $WWWDIR/gif/TK-strips/Entryz-Exitz
mv plots/muon/pos_Localy_T*.gif $WWWDIR/gif/TK-strips/Localy
mv plots/muon/pos_Localx_T*.gif $WWWDIR/gif/TK-strips/Localx

mv plots/muon/eloss_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/eloss
mv plots/muon/pos_Entryx-Exitx_*PIX*.gif $WWWDIR/gif/TK-pixels/Entryx-Exitx
mv plots/muon/pos_Entryy-Exity_*PIX*.gif $WWWDIR/gif/TK-pixels/Entryy-Exity
mv plots/muon/pos_Entryz-Exitz_*PIX*.gif $WWWDIR/gif/TK-pixels/Entryz-Exitz
mv plots/muon/pos_Localy_*PIX*.gif $WWWDIR/gif/TK-pixels/Localy
mv plots/muon/pos_Localx_*PIX*.gif $WWWDIR/gif/TK-pixels/Localx

mv plots/muon/Tof.gif       $WWWDIR/gif/ToF/
mv plots/muon/*summary*.gif $WWWDIR/gif/TK-summary

echo "...Done..."
