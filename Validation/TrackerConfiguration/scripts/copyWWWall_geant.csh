#! /bin/csh
eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src
cd ${DATADIR}/Validation/TrackerHits/test

setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/SimHits/) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/SimHits/

setenv WWWDIR /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/SimHits/${1}

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

mv plots/muon/*summary*.eps $WWWDIR/eps/TK-summary

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

mv plots/muon/*summary*.gif $WWWDIR/gif/TK-summary

echo "...Done..."
