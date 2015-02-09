#!/bin/bash

echo $1 $2 $3

if [ "$1" == "?" ] 
then
	echo "methode : ./relval_gedGsfE [a(nalyze),f(inalize),s(tore),p(ublish)] [i(nteractif),j(ob)] [r(eco),f(ast),p(ileup)]"
	echo "defaut = a j r"
	exit
fi

if [ "$1" != "a" ]
then
	if [ "$1" != "f" ] 
	then
		if [ "$1" != "s" ] 
		then
			if [ "$1" != "p" ]
            then
                echo "pas de choix etape" 
                echo "defaut = analyze"
                CHOIX_ETAPE='analyze'
                CHOIX_JOB='8nh'
            else
                echo "publish"
                CHOIX_ETAPE='publish'
                CHOIX_JOB='1nh'
            fi
        else
            echo "store"
#	        CHOIX_ETAPE='force'
            CHOIX_ETAPE='store'
            CHOIX_JOB='1nh'
		fi
	else
		echo "finalize"
		CHOIX_ETAPE='finalize'
        CHOIX_JOB='8nh'
	fi
else
	echo "analyze"
	CHOIX_ETAPE='analyze'
    CHOIX_JOB='8nh'
fi

echo "***" $1 $CHOIX_ETAPE

if [ "$2" != "i" ] 
then
	echo "pas d'interaction"
	echo "defaut = batch"
	CHOIX_INTERACTION="./electronBsub ${CHOIX_JOB} /afs/cern.ch/cms/utils/oval run ${CHOIX_ETAPE}.Val"
else
	echo "interaction"
	CHOIX_INTERACTION="/afs/cern.ch/cms/utils/oval run ${CHOIX_ETAPE}.Val"
fi

echo $2 $CHOIX_INTERACTION

if [ "$3" != "r" ] 
then
	if [ "$3" != "f" ] 
	then
		if [ "$3" != "p" ] 
		then
			echo "pas de choix calcul" 
			echo "defaut = FULL"
			CHOIX_CALCUL='Full'
		else
			echo "PILES PileUp"
			CHOIX_CALCUL='PileUp'
		fi
	else
		echo "FAST"
		CHOIX_CALCUL='Fast'
	fi
else
    echo "FULL"
    if [ "$1" != "p" ]
    then
        CHOIX_CALCUL='Full'
    else
        CHOIX_CALCUL='gedvsgedFull'
    fi
fi

echo $3 $CHOIX_CALCUL

case $CHOIX_CALCUL in
Full | gedvsgedFull) echo "Full"
	echo "--"
            if [ "$CHOIX_ETAPE" == "store" ]
            then
                echo "== store =="
                for var in `ls DQM*.root`
                    do
                        echo $var
                        i=${var:38}
                        #echo $i
                        #echo ${i:0:$((${#i}-12))}
                        j=${i:0:$((${#i}-12))}
                        echo electronHistos.$j.root
			var_final=electronHistos.$j.root
			cp $var $var_final
                    done
            fi

	for i in Pt10Startup_UP15 Pt1000Startup_UP15 Pt35Startup_UP15 TTbarStartup_13 ZEEStartup_13 QcdPt80Pt120Startup_13
#	for i in Pt1000Startup_UP15 TTbarStartup_13 ZEEStartup_13 QcdPt80Pt120Startup_13
#	for i in TTbarStartup_13 
		do 
			echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
#            ${COPIE_FICHIERS}${CHOIX_CALCUL}${i}_gedGsfE__RECO3.root electronHistos.Val${CHOIX_CALCUL}${i}_gedGsfE.root
#			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
PileUp) echo "PileUp"
	echo "++"
	for i in TTbarStartup ZEEStartup
		do 
			echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
#            ${COPIE_FICHIERS}${CHOIX_CALCUL}${i}_gedGsfE__RECO3.root electronHistos.Val${CHOIX_CALCUL}${i}_gedGsfE.root
#			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
Fast) echo "Fast"
	echo "**"
	for i in TTbarStartup ZEEStartup
#	for i in ZEEStartup
		do 
			echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
#            ${COPIE_FICHIERS}${CHOIX_CALCUL}${i}_gedGsfE__RECO3.root electronHistos.Val${CHOIX_CALCUL}${i}_gedGsfE.root
#			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
esac


