#!/bin/bash

echo $1 $2

if [ "$1" == "?" ] 
then
	echo "methode : ./finalize_gedGsfE [i(nteractif),j(ob)] [r(eco),f(ast),p(ileup)]"
	echo "defaut = j r"
	exit
fi

if [ "$1" != "i" ] 
then
	echo "pas d'interaction"
	echo "defaut = batch"
	CHOIX_INTERACTION='electronBsub 8nh /afs/cern.ch/cms/utils/oval run finalize.Val'
else
	echo "interaction"
	CHOIX_INTERACTION='/afs/cern.ch/cms/utils/oval run finalize.Val'
fi

#echo $1 $CHOIX_INTERACTION

if [ "$2" != "r" ] 
then
	if [ "$2" != "f" ] 
	then
		if [ "$2" != "p" ] 
		then
			echo "pas de choix calcul" 
			echo "defaut = FULL"
			CHOIX_CALCUL='Full'
		else
			#echo "PILES PileUp"
			CHOIX_CALCUL='PileUp'
		fi
	else
		#echo "FAST"
		CHOIX_CALCUL='Fast'
	fi
else
	#echo "FULL"
	CHOIX_CALCUL='Full'
fi

#echo $2 $CHOIX_CALCUL

case $CHOIX_CALCUL in
Full) echo "Full"
	echo "--"
	for i in Pt10Startup_UP15 Pt1000Startup_UP15 Pt35Startup_UP15 TTbarStartup_13 ZEEStartup_13 QcdPt80Pt120Startup_13
		do 
			echo " -- ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
PileUp) echo "PileUp"
	echo "++"
	for i in TTbarStartup ZEEStartup
		do 
			echo " ++ ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
Fast) echo "Fast"
	echo "**"
	for i in TTbarStartup ZEEStartup
		do 
			echo " ** ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
	;;
esac


