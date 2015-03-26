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
            CHOIX_ETAPE='force'
#            CHOIX_ETAPE='store'
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

echo "*** CHOIX_ETAPE : " $1 $CHOIX_ETAPE

if [ "$2" != "i" ] 
then
	echo "pas d'interaction"
	echo "defaut = batch"
	CHOIX_INTERACTION="./electronBsub ${CHOIX_JOB} /afs/cern.ch/cms/utils/oval run ${CHOIX_ETAPE}.Val"
else
	echo "interaction"
	CHOIX_INTERACTION="/afs/cern.ch/cms/utils/oval run ${CHOIX_ETAPE}.Val"
fi

echo "*** CHOIX_INTERACTION : " $2 $CHOIX_INTERACTION

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
			list="TTbarStartup ZEEStartup"
#			list="ZEEStartup"
#			list="TTbarStartup"
			for element in $list    
			do   
       			echo "element =" $element   
			done
    if [ ! -d "PU25" ];then
        echo "Creation of PU25 folder";
        mkdir PU25
    else
        echo "PU25 folder already created";
    fi
    if [ ! -d "PU50" ];then
        echo "Creation of PU50 folder";
        mkdir PU50
    else
        echo "PU50 folder already created";
    fi
			CHOIX_CALCUL='PileUp'
		fi
	else
		echo "FAST"
        list="TTbarStartup ZEEStartup"
#	    list="ZEEStartup "
#            list="TTbarStartup"
        for element in $list    
        do   
            echo "element =" $element   
        done
    if [ ! -d "FAST" ];then
        echo "Creation of FAST folder";
        mkdir FAST
    else
        echo "FAST folder already created";
    fi
		CHOIX_CALCUL='Fast'
	fi
else
    echo "FULL"
    list="Pt10Startup_UP15 Pt1000Startup_UP15 Pt35Startup_UP15 TTbarStartup_13 ZEEStartup_13 QcdPt80Pt120Startup_13"
#	list="Pt1000Startup_UP15 TTbarStartup_13 ZEEStartup_13 QcdPt80Pt120Startup_13"
#	list="Pt1000Startup_UP15 "
    for element in $list    
    do   
        echo "element =" $element   
    done
    if [ ! -d "GED" ];then
        echo "Creation of GED folder";
        mkdir GED
    else
        echo "GED folder already created";
    fi
    if [ "$1" != "p" ]
    then
        CHOIX_CALCUL='Full'
    else
        CHOIX_CALCUL='gedvsgedFull'
    fi
fi

echo "*** CHOIX_CALCUL : " $3 $CHOIX_CALCUL

echo "initialization done ... running"
echo "--"
if [ "$CHOIX_ETAPE" == "store" -o "$CHOIX_ETAPE" == "force" ]
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

if [ "$CHOIX_ETAPE" == "publish" ]
then
    echo "publish"
    if [ "$CHOIX_CALCUL" == "Fast" ]
    then
        echo "FAST"
	    for j in VsFull VsFast
        do
		    echo "---------- $j"
            for i in $list
		    do 
			    echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${j}${i}_gedGsfE"
				${CHOIX_INTERACTION}${CHOIX_CALCUL}${j}${i}_gedGsfE
		    done
        done
    else # no FAST
        echo "noFAST"
        for i in $list
	    do 
		    echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
    fi
    rm dd*.olog dqm*.root
else # no publish
    echo "no publish"
    for i in $list
		do 
			echo " == ${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE"
			${CHOIX_INTERACTION}${CHOIX_CALCUL}${i}_gedGsfE
		done
fi
            
