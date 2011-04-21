#!/bin/sh 

echo "This script produces all the histograms and plots..."
echo -e "Enter the REFERNCE.root file, the NEW.root file, the directory name/histogram title and the normalization ([REFERNCE.root] [NEW.root] [title] [normalization]): "
read reference new title norm

if [ -n "$reference" ]; then
    if [ -n "$new" ]; then
	if [ -n "$title" ]; then 
	    echo "processing "$reference" and "$new"."
	else 
	    echo "Input not correct!!"
	fi
    else 
	echo "Input not correct!!"
    fi
else
    echo "Input not correct!!"
fi

if [ -n $title ]; then
    if [ -e $reference ]; then
	if [ -e $new ]; then

	    TAG=(CaloJetTask_iterativeCone5CaloJets CaloJetTask_L2L3CorJetAK5Calo JPTJetTask_JetPlusTrackZSPCorJetIcone5 PFJetTask_iterativeCone5PFJets CaloJetTask_kt4CaloJets CaloJetTask_kt6CaloJets CaloJetTask_ak5CaloJets CaloJetTask_ak7CaloJets PFJetTask_ak5PFJets JPTJetTask_JetPlusTrackZSPCorJetAntiKt5)
	    TAGREF=(CaloJetTask_iterativeCone5CaloJets CaloJetTask_L2L3CorJetAK5Calo JPTJetTask_JetPlusTrackZSPCorJetIcone5 PFJetTask_iterativeCone5PFJets CaloJetTask_kt4CaloJets CaloJetTask_kt6CaloJets CaloJetTask_ak5CaloJets CaloJetTask_ak7CaloJets PFJetTask_ak5PFJets JPTJetTask_JetPlusTrackZSPCorJetAntiKt5)
	    FOLDER=(Icone5 CorrAK5 JPTIC5 PFlow kt4 kt6 AntiKt5 AntiKt7 PFlowAntiKt5 JPTAntiKt5)
	    ntag=${#TAG[@]}
	    echo "Number of module tags: " $ntag
	    for (( i=0;i<$ntag;i++ )); do
		echo "tag: " ${TAG[${i}]}

		mkdir $title
		mkdir $title/${FOLDER[${i}]}
		mkdir $title/${FOLDER[${i}]}/NewData
		mkdir $title/${FOLDER[${i}]}/RefData	    
		echo "folders are created"

		if [ -z $norm ]; then
		../../../../test/slc5_amd64_gcc434/compareHists $new $reference ${TAG[${i}]} ${TAGREF[${i}]} $title
		fi

		if [ $norm = "y" ]; then
		../../../../test/slc5_amd64_gcc434/compareHists $new $reference ${TAG[${i}]} ${TAGREF[${i}]} $title y
		fi

		bash make_thumbnails.sh *.gif
		mv *.gif $title/${FOLDER[${i}]}
		mv *.jpg $title/${FOLDER[${i}]}
		cp html/spacer.gif $title/${FOLDER[${i}]}

                # Icone PF
		if [ ${FOLDER[${i}]} = ${FOLDER[3]} ]; then
                cp html/htmlTemplate_PFUnCorr.html $title/${FOLDER[${i}]}
                # Icone Calo
                elif [ ${FOLDER[${i}]} = ${FOLDER[0]} ]; then
                    cp html/htmlTemplateUnCorr.html $title/${FOLDER[${i}]}
                 # kt4 Calo
                elif [ ${FOLDER[${i}]} = ${FOLDER[4]} ]; then
                    cp html/htmlTemplateUnCorr.html $title/${FOLDER[${i}]}
                 # kt6 Calo
                elif [ ${FOLDER[${i}]} = ${FOLDER[5]} ]; then
                    cp html/htmlTemplateUnCorr.html $title/${FOLDER[${i}]}
                # AntiKt PF
		elif [ ${FOLDER[${i}]} = ${FOLDER[8]} ]; then
		    cp html/htmlTemplate_PF.html $title/${FOLDER[${i}]}
                # JPT IC5
		elif [ ${FOLDER[${i}]} = ${FOLDER[2]} ]; then
		    cp html/JPTUnCorr.html $title/${FOLDER[${i}]}
                # JPT AK5
		elif [ ${FOLDER[${i}]} = ${FOLDER[9]} ]; then
		    cp html/JPT.html $title/${FOLDER[${i}]}
		else
		    cp html/htmlTemplate.html $title/${FOLDER[${i}]}
		fi

		../../../../test/slc5_amd64_gcc434/fixed_plotHists $new ${TAG[${i}]}
		mv *.gif $title/${FOLDER[${i}]}/NewData

		../../../../test/slc5_amd64_gcc434/fixed_plotHists $reference ${TAG[${i}]}
		mv *.gif $title/${FOLDER[${i}]}/RefData
	    done

	else
	    echo $new "not found!"
	fi
    else
	echo $refernce "not found!"
    fi
    
else
    echo "Enter a title!"
fi

if [ -d $title ]; then
    cp html/index.html $title
    cp $new       $title/new.root
    cp $reference $title/ref.root
else
    echo ".html files not copied!"
fi


