load "common.gnu"
set time
set title "${TEMPLATE_TITLE}"
set xlabel "${TEMPLATE_XLABEL}"
set output '${FIELD}.eps'
plot 'common.data' using 1:${INDEX} title "${FIELD}"
exit
