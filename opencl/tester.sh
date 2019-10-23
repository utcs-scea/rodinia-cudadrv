#!/usr/bin/env bash

source benchmarks

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    make clean && make
    for idx in `seq 1 15`; do
        exe ./run -p 0 -d 0
        exe echo
    done
    cd $DIR
    exe echo
    echo
done
