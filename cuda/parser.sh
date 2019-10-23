#!/usr/bin/env bash

source benchmarks

col="Init MemAlloc HtoD Exec DtoH Close API Total"

TMPFILE=$DIR/.tmp

cd $OUTDIR

echo -n "GPU,"; echo $col | tr ' ' ','
for b in $bm; do
    echo -n $b
    result=""
    echo -n > $TMPFILE
    for c in $col; do
        grep ^$c $OUTDIR/$b.txt | \
            awk '{ total += $2; count++ } END {
                   if (count > 0)
                       print total/count;
                   else
                       print -1
                 }' \
            >> $TMPFILE
    done
    echo -n ","; cat $TMPFILE | paste -sd "," -
    rm $TMPFILE
done
