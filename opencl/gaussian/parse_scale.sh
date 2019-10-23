DIR=.
OUTDIR=$DIR
scale=1
b="scale_${scale}"

grep "^[0-9]\+\.[0-9]\+" $OUTDIR/$b.txt | \
    awk '{ total += $1; count++ } END {
           if (count > 0)
               print total/count;
           else
               print -1
         }' \

grep "^Total:" $OUTDIR/$b.txt | \
    awk '{ total += $2; count++ } END {
           if (count > 0)
               print total/count;
           else
               print -1
         }' \
