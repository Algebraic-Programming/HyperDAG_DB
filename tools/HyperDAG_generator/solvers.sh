if [ $# -lt 4 ]; then
    echo "Usage: $0 <executable> <A> <L> <U>"
    exit 1
fi

executable = $1
A=$2
L=$3
U=$4

INPUT_FILENAME=$(basename "$A")
DIRNAME=$(dirname "$A")

$1 spmv -sink -reindex -output $DIRNAME/spmv_$INPUT_FILENAME -inputs $A
$1 lltsolver -sink -reindex -output $DIRNAME/lltsolver_$INPUT_FILENAME -inputs $L
$1 alltsolver -sink -reindex -output $DIRNAME/alltsolver_$INPUT_FILENAME -inputs $A $L
$1 lusolver -sink -reindex -output $DIRNAME/lusolver_$INPUT_FILENAME -inputs $L $U
$1 alusolver -sink -reindex -output $DIRNAME/alusolver_$INPUT_FILENAME -inputs $A $L $U
