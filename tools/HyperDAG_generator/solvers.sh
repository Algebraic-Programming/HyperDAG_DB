executable = $1
A=$2
L=$3
U=$4

$1 spmv -sink -reindex -output spmv_$A -inputs $A
$1 lltsolver -sink -reindex -output lltsolver_$A -inputs $L
$1 alltsolver -sink -reindex -output alltsolver_$A -inputs $A $L
$1 lusolver -sink -reindex -output lusolver_$A -inputs $L $U
$1 alusolver -sink -reindex -output alusolver_$A -inputs $A $L $U
