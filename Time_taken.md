## RTX 4060 Ti (8G)
---(resize=224)---

AlexNet training on cuda:0
loss 0.328, train acc 0.880, test acc 0.884
1947.0 examples/sec on cuda:0
Time taken: 0 h 5 min 8.2 sec

VGG-11 training on cuda:0
loss 0.172, train acc 0.936, test acc 0.920
1309.0 examples/sec on cuda:0
Time taken: 0 h 7 min 38.4 sec

NiN training on cuda:0
loss 0.430, train acc 0.840, test acc 0.840
1632.9 examples/sec on cuda:0
Time taken: 0 h 6 min 7.4 sec

## RTX 4070 Ti (12G)
---(resize=224)---

AlexNet training on cuda:0
loss 0.330, train acc 0.880, test acc 0.881
3175.8 examples/sec on cuda:0
Time taken: 0 h 3 min 8.9 sec

VGG-11 training on cuda:0
loss 0.194, train acc 0.927, test acc 0.907
2160.7 examples/sec on cuda:0
Time taken: 0 h 4 min 37.7 sec

NiN training on cuda:0
loss 0.557, train acc 0.805, test acc 0.795
2726.2 examples/sec on cuda:0
Time taken: 0 h 3 min 40.1 sec

---(resize=96)---

GoogLeNet training on cuda:0
loss 0.245, train acc 0.906, test acc 0.892
3923.3 examples/sec on cuda:0
Time taken: 0 h 2 min 32.9 sec

ResNet-18 training on cuda:0
loss 0.007, train acc 0.999, test acc 0.919
5908.0 examples/sec on cuda:0
Time taken: 0 h 1 min 41.6 sec

DenseNet training on cuda:0
loss 0.139, train acc 0.950, test acc 0.876
5386.8 examples/sec on cuda:0
Time taken: 0 h 1 min 51.4 sec

---(RNN scratch)---

RNN Scratch (sequential partitioning) is training on cuda:0 ...
Perplexity 1.1, 41219.6 tokens/sec, total time: 0 h 1 min 49.1 sec
time travelleryou can show black is white by argument said filby
traveller with a slight accession ofcheerfulness really thi

GRU (scratch) is training on cuda:0 ...
Perplexity 1.1, 13912.2 tokens/sec, total time: 0 h 4 min 54.5 sec
time travelleryou can show black is white by argument said filby
traveller with a slight accession ofcheerfulness really thi

LSTM (scratch) is training on cuda:0 ...
Perplexity 1.1, 9653.3 tokens/sec, total time: 0 h 5 min 49.2 sec
time traveller corming but bat lowk go sumowe ore a suint bur ma
traveller the fime wishollprous louing said filbycan a cube

---(RNN concise)---

RNN (concise) is training on cuda:0 ...
Perplexity 1.3, 397173.0 tokens/sec, total time: 0 h 0 min 12.3 sec
time travelleryou can showr sident any lagistan a some they toub
travelleroche a y at a that lias in the pras exsmat back of

GRU (concise) is training on cuda:0 ...
Perplexity 1.0, 249394.1 tokens/sec, total time: 0 h 0 min 18.4 sec
time travelleryou can show black is white by argument said filby
travelleryou can show black is white by argument said filby

LSTM (concise) is training on cuda:0 ...
Perplexity 1.0, 243102.1 tokens/sec, total time: 0 h 0 min 18.0 sec
time traveller for so it will be convenient to speak of himwas e
travelleryou can show black is white by argument said filby

Double-layer LSTM (concise) is training on cuda:0 ...
Perplexity 1.0, 147618.8 tokens/sec, total time: 0 h 0 min 30.4 sec
time traveller for so it will be convenient to speak of himwas e
traveller with a slight accession ofcheerfulness really thi

Double-layer Bi-LSTM (concise) is training on cuda:0 ...
Perplexity 1.1, 86238.1 tokens/sec, total time: 0 h 0 min 51.1 sec
time travellerererererererererererererererererererererererererer
travellerererererererererererererererererererererererererer
