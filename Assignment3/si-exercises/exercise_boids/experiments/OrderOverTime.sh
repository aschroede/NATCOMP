
#!/bin/sh
# node node-boids-template.js -f 400 -N 200 -i 10 -o 0 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime0
# node node-boids-template.js -f 400 -N 200 -i 10 -o 20 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime20
# node node-boids-template.js -f 400 -N 200 -i 10 -o 40 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime40
# node node-boids-template.js -f 400 -N 200 -i 10 -o 60 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime60
# node node-boids-template.js -f 400 -N 200 -i 10 -o 80 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime80
# node node-boids-template.js -f 400 -N 200 -i 10 -o 100 -c 1 -s 1 -a 1 -T 1000 -M experiments/OrderOverTime100


python3 orderParamAnalysis.py OrderOverTime0 OrderOverTime0/Metrics
python3 orderParamAnalysis.py OrderOverTime20 OrderOverTime20/Metrics
python3 orderParamAnalysis.py OrderOverTime40 OrderOverTime40/Metrics
python3 orderParamAnalysis.py OrderOverTime60 OrderOverTime60/Metrics
python3 orderParamAnalysis.py OrderOverTime80 OrderOverTime80/Metrics
python3 orderParamAnalysis.py OrderOverTime100 OrderOverTime100/Metrics/



python3 nnAnalysis.py OrderOverTime0/nearestNeighborParameter.csv OrderOverTime0/Metrics
python3 nnAnalysis.py OrderOverTime20/nearestNeighborParameter.csv OrderOverTime20/Metrics
python3 nnAnalysis.py OrderOverTime40/nearestNeighborParameter.csv OrderOverTime40/Metrics
python3 nnAnalysis.py OrderOverTime60/nearestNeighborParameter.csv OrderOverTime60/Metrics
python3 nnAnalysis.py OrderOverTime80/nearestNeighborParameter.csv OrderOverTime80/Metrics
python3 nnAnalysis.py OrderOverTime100/nearestNeighborParameter.csv OrderOverTime100/Metrics
