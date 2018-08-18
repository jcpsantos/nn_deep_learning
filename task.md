# Overview

The purpose of this task is to test your knowledge and capabilities using neural networks to solve problems. 

# The Task

The task is to create a neural network which takes a set of 10 points as inputs, and outputs
slope and the y-intercept of the best-fitting line for the given points. The points are noisy, i.e. they
won't fit perfectly on a line, so the net must figure out the best-fit line.

Note that this is a toy task, simple enough to be solved with other means. However, we're curious to see how
you use a neural network to solve the task.

**Important:** This task is simple enough to solve by other means. However, your result should be **one neural network**
that **any set of 10 points** can be fed into to get the answer. That is, you should **not** create any sort of 
algorithm that figures out the slope and intercept on a row-by-row basis. The end result is one network with one set
of trained weights.

## Data Format

### Inputs

The input is a CSV file consisting of the set of 10 points that need to be fit, along with an id
for each row. For example:

```
id,x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9
0,15.104372521694373,-68.3288669890148,-95.58279088853587,-181.47566659777777,39.29508013857384,-34.02032092432305,89.77504252047551,13.335549409789905,59.18020460838781,-27.745988520228575,58.18208066946731,-20.139440271026118,114.15065113682351,22.885664222726238,54.91953213577344,-40.08393846757413,96.94956199134829,-1.7529553733813685,43.73834621553149,-48.32482436599675
1,194.30922983731998,475.11773720087734,39.698795356827475,395.11744053173805,24.385304389555884,412.15259835657196,162.74730961624766,435.40436759448636,-209.97267989214552,268.71948435814704,-15.84739186808086,383.00361026054156,203.7405095041797,486.88150883694993,136.21677009353567,458.061140381253,-209.97936860663347,271.05932168687025,-205.44142240003345,274.8783816980477
2,33.0729609941203,35.65163224931255,59.05800945274443,37.780821108387244,-0.27154527365397296,32.15423141740265,-56.338112707260464,14.804835112815919,-9.506154372471283,23.591260263806447,-31.70490767293847,17.367742076974835,40.205569209665455,32.615782927114594,3.9894296538286502,37.07283430796646,8.756275693126607,27.872055475457028,6.629554851189012,29.855587062980288
3,-55.46550863969177,-228.00385690330634,-48.65131151123526,-207.41403511775863,-40.183517418595855,-183.64068630530306,-0.49320904423302814,-112.25435201910054,97.60087735014612,88.22355795258538,74.27606124699636,51.984610883989994,3.0154480080773194,-103.47131787074638,-107.67861258104709,-284.07696773414847,69.85213955467371,16.130733553958315,10.772837230897439,-82.78505105573907
4,57.29819709664967,-118.9914876627729,28.214930013580968,-119.72683218683812,12.136616487598763,-116.45726663409789,-47.826436194943106,-118.84602737605786,37.99794118961701,-119.85074936392166,-4.8688205078347915,-120.64096280943245,-80.97980135725717,-121.08356637995017,43.057504244329856,-119.13829385810006,-51.466973504692795,-123.550624029465,-75.81316358028273,-114.19119774294938
```

### Ground Truth

The ground truth format gives the desired slope and intercept for each row. For example, the truth corresponding to the
above rows is:

```
id,slope,intercept
0,0.9999146924638005,-84.03411330249395
1,0.4829914827651356,379.652537616225
2,0.20992194452988344,30.979968396764747
3,1.8889264676957538,-101.10620910949814
4,-0.08077458788042025,-120.17067810321352
```

## Training Data

The training data is a set of 100,000 rows along with the desired slope and intercept
for each one. 

[**Download Link**](https://www.dropbox.com/s/4tkjd8ltwa5ir5m/train_100k.zip?dl=0)

## Test Data

The test data is a set of 100,000 rows without the ground truths.

[**Download Link**](https://www.dropbox.com/s/scmpcc85u54omdt/test_100k.zip?dl=0)

## Evaluation

You'll be evaluated on two metrics: the mean squared errors of the slopes, and the
mean absolute errors of the intercepts. The attached [`evaluate.py`](https://gist.github.com/csaftoiu/dd39f1fdac52558053f9f8ffcce358f1#file-evaluate-py) file will 
take two ground-truth-format files and print out the evaluation. 

To give an example of the expected results, the best net we got had the following
evaluation on the data it was trained with:

```
$ python scripts/evaluate.py data/train_100k.truth.csv env/submission.train_100k.csv
Slope mse: 0.00682871657263
Intercept mae: 4.60931900967
```

# Submission

When you are done, submit the following:

* Your prediction on the test data
* All the code needed to train and run your network to produce that prediction from
  scratch, along with instructions on how to run the code
* A short description of the approach you took and how you arrived at the solution 
  you did

Good luck!
