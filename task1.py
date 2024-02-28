'''Task1'''
from pyspark import SparkContext
from pyspark.sql import SQLContext
from graphframes import *
#from pyspark.sql import SparkSession
import json
import sys
from itertools import combinations
import time
import random
import os

#os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

#sc.stop()
'''
threshold = int(sys.argv[1])
input_filepath = sys.argv[2]
output_filepath = sys.argv[3]'''

threshold = int(4)
input_filepath = '/content/ub_sample_data.csv'
output_filepath = '/content/out.txt'

#start = time.time()
sc= SparkContext(appName='Task1')
start = time.time()
# train data
lines = sc.textFile(input_filepath) # load train file into rdd
# Skip the header
header = lines.first()
lines = lines.filter(lambda x: x!= header)
# change each line to list
lines = lines.map(lambda x: x.strip().split(",")).cache()
# construct each node(user): (user,{bus_1,bus_2...})
all_node = lines.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set).collectAsMap()
# generate combo for every two nodes
combo_node = list(combinations(all_node.keys(),2))
# find nodes in the graph(with edges)
actural_node=[]
edge=[]
for u1, u2 in combo_node:
  e = all_node[u1].intersection(all_node[u2])
  if len(e) >= threshold:
    # undirected
    edge.append((u1,u2))
    edge.append((u2,u1))
    actural_node.append(u1)
    actural_node.append(u2)
# remove duplicated nodes
actural_node = set(actural_node)
# create vertices
sqlContext = SQLContext(sc)
vertices = sqlContext.createDataFrame([tuple([i]) for i in list(actural_node)], ["id"])
# create edges
edges = sqlContext.createDataFrame(edge, ["src", "dst"])
# create graph
g = GraphFrame(vertices, edges)
result = g.labelPropagation(maxIter=5)
# sort the answer by the size, first user id, and the user in each community
sorted_result = result.rdd.map(lambda x: (x[1],x[0])).groupByKey().mapValues(list).map(lambda x: sorted(x[1])).sortBy(lambda x: x).sortBy(lambda x: len(x)).collect()

with open(output_filepath, "w") as file:
  for i in sorted_result:
    file.write(str(i)[1:-1]+'\n')
end = time.time()
print('Duration: ',end-start)