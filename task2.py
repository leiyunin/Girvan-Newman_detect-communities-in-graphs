'''Task 2 complete code'''
from pyspark import SparkContext
from pyspark.sql import SQLContext
#from graphframes import *
#from pyspark.sql import SparkSession
import sys
from itertools import combinations
from collections import defaultdict
import time
import random
import os

#os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

#sc.stop()

'''threshold = int(sys.argv[1])
input_filepath = sys.argv[2]
betw_output_filepath = sys.argv[3]
comm_output_filepath = sys.argv[4]
'''


threshold = int(10)
input_filepath = '/content/ub_sample_data.csv'
betw_output_filepath = '/content/bett_out.txt'
comm_output_filepath = '/content/comm_out.txt'

#sc.stop()
#start = time.time()
#sc= SparkContext(appName='Task2')
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
# edge format: node as key, list of neighbor nodes as value
edge=defaultdict(list)
for u1, u2 in combo_node:
  e = all_node[u1].intersection(all_node[u2])
  if len(e) >= threshold:
    # undirected
    edge[u1].append(u2)
    edge[u2].append(u1)
    actural_node.append(u1)
    actural_node.append(u2)
# remove duplicated nodes
actural_node = list(set(actural_node))

'''BFS and count for contribution of each edge'''
def bfs(graph, nodes):
  D_list=[]
  parent={}
  level = {}
  for R in nodes:
    n_path = defaultdict(float)
    n_path[R]=1
    D=defaultdict(float)
    leaf={} # initial the dict for the graph
    visit= set()
    visit.add(R)
    level[R] = 0
    Q = []
    Q.append((R,0))
    while Q != []:
      R,L = Q.pop(0)
      for neighbor in graph[R]:
        if neighbor not in visit: # remember the path through nodes
          visit.add(neighbor)
          Q.append((neighbor,L+1))
          level[neighbor] = L + 1
          parent[neighbor]=R
          if R in leaf.keys():
            leaf[R].add(neighbor)
          else:
            leaf[R]={neighbor}
          #D[(R,neighbor)]=1
          n_path[neighbor]+=n_path[R]
        elif neighbor in visit and level[neighbor] == L + 1: # handle situation for same level nodes
          if R in leaf.keys():
            leaf[R].add(neighbor)
          else:
            leaf[R]={neighbor}
          n_path[neighbor]+=n_path[R]
      for n in leaf:
        for nn in leaf[n]:
          D[(n,nn)]=1
        #print(n_path[n]/n_path[nn])
    #tem_D={}
    for key in list(reversed(D.keys())):
      D[key]= D[(key)]*(n_path[key[0]]/n_path[key[1]]) # calculate the betweeness of each path on contribution
      first, second = key
      if second in leaf:
        for i in leaf[second]:
          #D[(second,i)]=(n_path[n]/n_path[nn])
          D[key]+=D[(second,i)]*(n_path[first]/n_path[second])
      # dict2[key]+=1
    D_list.append(D)

  '''Calculate betweeness'''
  # create a defaultdict to store summed values for each key
  summed_dict = defaultdict(float)
  # iterate over the list of dictionaries and sum the values for the same keys
  for dictionary in D_list:
      for key, value in dictionary.items():
          sorted_key = tuple(sorted(key))  # sort the tuple to handle (B, A) and (A, B) as the same key
          summed_dict[sorted_key] += value
  for key in summed_dict:
      summed_dict[key] /= 2
  # convert the defaultdict back to a regular dictionary if needed
  result_dict = dict(summed_dict)

  sorted_result=sorted(result_dict.items(), key=lambda x: (-x[1], x[0]))
  return sorted_result

result = bfs(edge,actural_node)

with open(betw_output_filepath, "w") as file:
  for user, bet in result:
    file.write(str(user)+','+str(round(bet,5))+'\n')

'''Modularity'''
cut_edge = edge # create a copy of the original graph
total_edge = len(result) # total number of edges in the original graph
mod1 = -1

while len(result) > 0:
    '''Find the conneted community'''
    community = []
    remaining_nodes = actural_node.copy()
    while remaining_nodes != []:
      R = remaining_nodes.pop(0)
      Q = []
      Q.append(R)
      visited = set()
      visited.add(R)
      while Q != []:
          R = Q.pop(0)
          for neighbor in cut_edge[R]:
              if neighbor not in visited:
                  remaining_nodes.remove(neighbor)
                  Q.append(neighbor)
                  visited.add(neighbor)
      #component = sorted(list(visited))
      community.append(sorted(list(visited)))
    '''Calculate the modularity'''
    mod2 = 0
    for s in community:
        for i in s:
            for j in s:
              # 1 if nodes i and j are connected, 0 otherwise
              A_ij = 1 if j in edge[i] else 0
              k_i = len(edge[i]) # degree of i
              k_j = len(edge[j]) # degree of j
              mod2 += A_ij - (k_i * k_j) / (2 * total_edge)
    mod2 = mod2/(2 * total_edge)
    #replace old modularity if new is higher
    if mod2 > mod1:
        mod1 = mod2
        res = community

    highest_b = result[0][1]
    # cut highest betweeness
    for edge_, bet in result:
        if bet >= highest_b:
            cut_edge[edge_[0]].remove(edge_[1])
            cut_edge[edge_[1]].remove(edge_[0])
    # calculate the new bet of graph
    result = bfs(cut_edge, edge)

result_ = sorted(res, key=lambda x: (len(x), x[0]))

with open(comm_output_filepath, "w") as file:
  for i in result_:
    file.write(str(i)[1:-1]+'\n')
end = time.time()
print('Duration: ',end-start)