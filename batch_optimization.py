import argparse
import os
import sys
import json
import gurobipy as gp
import math
import time
import random
import copy

class Solver():
    def __init__(self, orderlist, max_batch_size):
        self.order_ids = list(orderlist.keys()) # The ids for each order
        self.order_aisles = {} # The aisles for each order
        for o in self.order_ids:
            self.order_aisles[o] = set([p['aisle'] for p in orderlist[o]])
        # List of all aisles used by orders
        self.aisles = list(set([a for o in self.order_aisles.values() for a in o ]))
        self.aisles_orders = {} # The orders visiting each aisle
        for a in self.aisles:
            orders = []
            for o in self.order_ids:
                if a in self.order_aisles[o]:
                    orders.append(o)
            self.aisles_orders[a] = orders
        self.K = max_batch_size
        self.B = math.ceil(len(self.order_ids)/self.K)

    def MIP(self, initial_solution = False):
        model = gp.Model("Model")

        # If an order is part of batch
        x = model.addVars(self.order_ids, range(self.B), obj=0, vtype=gp.GRB.BINARY, name="x")

        # If a batch visits aisle
        y = model.addVars(range(self.B), self.aisles, obj=0, vtype=gp.GRB.BINARY, name="y")

        if initial_solution:
            single_batch_optimal = solver.greedy_optimal_single_batching()
            # Use greedy solution as starting solution
            for b, batch in enumerate(single_batch_optimal):
                for o in self.order_ids:
                    if o in batch:
                        x[o,b].Start = 1
                    else: 
                        x[o,b].Start = 0

            for b, batch in enumerate(single_batch_optimal):
                for a in self.aisles:
                    for o in self.aisles_orders[a]:
                        if o in batch:
                            y[b,a].Start = 1        

        # Number of aisles for each batch
        d = model.addVars(range(self.B), obj=1, vtype=gp.GRB.INTEGER, name="d")

        # Each batch cannot exceed the maximum size K
        model.addConstrs((gp.quicksum(x[o,b] for o in self.order_ids) <= self.K for b in range(self.B)), name='batch_size')

        # Each order must be in exactly one batch
        model.addConstrs((gp.quicksum(x[o,b] for b in range(self.B)) == 1 for o in self.order_ids), name='single_batch')

        model.addConstrs((y[b,a] >= x[o,b] for b in range(self.B) for a in self.aisles for o in self.aisles_orders[a]), name='batch_aisle')

        # Controlling the batch number of aisles variables
        model.addConstrs((d[b] == gp.quicksum(y[b,a] for a in self.aisles) for b in range(self.B)), name='num_aisles')

        model.setObjective(gp.quicksum(d[b] for b in range(self.B)))

        model.update()

        model.write("model.lp")

        model.optimize()

        batches = []
        for k in range(self.B):
            batch = []
            for o in self.order_ids:
                if x[o,k].X != 0:
                    batch.append(o)
            batches.append(batch)

        return batches
    
    def greedy_optimal_single_batching_with_seed(self):
        batches = []
        order_ids = copy.copy(self.order_ids)
        order_ids.sort(reverse = True,key=self.num_aisles)
        for b in range(self.B):
            batch = self.MIP_single_batch(order_ids, min(14,len(order_ids)), order_ids[0])
            batches.append(batch)
            for o in batch:
                order_ids.remove(o)
        return batches
    
    def greedy_optimal_single_batching(self):
        batches = []
        order_ids = copy.copy(self.order_ids)
        for b in range(self.B):
            batch = self.MIP_single_batch(order_ids, min(14,len(order_ids)))
            batches.append(batch)
            for o in batch:
                order_ids.remove(o)
        return batches

    def MIP_single_batch(self, order_ids, batch_size, seed = None):
        aisles = list(set([a for o in self.order_aisles if o in order_ids for a in self.order_aisles[o]]))
        model = gp.Model("Model")
        model.setParam('OutputFlag', 0)
        # If an order is part of the batch
        x = model.addVars(order_ids, obj=0, vtype=gp.GRB.BINARY, name="x")

        # If aisle is visited
        y = model.addVars(aisles, obj=1, vtype=gp.GRB.BINARY, name="y")     

        if seed is not None: # Fix the seed as part of the order
            model.addConstr(x[seed] == 1)

        # Batch must be of size batch_size
        model.addConstr((gp.quicksum(x[o] for o in order_ids) == batch_size), name='batch_size')

        model.addConstrs((y[a] >= x[o] for a in aisles for o in self.aisles_orders[a] if o in order_ids), name='batch_aisle')

        model.update()

        model.write("model.lp")

        model.setObjective(gp.quicksum(y[a] for a in aisles))

        model.optimize()
        batch = []
        for o in order_ids:
            if x[o].X != 0:
                batch.append(o)
        return batch

    def random_batches(self):
        order_ids = copy.copy(self.order_ids)
        batches = []
        for _ in range(self.B): 
            batch = []
            while len(batch) < min(self.K,len(batch) + len(order_ids)):
                random_order = random.choice(order_ids)
                batch.append(random_order)
                order_ids.remove(random_order)
            batches.append(batch)
        return batches

    def num_aisles(self, o):
        return len(self.order_aisles[o])

    def greedy_batches(self):
        order_ids = copy.copy(self.order_ids)
        # Sort orders on the nmber of aisles
        order_ids.sort(reverse = True,key=self.num_aisles)

        batches = []
        for _ in range(self.B): 
            batch = []
            batch_aisles = set(self.order_aisles[order_ids[0]])
            batch.append(order_ids[0]) # Select largest order
            batch.remove(order_ids[0])
            while len(batch) < min(self.K,len(batch) + len(order_ids)):
                best_order = None
                best_order_aisles = set()
                best_order_num = math.inf
                # Find the largest order with fewest number of aisles added
                for o in order_ids:
                    num_new_aisles = len(self.order_aisles[o] - batch_aisles) # The number of aisles added
                    if num_new_aisles < best_order_num:
                        best_order = o
                        best_order_aisles = self.order_aisles[o]
                        best_order_num = num_new_aisles
                    if num_new_aisles == 0: # Largest order (in aisles) with best aisles added
                        break
                batch.append(best_order)
                batch_aisles = batch_aisles.union(best_order_aisles)
                order_ids.remove(best_order)
            batches.append(batch)
        return batches

    def check_num_aisles(self, batches):
        num_aisles = []
        for b in batches:
            aisles = []
            for o in b:
                for a in self.order_aisles[o]:
                    aisles.append(a)
            num_aisles.append(len(set(aisles)))
        print(num_aisles)
        return sum(num_aisles)

def tsv2json(input_file, max_orders):
    file = open(input_file, 'r') 
    json_data = {}
    header = file.readline().strip().split('\t') # Read header
    i = 0
    for line in file:
        values = line.strip().split('\t')
        location = values[-1].split('-')
        # Split Hall 1 aisles into A and B sections
        if int(location[0]) > 22 or int(location[0]) == 1:
             # Aisles 48 and 52 are different names for the south and north side respectively of the same aisle
            if location[0] == '52':
                aisle = '48'
            else:
                aisle = location[0]
        elif int(location[0]) < 19 and int(location[0]) > 1:
            if int(location[1]) <= 34:
                aisle = location[0] + 'a'
            else:
                aisle = location[0] + 'b'
        elif location[0] == '19':
            if int(location[1])%2 != 0:
                if int(location[1]) <= 33:
                    aisle = location[0] + 'a'
                else:
                    aisle = location[0] + 'b'
            else:
                if int(location[1]) <= 18:
                    aisle = location[0] + 'a'
                else:
                    aisle = location[0] + 'b'
        elif location[0] == '22':
            if int(location[1]) <= 16:
                aisle = location[0] + 'a'
            else:
                aisle = location[0] + 'b'
        else:
            if int(location[1]) <= 18:
                aisle = location[0] + 'a'
            else:
                aisle = location[0] + 'b'
        data_entry = {
            header[2]: values[2],
            header[3]: values[3],
            'aisle': aisle,
            'section': location[1],
            'shelf': location[2]
            }

        # Check if the order number is already a key in the dictionary
        if values[1] in json_data:
            # If yes, append the entry to the existing list
            json_data[values[1]].append(data_entry)
        else:
            if len(json_data) > max_orders:
                break
            # If no, create a new list with the entry
            json_data[values[1]] = [data_entry]
    return json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch optimization.')
    parser.add_argument(dest="filepath", type=str, help='filepath')
    # Time limit and order number limit to be added

    args = parser.parse_args()
    orderlist = tsv2json(args.filepath, max_orders=102)
    solver = Solver(orderlist, max_batch_size=14)
    #batches_random = solver.random_batches()
    #batches_greedy = solver.greedy_batches()
    #single_batch_optimal = solver.greedy_optimal_single_batching()
    #single_batch_optimal_seed = solver.greedy_optimal_single_batching_with_seed()
    batches_MIP = solver.MIP(initial_solution=True)
    #print(f"Random: {solver.check_num_aisles(batches_random)}")
    #print(f"Greedy: {solver.check_num_aisles(batches_greedy)}")
    #print(f"Single batch: {solver.check_num_aisles(single_batch_optimal)}")
    #print(f"Single batch (seed): {solver.check_num_aisles(single_batch_optimal_seed)}")
    print(f"MIP: {solver.check_num_aisles(batches_MIP)}")
    print(batches_MIP)