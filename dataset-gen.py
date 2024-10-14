import random
import time
from sys import argv
random.seed(time.time())
n = int(argv[1])
diff = 5 #min delta 
answer = 13.1
foot_k = 0.3048
result = []
avg = round(answer / foot_k, 1)
avg_i = int(avg)
result.append(avg_i)
for i in range(1, n-1):
	if(sum(result) / len(result) > avg):
		x_n = random.randint(avg_i - diff, avg_i)
	else:
		x_n = random.randint(avg_i, avg_i + diff)
	result.append(x_n)
result.append(round(avg * n - sum(result)))
for i in range(n):
	print(result[i])
#print(result)
#print("avg in foot = " + str(sum(result) / len(result)))
#print("avg in m = " + str(sum(result) / len(result) * foot_k))
