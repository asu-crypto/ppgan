from ast import parse
import matplotlib.pyplot as plt
import pandas as pd

fr1 = open("eval_3_1.txt","r")
fr2 = open("eval_3_2.txt","r")
fr4 = open("eval_3_4.txt","r")
fr5 = open("eval_3_5.txt","r")

fr1_lines = fr1.read().splitlines()
fr2_lines = fr2.read().splitlines()
fr4_lines = fr4.read().splitlines()
fr5_lines = fr5.read().splitlines()

fr1.close()
fr2.close()
fr4.close()
fr5.close()

visualize_info = dict()

def parse_to_dict(lines, key, res_dict):
    iters = []
    fids = []
    for line in lines:
        iter_num, fid = line.split(",")
        iter_num = int(iter_num)
        fid = float(fid)
        fids.append(fid)
        iters.append(iter_num)
    res_dict[key] = {k:{"fid":v} for k,v in zip(iters, fids)}

parse_to_dict(fr1_lines, "3_1", visualize_info)
parse_to_dict(fr2_lines, "3_2", visualize_info)
parse_to_dict(fr4_lines, "3_4", visualize_info)
parse_to_dict(fr5_lines, "3_5", visualize_info)


fr1_lines = open("is_3_1.txt","r").read().splitlines()
fr2_lines = open("is_3_2.txt","r").read().splitlines()
# fr4_lines = open("is_3_4.txt","r").read().splitlines()
fr5_lines = open("is_3_5.txt","r").read().splitlines()

def parse_inception_data(lines, key, res_dict):
    for line in lines:
        iter_num, data = line.split(":")
        iter_num = int(iter_num)
        is_score = float(data.split(",")[0][1:])
        res_dict[key][iter_num]["is"] = is_score
parse_inception_data(fr1_lines, "3_1", visualize_info)
parse_inception_data(fr2_lines, "3_2", visualize_info)
# parse_inception_data(fr4_lines, "3_4", visualize_info)
parse_inception_data(fr5_lines, "3_5", visualize_info)
print(visualize_info)
X = [1000*i for i in range(1,51)]
# y_1_is = [visualize_info['3_1'][i]['is'] for i in X]
# y_2_is = [visualize_info['3_2'][i]['is'] for i in X]
# y_5_is = [visualize_info['3_5'][i]['is'] for i in X]
# plt.plot(X,y_1_is,label="is, normal training")
# plt.plot(X,y_2_is,label="is, secure D training")
# plt.plot(X,y_5_is,label="is, 1-layer secure D training")
# plt.legend()
# plt.xlabel("Iteration")
# plt.ylabel("Inception Score")

y_1_fid = [visualize_info['3_1'][i]['fid'] for i in X]
y_2_fid = [visualize_info['3_2'][i]['fid'] for i in X]
y_4_fid = [visualize_info['3_4'][i]['fid'] for i in X]
y_5_fid = [visualize_info['3_5'][i]['fid'] for i in X]
plt.plot(X,y_1_fid,label="fid, normal training")
plt.plot(X,y_2_fid,label="fid, secure D training")
plt.plot(X,y_5_fid,label="fid, 1-layer secure D training")
plt.plot(X,y_4_fid,label="fid, 2-layer secure D training")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("FID")

plt.show()