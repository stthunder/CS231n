# # Author : Sunita Nayak, Big Vision LLC
#
# #### Usage example: python3 downloadOI.py --classes 'Ice_cream,Cookie' --mode train
#
# import argparse
# import csv
# import subprocess
# import os
# from tqdm import tqdm
# import multiprocessing
# from multiprocessing import Pool as thread_pool
#
# cpu_count = multiprocessing.cpu_count()
#
# parser = argparse.ArgumentParser(description='Download Class specific images from OpenImagesV4')
# parser.add_argument("--mode", help="Dataset category - train, validation or test", required=True)
# parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
# parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count * 2)
# parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
# parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
# parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
# parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
# parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)
#
# args = parser.parse_args()
#
# run_mode = args.mode
#
# threads = args.nthreads
#
# classes = []
# for class_name in args.classes.split(','):
#     classes.append(class_name)
#
# with open('./class-descriptions-boxable.csv', mode='r') as infile:
#     reader = csv.reader(infile)
#     dict_list = {rows[1]: rows[0] for rows in reader}
#
# subprocess.run(['rm', '-rf', 'labels'])
# subprocess.run(['mkdir', 'labels'])
#
# subprocess.run(['rm', '-rf', 'JPEGImages'])
# subprocess.run(['mkdir', 'JPEGImages'])
#
# pool = thread_pool(threads)
# commands = []
# cnt = 0
#
# for ind in range(0, len(classes)):
#
#     class_name = classes[ind]
#     print("Class " + str(ind) + " : " + class_name)
#
#     subprocess.run(['mkdir', run_mode + '/' + class_name])
#
#     command = "grep " + dict_list[class_name.replace('_', ' ')] + " ./" + run_mode + "-annotations-bbox.csv"
#     class_annotations = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
#     class_annotations = class_annotations.splitlines()
#
#     for line in class_annotations:
#
#         line_parts = line.split(',')
#
#         # IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
#         if (args.occluded == 0 and int(line_parts[8]) > 0):
#             print("Skipped %s", line_parts[0])
#             continue
#         if (args.truncated == 0 and int(line_parts[9]) > 0):
#             print("Skipped %s", line_parts[0])
#             continue
#         if (args.groupOf == 0 and int(line_parts[10]) > 0):
#             print("Skipped %s", line_parts[0])
#             continue
#         if (args.depiction == 0 and int(line_parts[11]) > 0):
#             print("Skipped %s", line_parts[0])
#             continue
#         if (args.inside == 0 and int(line_parts[12]) > 0):
#             print("Skipped %s", line_parts[0])
#             continue
#
#         cnt = cnt + 1
#
#         command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + run_mode + '/' + \
#                   line_parts[0] + '.jpg ' + 'JPEGImages' + '/' + class_name + '/' + line_parts[0] + '.jpg'
#         commands.append(command)
#
#         with open('labels/%s.txt' % (line_parts[0]), 'a') as f:
#             f.write(' '.join([str(ind), str((float(line_parts[5]) + float(line_parts[4])) / 2),
#                               str((float(line_parts[7]) + float(line_parts[6])) / 2),
#                               str(float(line_parts[5]) - float(line_parts[4])),
#                               str(float(line_parts[7]) - float(line_parts[6]))]) + '\n')
#
# print("Annotation Count : " + str(cnt))
# commands = list(set(commands))
# print("Number of images to be downloaded : " + str(len(commands)))
#
# list(tqdm(pool.imap(os.system, commands), total=len(commands)))
#
# pool.close()
# pool.join()
# s = range(6)
# # print(s)
import numpy as np
# a = np.array([[1,2,3,4],[5,7,8,9]])
# X = np.array([[1,2],[4,5]])
# a[:,1] = X[0].T
# print(a[:,1])
# c = np.zeros((4,4))
# b = list(enumerate(a.tolist()))
# d = (0,0)
# print(a[d])
# print(a[0,0])

# d = c[b[1,:]]
# print(d)
# learning_rates = [1e-7, 5e-7]
# result = {}
# result[learning_rates[1],learning_rates[0]]=1
# print(result)
# print(len(learning_rates))
# X = np.array([[1,2,3,5],[3,4,6,8],[4,6,7,8]])
# X = X.reshape(((X.shape[0])*(X.shape[1])),1)
# # y = np.max(X,axis=1)
# # y = y.reshape((3,1))
# y = np.tile(X,(1,5))
# y = np.reshape(y,(3,4,5))
# print(y)
# y = np.sum(y,axis =2)
# print(y)
# y = np.reshape(y,(12,1))
# # # y =
# # y = np.sum(X,axis=1)
# # X = np.tile(X,(1,4))
# J = np.argmax(X,axis=1)
# print(y)


import numpy as np

# A = np.array([[1,2,3],[-1,-1,5],[4,5,5],[5,8,9]])
# B = np.array([[0,0],[0,0]])
# # temp = A > 0
# # a = np.argwhere(A>0)
# # print(a)
# # j  = a[:,0]
# # jj = a[:,1]
# # A[j,jj]= 5
# # A[a[1]] =5
# # c = np.max(A,B)actual
# # actual  = np.arange(2)
# print(np.sum(A,axis =1))
# A = np.random.rand(4,5)
# X_mean = np.mean(A,axis=0)
# X_mean = np.std(A,axis=0)
# bn_params = [{'mode': 'train'} for i in range(3)]
# bn_params[1] = 1
# X_mean = bn_params[1]
# A[1] = str(1)
# X = np.array([[[[1,2],[3,6]],[[4,5],[7,9]],[[3,7],[4,8]]],[[[1,2],[3,6]],[[4,5],[7,9]],[[3,7],[4,8]]]])
# print(X)
# X = np.array([[1,2],[3,4]])
# print(X[0:1,0:2])

# x = np.array([[[[7 4 3 1]
#                 [2 1 3 3]
#                 [3 1 2 3]
#                 [2 7 6 8]]
#
#                 [[6 5 8 1]
#                 [9 5 8 2]
#    [6 8 7 9]
#    [7 2 5 1]]
#
#   [[8 7 6 9]
#    [9 1 9 3]
#    [9 8 9 1]
#    [2 5 5 2]]]
#
#
#  [[[7 6 8 3]
#    [7 8 2 3]
#    [6 6 8 7]
#    [1 3 2 6]]
#
#   [[4 5 9 7]
#    [7 5 5 3]
#    [6 5 9 4]
#    [5 3 9 5]]
#
#   [[1 5 5 8]
#    [9 6 3 6]
#    [4 5 8 6]
#    [8 3 4 9]]]]
# )
# np.random.seed(0)
# x = np.random.randint(1,10,(2,3,2,2))
# # print(x)
# np.random.seed(1)
# w = np.random.randint(1,10,(1, 3, 2, 2))
# # print(w)
# # print('over')
# # X = X[:,1,0:1,0:1]
# stride, pad = 2, 1
#
# # print(x)
# # print(w)
# # print(w)
# X_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant')
# N, C, H, W = x.shape
# F, C, HH, WW = w.shape
#
# # print(W)
# HH_r = int(1 + (H + 2 * pad - HH) / stride)
# WW_r = int(1 + (H + 2 * pad - WW) / stride)
#
# W = X_pad.shape[3]
# H = X_pad.shape[2]
# # print(WW_r)
# # print(H,HH_r)
# out = np.zeros([N, F, HH_r, WW_r])
# for i in range(0, W - WW + 1, stride):
#     for j in range(0, H - HH + 1, stride):
#         for k in range(F):
#             M = X_pad[:, :, i:i + HH, j:j + WW] * w[k]
#             W = X_pad[:, :, i:i + HH, j:j + WW]
#             M = np.sum(M, axis=1)
#             M = np.sum(M, axis=2)
#             M = np.sum(M, axis=1)
#             out[:, k, int(j / stride), int(i / stride)] = M


# print(out)
# print(out)
# print(out)
# print(np.sum(X,axis=1))


# np.pad
# print(X.shape)
# mean = np.mean(X,axis=1)
# std  = np.std(X,axis=1)
# N = X.shape[0]
# mean = np.reshape(mean,(2,1))
# std = np.reshape(std,(2,1))
# X = X-mean
# print(X)
# ans = (N-1)*(std-(X)*np.sqrt(std))/(np.square(std))
# print(ans)

# np.random.seed(0)
# jvzhen = np.random.randint(1,10,(4,3,5,5))
# print(jvzhen)
# jvzhen = jvzhen.transpose(1,0,2,3)
# jjj    = jvzhen.transpose(0,2,3,1)
# print('////////////////////////////////////////////////')
# print(jvzhen)
# print()
# jvzhen = jvzhen.reshape(-1,3).T
# # jvzhen = jvzhen.T
# print(jvzhen)
# print(jvzhen - jjj.reshape(3,-1))
import torch
X = torch.randint(0,10,(3,2,3,3))
#
X = torch.reshape(X,(X.shape[0],X.shape[1],-1))
# N = W[X]
# print(N.shape)
print(X)
X_1 = X.permute((0,2,1))
print(X_1)
X_use = np.zeros((X.shape[0],X.shape[1],X.shape[1]))
for i in range(3):
    print(X_1[i].shape)
    print(X[i].shape)
    X_use[i]=torch.mm(X[i],X_1[i])



# print(torch.sum(X,axis = (0,2,3)))
# np.add.at(W,X[:],N[:])
# print(W)