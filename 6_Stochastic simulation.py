import numpy as np
import matplotlib.pyplot as plt

n=300
p=20
p1=10
sigma_x=0.2
rho_x=0
sigma_y=3
x_0=np.ones(21)/20
x_0[0]=1
x_0=x_0.reshape(21,1)
M=5000
beta = np.vstack((np.ones(1+p1).reshape(1+p1,1),np.zeros(p-p1).reshape(p-p1,1)))

# def gen_x():
#     mean = np.zeros(p)
#     cov = sigma_x*np.identity(p)
#     x = np.random.multivariate_normal(mean, cov)
#     i = 0
#     while(i<n-1):
#         mean = np.zeros(p)
#         cov = sigma_x*np.identity(p)
#         x_v = np.random.multivariate_normal(mean, cov)
#         x = np.vstack((x,x_v))
#         i = i + 1
#     x_1 = np.ones(n).reshape(n,1)
#     x = np.hstack((x_1,x))
#     return x
def gen_x():
    x = np.ones(n)
    for i in range(p):
        x = np.c_[x,np.random.normal(0,sigma_x,n)]
    return x

def gen_y(x):
    epsilon=np.random.normal(0, sigma_y, size=(n,1))
    y = np.dot(x,beta)+epsilon
    return y

def estimate(x,y,k):
    x_k = x[:,:(k+1)]
    beta_hat_k = np.dot(np.dot(np.linalg.inv(np.dot(x_k.T,x_k)),x_k.T),y)
    beta_hat_k = np.vstack((beta_hat_k,np.zeros(p-k).reshape(p-k,1)))
    y_0_hat_k = np.dot(x_0.T,beta_hat_k)[0,0]
    return  y_0_hat_k

bias_li=[]
var_li=[]
mse_li=[]

y_0_e = np.dot(x_0.T,beta)[0,0]

for k in range(1,p+1):
    print(k)
    y_0_hat_k_li = []
    # print(y_0_e)
    for m in range(M):
        # print(m)
        x = gen_x()
        y = gen_y(x)
        y_0_hat_k = estimate(x,y,k)
        # print(y_0_hat_k)
        y_0_hat_k_li.append(y_0_hat_k)
    bias = (np.mean(y_0_hat_k_li)-y_0_e)**2
    # print(bias)
    var = np.mean(np.square(y_0_hat_k_li - np.mean(y_0_hat_k_li)))
    # print(var)
    mse = np.mean(np.square(y_0_hat_k_li - y_0_e))
    # print(mse)
    bias_li.append(bias)
    var_li.append(var)
    mse_li.append(mse)

k_li = np.array(range(1,21))
# print(k_li.shape)
# print(bias_li)
# print(np.array(bias_li).shape)
argm = np.argmin(np.array(mse_li)) + 1
print(argm)
mi = np.min(np.array(mse_li))
ar_mi = []
ar_mi.append(argm)
ar_mi.append(mi)
armi = tuple(ar_mi)
plt.plot(k_li,np.array(bias_li), color = 'red', label = 'bias')
plt.plot(k_li,np.array(var_li), color = 'green', label = 'var')
plt.plot(k_li,np.array(mse_li), color = 'blue',label = 'mse')
plt.annotate("(%s,%s)"%armi, armi, xycoords='data', xytext=(10, 0.01), arrowprops=dict(arrowstyle='->')) 
plt.hlines(mi, 0, argm, colors='y', linestyle="--")
plt.vlines(argm, 0, mi, colors='y', linestyle="--")
plt.xlabel('k')
plt.ylabel('value')
plt.legend(['bias', 'var', 'mse'])
plt.show()






