import numpy as np


i1, i2 = 0.05, 0.10
t1, t2 = 0.01, 0.99

w1, w2 = 0.15, 0.20   
w3, w4 = 0.25, 0.30   
w5, w6 = 0.40, 0.45   
w7, w8 = 0.50, 0.55   

b1, b2 = 0.35, 0.60
eta = 0.5

# ============================================================
# Activation Function
# ============================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(out):
    return out * (1 - out)

# FORWARD PASS

# Hidden Layer
net_h1 = i1*w1 + i2*w2 + b1
net_h2 = i1*w3 + i2*w4 + b1
out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

# Output Layer
net_o1 = out_h1*w5 + out_h2*w6 + b2
net_o2 = out_h1*w7 + out_h2*w8 + b2
out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

print("=" * 50)
print("FORWARD PASS")
print("=" * 50)
print(f"net_h1 = {net_h1:.10f}")
print(f"net_h2 = {net_h2:.10f}")
print(f"out_h1 = {out_h1:.10f}")
print(f"out_h2 = {out_h2:.10f}")
print(f"net_o1 = {net_o1:.10f}")
print(f"net_o2 = {net_o2:.10f}")
print(f"out_o1 = {out_o1:.10f}")
print(f"out_o2 = {out_o2:.10f}")

# ERROR CAL
E_o1 = 0.5 * (t1 - out_o1)**2
E_o2 = 0.5 * (t2 - out_o2)**2
E_total = E_o1 + E_o2

print("\n" + "=" * 50)
print("ERROR CALCULATION")
print("=" * 50)
print(f"E_o1   = {E_o1:.10f}")
print(f"E_o2   = {E_o2:.10f}")
print(f"E_total= {E_total:.10f}")

# BACKWARD PASS — Output Layer

# Gradients for o1
dE_dout_o1   = -(t1 - out_o1)
dout_o1_dnet = sigmoid_deriv(out_o1)

dE_dout_o2   = -(t2 - out_o2)
dout_o2_dnet = sigmoid_deriv(out_o2)

dE_dnet_o1 = dE_dout_o1 * dout_o1_dnet
dE_dnet_o2 = dE_dout_o2 * dout_o2_dnet

# Update w5, w6, w7, w8
dE_dw5 = dE_dnet_o1 * out_h1
dE_dw6 = dE_dnet_o1 * out_h2
dE_dw7 = dE_dnet_o2 * out_h1
dE_dw8 = dE_dnet_o2 * out_h2

w5_new = w5 - eta * dE_dw5
w6_new = w6 - eta * dE_dw6
w7_new = w7 - eta * dE_dw7
w8_new = w8 - eta * dE_dw8

print("\n" + "=" * 50)
print("BACKWARD PASS — Output Layer")
print("=" * 50)
print(f"dE/dw5 = {dE_dw5:.10f}  =>  w5_new = {w5_new:.10f}")
print(f"dE/dw6 = {dE_dw6:.10f}  =>  w6_new = {w6_new:.10f}")
print(f"dE/dw7 = {dE_dw7:.10f}  =>  w7_new = {w7_new:.10f}")
print(f"dE/dw8 = {dE_dw8:.10f}  =>  w8_new = {w8_new:.10f}")

# BACKWARD PASS — Hidden Layer

# dE_total/dout_h1 = contribution from o1 + contribution from o2
dE_dout_h1 = (dE_dnet_o1 * w5) + (dE_dnet_o2 * w7)
dout_h1_dnet = sigmoid_deriv(out_h1)

dE_dout_h2 = (dE_dnet_o1 * w6) + (dE_dnet_o2 * w8)
dout_h2_dnet = sigmoid_deriv(out_h2)

dE_dw1 = dE_dout_h1 * dout_h1_dnet * i1
dE_dw2 = dE_dout_h1 * dout_h1_dnet * i2
dE_dw3 = dE_dout_h2 * dout_h2_dnet * i1
dE_dw4 = dE_dout_h2 * dout_h2_dnet * i2

w1_new = w1 - eta * dE_dw1
w2_new = w2 - eta * dE_dw2
w3_new = w3 - eta * dE_dw3
w4_new = w4 - eta * dE_dw4

print("\n" + "=" * 50)
print("BACKWARD PASS — Hidden Layer")
print("=" * 50)
print(f"dE/dw1 = {dE_dw1:.10f}  =>  w1_new = {w1_new:.10f}")
print(f"dE/dw2 = {dE_dw2:.10f}  =>  w2_new = {w2_new:.10f}")
print(f"dE/dw3 = {dE_dw3:.10f}  =>  w3_new = {w3_new:.10f}")
print(f"dE/dw4 = {dE_dw4:.10f}  =>  w4_new = {w4_new:.10f}")

# SUMMARY
print("\n" + "=" * 50)
print("SUMMARY — Updated Weights")
print("=" * 50)
print(f"w1: 0.15  =>  {w1_new:.9f}")
print(f"w2: 0.20  =>  {w2_new:.9f}")
print(f"w3: 0.25  =>  {w3_new:.9f}")
print(f"w4: 0.30  =>  {w4_new:.9f}")
print(f"w5: 0.40  =>  {w5_new:.9f}")
print(f"w6: 0.45  =>  {w6_new:.9f}")
print(f"w7: 0.50  =>  {w7_new:.9f}")
print(f"w8: 0.55  =>  {w8_new:.9f}")