#generates and prints all the coefficients for our model

alpha_L = 0.0024
alpha_T = 0.00024
KH = 2.54e-7
nu_w = 5.48e-4 

def dae(theta_a):
	return 8.44e-3 * (theta_a**1.5)/0.435

def dwe(theta_w):
	return 6.13e-5 * (theta_w - 0.04)*theta_w

def R(theta_a,theta_w):
	return theta_w+KH*theta_a

def Dr(theta_a,theta_w):
	return (theta_w*dwe(theta_w)+theta_w*alpha_T*nu_w+KH*theta_a*dae(theta_a))/R(theta_a,theta_w)
def Dz(theta_a,theta_w):
	return (theta_w*dwe(theta_w)+theta_w*alpha_L*nu_w+KH*theta_a*dae(theta_a))/R(theta_a,theta_w)
def D(theta_a,theta_w):
	return [[Dr(theta_a,theta_w),0,0],[0,Dr(theta_a,theta_w),0],[0,0,Dz(theta_a,theta_w)]]

def ve(theta_a,theta_w):
	return nu_w/R(theta_a,theta_w)

print(D(0.3,0.435-0.3))
print(ve(0.3,0.435-0.3))
